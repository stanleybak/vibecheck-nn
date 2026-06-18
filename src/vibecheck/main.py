"""CLI entry point for zonotope-based neural network verification."""

import argparse
import sys
import time

import numpy as np

from .network import ComputeGraph
from .vnnlib_loader import load_vnnlib
from .verify import zonotope_verify

_DTYPES = {'float32': np.float32, 'float64': np.float64,
           'f32': np.float32, 'f64': np.float64}


def main():
    parser = argparse.ArgumentParser(
        description='VibeCheck — Neural Network Verification via Zonotope Analysis')
    parser.add_argument('--net', required=True, help='Path to ONNX network')
    parser.add_argument('--spec', required=True, help='Path to VNNLIB specification')
    parser.add_argument('--dtype', default='float32', choices=list(_DTYPES),
                        help='Computation dtype (default: float32)')
    parser.add_argument('--mode', default='graph',
                        choices=['zonotope', 'bnb', 'milp', 'graph'],
                        help='Verification mode (default: graph)')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                        help='Device for BnB mode (default: gpu)')
    parser.add_argument('--bits', type=int, default=32, choices=[16, 32, 64],
                        help='Float precision for BnB mode (default: 32)')
    parser.add_argument('--bnb-order', default='bfs', choices=['bfs', 'dfs'],
                        help='BnB search order (default: bfs)')
    parser.add_argument('--timeout', type=float, default=30,
                        help='BnB timeout in seconds (default: 30)')
    parser.add_argument('--pgd-restarts', type=int, default=100,
                        help='PGD restarts for BnB (default: 100)')
    parser.add_argument('--disable-sat-finding', action='store_true',
                        help='Disable all counterexample search (PGD, MILP '
                             'witness). Soundness probe: on a SAT case the '
                             'verdict can then only come from the bounds/MILP '
                             "path, which must never return 'unsat'.")
    parser.add_argument('--config', default=None,
                        help='Per-benchmark YAML overrides on top of '
                             'default_settings(). When set, overrides take '
                             'precedence over CLI knobs; when omitted, '
                             'default_settings_for(graph, spec) auto-detects '
                             'a profile.')
    parser.add_argument('--results-file', default=None,
                        help='If set, write a single VNNCOMP-style line to '
                             "this file: 'unsat' (verified), 'sat' "
                             "(counterexample), 'unknown', or 'timeout'. "
                             'Sweep scripts MUST check this file rather than '
                             'inferring from exit code, so a no-op invocation '
                             "cannot masquerade as 'verified'.")
    parser.add_argument('--verbose', action='store_true',
                        help='Enable per-phase progress output (sets '
                             'print_progress). Loaded file paths, the config '
                             'used, the verdict, and a phase-timing summary '
                             'print regardless; --verbose adds the per-phase '
                             'blow-by-blow. Useful for competition logs.')
    parser.add_argument('--heartbeat', type=float, default=0.0,
                        help='Print a [heartbeat] line every N seconds with '
                             'the current phase, its in-phase elapsed time, '
                             'and GPU memory. 0 = off. Pinpoints a phase that '
                             'hangs (its end-of-phase print never fires).')
    parser.add_argument('--allow-unsafe-pkl-loading', action='store_true',
                        help='Allow loading a pre-parsed graph/spec from a '
                             'sidecar .pkl cache (written by prepare_instance.sh). '
                             'pickle is unsafe (arbitrary code execution on load), '
                             'so this is OFF by default; only enable for inputs '
                             'you produced yourself.')
    parser.add_argument('--write-pkl', action='store_true',
                        help='Parse --net/--spec into a pre-parse .pkl cache '
                             '(deterministic path keyed by the onnx/vnnlib/dtype) '
                             'and exit WITHOUT verifying. Used by '
                             'prepare_instance.sh; the timed run then loads it '
                             'via --allow-unsafe-pkl-loading.')
    args = parser.parse_args()

    if args.verbose:
        # Line-buffer stdout so per-phase progress flushes on every newline.
        # Without this, a redirected stdout is block-buffered and a hung/
        # SIGKILL'd run (e.g. a timed-out MILP build) loses its entire
        # buffered log → an undiagnosable empty file. The flushed marker
        # below also confirms logging is live before any heavy work starts.
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except AttributeError:   # pre-3.7 / non-TextIOWrapper stdout
            pass
        print('[verbose] line-buffered logging enabled', flush=True)

    if args.write_pkl:
        # Prepare step: parse + cache, no verification.
        from .preparse import write_cache
        path = write_cache(args.net, args.spec, _DTYPES[args.dtype])
        print(f'Wrote pre-parse cache: {path}')
        sys.exit(0)

    # Shared across _verify and this crash handler: tracks whether a 'sat'
    # (+counterexample) was already written to the results file, so a later
    # crash/timeout can't clobber a found counterexample.
    sat_state = {'emitted': False}
    # Pre-seed the results file with 'timeout': if the process is HARD-KILLED
    # for overrunning its budget (no clean exit, no Python exception — e.g. the
    # harness SIGKILLs it), this leaves the correct verdict behind instead of a
    # missing file (which a sweep aggregator counts as NO_FILE / not-solved). A
    # clean finish overwrites it with the real verdict; a crash overwrites it
    # with 'error'; an early within-tol counterexample overwrites it with 'sat'.
    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write('timeout\n')
    try:
        _verify(args, sat_state)
    except SystemExit:
        raise
    except BaseException:
        # Sweep aggregator requires a results-file on every run; a crash
        # without one shows up as NO_FILE and breaks the verdict count.
        # Write 'error' (NOT 'unknown'): a crash — unloadable/corrupt onnx,
        # unsupported op, an actual bug — is fundamentally different from a
        # sound verifier that ran and could not decide ('unknown'). Masking
        # crashes as 'unknown' hid a corrupt benchmark file (23 nn4sys
        # mscn_2048d_dual instances that *looked* like legitimate give-ups but
        # were really a failed ONNX load). The scorer treats 'error' as
        # not-solved, same as 'unknown' (no penalty), so this only adds
        # diagnosability. The traceback above carries the cause.
        import traceback
        traceback.print_exc()
        # Don't clobber a counterexample we already wrote (early within-tol /
        # real-CE write) with 'error' on a late crash.
        if args.results_file and not sat_state['emitted']:
            with open(args.results_file, 'w') as f:
                f.write('error\n')
        sys.exit(2)


def _verify(args, sat_state=None):
    dtype = _DTYPES[args.dtype]
    t_start = time.time()
    if sat_state is None:
        sat_state = {'emitted': False}

    # Fast path: load the pre-parsed graph+spec from prepare_instance.sh's
    # cache, skipping the (potentially multi-second) ONNX parse. Gated behind
    # the explicit unsafe-pkl flag; falls back to a normal parse on any miss.
    graph = spec = None
    if args.allow_unsafe_pkl_loading:
        from .preparse import load_cache, pkl_cache_path
        cached = load_cache(args.net, args.spec, dtype)
        if cached is not None:
            graph, spec = cached
            print(f'Loaded pre-parse cache: '
                  f'{pkl_cache_path(args.net, args.spec, dtype)}')
            print(f'  network: {args.net}')
            print(f'  spec:    {args.spec}')

    if graph is None:
        print(f'Loading network: {args.net}')
        graph = ComputeGraph.from_onnx(args.net, dtype=dtype)
    n_relu = len(graph.relu_nodes())
    forks = graph.fork_points()
    print(f'  {len(graph.nodes)} ops, {n_relu} ReLU layers, '
          f'{len(forks)} fork points, input shape: {graph.input_shape}')

    if spec is None:
        print(f'Loading spec: {args.spec}')
        spec = load_vnnlib(args.spec)
    print(f'  {spec.n_constraints} constraint(s), '
          f'{len(spec.disjuncts)} disjunct(s)')

    from . import heartbeat as _hb
    if args.heartbeat and args.heartbeat > 0:
        _hb.start(float(args.heartbeat))

    if args.mode == 'bnb':
        from .settings import default_settings
        from .verify_zono_bnb import zonotope_bnb_verify
        settings = default_settings(
            device=args.device,
            bits=args.bits,
            bnb_order=args.bnb_order,
            bnb_timeout=args.timeout,
            pgd_restarts=args.pgd_restarts,
        )
        if args.verbose:
            settings.print_progress = True
        graph.optimize(settings)
        print(f'Running BnB verification (device={args.device}, '
              f'bits={args.bits}, order={args.bnb_order})...')
        result, details = zonotope_bnb_verify(graph, spec, settings)
    elif args.mode == 'milp':
        from .settings import default_settings
        from .verify_milp import milp_verify
        settings = default_settings(
            device=args.device,
            bits=args.bits,
            total_timeout=args.timeout,
            pgd_restarts=args.pgd_restarts,
        )
        if args.verbose:
            settings.print_progress = True
        graph.optimize(settings)
        print(f'Running MILP verification (device={args.device}, '
              f'timeout={args.timeout}s)...')
        result, details = milp_verify(graph, spec, settings)
    elif args.mode == 'graph':
        from .verify_graph import verify_graph
        if args.config is not None:
            # Explicit per-benchmark YAML: load → use as overrides on top of
            # default_settings(). CLI knobs (device/bits/timeout/...) apply
            # too, but YAML overrides win when there's a conflict.
            from .settings import default_settings
            from .config_loader import load_config
            yaml_overrides = load_config(args.config)
            cli_overrides = dict(
                device=args.device, bits=args.bits,
                total_timeout=args.timeout, pgd_restarts=args.pgd_restarts)
            cli_overrides.update(yaml_overrides)
            if args.disable_sat_finding:  # CLI soundness probe wins over YAML
                cli_overrides['disable_sat_finding'] = True
            settings = default_settings(**cli_overrides)
            settings._profile = f'config:{args.config}'
        else:
            from .config_profiles import default_settings_for
            settings = default_settings_for(
                graph, spec,
                device=args.device,
                bits=args.bits,
                total_timeout=args.timeout,
                pgd_restarts=args.pgd_restarts,
            )
            if args.disable_sat_finding:
                settings.disable_sat_finding = True
        if args.verbose:
            settings.print_progress = True
        # SAT/counterexample policy: the pipeline may find a *within-tolerance*
        # counterexample (a near-boundary point that violates only within the
        # VNNCOMP atol) before it finds a clear one or proves unsat. Hand it a
        # sink that writes that CE to the results file IMMEDIATELY (so a later
        # timeout/crash can't lose it) without committing the run — the search
        # continues for a clear CE or an unsat proof. `_emit_result`'s
        # never-downgrade rule keeps that early 'sat' unless a later 'sat'
        # (clearer CE) or 'unsat' (proof — the near-miss was spurious) overrides.
        settings.sat_tol = float(
            settings.sat_validate_atol
            if 'sat_validate_atol' in settings else 1e-4)
        settings.result_sink = (lambda w: _emit_result(
            args, spec, 'sat', w, sat_state)) if args.results_file else None
        graph.optimize(settings)
        print(f'Running graph verification (device={args.device}, '
              f'impl={settings.graph_impl}, profile={settings._profile}, '
              f'timeout={args.timeout}s)...')
        result, details = verify_graph(graph, spec, settings)
    else:
        print('Running zonotope analysis...')
        result, details = zonotope_verify(graph, spec)

    _hb.stop()
    t_total = time.time() - t_start

    print(f'\nResult: {result}')
    if 'worst_margin' in details:
        print(f'  Worst margin: {details["worst_margin"]:.6f}')
        for i, margin in details['margins'].items():
            status = 'SAFE' if margin > 0 else 'UNKNOWN'
            print(f'  Disjunct {i}: margin={margin:.6f} [{status}]')
    if 'n_evals' in details:
        print(f'  BnB evals: {details["n_evals"]}')
    if 'volume_proven' in details:
        print(f'  Volume proven: {details["volume_proven"]:.1%}')
    if args.mode == 'graph':
        phase_timing = details.get('timing', {})
        if phase_timing:
            parts = [f'{k.replace("phase", "p").split("_", 1)[0]}={v:.2f}s'
                     if k.startswith('phase') else f'{k}={v:.2f}s'
                     for k, v in phase_timing.items()
                     if isinstance(v, (int, float))]
            print('  Timing: ' + '  '.join(parts))
        n_splits = details.get('n_splits', {})
        if n_splits:
            split_str = ' '.join(f'{k}:{v}' for k, v in n_splits.items() if v > 0)
            print(f'  Splits: {split_str}')
        if 'build_time_total' in details:
            print(f'  Build time total: {details["build_time_total"]:.2f}s')
        per_layer = details.get('per_layer_timing', {})
        if per_layer:
            print('  Per-layer tightening:')
            for name, t in per_layer.items():
                wz = t.get('width_zono', 0.0)
                wa = t.get('width_adapt', 0.0)
                wl = t.get('width_lp', 0.0)
                wf = t.get('width_final', 0.0)
                print(f'    {name}: '
                      f'build={t["build"]:.2f}s '
                      f'probe={t["probe"]:.2f}s '
                      f'solve={t["solve"]:.2f}s  '
                      f'width zono={wz:.3f}→adapt={wa:.3f}'
                      f'→lp={wl:.3f}→final={wf:.3f}')
    print(f'  Time: {t_total:.2f}s')

    if args.results_file:
        # VNNCOMP convention: file contents are the authoritative verdict.
        # 'unsat' = property holds; 'sat' = counterexample found; else
        # 'unknown'/'timeout'.
        verdict_map = {
            'verified': 'unsat',
            'sat': 'sat',
            'unknown': 'unknown',
            'timeout': 'timeout',
        }
        line = verdict_map.get(result, f'unknown ({result})')
        # VNNCOMP distinguishes 'timeout' (ran out of the wall budget without
        # deciding) from a give-up 'unknown'. The pipeline returns 'unknown'
        # on its time-budget-exhausted paths but flags details['timed_out'];
        # fall back to comparing elapsed wall against the budget for any path
        # that doesn't set the flag.
        timed_out = (isinstance(details, dict) and details.get('timed_out'))
        _w_final = (details.get('witness') if isinstance(details, dict)
                    else None)
        # UNIVERSAL SAT-validation chokepoint (matches the VNNCOMP scoring
        # step): before emitting `sat`, replay the witness through CPU
        # onnxruntime and confirm it violates the spec within
        # `sat_validate_atol`. verify_graph already validates internally at
        # `_finalize` (this re-check is idempotent), but the milp/bnb/hybrid
        # and conv→milp auto-route paths do NOT — this is their gate. Only a
        # witnessed `sat` that ORT rejects is downgraded (no false SAT, no
        # bogus CE file); witness-less results are left untouched. Skipped
        # only when sat-validation is explicitly disabled (soundness probes).
        if (line == 'sat' and _w_final is not None
                and not bool(getattr(settings, 'skip_sat_validation', False))):
            from .verify_graph import _validate_sat_witness
            _atol_f = float(settings.sat_validate_atol
                            if 'sat_validate_atol' in settings else 1e-4)
            _ok_f, _vinfo_f = _validate_sat_witness(
                getattr(graph, 'onnx_path', None), spec, _w_final, atol=_atol_f)
            if not _ok_f:
                print('  [validate] final SAT witness rejected by ORT '
                      f'({_vinfo_f.get("reason")}) — downgrading, not emitting SAT')
                line = 'timeout' if timed_out else 'unknown'
                _w_final = None
        if line == 'unknown' and (
                timed_out or (args.timeout is not None and t_total >= args.timeout - 2.0)):
            line = 'timeout'
        _emit_result(args, spec, line, _w_final, sat_state)

    sys.exit(0 if result == 'verified' else 1)


def _emit_result(args, spec, line, witness, sat_state):
    """Write the VNNCOMP results file: verdict on line 1, then (for 'sat') the
    counterexample s-expression. Never-downgrade rule: once a 'sat' (with a
    counterexample) has been written, a later 'timeout'/'unknown'/'error' will
    NOT overwrite it — a found counterexample is sticky against running out of
    budget. A later 'sat' (a clearer counterexample) or 'unsat' (a proof — the
    earlier within-tolerance near-miss was spurious) DOES override.

    Called both for the final verdict and, mid-run, by `settings.result_sink`
    to persist a within-tolerance counterexample early. Idempotent/repeatable.
    """
    if not args.results_file:
        return
    if sat_state.get('emitted') and line not in ('sat', 'unsat'):
        return  # keep the counterexample we already wrote
    ce = None
    if line == 'sat' and witness is not None:
        # VNNCOMP `sat` carries the counterexample: verdict on line 1, then an
        # s-expression `((X_0 <v>) ... (Y_0 <v>) ...)` over every input then
        # output dim. The harness re-runs the ONNX on the X's to confirm. We
        # reuse the same ORT forward the soundness validator uses so Y matches.
        ce = _counterexample_sexpr(args.net, spec, witness)
        if ce is None:
            print('  [warn] sat verdict but could not build a counterexample '
                  '(no ORT output); results file has the verdict only.')
    with open(args.results_file, 'w') as f:
        f.write(line + '\n')
        if ce is not None:
            f.write(ce + '\n')
    if line == 'sat':
        sat_state['emitted'] = True


def _counterexample_sexpr(onnx_path, spec, witness):
    """Build the VNNCOMP counterexample s-expression for a SAT witness.

    Returns `((X_0 <v>) ... (Y_0 <v>) ...)` (one atom per line) or None if the
    ONNX output can't be computed (e.g. onnxruntime missing). Y is obtained from
    the same ORT forward the soundness validator runs, so it matches the
    scoring harness's recomputed output within tolerance.
    """
    import numpy as np
    from .verify_graph import _validate_sat_witness
    x = np.asarray(witness).flatten().astype(np.float64)
    # _validate_sat_witness runs ORT and stashes the output in info['out'].
    _, info = _validate_sat_witness(onnx_path, spec, witness)
    y = info.get('out')
    if y is None:
        return None
    y = np.asarray(y).flatten().astype(np.float64)
    atoms = [f'(X_{i} {v:.17g})' for i, v in enumerate(x)]
    atoms += [f'(Y_{j} {v:.17g})' for j, v in enumerate(y)]
    return '(' + '\n'.join(atoms) + ')'


if __name__ == '__main__':
    main()
