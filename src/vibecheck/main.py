"""CLI entry point for zonotope-based neural network verification."""

import argparse
import os
import sys
import time

import numpy as np

from .network import ComputeGraph
from .vnnlib_loader import load_vnnlib
from .verify import zonotope_verify

_DTYPES = {'float32': np.float32, 'float64': np.float64,
           'f32': np.float32, 'f64': np.float64}


def _require_input_file(path, label):
    """Exit cleanly (code 1) with an informative message if an input file is
    missing. A missing/typo'd --net or --spec path is one of the most common
    mistakes, and a clear early error beats a deep stack trace. A `.gz` sibling
    counts as present (the benchmarks ship gzipped; the loaders decompress)."""
    if not path or not (os.path.isfile(path) or os.path.isfile(str(path) + '.gz')):
        print(f'Error: {label} file not found: {path!r}\n'
              f'       check the path for typos (a .gz of the same name is also '
              f'accepted).', file=sys.stderr)
        sys.exit(1)


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
    parser.add_argument('--set', action='append', default=[], dest='set_kv',
                        metavar='KEY=VALUE',
                        help='Override any single setting by name (repeatable). '
                             'VALUE is YAML-coerced, e.g. '
                             '--set phase8_fast_dual_ascent_K=2 '
                             '--set phase8_fast_dual_ascent_ls=subgrad '
                             '--set dump_bnb_dir=/tmp/dumps. Applied AFTER --config '
                             '(so --set wins); the key must exist in default_settings().')
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
    parser.add_argument('--prepare-pkl-unsafe', action='store_true',
                        help='Untimed prepare step (used by prepare_instance.sh), then '
                             'exit WITHOUT verifying. For a normal model: parse --net/'
                             '--spec into SEPARATE per-source .pkl caches (<onnx>.pkl and '
                             '<vnnlib>.pkl, keyed by content sha1) that the timed run loads '
                             'via --allow-unsafe-pkl-loading. For an INT8-quantized model '
                             '(DequantizeLinear/QuantizeLinear): fold the float (STE) + '
                             'fake-quant surrogates instead (that model uses the '
                             'surrogate-attack path, not the graph load). UNSAFE: the .pkl '
                             'is loaded via pickle (arbitrary code exec) — only use in a '
                             'trusted directory you control. -unsafe is in the name to '
                             'make that explicit.')
    args = parser.parse_args()

    # Parse --set KEY=VALUE overrides once (validated against default_settings());
    # applied to the built settings at every construction site below + in the
    # attack-mode paths. Stored on args so the attack helpers can see them too.
    from .config_loader import parse_set_overrides
    args.set_overrides = parse_set_overrides(args.set_kv)

    # Clean error for a missing --net (a path typo is the most common mistake).
    # The spec is checked later — the quantized prepare path doesn't use it.
    _require_input_file(args.net, 'network (--net)')

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
        # Gurobi license visibility: graph mode uses Gurobi LP/MILP, so flag
        # whether a real license file is installed vs the bundled size-limited
        # one. Cheap filesystem check only (no Env init) -- this runs per timed
        # instance under run_instance.sh's default --verbose.
        _lic = os.environ.get('GRB_LICENSE_FILE')
        _candidates = ([_lic] if _lic else []) + [
            '/opt/gurobi/gurobi.lic', os.path.expanduser('~/gurobi.lic')]
        _found = next((p for p in _candidates if p and os.path.isfile(p)), None)
        print(f'[verbose] gurobi license file: {_found}' if _found
              else '[verbose] gurobi license file: none found '
                   '(gurobipy bundled size-limited license)', flush=True)

    if args.prepare_pkl_unsafe:
        # Untimed prepare step: parse + cache, no verification. For a QUANTIZED model the
        # graph pre-parse fails on the dequantized (non-constant) conv kernel after ~70s —
        # that model uses the surrogate-attack path, not the graph load — so fold the
        # float (STE) + fake-quant surrogates here (the actual untimed prep) instead of a
        # cache that can't be used. Non-quantized: write the per-source pre-parse caches.
        from . import surrogate_pgd as sp
        if sp.has_quantized_ops(args.net):
            p = _surrogate_path(args.net)
            sp.build_float_surrogate(args.net, p)
            fq = (p[:-5] if p.endswith('.onnx') else p) + '_fq.onnx'
            sp.build_fakequant_surrogate(args.net, fq)
            print(f'Quantized model: built surrogates (skipped graph pre-parse): {p}, {fq}')
            sys.exit(0)
        _require_input_file(args.spec, 'spec (--spec)')   # non-quant prepare needs the spec
        from .preparse import write_cache
        onnx_pkl, vnnlib_pkl = write_cache(args.net, args.spec, _DTYPES[args.dtype])
        print(f'Wrote pre-parse caches:\n  graph: {onnx_pkl}\n  spec:  {vnnlib_pkl}')
        sys.exit(0)

    # Resolve the counterexample on-disk FORMAT version once, from the original spec
    # (BEFORE _verify rewrites args.spec for pair/augment) + the config's
    # `counterexample_format` ('auto' -> match the input vnnlib version). Both the graph
    # and surrogate emit paths read args.cex_version.
    _cf = 'auto'
    if args.config:
        from .config_loader import load_config
        _cf = load_config(args.config).get('counterexample_format', 'auto')
    try:
        args.cex_version = _resolve_cex_version(_cf, args.spec)
        # The spec-declared I/O names/dtypes/shapes for a v2 cex, resolved ONCE here so all
        # three emit paths use them (not the ONNX node names). See _resolve_cex_io_meta.
        args.cex_io_decls = _resolve_cex_io_meta(args.spec)
    except OSError:
        # 'auto' reads the spec head to detect its version; if it isn't readable yet (a dummy
        # path with a monkeypatched loader, or a not-yet-materialized file), fall back to v1
        # FORMAT. This is cosmetic only — a genuinely-missing spec still fails LOUDLY at the
        # verification load (graph/spec loaders), which the crash handler records as 'error'.
        args.cex_version = '1.0'
        args.cex_io_decls = None

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

    # Clean error for missing input files (typo'd path) before any heavy work.
    _require_input_file(args.net, 'network (--net)')
    _require_input_file(args.spec, 'spec (--spec)')

    # Network-pair benchmarks (isomorphic_acasxu / monotonic_acasxu): `--net` is a
    # pair list `[('f',a),('g',b)]` and `--spec` relates the two nets. Convert to a
    # single MERGED onnx + v1 spec up front, then verify normally. Must run BEFORE the
    # graph load. See network_pair.py (the merge is exact + onnxruntime-oracle-gated).
    _maybe_network_pair(args)

    # Nonlinear v2 specs (adaptive_cruise_control): transpile to an augmented
    # ONNX + linear v1 spec up front, then verify normally. Must run BEFORE the
    # graph load. See nonlinear_augment.py (transpile is ORT-oracle-gated).
    _maybe_nonlinear_augment(args)

    # Surrogate-attack mode (incomplete; INT8-quantized / unsupported ONNX). Must run
    # BEFORE the graph load below, which would fail on DequantizeLinear/QuantizeLinear.
    # Gated on an explicit config with surrogate_attack=True AND the ONNX having
    # quantized ops; emits the verdict and exits. See surrogate_pgd.py.
    _surr = _maybe_surrogate_attack(args, sat_state)
    if _surr is not None:
        sys.exit(_surr)

    # Sign-BNN attack mode (incomplete; binarized nets with `Sign` activations vibecheck can't
    # bound). Gated on config sign_attack=True AND the ONNX having `Sign` ops; emits the verdict
    # via the surrogate-attack emit path (same witness shape). See sign_attack.py.
    _sgn = _maybe_sign_attack(args, sat_state)
    if _sgn is not None:
        sys.exit(_sgn)

    # Generic onnx2torch PGD attack mode (incomplete; differentiable nets vibecheck can't bound
    # soundly/cheaply, e.g. collins_aerospace YOLOv5-nano). Gated on config torch_attack=True;
    # emits via the surrogate-attack emit path (same witness shape). See torch_attack.py.
    _tch = _maybe_torch_attack(args, sat_state)
    if _tch is not None:
        sys.exit(_tch)

    # cctsdb_yolo custom handler (COMPLETE; YOLO patch nets vibecheck/onnx2torch can't load).
    # Gated on config cctsdb_yolo=True; enumerates the integer patch-position grid through the
    # original net on ORT-CPU. Emits via the surrogate emit path. See cctsdb_yolo.py.
    _cct = _maybe_cctsdb_yolo(args, sat_state)
    if _cct is not None:
        sys.exit(_cct)

    # Fast path: load the pre-parsed graph and/or spec from prepare_instance.sh's
    # per-source caches, skipping the (potentially multi-second) ONNX parse. The
    # graph and spec are cached separately (keyed by content sha1), so each is
    # reused independently; falls back to a normal parse on any miss. Gated behind
    # the explicit unsafe-pkl flag.
    graph = spec = None
    if args.allow_unsafe_pkl_loading:
        from .preparse import load_cache
        graph, spec = load_cache(args.net, args.spec, dtype)
        if graph is not None:
            print(f'Loaded pre-parse graph cache for {os.path.basename(args.net)}')
        if spec is not None:
            print(f'Loaded pre-parse spec cache for {os.path.basename(args.spec)}')

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
        settings.update(args.set_overrides)
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
        settings.update(args.set_overrides)
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
        settings.update(args.set_overrides)   # --set wins over config/profile
        if args.verbose:
            settings.print_progress = True
        # SAT policy (VNN-COMP 2026 output-strict): VC only ever emits a `sat` for a
        # GENUINE counterexample (output strictly violates, input in-box), and it
        # returns immediately when it finds one — there is no within-output-tolerance
        # "emit early then keep searching" path any more, so no result_sink is needed.
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
        # onnxruntime and confirm it violates the spec. The 2026 rule applies
        # the 1e-4 tolerance only to the INPUT box (`sat_validate_atol`); the
        # replayed OUTPUT must violate with NO tolerance (`sat_validate_out_atol`,
        # default 0.0 = strict). verify_graph already validates internally at
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
            # Output tolerance is FIXED at 0.0 (VNN-COMP 2026 rule: the replayed
            # output must violate with NO tolerance) — NOT configurable, so a
            # config can never loosen it. Only the INPUT box gets `atol`.
            _ok_f, _vinfo_f = _validate_sat_witness(
                getattr(graph, 'onnx_path', None), spec, _w_final,
                atol=_atol_f, out_atol=0.0)
            if not _ok_f:
                print('  [validate] final SAT witness rejected by ORT '
                      f'({_vinfo_f.get("reason")}) — downgrading, not emitting SAT')
                line = 'timeout' if timed_out else 'unknown'
                _w_final = None
            else:
                if _vinfo_f.get('witness_inbox') is not None:
                    # Emit the float32-safe in-box witness (clamped strictly inside
                    # the box) — not the raw one whose edge can round outside.
                    _w_final = _vinfo_f['witness_inbox']
                if args.verbose:
                    _log_cex_values(_w_final, _vinfo_f.get('out'))
            # NOTE: VC's own `_validate_sat_witness` above (input box <=atol, output
            # STRICT at out_atol=0) is the production gate — it is ~4 ms even on a
            # 1.27M-dim spec, vs ~24 s for the vendored competition checker (which
            # Python-evaluates every input-bound assertion). We keep the vendored
            # `vnncomp_cex_v2` module only as a TEST ORACLE: tests assert VC's
            # verdict is bit-identical to the competition checker (see
            # tests/test_competition_cex_equiv.py), so we get the competition's
            # exact semantics without paying its per-assertion cost in the sweep.
        if line == 'unknown' and (
                timed_out or (args.timeout is not None and t_total >= args.timeout - 2.0)):
            line = 'timeout'
        _emit_result(args, spec, line, _w_final, sat_state,
                     settings.counterexample_precision)

    sys.exit(0 if result == 'verified' else 1)


def _maybe_network_pair(args):
    """If `--net` is a network-pair list-string (isomorphic/monotonic_acasxu),
    convert the pair to a single merged ONNX + v1 spec and rewrite args.net/args.spec
    in place so the normal pipeline verifies it. No-op otherwise."""
    from . import network_pair as npair
    if not npair.is_network_pair_net_field(args.net):
        return
    merged_onnx, merged_spec = npair.build_merged_instance(args.net, args.spec)
    print(f'Network-pair instance: merged {npair.detect_kind(npair._read_vnnlib_text(args.spec))} '
          f'pair -> {merged_onnx}')
    args.net = merged_onnx
    args.spec = merged_spec


def _maybe_nonlinear_augment(args):
    """If `--spec` is a NONLINEAR v2 spec (degree>=2 polynomial atoms / X*Y
    coupling, e.g. adaptive_cruise_control), transpile to an augmented ONNX (runs
    the original net f, then computes each constraint polynomial p_c(X,Y) as an
    extra output) + a LINEAR v1 DNF spec, and rewrite args.net/args.spec in place
    so the normal pipeline verifies it. ORT-oracle-gated (augmented output ==
    polynomial). No-op for linear specs. Must run BEFORE the graph load.

    Stashes the ORIGINAL net path on args so the emitted counterexample carries
    the original net's output Y (the augmented net's outputs are the constraint
    polynomials, a different dimension; the VNNCOMP scorer ignores the solver's Y
    but the cex must still have the original output shape)."""
    from . import nonlinear_augment as nla
    # This is a best-effort PRE-DETECTION of a nonlinear spec. Catch OSError (missing /
    # monkeypatched-dummy path) AND ValueError (e.g. `_read_vnnlib_text` opening a `.gz`
    # spec as text -> UnicodeDecodeError, a ValueError subclass) and SKIP: a genuinely-bad
    # spec is NOT swallowed — it still raises LOUDLY at the verification load (graph/spec
    # loaders, which DO handle .gz, or the surrogate `parse_box_and_output`), recorded as
    # 'error'. This only avoids a noisy pre-read; it never hides a verification error.
    try:
        text = nla._read_vnnlib_text(args.spec)
    except (OSError, ValueError):
        return
    if not nla.is_nonlinear_v2_spec(text):
        return
    aug_onnx, aug_spec = nla.build_augmented_instance(args.net, args.spec)
    print(f'Nonlinear v2 spec: augmented {args.net} -> {aug_onnx}')
    args.orig_net_for_cex = args.net
    # Preserve the ORIGINAL net+spec so the competition CE validator
    # (_validate_cex_v2_competition) replays/parses the real v2 instance, not the
    # augmented v1 net/spec we hand the internal pipeline.
    args.orig_spec_for_cex = args.spec
    args.net = aug_onnx
    args.spec = aug_spec


def _maybe_surrogate_attack(args, sat_state):
    """If surrogate-attack mode is enabled (config flag) AND the ONNX is quantized,
    run gradient-PGD via a continuous float surrogate, emit the verdict, and return a
    process exit code. Otherwise return None to fall through to normal verification.

    Incomplete/attack-only: returns only sat/timeout/unknown (never verified), with the
    counterexample validated on the ORIGINAL model via CPU onnxruntime."""
    if args.config is None:
        return None
    from .config_loader import load_config
    overrides = load_config(args.config)
    if not overrides.get('surrogate_attack', False):
        return None
    from . import surrogate_pgd as sp
    if not sp.has_quantized_ops(args.net):
        return None  # surrogate mode only engages for quantized models
    from .settings import default_settings
    overrides.setdefault('total_timeout', args.timeout)
    settings = default_settings(**overrides)
    settings.update(getattr(args, 'set_overrides', {}) or {})
    timeout = float(args.timeout if args.timeout else 100.0)
    print(f'Surrogate-attack mode: quantized ONNX -> PGD via float surrogate '
          f'(restarts={settings.surrogate_attack_restarts}, '
          f'steps={settings.surrogate_attack_steps}, timeout={timeout}s)')
    verdict, witness = sp.surrogate_attack(
        args.net, args.spec, settings, timeout,
        surrogate_path=_surrogate_path(args.net),
        log=(print if args.verbose else (lambda _m: None)))
    if args.results_file:
        _emit_surrogate_result(args, verdict, witness, sat_state,
                               settings.counterexample_precision)
    print(f'\nResult (surrogate-attack): {verdict}')
    return 1  # never 'verified' in this incomplete mode


def _maybe_sign_attack(args, sat_state):
    """If sign-attack mode is enabled (config flag) AND the ONNX has `Sign` ops (a BNN),
    run STE-PGD to find an adversarial counterexample, emit the verdict (reusing the
    surrogate emit path — same witness shape, ORT-validated on args.net), and return a
    process exit code. Otherwise return None to fall through to normal verification.
    Incomplete/attack-only: returns only sat/timeout/unknown (never verified)."""
    if args.config is None:
        return None
    from .config_loader import load_config
    overrides = load_config(args.config)
    if not overrides.get('sign_attack', False):
        return None
    from . import sign_attack as sa
    if not sa.has_sign_ops(args.net):
        return None  # sign mode only engages for binarized (Sign) nets
    from .settings import default_settings
    overrides.setdefault('total_timeout', args.timeout)
    settings = default_settings(**overrides)
    settings.update(getattr(args, 'set_overrides', {}) or {})
    timeout = float(args.timeout if args.timeout else 100.0)
    print(f'Sign-BNN attack mode: STE-PGD on Sign surrogate '
          f'(restarts={settings.sign_attack_restarts}, steps={settings.sign_attack_steps}, '
          f'timeout={timeout}s)')
    verdict, witness = sa.sign_attack(
        args.net, args.spec, settings, timeout,
        log=(print if args.verbose else (lambda _m: None)))
    if args.results_file:
        _emit_surrogate_result(args, verdict, witness, sat_state,
                               settings.counterexample_precision)
    print(f'\nResult (sign-attack): {verdict}')
    return 1  # never 'verified' in this incomplete mode


def _maybe_torch_attack(args, sat_state):
    """If generic torch-attack mode is enabled (config flag), run onnx2torch PGD to find an
    adversarial counterexample, emit the verdict (reusing the surrogate emit path — same witness
    shape, ORT-validated on args.net), and return a process exit code. Otherwise return None to
    fall through to normal verification. Incomplete/attack-only (never verified)."""
    if args.config is None:
        return None
    from .config_loader import load_config
    overrides = load_config(args.config)
    if not overrides.get('torch_attack', False):
        return None
    from . import torch_attack as ta
    from .settings import default_settings
    overrides.setdefault('total_timeout', args.timeout)
    settings = default_settings(**overrides)
    settings.update(getattr(args, 'set_overrides', {}) or {})
    timeout = float(args.timeout if args.timeout else 100.0)
    print(f'Torch-attack mode: onnx2torch PGD '
          f'(restarts={settings.torch_attack_restarts}, steps={settings.torch_attack_steps}, '
          f'timeout={timeout}s)')
    verdict, witness = ta.torch_attack(
        args.net, args.spec, settings, timeout,
        log=(print if args.verbose else (lambda _m: None)))
    if args.results_file:
        _emit_surrogate_result(args, verdict, witness, sat_state,
                               settings.counterexample_precision)
    print(f'\nResult (torch-attack): {verdict}')
    return 1  # never 'verified' in this incomplete mode


def _maybe_cctsdb_yolo(args, sat_state):
    """If cctsdb_yolo mode is enabled (config flag), verify the YOLO patch instance COMPLETELY
    by enumerating the integer patch-position grid through the original net on ORT-CPU, emit the
    verdict (surrogate emit path; sat witness ORT-validated on args.net), and return a process
    exit code. Otherwise None. Returns sat/unsat/timeout (complete: unsat = every placement safe)."""
    if args.config is None:
        return None
    from .config_loader import load_config
    overrides = load_config(args.config)
    if not overrides.get('cctsdb_yolo', False):
        return None
    from . import cctsdb_yolo as cy
    from .settings import default_settings
    overrides.setdefault('total_timeout', args.timeout)
    settings = default_settings(**overrides)
    settings.update(getattr(args, 'set_overrides', {}) or {})
    timeout = float(args.timeout if args.timeout else 100.0)
    print(f'cctsdb_yolo mode: discrete patch-position enumeration (timeout={timeout}s)')
    verdict, witness = cy.cctsdb_yolo_verify(
        args.net, args.spec, settings, timeout,
        log=(print if args.verbose else (lambda _m: None)))
    if args.results_file:
        _emit_surrogate_result(args, verdict, witness, sat_state,
                               settings.counterexample_precision)
    print(f'\nResult (cctsdb_yolo): {verdict}')
    return 0 if verdict == 'unsat' else 1


def _surrogate_path(onnx_path):
    """Deterministic path for the folded float surrogate (shared by prepare/run)."""
    import hashlib
    import tempfile
    h = hashlib.md5(os.path.abspath(onnx_path).encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f'vibecheck_surrogate_{h}.onnx')


def _cex_sexpr(x_flat, y_flat, fmt='.17g'):
    """Build the VNNLIB 1.0 counterexample s-expression `((X_0 v) ... (Y_0 v) ...)`
    from flattened input/output arrays. `fmt` is the per-value precision from the
    `counterexample_precision` setting (default '.17g', round-trips float64 losslessly)."""
    atoms = [f'(X_{i} {v:{fmt}})' for i, v in enumerate(x_flat)]
    atoms += [f'(Y_{j} {v:{fmt}})' for j, v in enumerate(y_flat)]
    return '(' + '\n'.join(atoms) + ')'


# TensorProto elem_type -> the dtype string a v2 counterexample writes (FLOAT/DOUBLE/FLOAT16).
_ONNX_DT = {1: 'float32', 11: 'float64', 10: 'float16'}


def _onnx_io_meta(onnx_path):
    """(inputs, outputs) — each a list of (name, dtype_str, shape, size) for the ONNX's free
    inputs and outputs in graph order: the per-tensor structure a v2 counterexample needs."""
    import numpy as np
    from . import surrogate_pgd as sp
    m = sp._load_onnx_model(onnx_path)
    init = {i.name for i in m.graph.initializer}

    def meta(vi):
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        return (vi.name, _ONNX_DT.get(vi.type.tensor_type.elem_type, 'float32'),
                shape, int(np.prod(shape)) if shape else 1)

    init_names = init
    ins = [meta(i) for i in m.graph.input if i.name not in init_names]
    outs = [meta(o) for o in m.graph.output]
    return ins, outs


def _cex_v2(ins_meta, outs_meta, x_flat, y_flat, fmt):
    """Build the VNNLIB 2.0 counterexample: per-tensor `NAME dtype [d0,d1,...]` header then
    the tensor's C-order values (one per line) — every input (the flat X split by input size)
    then every output (the flat Y split by output size). Mirrors the VNN-COMP 2026 v2 format."""
    lines = []
    for meta, flat in ((ins_meta, x_flat), (outs_meta, y_flat)):
        off = 0
        for name, dt, shape, size in meta:
            lines.append(f"{name} {dt} [{','.join(str(d) for d in shape)}]")
            lines.extend(f'{v:{fmt}}' for v in flat[off:off + size])
            off += size
    return '\n'.join(lines)


def _format_cex(version, onnx_path, x_flat, y_flat, fmt, io_meta=None):
    """Dispatch the counterexample to the v1 (flat X_i/Y_i s-expr) or v2 (per-tensor) format
    per the resolved spec version. For v2 the per-tensor headers MUST MIRROR the spec's
    `declare-input`/`declare-output` — name, dtype (e.g. `real`/`float32`, echoed verbatim),
    and shape — so the v2 validator accepts them (`io_meta` = the spec-declared tensors).
    Only if the spec didn't declare them (`io_meta is None`) do we fall back to the ONNX node
    metadata. The values under each header are plain numbers regardless. Logs the source."""
    if version == '2.0':
        ins, outs = io_meta if io_meta is not None else _onnx_io_meta(onnx_path)
        _src = 'spec-declared tensors' if io_meta is not None else 'ONNX node metadata'
        print(f'  [counterexample] format=v2.0 (per-tensor: "NAME dtype [shape]" + '
              f'C-order values per input then output; using {_src})', flush=True)
        return _cex_v2(ins, outs, x_flat, y_flat, fmt)
    print('  [counterexample] format=v1.0 (flat s-expr: ((X_i <v>) ... (Y_j <v>)))',
          flush=True)
    return _cex_sexpr(x_flat, y_flat, fmt)


def _vnnlib_version(spec_path):
    """Detect a VNNLIB spec's version ('2.0' vs '1.0') from its head (handles .gz, and the
    instances.csv-style plain name when only the .gz is on disk)."""
    import gzip
    if not os.path.exists(spec_path) and os.path.exists(spec_path + '.gz'):
        spec_path = spec_path + '.gz'
    opener = gzip.open if spec_path.endswith('.gz') else open
    with opener(spec_path, 'rt') as fh:
        txt = fh.read(4096)
    return '2.0' if ('vnnlib-version' in txt or 'declare-network' in txt
                     or 'declare-input' in txt) else '1.0'


def _resolve_cex_version(cf, spec_path):
    """Resolve the on-disk counterexample format version from the `counterexample_format`
    setting value: explicit '1'/'2', else 'auto' -> match the input spec's vnnlib version."""
    cf = str(cf).lower()
    if cf in ('1', '1.0', 'v1'):
        return '1.0'
    if cf in ('2', '2.0', 'v2'):
        return '2.0'
    return _vnnlib_version(spec_path)


def _resolve_cex_io_meta(spec_path):
    """The SPEC-declared I/O for a v2 counterexample, as
    ``((name, dtype, shape, size) inputs..., (...) outputs...)`` — so EVERY emit path
    (standard / augmented / surrogate-multi-input) writes the vnnlib's variable names
    (X / X1,X2 / Y) AND mirrors the spec's declared dtype + shape, instead of the ONNX node
    metadata. The cex header MUST match the spec's `declare-input`/`declare-output` (the v2
    validator compares them, so `real`/`float32` is echoed verbatim); the values underneath
    are plain numbers regardless. Parsed CHEAPLY from just the `declare-network` header so a
    spec with millions of input-bound asserts is never fully read. Returns ``None`` for v1 /
    on a read error (the cex then keeps the ONNX node names). Resolved ONCE in ``main`` from
    the ORIGINAL spec (before any pair/augment rewrite of ``args.spec``)."""
    import gzip
    import re
    p = spec_path
    if not os.path.exists(p) and os.path.exists(p + '.gz'):
        p = p + '.gz'
    try:
        opener = gzip.open if p.endswith('.gz') else open
        with opener(p, 'rt') as fh:
            head = fh.read(16384)
    except OSError:
        return None
    if 'declare-network' not in head:
        return None
    ins, outs = [], []
    for kind, name, dt, shp in re.findall(
            r'\(declare-(input|output)\s+(\S+)\s+(\S+)\s+\[([^\]]*)\]', head):
        shape = tuple(int(s.strip()) for s in shp.split(',') if s.strip())
        size = 1
        for d in shape:
            size *= d
        (ins if kind == 'input' else outs).append((name, dt, shape, size))
    return (tuple(ins), tuple(outs)) if (ins or outs) else None


def _emit_surrogate_result(args, verdict, witness, sat_state, cex_fmt='.17g'):
    """Write the surrogate verdict; for sat, a counterexample over the multi-input witness
    (ORIGINAL-model order) + the ORT-CPU output, in the format matching the input spec's
    vnnlib version (`args.cex_version`; smart_turn is v2 -> per-tensor blocks). Honors the
    never-downgrade rule."""
    import numpy as np
    from . import surrogate_pgd as sp
    if verdict == 'sat' and witness is not None:
        x = np.concatenate([np.asarray(w).ravel() for w in witness]).astype(np.float64)
        y = np.asarray(sp._ort_eval(args.net, witness)).ravel().astype(np.float64)
        # Defense-in-depth: the surrogate/attack paths bypass the universal
        # _validate_sat_witness gate, so re-verify here (in float64) that the
        # ORT-recomputed output GENUINELY (strictly) violates the spec before
        # committing 'sat'. Catches any search-side acceptance bug — e.g. a point
        # sitting exactly on a strict `>`/`<` threshold (margin 0) is NOT a CE.
        _m = None
        _spec_path = getattr(args, 'spec', None)
        if _spec_path:
            try:
                _odnf = sp.parse_box_and_output(_spec_path).out_dnf
                _m = max(min((float(y[i]) - rhs) if op == 'gt' else (rhs - float(y[i]))
                             for i, op, rhs in clause) for clause in _odnf)
            except (NotImplementedError, OSError, IndexError, ValueError,
                    AttributeError):
                _m = None                   # can't re-check -> don't block (rare path)
        if _m is not None and _m <= 0.0:
            print(f'  [surrogate] witness does NOT strictly violate spec '
                  f'(margin={_m:.3e}) — NOT emitting SAT', flush=True)
            verdict = 'timeout'
        else:
            if getattr(args, 'verbose', False):
                _log_cex_values(x, y)
            ce = _format_cex(getattr(args, 'cex_version', '1.0'), args.net, x, y, cex_fmt,
                             io_meta=getattr(args, 'cex_io_decls', None))
            with open(args.results_file, 'w') as f:
                f.write('sat\n' + ce + '\n')
            sat_state['emitted'] = True
            return
    if not sat_state.get('emitted'):
        with open(args.results_file, 'w') as f:
            f.write(verdict + '\n')


def _log_cex_values(x, y, log=print):
    """Verbose preview of a validated counterexample: the first few input and output
    values (X on one line, Y on the next, each truncated with '...' if longer)."""
    import numpy as np

    def _fmt(a):
        if a is None:
            return '(none)'
        flat = np.asarray(a, dtype=np.float64).ravel()
        head = ', '.join(f'{v:.6g}' for v in flat[:3])
        return head + (' ...' if flat.size > 3 else '')

    log(f'  [counterexample] validated witness  X: {_fmt(x)}')
    log(f'  [counterexample] validated witness  Y: {_fmt(y)}')


def _emit_result(args, spec, line, witness, sat_state, cex_fmt='.17g'):
    """Write the VNNCOMP results file: verdict on line 1, then (for 'sat') the
    counterexample s-expression. Never-downgrade rule: once a 'sat' (with a
    counterexample) has been written, a later 'timeout'/'unknown'/'error' will
    NOT overwrite it — a found counterexample is sticky against running out of
    budget. A later 'sat' (a clearer counterexample) or 'unsat' (a proof) DOES
    override. Idempotent/repeatable.
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
        _ver = getattr(args, 'cex_version', '1.0')
        _io = getattr(args, 'cex_io_decls', None)   # spec-declared v2 names (resolved once)
        _orig = getattr(args, 'orig_net_for_cex', None)
        if _orig is not None:
            # Nonlinear-augmented instance: the loaded net's outputs are the
            # constraint polynomials, not the original net's output. Emit Y from
            # the ORIGINAL net so the cex has the benchmark's true output shape.
            ce = _counterexample_sexpr_orig(_orig, witness, cex_fmt, _ver, io_meta=_io)
        else:
            ce = _counterexample_sexpr(args.net, spec, witness, cex_fmt, _ver, io_meta=_io)
        if ce is None:
            print('  [warn] sat verdict but could not build a counterexample '
                  '(no ORT output); results file has the verdict only.')
    # ATOMIC write: a hard process kill at the deadline (run_instance.sh's
    # `timeout`) can strike mid-write. A plain truncate+write would then leave a
    # 'sat' with a partial/missing counterexample, which the official scorer
    # rejects. Write to a temp file in the same dir + os.replace (atomic rename
    # on POSIX) so the results file is ALWAYS either the prior content or the
    # complete new content — never a torn 'sat'.
    tmp = f'{args.results_file}.tmp'
    with open(tmp, 'w') as f:
        f.write(line + '\n')
        if ce is not None:
            f.write(ce + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.results_file)
    if line == 'sat':
        sat_state['emitted'] = True


def _validate_cex_v2_competition(args, spec, witness, cex_fmt='.17g'):
    """Validate the v2 counterexample vibecheck would emit using the VENDORED
    competition checker (`vnncomp_cex_v2.validate_cex_v2`, bit-identical to the
    VNN-COMP 2026 scorer). Builds the exact CE string we'd write, drops it in a
    temp file, and runs the scorer's validator against the original --net/--spec.

    Returns (accepted, result_str, message). `accepted` is True iff the scorer
    would award the instance (CORRECT or CORRECT_UP_TO_TOLERANCE). A build
    failure (no CE) returns (False, 'no_ce', ...) so a non-validatable witness is
    never emitted.
    """
    import tempfile
    from .vnncomp_cex_v2 import validate_cex_v2, ACCEPTED_RESULTS
    _io = getattr(args, 'cex_io_decls', None)
    _orig = getattr(args, 'orig_net_for_cex', None)
    # The competition replays/parses the ORIGINAL v2 instance. For augmented runs
    # args.net/args.spec are the internal augmented v1 versions, so fall back to the
    # preserved originals (orig_net_for_cex / orig_spec_for_cex).
    val_net = _orig if _orig is not None else args.net
    val_spec = getattr(args, 'orig_spec_for_cex', None) or args.spec
    if _orig is not None:
        ce = _counterexample_sexpr_orig(_orig, witness, cex_fmt, '2.0', io_meta=_io)
    else:
        ce = _counterexample_sexpr(args.net, spec, witness, cex_fmt, '2.0', io_meta=_io)
    if ce is None:
        return False, 'no_ce', 'could not build counterexample (no ORT output)'
    tf = tempfile.NamedTemporaryFile('w', suffix='.counterexample', delete=False)
    try:
        tf.write(ce + '\n')
        tf.close()
        res, msg = validate_cex_v2(val_net, val_spec, tf.name)
    finally:
        try:
            os.remove(tf.name)
        except OSError:
            pass
    return (res in ACCEPTED_RESULTS), res, msg


def _counterexample_sexpr(onnx_path, spec, witness, cex_fmt='.17g', version='1.0',
                          io_meta=None):
    """Build the counterexample for a SAT witness in the v1 (flat) or v2 (per-tensor)
    format. Returns the cex string or None if the ONNX output can't be computed (e.g.
    onnxruntime missing). Y is obtained from the same ORT forward the soundness validator
    runs, so it matches the scoring harness's recomputed output within tolerance.
    """
    import numpy as np
    from .verify_graph import _validate_sat_witness
    x = np.asarray(witness).flatten().astype(np.float64)
    # _validate_sat_witness runs ORT and stashes the output in info['out'].
    _, info = _validate_sat_witness(onnx_path, spec, witness)
    y = info.get('out')
    if y is None:
        return None
    # Write the float32-safe in-box witness as X (same point that produced Y
    # via ORT), so the scorer's box check passes despite FP edge rounding.
    if info.get('witness_inbox') is not None:
        x = np.asarray(info['witness_inbox']).flatten().astype(np.float64)
    y = np.asarray(y).flatten().astype(np.float64)
    # v2: emit with the SPEC's declared I/O names (io_meta), not the ONNX node names.
    return _format_cex(version, onnx_path, x, y, cex_fmt, io_meta=io_meta)


def _counterexample_sexpr_orig(orig_onnx, witness, cex_fmt='.17g', version='1.0',
                               io_meta=None):
    """Counterexample for a nonlinear-AUGMENTED instance: X is the witness, Y is the
    ORIGINAL net's output recomputed in float32 CPU ORT (the same arithmetic the VNNCOMP
    scorer uses), formatted per the spec version (v2 uses io_meta = the original spec's
    declared I/O names). Returns None if ORT can't run."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    x = np.asarray(witness).flatten().astype(np.float64)
    try:
        if orig_onnx.endswith('.gz'):
            import gzip
            with gzip.open(orig_onnx, 'rb') as _f:
                sess = ort.InferenceSession(
                    _f.read(), providers=['CPUExecutionProvider'])
        else:
            sess = ort.InferenceSession(
                orig_onnx, providers=['CPUExecutionProvider'])
        in_meta = sess.get_inputs()[0]
        in_shape = [d if isinstance(d, int) and d > 0 else 1
                    for d in in_meta.shape]
        y = sess.run(None, {in_meta.name: x.reshape(in_shape).astype(np.float32)})[0]
    except (RuntimeError, OSError, ValueError):
        return None
    return _format_cex(version, orig_onnx, x,
                       np.asarray(y).flatten().astype(np.float64), cex_fmt,
                       io_meta=io_meta)


if __name__ == '__main__':
    main()
