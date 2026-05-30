"""relusplitter benchmark sweep harness.

Runs vibecheck on a chosen subset of the vnncomp2025 relusplitter
benchmark, one subprocess per case (so OOM / Gurobi crashes / runaway
allocs don't take down the orchestrator). Each subprocess is wrapped in
`systemd-run --user --scope -p MemoryMax=<cap>` when --memory-max is set.

Subsets:
    --set iter        : 11-case smoke set the cleanup plan iterates on
    --set full        : every line in instances.csv (~220 cases)
    --set soundness   : 20 AB-CROWN-confirmed SAT cases (sound check)
    --set custom      : pass --cases <id>,<id>,... explicitly

Output: a JSON file with one record per case (status, wall_s,
settings_hash, override_json, etc.). Use --diff <baseline.json> to
compare two runs side-by-side. Per-case raw output is kept in
<out_dir>/raw/<id>.json so any unexpected verdict can be inspected
directly.

Remote mode (--remote stan@HOST): rsyncs the local repo to
~/Desktop/temp/vibecheck-temp on the remote, then runs the same harness
there over ssh. Raw JSONs are streamed back to the local --out path.

Override format: --override KEY=VALUE (repeatable). Booleans, ints,
floats are auto-parsed; quote strings if they collide. For tuples /
lists pass --override-json '{"k": [...]}' .
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = Path(
    '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter')
DEFAULT_REMOTE_REPO = '~/Desktop/temp/vibecheck-temp'
DEFAULT_REMOTE_BENCH = (
    '~/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter')
DEFAULT_AB_RESULTS = Path(
    '/home/stan/repositories/vnncomp2025_results/alpha_beta_crown/'
    '2025_relusplitter/results.csv')
REMOTE_HOST_DEFAULT = 'stan@100.83.144.97'

# 11-case iteration set from the cleanup plan. The 'expected' column is
# the verdict the cleanup plan must preserve (the "regression detector"
# rows are deliberately left as 'unknown' — they're the regression
# canaries, not features under test).
ITER_CASES: list[tuple[str, str, str, str]] = [
    # (id, onnx, vnnlib, expected)
    ('mnist_256x4_prop_5_0.05',
     'onnx/mnist_fc_vnncomp2022_mnist-net_256x4.onnx',
     'vnnlib/mnist_fc_vnncomp2022_prop_5_0.05.vnnlib',
     'verified'),
    ('mnist_256x6_prop_4_0.03',
     'onnx/mnist_fc_vnncomp2022_mnist-net_256x6.onnx',
     'vnnlib/mnist_fc_vnncomp2022_prop_4_0.03.vnnlib',
     'verified'),
    ('mnist_256x6_prop_5_0.05',
     'onnx/mnist_fc_vnncomp2022_mnist-net_256x6.onnx',
     'vnnlib/mnist_fc_vnncomp2022_prop_5_0.05.vnnlib',
     'unknown'),  # regression detector
    ('oval21_deep_kw_img4740',
     'onnx/oval21-benchmark_cifar_deep_kw.onnx',
     'vnnlib/oval21-benchmark_cifar_deep_kw-img4740-eps0.01647058823529412.vnnlib',
     'verified'),
    ('oval21_deep_kw_img3039_RSPLITTER',
     'onnx/oval21-benchmark_cifar_deep_kw_RSPLITTER_'
     'oval21-benchmark_cifar_deep_kw-img3039-eps0.035686274509803925.onnx',
     'vnnlib/oval21-benchmark_cifar_deep_kw-img3039-eps0.035686274509803925.vnnlib',
     'verified'),
    ('oval21_deep_kw_img5988_RSPLITTER',
     'onnx/oval21-benchmark_cifar_deep_kw_RSPLITTER_'
     'oval21-benchmark_cifar_deep_kw-img5988-eps0.037516339869281046.onnx',
     'vnnlib/oval21-benchmark_cifar_deep_kw-img5988-eps0.037516339869281046.vnnlib',
     'verified'),
    ('cifar_biasfield_0',
     'onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_0.onnx',
     'vnnlib/cifar_biasfield_vnncomp2022_prop_0.vnnlib',
     'verified'),
    ('cifar_biasfield_28',
     'onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_28.onnx',
     'vnnlib/cifar_biasfield_vnncomp2022_prop_28.vnnlib',
     'verified'),
    ('cifar_biasfield_40',
     'onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_40.onnx',
     'vnnlib/cifar_biasfield_vnncomp2022_prop_40.vnnlib',
     'sat'),
    ('cifar_biasfield_70',
     'onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_70.onnx',
     'vnnlib/cifar_biasfield_vnncomp2022_prop_70.vnnlib',
     'unknown'),  # regression detector
    ('mnist_256x4_prop_0_0.05',
     'onnx/mnist_fc_vnncomp2022_mnist-net_256x4.onnx',
     'vnnlib/mnist_fc_vnncomp2022_prop_0_0.05.vnnlib',
     'sat'),
]


def _id_for(net_rel: str, spec_rel: str) -> str:
    """Stable per-case id used in JSON records and raw filenames."""
    n = Path(net_rel).stem
    s = Path(spec_rel).stem
    return f'{n}__{s}'


def load_full_set(bench: Path, expected_csv: Path | None) -> list[dict]:
    """Return one record per row in instances.csv with timeout + expected.

    `expected` is filled from the AB-CROWN reference results when
    available (sat / unsat / timeout / unknown), so --diff can flag both
    soundness regressions (vibecheck=verified for AB=sat) and lost
    captures.
    """
    instances = bench / 'instances.csv'
    assert instances.is_file(), f'no such file: {instances}'

    expected_map: dict[str, str] = {}
    if expected_csv is not None and expected_csv.is_file():
        with open(expected_csv) as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                _bench, net, spec, _prep, verdict = row[:5]
                key = _id_for(Path(net).name, Path(spec).name)
                expected_map[key] = {
                    'unsat': 'verified',
                    'sat': 'sat',
                    'timeout': 'unknown',
                }.get(verdict, verdict)

    out = []
    with open(instances) as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[0]:
                continue
            net_rel, spec_rel, t = row[0], row[1], row[2]
            cid = _id_for(net_rel, spec_rel)
            out.append({
                'id': cid,
                'net_rel': net_rel,
                'spec_rel': spec_rel,
                'timeout': float(t),
                'expected': expected_map.get(cid, ''),
            })
    return out


def load_iter_set(bench: Path) -> list[dict]:
    """Return iter-set records using the 11 cases hard-coded above."""
    out = []
    for cid, net_rel, spec_rel, expected in ITER_CASES:
        out.append({
            'id': cid,
            'net_rel': net_rel,
            'spec_rel': spec_rel,
            'timeout': 180.0,
            'expected': expected,
        })
    return out


def load_soundness_set(bench: Path,
                       ab_results: Path) -> list[dict]:
    """Return all AB-CROWN-confirmed SAT cases (relative paths).

    These are the soundness probe: with `disable_sat_finding=True`,
    vibecheck must NEVER return 'verified' on any of them. 'unknown' is
    fine; 'sat' is impossible because the override disables PGD; only
    'unknown' or 'verified' (= soundness bug) are observable.
    """
    assert ab_results.is_file(), f'no such file: {ab_results}'
    out = []
    with open(ab_results) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            _bench, net_abs, spec_abs, _prep, verdict = row[:5]
            if verdict.strip() != 'sat':
                continue
            net_rel = 'onnx/' + Path(net_abs).name
            spec_rel = 'vnnlib/' + Path(spec_abs).name
            cid = _id_for(net_rel, spec_rel)
            out.append({
                'id': cid,
                'net_rel': net_rel,
                'spec_rel': spec_rel,
                'timeout': 180.0,
                'expected': 'sat',
            })
    return out


def parse_overrides(pairs: list[str], override_json: str) -> dict:
    """Combine --override KEY=VALUE pairs with --override-json JSON dict."""
    base: dict = json.loads(override_json) if override_json else {}
    for kv in pairs:
        assert '=' in kv, f'override must be KEY=VALUE: {kv!r}'
        k, v = kv.split('=', 1)
        # Coerce: bool first (case-insensitive), then int, then float, else str.
        if v.lower() in ('true', 'false'):
            base[k] = (v.lower() == 'true')
        else:
            try:
                base[k] = int(v)
            except ValueError:
                try:
                    base[k] = float(v)
                except ValueError:
                    base[k] = v
    return base


def _wait_gpu_free(min_free_mib: int = 8000, timeout_s: float = 30.0,
                    poll_s: float = 0.5) -> None:
    """Poll nvidia-smi until the GPU has at least `min_free_mib` free or
    `timeout_s` elapses. Required between sweep cases on the RTX 3080
    server because Phase 8's `multiprocessing.Pool` workers and the
    parent's CUDA context can take several seconds to fully release
    after the parent subprocess exits — even when no compute apps are
    listed by `nvidia-smi --query-compute-apps`. Without this, the next
    subprocess's first GPU allocation reliably fails with
    `torch.AcceleratorError: out of memory` on cifar_biasfield-class
    workloads. No-op when nvidia-smi is unavailable (CPU-only machines).
    """
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, timeout=5).decode().strip()
            free_mib = int(out.splitlines()[0])
            if free_mib >= min_free_mib:
                return
        except (subprocess.SubprocessError, ValueError, OSError):
            return  # no GPU or query failed; nothing to wait on
        time.sleep(poll_s)


def _resolve_path(base: Path, rel: str) -> str:
    """Return absolute path; if missing, try the .gz variant.

    The local benchmark checkout keeps every file gzipped; the remote has
    them uncompressed. Both onnx_loader.load_onnx and
    vnnlib_loader.load_vnnlib transparently handle .gz, so callers don't
    need to decompress first.
    """
    p = base / rel
    if p.is_file():
        return str(p)
    pg = base / (rel + '.gz')
    if pg.is_file():
        return str(pg)
    return str(p)  # let downstream fail with a meaningful error


def _build_local_cmd(case: dict, bench: Path, raw_dir: Path,
                     override_json: str, memory_max: str | None) -> list[str]:
    """Build the local `systemd-run | python` command for one case."""
    net = _resolve_path(bench, case['net_rel'])
    spec = _resolve_path(bench, case['spec_rel'])
    out = str(raw_dir / f'{case["id"]}.json')
    py = str(REPO_ROOT / '.venv/bin/python')
    inner = [py, '-m', 'tests._run_one_case',
             '--net', net,
             '--spec', spec,
             '--timeout', str(case['timeout']),
             '--out', out,
             '--id', case['id'],
             '--expected', case['expected'],
             '--override-json', override_json]
    if memory_max:
        # Wrap in a transient cgroup so a runaway allocation can't kill
        # the orchestrator. --pipe so stdout/stderr come through.
        return ['systemd-run', '--user', '--scope', '--quiet',
                '-p', f'MemoryMax={memory_max}',
                '--', *inner]
    return inner


def _build_remote_cmd(case: dict, remote: str, remote_repo: str,
                      remote_bench: str, override_json: str,
                      memory_max: str | None) -> list[str]:
    """Build the ssh command running _run_one_case on the remote.

    Result JSON is written into <remote_repo>/.sweep_raw/<id>.json on the
    remote; orchestrator pulls it back via scp afterwards.
    """
    net = f'{remote_bench}/{case["net_rel"]}'
    spec = f'{remote_bench}/{case["spec_rel"]}'
    raw_remote = f'{remote_repo}/.sweep_raw/{case["id"]}.json'

    inner_parts = [
        'cd', remote_repo, '&&',
        '.venv/bin/python', '-m', 'tests._run_one_case',
        '--net', net,
        '--spec', spec,
        '--timeout', str(case['timeout']),
        '--out', raw_remote,
        '--id', shlex.quote(case['id']),
        '--expected', shlex.quote(case['expected']),
        '--override-json', shlex.quote(override_json),
    ]
    if memory_max:
        inner_parts = (['systemd-run', '--user', '--scope', '--quiet',
                        '-p', f'MemoryMax={memory_max}', '--']
                       + inner_parts)
    inner = ' '.join(inner_parts)
    return ['ssh', remote, inner]


def rsync_to_remote(remote: str, remote_repo: str) -> None:
    """rsync the local repo over to the remote machine."""
    src = str(REPO_ROOT) + '/'
    dst = f'{remote}:{remote_repo.rstrip("/")}/'
    print(f'[rsync] {src} -> {dst}', flush=True)
    subprocess.check_call([
        'rsync', '-az',
        '--exclude', '.venv',
        '--exclude', '__pycache__',
        '--exclude', '.git',
        '--exclude', '.sweep_raw',
        '--exclude', 'scratch',
        src, dst,
    ])


def fetch_remote_raw(remote: str, remote_repo: str,
                     case_id: str, raw_dir: Path) -> dict | None:
    """scp one case's raw JSON back; return parsed record or None."""
    raw_local = raw_dir / f'{case_id}.json'
    raw_remote = f'{remote}:{remote_repo}/.sweep_raw/{case_id}.json'
    p = subprocess.run(['scp', '-q', raw_remote, str(raw_local)],
                       capture_output=True)
    if p.returncode != 0 or not raw_local.is_file():
        return None
    with open(raw_local) as f:
        return json.load(f)


def run_one(case: dict, *, bench: Path, raw_dir: Path,
            override_json: str, memory_max: str | None,
            remote: str | None, remote_repo: str,
            remote_bench: str, hard_cap_extra: float = 30.0) -> dict:
    """Run a single case (local or remote subprocess); return the record.

    Wall cap: case['timeout'] + hard_cap_extra. If the subprocess
    overshoots, we kill it and synthesize a 'timeout' record so the
    sweep can keep going.
    """
    out_path = raw_dir / f'{case["id"]}.json'
    if remote is None:
        cmd = _build_local_cmd(case, bench, raw_dir, override_json,
                               memory_max)
    else:
        cmd = _build_remote_cmd(case, remote, remote_repo, remote_bench,
                                override_json, memory_max)

    t0 = time.perf_counter()
    wall_cap = case['timeout'] + hard_cap_extra
    try:
        subprocess.run(cmd, capture_output=True, timeout=wall_cap)
        _wait_gpu_free()
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return {
            'id': case['id'],
            'net': case['net_rel'],
            'spec': case['spec_rel'],
            'expected': case['expected'],
            'override_json': json.loads(override_json),
            'status': 'timeout',
            'wall_s': wall,
            'error': f'subprocess wall > {wall_cap}s',
            'timing': {},
            'phase': None,
            'remaining': None,
            'settings_hash': '',
        }

    if remote is not None:
        rec = fetch_remote_raw(remote, remote_repo, case['id'], raw_dir)
    else:
        rec = json.loads(out_path.read_text()) if out_path.is_file() else None

    if rec is None:
        wall = time.perf_counter() - t0
        return {
            'id': case['id'],
            'net': case['net_rel'],
            'spec': case['spec_rel'],
            'expected': case['expected'],
            'override_json': json.loads(override_json),
            'status': 'crash',
            'wall_s': wall,
            'error': 'no JSON output produced',
            'timing': {},
            'phase': None,
            'remaining': None,
            'settings_hash': '',
        }
    return rec


def diff_against(out_records: list[dict], baseline_path: Path) -> dict:
    """Compare current sweep records to a baseline JSON file."""
    base = json.loads(baseline_path.read_text())
    base_by_id = {r['id']: r for r in base['records']}
    cur_by_id = {r['id']: r for r in out_records}
    all_ids = sorted(set(base_by_id) | set(cur_by_id))
    summary = {'regressions': [], 'gains': [], 'soundness_breaks': [],
               'wall_delta_s': 0.0, 'changed_status': []}
    for cid in all_ids:
        b = base_by_id.get(cid)
        c = cur_by_id.get(cid)
        if b is None or c is None:
            continue
        bs, cs = b.get('status'), c.get('status')
        if bs == cs:
            summary['wall_delta_s'] += (c.get('wall_s', 0)
                                        - b.get('wall_s', 0))
            continue
        summary['changed_status'].append((cid, bs, cs))
        if bs == 'verified' and cs != 'verified':
            summary['regressions'].append((cid, bs, cs))
        elif bs != 'verified' and cs == 'verified':
            summary['gains'].append((cid, bs, cs))
        # Soundness: a case AB-CROWN says 'sat' must never become
        # 'verified'. The expected field carries the AB verdict.
        if c.get('expected') == 'sat' and cs == 'verified':
            summary['soundness_breaks'].append(cid)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--set', required=True,
                   choices=['iter', 'full', 'soundness', 'custom'])
    p.add_argument('--cases', default='',
                   help='Comma-separated case ids for --set custom')
    p.add_argument('--bench', default=str(DEFAULT_BENCH))
    p.add_argument('--ab-results', default=str(DEFAULT_AB_RESULTS))
    p.add_argument('--out', default='/tmp/vibecheck_runs/sweep.json')
    p.add_argument('--memory-max', default='14G',
                   help='cgroup MemoryMax per case; "" to disable')
    p.add_argument('--remote', default='',
                   help=f'ssh target (e.g. {REMOTE_HOST_DEFAULT}); '
                        '"" runs locally')
    p.add_argument('--remote-repo', default=DEFAULT_REMOTE_REPO)
    p.add_argument('--remote-bench', default=DEFAULT_REMOTE_BENCH)
    p.add_argument('--no-rsync', action='store_true',
                   help='Skip rsync push when --remote is set')
    p.add_argument('--override', action='append', default=[])
    p.add_argument('--override-json', default='')
    p.add_argument('--diff', default='',
                   help='Compare against baseline JSON file')
    p.add_argument('--no-run', action='store_true',
                   help='Just print the case list')
    args = p.parse_args()

    bench = Path(args.bench)
    out_path = Path(args.out)
    raw_dir = out_path.parent / f'{out_path.stem}_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.set == 'iter':
        cases = load_iter_set(bench)
    elif args.set == 'full':
        cases = load_full_set(bench, Path(args.ab_results))
    elif args.set == 'soundness':
        cases = load_soundness_set(bench, Path(args.ab_results))
        # Force the soundness-probe override on top of user overrides.
        args.override.append('disable_sat_finding=true')
    else:
        wanted = {c.strip() for c in args.cases.split(',') if c.strip()}
        full = load_full_set(bench, Path(args.ab_results))
        cases = [c for c in full if c['id'] in wanted]

    overrides = parse_overrides(args.override, args.override_json)
    override_json = json.dumps(overrides, sort_keys=True)
    memory_max = args.memory_max or None

    print(f'[sweep] set={args.set} n={len(cases)} '
          f'remote={args.remote or "(local)"} '
          f'memory_max={memory_max} '
          f'overrides={overrides}', flush=True)
    if args.no_run:
        for c in cases:
            print(f'  {c["id"]:<60} timeout={c["timeout"]:.0f}s '
                  f'expected={c["expected"]!r}')
        return 0

    if args.remote:
        if not args.no_rsync:
            rsync_to_remote(args.remote, args.remote_repo)
        # Pre-create the remote raw dir.
        subprocess.run(['ssh', args.remote,
                        f'mkdir -p {args.remote_repo}/.sweep_raw'])

    records: list[dict] = []
    t_start = time.perf_counter()
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        rec = run_one(case, bench=bench, raw_dir=raw_dir,
                      override_json=override_json,
                      memory_max=memory_max,
                      remote=args.remote or None,
                      remote_repo=args.remote_repo,
                      remote_bench=args.remote_bench)
        wall = time.perf_counter() - t0
        ok = '✓' if rec.get('status') == rec.get('expected', '') else '·'
        if rec.get('expected') == '':
            ok = '?'
        print(f'[{i+1:3d}/{len(cases)}] {ok} {case["id"]:<60} '
              f'status={rec.get("status"):<10} '
              f'wall={wall:6.1f}s expected={rec.get("expected","-"):<10}',
              flush=True)
        records.append(rec)
        # Persist incrementally so a crash mid-sweep doesn't lose state.
        out_path.write_text(json.dumps({
            'set': args.set,
            'overrides': overrides,
            'records': records,
            'partial': True,
        }, indent=2))

    elapsed = time.perf_counter() - t_start
    final = {
        'set': args.set,
        'overrides': overrides,
        'records': records,
        'wall_total_s': elapsed,
        'partial': False,
    }
    out_path.write_text(json.dumps(final, indent=2))

    # Tally.
    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r.get('status', '?')] = (
            status_counts.get(r.get('status', '?'), 0) + 1)
    n_verified = status_counts.get('verified', 0)
    n_match = sum(1 for r in records
                  if r.get('expected') and r.get('status') == r.get('expected'))
    print(f'\n[done] wall={elapsed:.1f}s verified={n_verified}/{len(records)} '
          f'match-expected={n_match}/{len(records)} '
          f'status-counts={status_counts}', flush=True)

    soundness_breaks = [r['id'] for r in records
                        if r.get('expected') == 'sat' and r.get('status') == 'verified']
    if soundness_breaks:
        print(f'[!! SOUNDNESS] {len(soundness_breaks)} cases AB=sat but '
              f'vibecheck=verified:', flush=True)
        for cid in soundness_breaks:
            print(f'    {cid}', flush=True)

    if args.diff:
        summary = diff_against(records, Path(args.diff))
        print('\n[diff vs baseline]', flush=True)
        print(f'  regressions     : {len(summary["regressions"])}')
        for cid, bs, cs in summary['regressions']:
            print(f'    - {cid}: {bs} -> {cs}')
        print(f'  gains           : {len(summary["gains"])}')
        for cid, bs, cs in summary['gains']:
            print(f'    + {cid}: {bs} -> {cs}')
        print(f'  soundness_breaks: {len(summary["soundness_breaks"])}')
        for cid in summary['soundness_breaks']:
            print(f'    !! {cid}')
        print(f'  wall_delta_s    : {summary["wall_delta_s"]:+.1f}s '
              f'(over status-unchanged cases)')
        if summary['regressions'] or summary['soundness_breaks']:
            return 2
    if soundness_breaks:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
