#!/usr/bin/env python
"""Subprocess-isolated integration-test runner.

Each case runs in its OWN `python -m vibecheck.main` subprocess and the verdict
is read from `--results-file` (VNNCOMP convention: unsat/sat/unknown/timeout) —
NEVER from exit code or stdout. This gives:

  * isolation — a fresh process + fresh CUDA context per case, so GPU memory does
    NOT accumulate across cases (the in-process pytest harness shares one process
    and OOMs on the tail of a long suite on small GPUs).
  * production parity — exercises the exact CLI path users run.
  * trustworthy verdicts — file contents only, per the repo's verdict-file rule.

Cases are sourced from the pytest integration modules themselves
(`tests/integration/test_<benchmark>.py`: CASES / CONFIG_YAML / BENCHMARK_DIR),
so this stays in sync with the canonical case list.

Usage:
  scripts/integration_runner.py malbeware
  scripts/integration_runner.py tinyimagenet_2024 --only prop_1175 --logdir /tmp/it
  scripts/integration_runner.py dist_shift_2023 --root /path/to/vnncomp2025_benchmarks
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# VNNCOMP results-file verdict -> vibecheck verdict (main.py verdict_map inverse).
FILE_TO_VC = {
    'unsat': 'verified',
    'sat': 'sat',
    'unknown': 'unknown',
    'timeout': 'timeout',
}


def _load_module(benchmark):
    path = REPO / 'tests' / 'integration' / f'test_{benchmark}.py'
    if not path.exists():
        sys.exit(f'no integration module for {benchmark!r}: {path}')
    # Import as a real package so `from ._runner import run_case` resolves
    # (tests/ and tests/integration/ work as PEP-420 namespace packages).
    import importlib
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module(f'tests.integration.test_{benchmark}')
    except Exception as e:                    # noqa: BLE001 - report and stop
        sys.exit(f'failed to import {path}: {e}')


def _resolve_root(arg_root):
    if arg_root:
        root = Path(arg_root)
    else:
        import yaml
        paths = yaml.safe_load((REPO / 'tests' / 'paths.yaml').read_text())
        root = Path(paths['vnncomp_benchmarks'])
    if (root / 'benchmarks').is_dir():
        root = root / 'benchmarks'
    return root


def _resolve_file(base):
    """Return an existing path, swapping .gz suffix; gunzip a .gz-only file."""
    if base.exists():
        return base
    gz = Path(str(base) + '.gz')
    unz = Path(str(base)[:-3]) if str(base).endswith('.gz') else None
    if gz.exists():
        subprocess.run(['gunzip', '-kf', str(gz)], check=True)
        return base
    if unz and unz.exists():
        return unz
    return base  # let the subprocess report the missing file


def run(benchmark, only, root, logdir):
    mod = _load_module(benchmark)
    cases = mod.CASES
    bench_dir = root / mod.BENCHMARK_DIR
    config = REPO / 'configs' / mod.CONFIG_YAML
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    npass = nfail = 0
    print(f'== {benchmark}: {len(cases)} cases (config={mod.CONFIG_YAML}) '
          f'subprocess-isolated ==')
    for i, case in enumerate(cases):
        desc = case['desc']
        if only and only not in desc:
            continue
        net = _resolve_file(bench_dir / case['net'])
        vnn = _resolve_file(bench_dir / case['vnnlib'])
        rfile = logdir / f'verdict_{benchmark}_{i}.txt'
        log = logdir / f'log_{benchmark}_{i}.txt'
        if rfile.exists():
            rfile.unlink()
        cmd = [sys.executable, '-m', 'vibecheck.main',
               '--net', str(net), '--spec', str(vnn),
               '--config', str(config),
               '--timeout', str(case['timeout']),
               '--device', 'gpu', '--bits', '32',
               '--pgd-restarts', str(case.get('pgd_restarts', 100)),
               '--results-file', str(rfile)]
        # External wall-clock kill = 4x the case budget + 60s. A case that hits
        # this overran its own --timeout by a wide margin: a timeout-enforcement
        # bug (some op not polling the clock), surfaced rather than hung on.
        safety = case['timeout'] * 4 + 60
        t0 = time.perf_counter()
        killed = False
        with open(log, 'w') as lf:
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               cwd=str(REPO), timeout=safety)
            except subprocess.TimeoutExpired:
                killed = True
        wall = time.perf_counter() - t0

        # Verdict from the FILE ONLY.
        if not rfile.exists():
            verdict = '<no-results-file>'
        else:
            raw = rfile.read_text().strip().split()[0] if rfile.read_text().strip() else ''
            verdict = FILE_TO_VC.get(raw, f'unknown({raw})')

        expected = case['expected']
        max_wall = case.get('max_wall_s')
        ok_verdict = (verdict == expected)
        ok_wall = (max_wall is None or wall <= max_wall)
        ok = ok_verdict and ok_wall
        npass += ok
        nfail += (not ok)
        flag = 'PASS' if ok else 'FAIL'
        wall_note = '' if ok_wall else f' WALL>{max_wall}s'
        kill_note = (f'  !! KILLED at {wall:.0f}s (overran --timeout={case["timeout"]}s '
                     f'-> timeout-enforcement bug)') if killed else ''
        print(f'  [{flag}] {desc}\n'
              f'         verdict={verdict} expected={expected} '
              f'wall={wall:.1f}s{wall_note}  log={log}{kill_note}')
    print(f'== {benchmark}: {npass} pass, {nfail} fail ==')
    return nfail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('benchmark', help='e.g. malbeware, tinyimagenet_2024, dist_shift_2023')
    ap.add_argument('--only', default=None, help='substring filter on case desc')
    ap.add_argument('--root', default=None, help='vnncomp benchmarks root (else tests/paths.yaml)')
    ap.add_argument('--logdir', default='/tmp/vibecheck_integration', help='per-case logs + verdict files')
    args = ap.parse_args()
    sys.exit(1 if run(args.benchmark, args.only, _resolve_root(args.root), args.logdir) else 0)


if __name__ == '__main__':
    main()
