#!/usr/bin/env python3
"""Completeness audit: run vibecheck on every instance of a benchmark, compare
the verdict (FROM --results-file ONLY) to AB-CROWN's published verdict, and flag
cases where ABC solved (sat/unsat) within its time but vibecheck did NOT.

Usage: audit_benchmark.py <benchmark> [time_cap]
Server1 paths hard-coded.
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
BENCH = HOME / 'repositories' / 'vnncomp2025_benchmarks' / 'benchmarks'
ABCDIR = HOME / 'repositories' / 'vnncomp2025_results' / 'alpha_beta_crown'
VC = HOME / 'Desktop' / 'temp' / 'vibecheck-temp'
VIBE = VC / '.venv' / 'bin' / 'python'

FILE_TO_VC = {'unsat': 'verified', 'sat': 'sat', 'unknown': 'unknown',
              'timeout': 'timeout'}


def load_abc(b):
    m = {}
    fp = ABCDIR / f'2025_{b}' / 'results.csv'
    if not fp.exists():
        return m
    for line in fp.read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 6:
            vnn_base = os.path.basename(p[2])
            m[vnn_base] = (p[4], p[5])
    return m


def load_instances(b):
    fp = BENCH / b / 'instances.csv'
    out = []
    for line in fp.read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3 and p[0]:
            out.append((p[0], p[1], float(p[2])))
    return out


def resolve(p):
    p = Path(p)
    if p.exists():
        return p
    gz = Path(str(p) + '.gz')
    if gz.exists():
        return gz
    if str(p).endswith('.gz') and Path(str(p)[:-3]).exists():
        return Path(str(p)[:-3])
    return p


def main():
    b = sys.argv[1]
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else None
    config = VC / 'configs' / f'{b}.yaml'
    abc = load_abc(b)
    insts = load_instances(b)
    out_csv = HOME / 'persistent_runs' / 'results' / f'audit_{b}.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['onnx', 'vnnlib', 'abc', 'abc_t', 'vc',
                                'vc_wall', 'miss'])
    n_miss = 0
    n_abc_solved = 0
    for i, (onnx_rel, vnn_rel, tmo) in enumerate(insts):
        if cap:
            tmo = min(tmo, cap)
        vnn_base = os.path.basename(vnn_rel)
        abc_res, abc_t = abc.get(vnn_base, ('unknown', ''))
        onnx = resolve(BENCH / b / onnx_rel)
        vnn = resolve(BENCH / b / vnn_rel)
        rfile = Path(f'/tmp/audit_{b}_{i}.txt')
        if rfile.exists():
            rfile.unlink()
        cmd = [str(VIBE), '-m', 'vibecheck.main', '--net', str(onnx),
               '--spec', str(vnn), '--timeout', str(tmo), '--device', 'gpu',
               '--bits', '32', '--results-file', str(rfile)]
        if config.exists():
            cmd += ['--config', str(config)]
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=str(VC), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=tmo + 60)
        except subprocess.TimeoutExpired:
            pass
        wall = time.time() - t0
        raw = rfile.read_text().strip().split()[0] if rfile.exists() and \
            rfile.read_text().strip() else ''
        vc = FILE_TO_VC.get(raw, 'error' if not raw else f'other({raw})')
        abc_solved = abc_res in ('sat', 'unsat')
        vc_ok = (vc == 'sat' and abc_res == 'sat') or \
                (vc == 'verified' and abc_res == 'unsat')
        miss = abc_solved and not vc_ok
        if abc_solved:
            n_abc_solved += 1
        if miss:
            n_miss += 1
        with open(out_csv, 'a', newline='') as f:
            csv.writer(f).writerow([onnx_rel, vnn_rel, abc_res, abc_t, vc,
                                    f'{wall:.1f}', 'MISS' if miss else ''])
        tag = ' *** MISS ***' if miss else ''
        print(f'[{i+1}/{len(insts)}] {vnn_base[:40]:40s} abc={abc_res:7s} '
              f'vc={vc:9s} ({wall:.0f}s){tag}', flush=True)
    print(f'\n=== {b}: {n_miss} MISSES of {n_abc_solved} ABC-solved '
          f'({len(insts)} total) ===', flush=True)


if __name__ == '__main__':
    main()
