"""Local completeness audit: run vibecheck on every instance of a benchmark,
verdict FROM --results-file ONLY, compare to AB-CROWN's published verdict, flag
cases ABC solved (sat/unsat) within time but vibecheck did NOT.

Usage: local_audit.py <benchmark> [time_cap] [max_cases]
Local paths. Run under a memory cap for safety:
  systemd-run --user --scope -p MemoryMax=8G .venv/bin/python scratch/local_audit.py dist_shift_2023 60
"""
import csv, os, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
BENCH = HOME / 'repositories' / 'vnncomp2025_benchmarks' / 'benchmarks'
ABCDIR = HOME / 'repositories' / 'vnncomp2025_results' / 'alpha_beta_crown'
VC = HOME / 'repositories' / 'vibecheck'
VIBE = VC / '.venv' / 'bin' / 'python'
FILE_TO_VC = {'unsat': 'verified', 'sat': 'sat', 'unknown': 'unknown', 'timeout': 'timeout'}


def load_abc(b):
    m = {}
    fp = ABCDIR / f'2025_{b}' / 'results.csv'
    if not fp.exists():
        return m
    for line in fp.read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 6:
            m[(os.path.basename(p[1]), os.path.basename(p[2]))] = (p[4], p[5])
    return m


def load_instances(b):
    out = []
    for line in (BENCH / b / 'instances.csv').read_text().splitlines():
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
    max_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    config = VC / 'configs' / f'{b}.yaml'
    abc = load_abc(b)
    insts = load_instances(b)[:max_cases]
    out_csv = VC / 'scratch' / f'audit_{b}_local.csv'
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['onnx', 'vnnlib', 'abc', 'abc_t', 'vc', 'vc_wall', 'miss'])
    n_miss = n_abc = 0
    for i, (onnx_rel, vnn_rel, tmo) in enumerate(insts):
        if cap:
            tmo = min(tmo, cap)
        abc_res, abc_t = abc.get((os.path.basename(onnx_rel), os.path.basename(vnn_rel)), ('unknown', ''))
        onnx, vnn = resolve(BENCH / b / onnx_rel), resolve(BENCH / b / vnn_rel)
        rfile = Path(f'/tmp/laudit_{b}_{i}.txt')
        if rfile.exists():
            rfile.unlink()
        cmd = [str(VIBE), '-m', 'vibecheck.main', '--net', str(onnx), '--spec', str(vnn),
               '--timeout', str(tmo), '--device', 'gpu', '--bits', '32', '--results-file', str(rfile)]
        if config.exists():
            cmd += ['--config', str(config)]
        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=str(VC), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=tmo + 60)
        except subprocess.TimeoutExpired:
            pass
        wall = time.time() - t0
        raw = rfile.read_text().strip().split()[0] if rfile.exists() and rfile.read_text().strip() else ''
        vc = FILE_TO_VC.get(raw, 'error' if not raw else f'other({raw})')
        abc_solved = abc_res in ('sat', 'unsat')
        vc_ok = (vc == 'sat' and abc_res == 'sat') or (vc == 'verified' and abc_res == 'unsat')
        miss = abc_solved and not vc_ok
        n_abc += abc_solved
        n_miss += miss
        with open(out_csv, 'a', newline='') as f:
            csv.writer(f).writerow([onnx_rel, vnn_rel, abc_res, abc_t, vc, f'{wall:.1f}', 'MISS' if miss else ''])
        tag = ' *** MISS ***' if miss else ''
        print(f'[{i+1}/{len(insts)}] {os.path.basename(vnn_rel)[:36]:36s} abc={abc_res:7s} '
              f'vc={vc:9s} ({wall:.0f}s){tag}', flush=True)
    print(f'\n=== {b}: {n_miss} MISSES of {n_abc} ABC-solved ({len(insts)} total) ===', flush=True)


if __name__ == '__main__':
    main()
