"""Full acasxu sweep: v3 (parallel + attack) vs nnenum (default, 16-proc).
Verdicts read from per-case results-files (NEVER exit code). Flags disagreements.
Writes sweep_acasxu.csv incrementally. Usage: sweep_acasxu.py [WORKERS] [TIMEOUT_CAP]"""
import sys, os, subprocess, time, csv

ROOT = '/home/stan/repositories/vibecheck/scratch/nnenum_style'
BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
VPY = '/home/stan/repositories/vibecheck/.venv/bin/python'
NPY = os.path.expanduser('~/repositories/nnenum/.venv/bin/python')
NSRC = os.path.expanduser('~/repositories/nnenum/src')

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
TO_CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0


def parse_instances():
    rows = []
    with open(f'{BENCH}/instances.csv') as f:
        for onnx, vnnlib, to in csv.reader(f):
            net = onnx.split('run2a_')[1].split('_batch')[0]
            prop = vnnlib.split('/')[1].replace('.vnnlib', '')
            rows.append((net, prop, min(float(to), TO_CAP)))
    return rows


def run_v3(net, prop, to):
    rf = f'/tmp/sweep_v3_{net}_{prop}.txt'
    if os.path.exists(rf):
        os.remove(rf)
    t0 = time.time()
    try:
        subprocess.run([VPY, '-u', f'{ROOT}/star_bab_v3.py', net, prop,
                        '--timeout', str(to), '--workers', str(WORKERS),
                        '--results-file', rf],
                       cwd=ROOT, timeout=to + 60, capture_output=True)
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0
    dt = time.time() - t0
    v = open(rf).read().strip() if os.path.exists(rf) else 'NORESULT'
    return v, dt


def run_nnenum(net, prop, to):
    rf = f'/tmp/sweep_nn_{net}_{prop}.txt'
    if os.path.exists(rf):
        os.remove(rf)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=NSRC)
    t0 = time.time()
    try:
        subprocess.run([NPY, f'{ROOT}/nnenum_timed.py', net, prop, 'default',
                        str(WORKERS), str(to), rf],
                       cwd=ROOT, timeout=to + 60, capture_output=True, env=env)
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0
    dt = time.time() - t0
    v = open(rf).read().strip() if os.path.exists(rf) else 'NORESULT'
    return v, dt


def main():
    rows = parse_instances()
    out = f'{ROOT}/sweep_acasxu.csv'
    with open(out, 'w') as f:
        f.write('net,prop,v3,v3_s,nnenum,nn_s,status\n')
    n_agree = n_disagree = n_v3to = 0
    for i, (net, prop, to) in enumerate(rows):
        v3v, v3t = run_v3(net, prop, to)
        nnv, nnt = run_nnenum(net, prop, to)
        # status: AGREE / DISAGREE(critical if v3 unsat vs nnenum sat or vice versa)
        both = {v3v, nnv}
        if v3v in ('unsat', 'sat') and nnv in ('unsat', 'sat'):
            status = 'AGREE' if v3v == nnv else 'DISAGREE!!'
            n_agree += (v3v == nnv); n_disagree += (v3v != nnv)
        elif v3v == 'timeout':
            status = 'v3_timeout'; n_v3to += 1
        else:
            status = f'inconclusive({v3v}/{nnv})'
        line = f'{net},{prop},{v3v},{v3t:.1f},{nnv},{nnt:.1f},{status}'
        with open(out, 'a') as f:
            f.write(line + '\n')
        print(f'[{i+1}/{len(rows)}] {line}', flush=True)
    print(f'\nDONE: agree={n_agree} disagree={n_disagree} v3_timeout={n_v3to}',
          flush=True)


if __name__ == '__main__':
    main()
