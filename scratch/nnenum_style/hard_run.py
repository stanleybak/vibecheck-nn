"""Run the hard set: v3 (parallel nw=14) vs nnenum default vs nnenum matched.
Verdicts from per-case results-files (never exit code). Writes hard_run.csv."""
import sys, os, subprocess, time

ROOT = '/home/stan/repositories/vibecheck/scratch/nnenum_style'
VPY = '/home/stan/repositories/vibecheck/.venv/bin/python'
NPY = os.path.expanduser('~/repositories/nnenum/.venv/bin/python')
NSRC = os.path.expanduser('~/repositories/nnenum/src')
TO = 200.0
NW = 14

CASES = []
for ln in open(f'{ROOT}/hard_set.txt'):
    ln = ln.split('#')[0].split()
    if len(ln) >= 2:
        CASES.append((ln[0], ln[1]))


def runv3(net, prop):
    rf = f'/tmp/hard_v3_{net}_{prop}.txt'
    if os.path.exists(rf): os.remove(rf)
    t0 = time.time()
    try:
        subprocess.run([VPY, '-u', f'{ROOT}/star_bab_v3.py', net, prop,
                        '--timeout', str(TO), '--workers', str(NW),
                        '--results-file', rf], cwd=ROOT, timeout=TO + 90,
                       capture_output=True)
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0
    dt = time.time() - t0
    v = open(rf).read().strip() if os.path.exists(rf) else 'NORESULT'
    return v, dt


def runnn(net, prop, mode):
    rf = f'/tmp/hard_nn_{mode}_{net}_{prop}.txt'
    if os.path.exists(rf): os.remove(rf)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=NSRC)
    t0 = time.time()
    try:
        subprocess.run([NPY, f'{ROOT}/nnenum_timed.py', net, prop, mode, '16',
                        str(TO), rf], cwd=ROOT, timeout=TO + 90,
                       capture_output=True, env=env)
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0
    dt = time.time() - t0
    v = open(rf).read().strip() if os.path.exists(rf) else 'NORESULT'
    return v, dt


rows = []
with open(f'{ROOT}/hard_run.csv', 'w') as f:
    f.write('net,prop,v3,v3_s,nn_default,nnd_s,nn_matched,nnm_s\n')
    for net, prop in CASES:
        v3v, v3t = runv3(net, prop)
        ndv, ndt = runnn(net, prop, 'default')
        nmv, nmt = runnn(net, prop, 'matched')
        line = f'{net},{prop},{v3v},{v3t:.1f},{ndv},{ndt:.1f},{nmv},{nmt:.1f}'
        f.write(line + '\n'); f.flush()
        rows.append((net, prop, v3v, v3t, ndv, ndt, nmv, nmt))
        print(line, flush=True)

print("\n%-12s %-18s %-18s %-18s" % ('case', 'v3 (nw=14)', 'nnenum default', 'nnenum matched'))
for net, prop, v3v, v3t, ndv, ndt, nmv, nmt in rows:
    print("%-12s %-18s %-18s %-18s" % (
        f'{net} {prop}', f'{v3v} {v3t:.1f}s', f'{ndv} {ndt:.1f}s', f'{nmv} {nmt:.1f}s'))
