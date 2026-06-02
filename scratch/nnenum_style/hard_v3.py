import subprocess, os, time
ROOT='/home/stan/repositories/vibecheck/scratch/nnenum_style'
VPY='/home/stan/repositories/vibecheck/.venv/bin/python'
CASES=[l.split('#')[0].split()[:2] for l in open(f'{ROOT}/hard_set.txt') if l.split('#')[0].split()]
with open(f'{ROOT}/hard_v3.out','w') as f:
    for net,prop in CASES:
        rf=f'/tmp/hv3_{net}_{prop}.txt'
        if os.path.exists(rf): os.remove(rf)
        t0=time.time()
        try:
            subprocess.run([VPY,'-u',f'{ROOT}/star_bab_v3.py',net,prop,'--timeout','116','--workers','14','--results-file',rf],cwd=ROOT,timeout=170,capture_output=True)
        except subprocess.TimeoutExpired: pass
        dt=time.time()-t0
        v=open(rf).read().strip() if os.path.exists(rf) else 'NORESULT'
        within = 'YES' if (v in ('unsat','sat') and dt<=116) else ('TIMEOUT' if dt>116 or v=='timeout' else v)
        line=f"{net} {prop}: verdict={v} wall={dt:.1f}s within_116s={within}"
        f.write(line+"\n"); f.flush(); print(line, flush=True)
