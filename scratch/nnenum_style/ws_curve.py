import star_bab_v3 as v3, time, os
base = 157.0
with open('ws_curve.out','w') as f:
    for nw in [2,4,6,8,16]:
        t0=time.time(); s,dt,st=v3.verify('3_3','prop_2',gen_cap=None,multizono=True,witness_contract=True,single_bound=True,n_workers=nw,chunk=1,timeout=160)
        w=time.time()-t0
        line=f"nw={nw:2d}: {'SAFE' if s else s} {w:5.1f}s speedup={base/w:4.1f}x eff={100*base/w/nw:.0f}% LP={v3.LP_SOLVES[0]}"
        f.write(line+"\n"); f.flush(); print(line, flush=True)
