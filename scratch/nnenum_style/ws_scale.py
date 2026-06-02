"""Work-stealing scaling on 3_3 prop_2. Writes ws_scale.out incrementally."""
import star_bab_v3 as v3
import time

base = None
with open('ws_scale.out', 'w') as f:
    for nw in [1, 8, 16]:
        t0 = time.time()
        safe, dt, stats = v3.verify('3_3', 'prop_2', gen_cap=None, multizono=True,
                                    witness_contract=True, single_bound=True,
                                    n_workers=nw, chunk=1, timeout=400)
        w = time.time() - t0
        if nw == 1:
            base = w
        r = 'SAFE' if safe is True else ('SAT' if safe is False else 'TIMEOUT')
        line = (f"nw={nw:2d}: {r} {w:.1f}s speedup={base/w:.1f}x "
                f"splits={stats['splits']} LP={v3.LP_SOLVES[0]}")
        f.write(line + "\n"); f.flush()
        print(line, flush=True)
