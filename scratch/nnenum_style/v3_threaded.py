"""v3 single- vs multi-threaded on a hard acasxu case. Wall time + num LPs.
Writes incrementally to v3_threaded.out. Usage: v3_threaded.py NET PROP TIMEOUT W1 W2..."""
import sys, time
import star_bab_v3 as v3

net = sys.argv[1]; prop = sys.argv[2]
TO = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
workers = [int(x) for x in sys.argv[4:]] or [1, 8, 16]

cfg = dict(gen_cap=None, multizono=True, witness_contract=True, single_bound=True)
out = f'v3_threaded_{net}_{prop}.out'
with open(out, 'w') as f:
    f.write(f"# v3 {net} {prop}, timeout={TO}s, config={cfg}\n"); f.flush()
    for nw in workers:
        t0 = time.time()
        safe, dt, stats = v3.verify(net, prop, n_workers=nw, timeout=TO, **cfg)
        wall = time.time() - t0
        r = 'SAFE' if safe else ('TIMEOUT' if safe is None else 'NOTSAFE')
        line = (f"n_workers={nw:2d}: {r:8s} wall={wall:7.1f}s solve={dt:7.1f}s "
                f"splits={stats['splits']:6d} LP={v3.LP_SOLVES[0]:8d} "
                f"frontier={stats.get('frontier','-')}")
        f.write(line + "\n"); f.flush()
        print(line, flush=True)
