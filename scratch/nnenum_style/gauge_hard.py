"""Gauge which acasxu cases are hard (long-running) for v3 single-threaded.
Writes one line per case to gauge_hard.out immediately (flush), so a kill leaves
partial results. Per-case timeout keeps it bounded."""
import sys, time
import star_bab_v3 as v3

CASES = [('2_2', 'prop_3'), ('3_3', 'prop_2'), ('4_2', 'prop_2'),
         ('1_9', 'prop_7'), ('2_2', 'prop_2'), ('5_3', 'prop_2'),
         ('4_6', 'prop_1'), ('3_3', 'prop_9')]
PER = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

with open('gauge_hard.out', 'w') as f:
    f.write(f"# per-case timeout = {PER}s\n"); f.flush()
    for net, prop in CASES:
        t0 = time.time()
        try:
            safe, dt, stats = v3.verify(net, prop, gen_cap=None, multizono=True,
                                        witness_contract=True, single_bound=True,
                                        timeout=PER)
            r = 'SAFE' if safe else ('TIMEOUT' if safe is None else 'NOTSAFE')
            line = (f"{net} {prop}: {r} {dt:.1f}s splits={stats['splits']} "
                    f"LP={v3.LP_SOLVES[0]}")
        except Exception as e:
            line = f"{net} {prop}: ERR {type(e).__name__}: {e}"
        f.write(line + "\n"); f.flush()
        print(line, flush=True)
