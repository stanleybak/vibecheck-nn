"""_tighten_layer_parallel must respect its wall-clock deadline.

The per-neuron `timeout` bounds ONE Gurobi solve; a full layer is hundreds
of solves through a blocking pool — without the deadline the phase blew
through total_timeout 2x+ (cora: 30s budget -> 70s walls). The fix consumes
results via imap_unordered(chunksize=1).next(timeout) and terminates the
pool at the wall; partial results are each valid bounds (sound).

The worker is monkeypatched to a sleeper — fork-start children inherit the
patched module state on Linux, so no Gurobi is involved.
"""
import time

import numpy as np

import vibecheck.verify_milp as vm


def _slow_solver(args):
    idx = args[0]
    time.sleep(0.4)
    return idx, -1.0, 1.0, None, False


def test_deadline_stops_early_with_partial_results(monkeypatch):
    monkeypatch.setattr(vm, '_solve_neuron_both', _slow_solver)
    n = 16
    layers_np = [{'type': 'fc', 'W': np.eye(n), 'bias': np.zeros(n)}]
    bounds = {0: (np.full(n, -2.0), np.full(n, 2.0))}   # all unstable
    x_lo = np.full(n, -1.0)
    x_hi = np.full(n, 1.0)

    t0 = time.perf_counter()
    new_lo, new_hi, any_timeout = vm._tighten_layer_parallel(
        layers_np, x_lo, x_hi, bounds, 0,
        use_milp=False, timeout=5.0, n_cores=2,
        lp_per_worker=True, witness_n_random=0,
        deadline=time.perf_counter() + 1.0)
    wall = time.perf_counter() - t0

    # 16 tasks x 0.4s on 2 cores = 3.2s of work; the 1.0s deadline must
    # cut it off well before that (some slack for pool startup/teardown).
    assert wall < 2.5, f'deadline not enforced: wall={wall:.2f}s'
    assert any_timeout, 'early stop must be reported as a timeout'
    # whatever did complete was applied; bounds stay valid either way
    assert np.all(new_lo >= -2.0) and np.all(new_hi <= 2.0)
