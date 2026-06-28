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
import pytest

import vibecheck.verify_milp as vm


def _slow_solver(args):
    idx = args[0]
    time.sleep(0.4)
    return idx, -1.0, 1.0, None, False


def _numeric_trouble_solver(args):
    # Mirror a worker that already retried at NumericFocus=3 and STILL tripped
    # numeric trouble: it re-raises, surfacing in the parent at it.next()
    # (deadline path) or pool.map (no-deadline path).
    raise vm.GurobiNumericTrouble('simulated numeric trouble')


@pytest.fixture(autouse=True)
def _reset_nf_globals():
    """Each test starts with the risky flag OFF and an empty event log."""
    vm._shared_nf_retry_risky = False
    vm._numeric_trouble_events = []
    yield
    vm._shared_nf_retry_risky = False
    vm._numeric_trouble_events = []


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


# --- default (flag OFF): numeric trouble must PROPAGATE, never be masked ----
def test_numeric_trouble_propagates_when_disabled_deadline(monkeypatch):
    """With the risky flag OFF (default), a worker GurobiNumericTrouble is
    NOT masked — it re-raises so main records `error` and the numeric problem
    is surfaced, not hidden."""
    monkeypatch.setattr(vm, '_solve_neuron_both', _numeric_trouble_solver)
    n = 8
    layers_np = [{'type': 'fc', 'W': np.eye(n), 'bias': np.zeros(n)}]
    bounds = {0: (np.full(n, -2.0), np.full(n, 2.0))}
    with pytest.raises(vm.GurobiNumericTrouble):
        vm._tighten_layer_parallel(
            layers_np, np.full(n, -1.0), np.full(n, 1.0), bounds, 0,
            use_milp=False, timeout=5.0, n_cores=2, lp_per_worker=True,
            witness_n_random=0, deadline=time.perf_counter() + 5.0)


def test_numeric_trouble_propagates_when_disabled_map(monkeypatch):
    """Same, on the no-deadline pool.map() path: flag OFF -> propagate."""
    monkeypatch.setattr(vm, '_solve_neuron_both', _numeric_trouble_solver)
    n = 6
    layers_np = [{'type': 'fc', 'W': np.eye(n), 'bias': np.zeros(n)}]
    bounds = {0: (np.full(n, -2.0), np.full(n, 2.0))}
    with pytest.raises(vm.GurobiNumericTrouble):
        vm._tighten_layer_parallel(
            layers_np, np.full(n, -1.0), np.full(n, 1.0), bounds, 0,
            use_milp=False, timeout=5.0, n_cores=2, lp_per_worker=True,
            witness_n_random=0, deadline=None)


# --- risky flag ON: worker retried + still dirty -> skip + loud log + event -
def test_numeric_trouble_skipped_deadline_path(monkeypatch, capsys):
    """With the risky flag ON, a worker that retried at NumericFocus=3 and
    STILL hit numeric trouble has its neuron tightening dropped (keeping the
    looser, sound pre-tightening bound); the rest keep going and the event is
    logged loudly + recorded for details. (metaroom 6cnn_ry_39_6 dead-end.)"""
    monkeypatch.setattr(vm, '_solve_neuron_both', _numeric_trouble_solver)
    vm._shared_nf_retry_risky = True
    n = 8
    layers_np = [{'type': 'fc', 'W': np.eye(n), 'bias': np.zeros(n)}]
    lo, hi = np.full(n, -2.0), np.full(n, 2.0)         # all unstable
    bounds = {0: (lo.copy(), hi.copy())}

    new_lo, new_hi, any_timeout = vm._tighten_layer_parallel(
        layers_np, np.full(n, -1.0), np.full(n, 1.0), bounds, 0,
        use_milp=False, timeout=5.0, n_cores=2,
        lp_per_worker=True, witness_n_random=0,
        deadline=time.perf_counter() + 5.0)

    # No result survived -> looser incoming bounds kept verbatim.
    assert np.array_equal(new_lo, lo) and np.array_equal(new_hi, hi)
    assert 'NUMERIC TROUBLE' in capsys.readouterr().out
    assert vm._numeric_trouble_events, 'event must be recorded for details'
    assert vm._numeric_trouble_events[0]['phase'] == 'tighten'


def test_numeric_trouble_skipped_map_path(monkeypatch, capsys):
    """Same, on the no-deadline pool.map() path: it has no per-task
    isolation, so the whole layer's tightening is dropped wholesale (still
    sound — keeps the looser incoming bounds)."""
    monkeypatch.setattr(vm, '_solve_neuron_both', _numeric_trouble_solver)
    vm._shared_nf_retry_risky = True
    n = 6
    layers_np = [{'type': 'fc', 'W': np.eye(n), 'bias': np.zeros(n)}]
    lo, hi = np.full(n, -2.0), np.full(n, 2.0)
    bounds = {0: (lo.copy(), hi.copy())}

    new_lo, new_hi, any_timeout = vm._tighten_layer_parallel(
        layers_np, np.full(n, -1.0), np.full(n, 1.0), bounds, 0,
        use_milp=False, timeout=5.0, n_cores=2,
        lp_per_worker=True, witness_n_random=0, deadline=None)

    assert np.array_equal(new_lo, lo) and np.array_equal(new_hi, hi)
    assert not any_timeout
    assert 'NUMERIC TROUBLE' in capsys.readouterr().out
    assert vm._numeric_trouble_events


# --- _optimize_nf_retry unit behavior (in-process, no multiprocessing) ------
class _FakeModel:
    def __init__(self, fail_times):
        self.fail_times = fail_times      # how many optimize calls trip trouble
        self.calls = 0
        self.params = {}

    def setParam(self, k, v):
        self.params[k] = v


def _make_optimize(model):
    def _optimize(m):
        m.calls += 1
        if m.calls <= m.fail_times:
            raise vm.GurobiNumericTrouble('boom')
    return _optimize


def test_optimize_nf_retry_disabled_propagates(monkeypatch):
    """Flag OFF: no retry, the first trouble propagates (NumericFocus untouched)."""
    m = _FakeModel(fail_times=1)
    monkeypatch.setattr(vm, 'optimize_checked', _make_optimize(m))
    vm._shared_nf_retry_risky = False
    with pytest.raises(vm.GurobiNumericTrouble):
        vm._optimize_nf_retry(m)
    assert m.calls == 1 and 'NumericFocus' not in m.params


def test_optimize_nf_retry_recovers(monkeypatch, capsys):
    """Flag ON: first solve trips trouble, retry at NumericFocus=3 is clean."""
    m = _FakeModel(fail_times=1)
    monkeypatch.setattr(vm, 'optimize_checked', _make_optimize(m))
    vm._shared_nf_retry_risky = True
    used = vm._optimize_nf_retry(m)
    assert used is True and m.calls == 2
    assert m.params.get('NumericFocus') == 3 and m.params.get('ScaleFlag') == 2
    assert 'RESOLVED by NumericFocus=3' in capsys.readouterr().out


def test_optimize_nf_retry_still_dirty_reraises(monkeypatch):
    """Flag ON: if the NumericFocus=3 retry ALSO trips trouble it re-raises —
    a still-dirty bound is never trusted."""
    m = _FakeModel(fail_times=2)
    monkeypatch.setattr(vm, 'optimize_checked', _make_optimize(m))
    vm._shared_nf_retry_risky = True
    with pytest.raises(vm.GurobiNumericTrouble):
        vm._optimize_nf_retry(m)
    assert m.calls == 2 and m.params.get('NumericFocus') == 3
