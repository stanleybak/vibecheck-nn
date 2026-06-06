"""``parse_problem_gpu`` must produce a Problem identical to ``parse_problem``.

The GPU parser builds the split matrix ``a_g`` directly on the device via a
flat scatter (avoiding the dense ``(n_unstable, n_gens)`` host alloc + upload)
and computes ``d_t`` via a sparse matvec instead of densifying ``obj_G_out_csr``.
Both are semantics-preserving, so every Problem field must match: ``a_g`` exactly
in float32, the scalar vectors to float64 tolerance. We exercise it on the CPU
device so the test runs without a GPU.
"""
import numpy as np
import scipy.sparse as sp
import torch

from vibecheck.fast_dual_ascent import parse_problem, parse_problem_gpu


def _make_state(rng, n_gens, n_unstable, n_input):
    n_out = 4
    obj_c_out = rng.standard_normal(n_out)
    obj_G = rng.standard_normal((n_out, n_gens))
    # Zero most of obj_G to keep it genuinely sparse (matches real states).
    obj_G[rng.random((n_out, n_gens)) < 0.7] = 0.0
    unstable = []
    col = n_input
    for k in range(n_unstable):
        lo_k = float(-abs(rng.standard_normal()) - 0.1)
        hi_k = float(abs(rng.standard_normal()) + 0.1)
        lam = hi_k / (hi_k - lo_k)
        mu = -hi_k * lo_k / (2.0 * (hi_k - lo_k))
        s = int(rng.integers(2, max(3, n_input)))
        idx = np.sort(rng.choice(n_input, size=s, replace=False)).astype(np.int64)
        val = rng.standard_normal(s)
        unstable.append(dict(
            layer_idx=1 + (k % 2), neuron_idx=k, c_in=float(rng.standard_normal()),
            lo=lo_k, hi=hi_k, lam=lam, mu=mu, e_new_col=col, form='alpha_zono',
            row_indices=idx, row_values=val))
        col += 1
    state = dict(
        n_gens=n_gens, n_input=n_input, formulation='sparse',
        unstable_list=unstable, stable_list=[],
        obj_c_out=obj_c_out, obj_G_out_csr=sp.csr_matrix(obj_G),
        sigmoid_tanh_layer_ids=set())
    return state


def _assert_problems_match(pc, pg):
    # a_g: GPU parser builds float32 directly; CPU parser is float64 cast to
    # float32 at upload time — must be bit-identical in float32.
    ag_cpu = torch.as_tensor(np.asarray(pc.a_g), dtype=torch.float32)
    ag_gpu = pg.a_g.to(torch.float32).cpu() if torch.is_tensor(pg.a_g) else \
        torch.as_tensor(pg.a_g, dtype=torch.float32)
    assert torch.equal(ag_cpu, ag_gpu), \
        f'a_g differs (max {float((ag_cpu - ag_gpu).abs().max()):.2e})'
    assert pc.n_gens == pg.n_gens
    np.testing.assert_array_equal(pc.e_new_col, pg.e_new_col)
    np.testing.assert_array_equal(pc.e_lb, pg.e_lb)
    np.testing.assert_array_equal(pc.e_hi, pg.e_hi)
    assert abs(pc.c0 - pg.c0) < 1e-12
    for f in ('d_t', 'ratio_off', 'ratio_on', 'c0_off', 'c0_on', 'c_in', 'z_lo', 'z_hi'):
        a = np.asarray(getattr(pc, f), float); b = np.asarray(getattr(pg, f), float)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=f)
    # derived bound must agree
    assert abs(pc.root_bound - pg.root_bound) < 1e-9


def test_parse_gpu_matches_cpu_all_scored():
    rng = np.random.default_rng(2024)
    state = _make_state(rng, n_gens=50, n_unstable=20, n_input=12)
    qw = rng.standard_normal(4); qb = float(rng.standard_normal())
    keys = [(u['layer_idx'], u['neuron_idx']) for u in state['unstable_list']]
    pc = parse_problem(state, qw, qb, keys)
    pg = parse_problem_gpu(state, qw, qb, keys, torch.device('cpu'))
    _assert_problems_match(pc, pg)


def test_parse_gpu_matches_cpu_scored_subset_reordered():
    """scored_keys is a reordered subset — a_g rows must follow scored order."""
    rng = np.random.default_rng(7)
    state = _make_state(rng, n_gens=40, n_unstable=16, n_input=10)
    qw = rng.standard_normal(4); qb = 0.25
    keys = [(u['layer_idx'], u['neuron_idx']) for u in state['unstable_list']]
    rng.shuffle(keys)
    keys = keys[:9]  # subset, reordered
    pc = parse_problem(state, qw, qb, keys)
    pg = parse_problem_gpu(state, qw, qb, keys, torch.device('cpu'))
    assert pc.n_splits == pg.n_splits == 9
    _assert_problems_match(pc, pg)


def test_parse_gpu_empty_scored_keys():
    rng = np.random.default_rng(1)
    state = _make_state(rng, n_gens=20, n_unstable=5, n_input=6)
    qw = rng.standard_normal(4); qb = 0.0
    pc = parse_problem(state, qw, qb, [])
    pg = parse_problem_gpu(state, qw, qb, [], torch.device('cpu'))
    assert pc.n_splits == pg.n_splits == 0
    _assert_problems_match(pc, pg)
