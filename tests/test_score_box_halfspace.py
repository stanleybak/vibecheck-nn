"""Unit tests for ``score_box_halfspace_delta_lb`` (Phase-8 branching scores).

The scorer computes, per unstable neuron, the worst-child ``delta_LB`` via a
box+halfspace Lagrangian dual. It was rewritten to operate on each neuron's
*sparse* generator support (the halfspace ``a`` is the neuron's pre-ReLU row,
which is zero off-support) instead of materialising a dense ``n_gens`` vector
per neuron. These tests pin that the sparse path is bit-identical to the dense
reference (the old algorithm: build the full ``row_full`` and call
``box_halfspace.lagrangian_min``), and exercise the degenerate-``mu`` fallback
and the infeasible (``+inf``) branch.
"""
import numpy as np
import scipy.sparse as sp

from vibecheck.box_halfspace import lagrangian_min
from vibecheck.verify_gen_lp import score_box_halfspace_delta_lb


def _dense_reference(state, qw, qb, ew_per_relu):
    """The pre-rewrite dense scorer — ground truth for the sparse version."""
    obj_c_out = np.asarray(state['obj_c_out'], dtype=np.float64)
    obj_G_out = state['obj_G_out_csr'].toarray().astype(np.float64)
    n_gens = state['n_gens']
    qw = np.asarray(qw, dtype=np.float64)
    qb = float(qb)
    d_obj_base = qw @ obj_G_out
    c0_obj_base = float(qw @ obj_c_out + qb)
    baseline_lb = c0_obj_base - float(np.sum(np.abs(d_obj_base)))
    scores = {}
    skip = state.get('sigmoid_tanh_layer_ids', set())
    for u in state.get('unstable_list', []):
        li = u['layer_idx']
        if li in skip:
            continue
        j = u['neuron_idx']
        c_in_k = float(u['c_in'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)
        row_val = np.asarray(u['row_values'], dtype=np.float64)
        e_new_col = int(u['e_new_col'])
        lo_k = float(u['lo']); hi_k = float(u['hi'])
        if 'lam' in u and 'mu' in u:
            lam = float(u['lam']); mu = float(u['mu'])
        else:
            if hi_k <= lo_k or hi_k <= 0 or lo_k >= 0:
                scores[(li, j)] = 0.0
                continue
            lam = hi_k / (hi_k - lo_k)
            mu = -hi_k * lo_k / (2.0 * (hi_k - lo_k))
        if mu > 1e-12:
            ew_k = float(d_obj_base[e_new_col]) / mu
        else:
            ew_l = ew_per_relu.get(li)
            if ew_l is None or j >= len(ew_l):
                scores[(li, j)] = 0.0
                continue
            ew_k = float(ew_l[j])
        row_full = np.zeros(n_gens, dtype=np.float64)
        row_full[row_idx] = row_val
        d_off = d_obj_base - ew_k * lam * row_full
        d_off[e_new_col] -= ew_k * mu
        c0_off = c0_obj_base - ew_k * (lam * c_in_k + mu)
        lb_off = lagrangian_min(d_off, c0_off, row_full, -c_in_k)
        d_on = d_obj_base + ew_k * (1.0 - lam) * row_full
        d_on[e_new_col] -= ew_k * mu
        c0_on = c0_obj_base + ew_k * ((1.0 - lam) * c_in_k - mu)
        lb_on = lagrangian_min(d_on, c0_on, -row_full, c_in_k)
        scores[(li, j)] = float(min(lb_off, lb_on)) - baseline_lb
    return scores


def _make_state(rng, n_gens, n_unstable, n_input, degenerate=(), infeasible=()):
    """Build a synthetic alpha_zono state.

    degenerate: neuron indices to give mu≈0 (forces the ew fallback path).
    infeasible: neuron indices set up so the off-side sub-LP is infeasible
                (c_in > Σ|row_val| ⇒ min a·e > β = −c_in over the box), which
                drives ``_lmin_sparse`` through its breakpoints-exhausted
                ``+inf`` return.
    """
    n_out = 3
    obj_c_out = rng.standard_normal(n_out)
    obj_G = rng.standard_normal((n_out, n_gens))
    unstable = []
    col = n_input
    for k in range(n_unstable):
        lo_k = float(-abs(rng.standard_normal()) - 0.1)
        hi_k = float(abs(rng.standard_normal()) + 0.1)
        lam = hi_k / (hi_k - lo_k)
        if k in degenerate:
            mu = 0.0
        else:
            mu = -hi_k * lo_k / (2.0 * (hi_k - lo_k))
        # Support over a random subset of the input generators (these precede
        # the new slack column, matching real pre-ReLU rows).
        s = int(rng.integers(2, max(3, n_input)))
        idx = np.sort(rng.choice(n_input, size=s, replace=False)).astype(np.int64)
        val = rng.standard_normal(s)
        c_in = float(rng.standard_normal())
        if k in infeasible:
            val = np.array([0.2, 0.2], dtype=np.float64)
            idx = np.sort(rng.choice(n_input, size=2, replace=False)).astype(np.int64)
            c_in = 5.0  # > Σ|val| = 0.4 ⇒ off-side halfspace infeasible vs box
        unstable.append(dict(
            layer_idx=1 + (k % 2), neuron_idx=k, c_in=c_in,
            lo=lo_k, hi=hi_k, lam=lam, mu=mu, e_new_col=col, form='alpha_zono',
            row_indices=idx, row_values=val))
        col += 1
    state = dict(
        n_gens=n_gens, n_input=n_input, formulation='sparse',
        unstable_list=unstable, stable_list=[],
        obj_c_out=obj_c_out, obj_G_out_csr=sp.csr_matrix(obj_G),
        sigmoid_tanh_layer_ids=set())
    return state


def test_sparse_matches_dense_reference():
    rng = np.random.default_rng(12345)
    n_input = 12
    state = _make_state(rng, n_gens=40, n_unstable=20, n_input=n_input)
    qw = rng.standard_normal(3); qb = float(rng.standard_normal())
    got = score_box_halfspace_delta_lb(state, qw, qb, {})
    ref = _dense_reference(state, qw, qb, {})
    assert set(got) == set(ref)
    for key in ref:
        a, b = ref[key], got[key]
        if np.isinf(a) and np.isinf(b):
            continue
        assert abs(a - b) < 1e-10, (key, a, b)


def test_degenerate_mu_uses_ew_fallback():
    """mu≈0 neurons must fall back to ew_per_relu, matching the dense path."""
    rng = np.random.default_rng(7)
    n_input = 10
    state = _make_state(rng, n_gens=30, n_unstable=12, n_input=n_input,
                        degenerate=(2, 5, 9))
    qw = rng.standard_normal(3); qb = 0.3
    # ew_per_relu must cover both layer_idx 1 and 2 for the fallback neurons.
    ew = {1: rng.standard_normal(12), 2: rng.standard_normal(12)}
    got = score_box_halfspace_delta_lb(state, qw, qb, ew)
    ref = _dense_reference(state, qw, qb, ew)
    assert set(got) == set(ref)
    for key in ref:
        assert abs(ref[key] - got[key]) < 1e-10, (key, ref[key], got[key])


def test_infeasible_subproblem_matches_reference():
    """An off-side sub-LP that is infeasible vs the box drives the
    breakpoints-exhausted ``+inf`` branch inside the dual. The final score
    still collapses to the feasible on-side via ``min``, so we assert the
    sparse path agrees with the dense reference (which exercises the same
    internal ``+inf``)."""
    rng = np.random.default_rng(99)
    n_input = 8
    state = _make_state(rng, n_gens=25, n_unstable=10, n_input=n_input,
                        infeasible=(1, 4))
    qw = rng.standard_normal(3); qb = -0.2
    got = score_box_halfspace_delta_lb(state, qw, qb, {})
    ref = _dense_reference(state, qw, qb, ew_per_relu={})
    assert set(got) == set(ref)
    for key in ref:
        a, b = ref[key], got[key]
        if np.isinf(a) and np.isinf(b):
            continue
        assert abs(a - b) < 1e-10, (key, a, b)
    # The infeasible neurons must still produce a finite score (on-side feasible)
    # and that score must match the dense reference exactly.
    for k in (1, 4):
        assert np.isfinite(got[(1 + (k % 2), k)])


def test_phase1_form_recomputes_lam_mu():
    """A phase1-form state omits lam/mu; the scorer must recompute them from
    (lo, hi) and skip genuinely-stable entries (hi≤0 or lo≥0) with score 0.
    Matches the dense reference, which recomputes identically."""
    rng = np.random.default_rng(21)
    n_input = 6
    state = _make_state(rng, n_gens=20, n_unstable=6, n_input=n_input)
    # Strip lam/mu to force the recompute branch.
    for u in state['unstable_list']:
        u.pop('lam'); u.pop('mu')
    # Make one entry stable (hi ≤ 0) → must be skipped with score 0.
    state['unstable_list'][2]['lo'] = -2.0
    state['unstable_list'][2]['hi'] = -0.5
    qw = rng.standard_normal(3); qb = 0.1
    got = score_box_halfspace_delta_lb(state, qw, qb, {})
    ref = _dense_reference(state, qw, qb, {})
    assert set(got) == set(ref)
    for key in ref:
        a, b = ref[key], got[key]
        if np.isinf(a) and np.isinf(b):
            continue
        assert abs(a - b) < 1e-10, (key, a, b)
    li2 = state['unstable_list'][2]['layer_idx']
    assert got[(li2, 2)] == 0.0


def test_zero_objective_support_infeasible_branch():
    """Hand-crafted single neuron: objective is zero on the row support and
    ew=0, so every breakpoint is invalid (d=0 ⇒ not valid.any()) while gp0>0
    (c_in > Σ|a|) ⇒ the dual is infeasible (+inf). Exercises the early
    ``not valid.any(): return inf`` guard in ``_lmin_sparse``."""
    n_gens = 5
    # obj_G_out: row 0 has zeros on the support cols {0,1} so d_base there is 0;
    # column 2 (= e_new_col) is nonzero so d_base[enc] ≠ 0.
    obj_G = np.zeros((1, n_gens), dtype=np.float64)
    obj_G[0, 2] = 1.0
    state = dict(
        n_gens=n_gens, n_input=2, formulation='sparse', stable_list=[],
        obj_c_out=np.array([0.0]), obj_G_out_csr=sp.csr_matrix(obj_G),
        sigmoid_tanh_layer_ids=set(),
        unstable_list=[dict(
            layer_idx=1, neuron_idx=0, c_in=10.0,  # ≫ Σ|row_val| = 0.4 ⇒ gp0>0
            lo=-1.0, hi=1.0, lam=0.5, mu=0.0,       # mu=0 ⇒ ew fallback
            e_new_col=2, form='alpha_zono',
            row_indices=np.array([0, 1], dtype=np.int64),
            row_values=np.array([0.2, 0.2], dtype=np.float64))])
    qw = np.array([1.0]); qb = 0.0
    ew = {1: np.array([0.0])}  # ew_k = 0 ⇒ d_off on support stays 0
    got = score_box_halfspace_delta_lb(state, qw, qb, ew)
    ref = _dense_reference(state, qw, qb, ew)
    assert set(got) == set(ref)
    for key in ref:
        a, b = ref[key], got[key]
        if np.isinf(a) and np.isinf(b):
            continue
        assert abs(a - b) < 1e-10, (key, a, b)


def test_degenerate_mu_missing_ew_scores_zero():
    """A degenerate (mu≈0) neuron whose layer is absent from ew_per_relu has
    no resolvable ew_k and must score 0 (both sparse and dense paths)."""
    n_gens = 6
    obj_G = np.random.default_rng(1).standard_normal((1, n_gens))
    state = dict(
        n_gens=n_gens, n_input=3, formulation='sparse', stable_list=[],
        obj_c_out=np.array([0.0]), obj_G_out_csr=sp.csr_matrix(obj_G),
        sigmoid_tanh_layer_ids=set(),
        unstable_list=[dict(
            layer_idx=7, neuron_idx=0, c_in=0.1, lo=-1.0, hi=1.0,
            lam=0.5, mu=0.0, e_new_col=3, form='alpha_zono',
            row_indices=np.array([0, 1], dtype=np.int64),
            row_values=np.array([0.3, -0.2], dtype=np.float64))])
    qw = np.array([1.0]); qb = 0.0
    got = score_box_halfspace_delta_lb(state, qw, qb, ew_per_relu={})  # no layer 7
    assert got[(7, 0)] == 0.0
    assert got == _dense_reference(state, qw, qb, ew_per_relu={})


def test_skip_layers_excluded():
    rng = np.random.default_rng(3)
    state = _make_state(rng, n_gens=20, n_unstable=8, n_input=6)
    state['sigmoid_tanh_layer_ids'] = {2}
    qw = rng.standard_normal(3); qb = 0.0
    got = score_box_halfspace_delta_lb(state, qw, qb, {})
    assert all(li != 2 for (li, _j) in got)
    assert got, "layer-1 neurons should still be scored"
