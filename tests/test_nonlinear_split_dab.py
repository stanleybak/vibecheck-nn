"""Integration tests for nonlinear splits in the GPU dual-ascent BaB.

`build_nonlinear_split_caches` must produce LP caches that the band-agnostic
`_batched_dual_ascent` solver consumes soundly. Validated against an exact
scipy LP oracle (weak duality: dual best_g ≤ LP_min, never over-certifies) and
against a brute-force sample of the TRUE network (the relaxed node LP must
lower-bound the true sub-domain minimum). Also: splitting tightens
(min over children ≥ parent). CPU float64; GPU-equivalence covered separately.
"""
import numpy as np
import torch
from scipy.optimize import linprog

from vibecheck.nonlinear_relax import REGISTRY
from vibecheck.nl_pow import PowRelax
from vibecheck.nonlinear_split_planes import split_point
from vibecheck.nonlinear_split_dual import (backward_sensitivity,
                                            band_change_correction,
                                            split_halfspace)
from vibecheck import dual_ascent_bab as dab

_F64 = torch.float64


def _make_instance(relax, op_type, lo, hi, seed, n_in=4):
    """One nonlinear neuron: n_in input symbols + 1 e_new symbol.

    Returns (splittable dict, d_np, c0, n_gens, parent-band, g_k, extra, c_const)
    with a chosen so z = c_in + a·e ranges exactly over [lo,hi].
    """
    gen = np.random.default_rng(seed)
    a = gen.standard_normal(n_in)
    a = a * ((0.5 * (hi - lo)) / np.abs(a).sum())
    c_in = 0.5 * (lo + hi)
    lam, mu, delta = (float(t) for t in relax.affine_band_alpha(
        torch.tensor(lo, dtype=_F64), torch.tensor(hi, dtype=_F64),
        torch.tensor(0.5, dtype=_F64)))
    g_k = float(gen.standard_normal()) * 1.5
    extra = gen.standard_normal(n_in)
    c_const = float(gen.standard_normal())
    n_gens = n_in + 1
    e_new_col = n_in
    d_np = np.zeros(n_gens)
    d_np[:n_in] = g_k * lam * a + extra
    d_np[e_new_col] = g_k * delta
    c0 = g_k * (lam * c_in + mu) + c_const
    sp = dict(relax=relax, op_type=op_type, lo=lo, hi=hi, c_in=c_in,
              row_indices=np.arange(n_in), row_values=a, e_new_col=e_new_col,
              lam=lam, mu=mu, delta=delta)
    return sp, d_np, c0, n_gens, (lam, mu, delta), g_k, extra, c_const, a


def _lp_min(d, c0, A_ub, b_ub, n_gens):
    """Exact box-LP minimum of c0 + d·e s.t. A_ub·e ≤ b_ub, e ∈ [-1,1]^n."""
    bounds = [(-1.0, 1.0)] * n_gens
    res = linprog(c=d, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    assert res.success, res.message
    return c0 + float(res.fun)


_CASES = [
    (PowRelax(2), 'pow', -2.0, 3.0),
    (REGISTRY['Sigmoid'](), 'sigmoid', -3.0, 4.0),
    (REGISTRY['Tanh'](), 'tanh', -4.0, 2.0),
    (REGISTRY['Sin'](), 'sin', 0.3, 2.4),
    (REGISTRY['Cos'](), 'cos', -1.0, 1.6),
]


def _dual_best_g(d_np, c0, A_rows, b_rows, n_gens, max_iter=200):
    """Run the batched dual-ascent on a single node; return best_g (lower bd)."""
    device = torch.device('cpu')
    m = A_rows.shape[0]
    d_b = torch.as_tensor(d_np, dtype=_F64).unsqueeze(0)
    c0_b = torch.tensor([c0], dtype=_F64)
    A_b = torch.as_tensor(A_rows, dtype=_F64).reshape(1, m, n_gens)
    b_b = torch.as_tensor(b_rows, dtype=_F64).reshape(1, m)
    lam_b = torch.zeros(1, m, dtype=_F64)
    e_lb = torch.full((n_gens,), -1.0, dtype=_F64)
    e_hi = torch.full((n_gens,), 1.0, dtype=_F64)
    width = e_hi - e_lb
    best_g, _, _, _ = dab._batched_dual_ascent(
        d_b, c0_b, A_b, b_b, lam_b, torch.ones(1, dtype=torch.bool),
        e_lb, e_hi, width, max_iter=max_iter, repair_steps=5,
        feas_tol=1e-9, tol=1e-12, dtype=_F64, device=device)
    return float(best_g[0])


def test_caches_match_per_neuron_correction():
    # The assembled caches must equal the standalone band correction (the
    # builder just packs band_change_correction + split_halfspace into arrays).
    from vibecheck.nonlinear_split_dual import (band_change_correction,
                                                split_halfspace)
    for relax, op_type, lo, hi in _CASES:
        sp, d_np, c0, n_gens, _, _, _, _, a = _make_instance(
            relax, op_type, lo, hi, seed=7)
        hs_A, hs_b, d_corr, c0_corr = dab.build_nonlinear_split_caches(
            [sp], d_np, c0, n_gens, torch.device('cpu'), _F64)
        p = float(split_point(op_type, lo, hi))
        g_k = float(backward_sensitivity(d_np[sp['e_new_col']], sp['delta']))
        for side, (clo, chi, name) in enumerate(
                ((lo, p, 'left'), (p, hi, 'right'))):
            # default cache uses the PARENT slope (alpha=None): offset re-fit.
            lam_n, mu_n, delta_n = (float(t) for t in relax.affine_band(
                torch.tensor(clo, dtype=_F64), torch.tensor(chi, dtype=_F64),
                lam=torch.tensor(sp['lam'], dtype=_F64)))
            dcr, dce, c0c = band_change_correction(
                g_k, sp['c_in'], a, sp['lam'], sp['mu'], sp['delta'],
                lam_n, mu_n, delta_n)
            exp_d = np.zeros(n_gens)
            exp_d[:len(a)] = np.asarray(dcr)
            exp_d[sp['e_new_col']] = float(dce)
            assert torch.allclose(d_corr[0, side],
                                  torch.as_tensor(exp_d, dtype=_F64), atol=1e-12)
            assert abs(float(c0_corr[0, side]) - float(c0c)) < 1e-12
            hs_row, hs_rhs = split_halfspace(sp['c_in'], a, p, name)
            exp_hs = np.zeros(n_gens); exp_hs[:len(a)] = np.asarray(hs_row)
            assert torch.allclose(hs_A[0, side, 0],
                                  torch.as_tensor(exp_hs, dtype=_F64), atol=1e-12)
            assert abs(float(hs_b[0, side, 0]) - float(hs_rhs)) < 1e-12
            assert float(hs_b[0, side, 1]) == 1.0       # no-op pad row
            assert torch.count_nonzero(hs_A[0, side, 1]) == 0


def test_dual_sound_vs_lp_and_relaxation_sound_vs_truth():
    for relax, op_type, lo, hi in _CASES:
        sp, d_np, c0, n_gens, (lam, mu, delta), g_k, extra, c_const, a = \
            _make_instance(relax, op_type, lo, hi, seed=3)
        hs_A, hs_b, d_corr, c0_corr = dab.build_nonlinear_split_caches(
            [sp], d_np, c0, n_gens, torch.device('cpu'), _F64)
        p = float(split_point(op_type, lo, hi))
        e = (np.random.default_rng(9).random((200000, len(a))) * 2 - 1)
        z = sp['c_in'] + e @ a
        for side, (clo, chi) in enumerate(((lo, p), (p, hi))):
            d_side = d_np + d_corr[0, side].numpy()
            c0_side = c0 + float(c0_corr[0, side])
            A_ub = hs_A[0, side, 0:1].numpy()       # the single real halfspace
            b_ub = hs_b[0, side, 0:1].numpy()
            lp = _lp_min(d_side, c0_side, A_ub, b_ub, n_gens)
            # weak duality: dual lower-bounds the LP, never over-certifies.
            bg = _dual_best_g(d_side, c0_side, A_ub, b_ub, n_gens)
            assert bg <= lp + 1e-6, f'{op_type} {side}: dual {bg} > LP {lp}'
            assert bg >= lp - 1e-3, f'{op_type} {side}: dual loose {bg} vs {lp}'
            # relaxation sound vs TRUE network on this sub-domain.
            keep = (z >= clo - 1e-12) & (z <= chi + 1e-12)
            if keep.sum() == 0:
                continue
            # TRUE objective: c0/d hold the band form (g_k·(λz+μ)+δ·e_new); the
            # real network contributes g_k·f(z) instead. The non-k parts are
            # c_const (the c0 remainder) + extra·e (the d remainder on inputs).
            true_obj = (c_const + e[keep] @ extra + g_k * relax.func(
                torch.as_tensor(z[keep], dtype=_F64)).numpy())
            assert lp <= float(true_obj.min()) + 1e-6, (
                f'{op_type} {side}: LP {lp} not ≤ true min {true_obj.min()}')


def test_alpha_given_band_is_sound():
    # The optional α-chosen-slope path (alpha != None) must still be a sound
    # relaxation, even though it may not tighten vs the parent.
    relax = REGISTRY['Tanh']()
    sp, d_np, c0, n_gens, (lam, mu, delta), g_k, extra, c_const, a = \
        _make_instance(relax, 'tanh', -4.0, 2.0, seed=8)
    hs_A, hs_b, d_corr, c0_corr = dab.build_nonlinear_split_caches(
        [sp], d_np, c0, n_gens, torch.device('cpu'), _F64,
        alpha_l=0.5, alpha_r=0.5)
    p = float(split_point('tanh', -4.0, 2.0))
    e = (np.random.default_rng(2).random((100000, len(a))) * 2 - 1)
    z = sp['c_in'] + e @ a
    for side, (clo, chi) in enumerate(((-4.0, p), (p, 2.0))):
        d_side = d_np + d_corr[0, side].numpy()
        c0_side = c0 + float(c0_corr[0, side])
        lp = _lp_min(d_side, c0_side, hs_A[0, side, 0:1].numpy(),
                     hs_b[0, side, 0:1].numpy(), n_gens)
        keep = (z >= clo - 1e-12) & (z <= chi + 1e-12)
        if keep.sum() == 0:
            continue
        true_obj = (c_const + e[keep] @ extra + g_k * relax.func(
            torch.as_tensor(z[keep], dtype=_F64)).numpy())
        assert lp <= float(true_obj.min()) + 1e-6


def test_split_tightens_vs_parent():
    for relax, op_type, lo, hi in _CASES:
        sp, d_np, c0, n_gens, _, _, _, _, a = _make_instance(
            relax, op_type, lo, hi, seed=4)
        parent_lp = _lp_min(d_np, c0, None, None, n_gens)
        hs_A, hs_b, d_corr, c0_corr = dab.build_nonlinear_split_caches(
            [sp], d_np, c0, n_gens, torch.device('cpu'), _F64)
        child = []
        for side in (0, 1):
            d_side = d_np + d_corr[0, side].numpy()
            c0_side = c0 + float(c0_corr[0, side])
            child.append(_lp_min(d_side, c0_side, hs_A[0, side, 0:1].numpy(),
                                 hs_b[0, side, 0:1].numpy(), n_gens))
        assert min(child) >= parent_lp - 1e-6, (
            f'{op_type}: split LOOSER than parent: min {min(child)} '
            f'< parent {parent_lp}')


def _node_best_g(sp, d_np, c0, n_gens, g_k, clo, chi, max_iter=8,
                 slope='chord'):
    """Dual lower bound for the node restricting neuron sp's pre-activation to
    z ∈ [clo,chi]: re-tighten the band on the sub-interval and pin z with two
    halfspaces (z≥clo, z≤chi). Pure end-to-end use of the dual-correction
    primitives + the batched solver. `slope='chord'` re-fits the sub-interval
    chord (δ shrinks quadratically -> fast BaB convergence); `slope='parent'`
    inherits the parent λ (guaranteed-tightening default, but linear shrink)."""
    a = sp['row_values']; c_in = sp['c_in']
    lam_arg = (None if slope == 'chord'
               else torch.tensor(sp['lam'], dtype=_F64))
    lam_n, mu_n, delta_n = (float(t) for t in sp['relax'].affine_band(
        torch.tensor(clo, dtype=_F64), torch.tensor(chi, dtype=_F64),
        lam=lam_arg))
    dcr, dce, c0c = band_change_correction(
        g_k, c_in, a, sp['lam'], sp['mu'], sp['delta'], lam_n, mu_n, delta_n)
    d_node = d_np.copy()
    d_node[:len(a)] += np.asarray(dcr)
    d_node[sp['e_new_col']] += float(dce)
    c0_node = c0 + float(c0c)
    rows, rhs = [], []
    lo_row, lo_b = split_halfspace(c_in, a, clo, 'right')   # z ≥ clo
    hi_row, hi_b = split_halfspace(c_in, a, chi, 'left')    # z ≤ chi
    for r, b in ((lo_row, lo_b), (hi_row, hi_b)):
        full = np.zeros(n_gens); full[:len(a)] = np.asarray(r)
        rows.append(full); rhs.append(float(b))
    A = np.stack(rows); b = np.asarray(rhs)
    return _dual_best_g(d_node, c0_node, A, b, n_gens, max_iter=max_iter)


def _certify_by_bisection(sp, d_np, c0, n_gens, g_k, clo, chi, depth, tol=1e-7):
    """Recursively bisect z∈[clo,chi]; certified iff every leaf's dual bound>0.
    Sound: children cover the parent (z≤m or z≥m). Returns (certified, nodes)."""
    bg = _node_best_g(sp, d_np, c0, n_gens, g_k, clo, chi)
    if bg > tol:
        return True, 1
    if depth == 0:
        return False, 1
    m = 0.5 * (clo + chi)
    okL, nL = _certify_by_bisection(sp, d_np, c0, n_gens, g_k, clo, m,
                                    depth - 1, tol)
    if not okL:
        return False, 1 + nL
    okR, nR = _certify_by_bisection(sp, d_np, c0, n_gens, g_k, m, chi,
                                    depth - 1, tol)
    return okR, 1 + nL + nR


def test_bisection_closes_safe_and_never_certifies_unsafe():
    # End-to-end: dual-ascent + nonlinear split must CLOSE a truly-safe gap the
    # root can't, and must NEVER certify a truly-unsafe instance (soundness).
    # 1-D pre-activation (n_in=1) so a fine grid gives true_min EXACTLY — the
    # safety shift then sits where we intend (random multi-D sampling misses
    # sharp corners and mis-sets the margin).
    eg = torch.linspace(-1.0, 1.0, 40001, dtype=_F64)
    MARGIN = 0.3
    n_closed = 0
    for relax, op_type, lo, hi in _CASES:
        # find a seed whose ROOT relaxation has a genuine gap at this margin
        # (root can't certify the +MARGIN-safe instance) so the split is what
        # closes it; not every op/seed has a wide gap.
        for seed in range(100, 130):
            sp, d_np, c0, n_gens, band, g_k, extra, c_const, a = \
                _make_instance(relax, op_type, lo, hi, seed=seed, n_in=1)
            z = sp['c_in'] + a[0] * eg
            true = c_const + extra[0] * eg + g_k * relax.func(z)
            true_min = float(true.min())
            off_safe = MARGIN - true_min
            root_g = _node_best_g(sp, d_np, c0, n_gens, g_k, lo, hi)
            if root_g + off_safe <= 0:
                break
        else:
            continue                       # no gap instance for this op
        n_closed += 1
        # SAFE: root fails, but bisecting the pre-activation must certify.
        okS, _ = _certify_by_bisection(sp, d_np, c0 + off_safe, n_gens,
                                       g_k, lo, hi, depth=22)
        assert okS, f'{op_type}: bisection failed to close a safe (gap) case'
        # UNSAFE: shift the same instance so true min = -MARGIN -> a real
        # counterexample exists, so NO sound relaxation may certify.
        off_unsafe = -MARGIN - true_min
        okU, _ = _certify_by_bisection(sp, d_np, c0 + off_unsafe, n_gens,
                                       g_k, lo, hi, depth=18)
        assert not okU, (
            f'{op_type}: FALSE-CERTIFIED an unsafe case '
            f'(true_min<0) — soundness violation')
    # at least some op exercised a real root gap closed by splitting.
    assert n_closed >= 1, 'no case exercised a real root gap closed by splitting'
