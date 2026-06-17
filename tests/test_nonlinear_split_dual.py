"""Dual-space split-correction tests (nonlinear_split_dual).

The correction must EXACTLY reproduce an honest zonotope re-propagation with the
child band (algebraic equivalence — the primary test), and the corrected
objective + side halfspace must SOUNDLY lower-bound the true objective over the
split sub-domain (sampled). Pure CPU float64; trivially GPU-equivalent.
"""
import torch

from vibecheck.nonlinear_relax import REGISTRY
from vibecheck.nl_pow import PowRelax
from vibecheck.nonlinear_split_planes import op_planes, split_point
from vibecheck.nonlinear_split_dual import (backward_sensitivity,
                                            band_change_correction,
                                            split_halfspace)

_F64 = torch.float64
_OPS = [
    (PowRelax(2), 'pow', (-2.0, 3.0)),
    (PowRelax(2), 'pow', (0.5, 2.5)),
    (REGISTRY['Sigmoid'](), 'sigmoid', (-3.0, 4.0)),
    (REGISTRY['Tanh'](), 'tanh', (-4.0, 2.0)),
    (REGISTRY['Sin'](), 'sin', (0.3, 2.4)),
    (REGISTRY['Cos'](), 'cos', (-1.0, 1.6)),
]


def _band(relax, lo, hi, alpha):
    """Return (lam, mu, delta) scalars for the affine band over [lo,hi]."""
    a = None if alpha is None else torch.tensor(alpha, dtype=_F64)
    lam, mu, delta = relax.affine_band_alpha(
        torch.tensor(lo, dtype=_F64), torch.tensor(hi, dtype=_F64),
        a if a is not None else torch.tensor(0.5, dtype=_F64))
    return float(lam), float(mu), float(delta)


def _scenario(relax, lo, hi, seed):
    """Build a concrete 1-neuron zonotope + scalar objective.

    Returns a dict with the input row `a` (over n_in symbols), `c_in`, the
    parent band, the backward sensitivity `g_k`, the objective (d over
    [a-cols, e_new-col] and c0), and a sampler for the exact pre-activation z.
    `a` and `c_in` are chosen so z's range is exactly [lo,hi].
    """
    gen = torch.Generator().manual_seed(seed)
    n_in = 4
    a = torch.randn(n_in, generator=gen, dtype=_F64)
    rad = float(a.abs().sum())
    c_in = 0.5 * (lo + hi)
    scale = (0.5 * (hi - lo)) / rad           # so z-range == [lo,hi]
    a = a * scale
    lam, mu, delta = _band(relax, lo, hi, 0.5)
    g_k = float(torch.randn((), generator=gen, dtype=_F64)) * 1.7
    extra = torch.randn(n_in, generator=gen, dtype=_F64)   # other neurons' share
    c_const = float(torch.randn((), generator=gen, dtype=_F64))
    # objective d over columns [0..n_in) = a-cols, [n_in] = e_new col.
    d = torch.empty(n_in + 1, dtype=_F64)
    d[:n_in] = g_k * lam * a + extra
    d[n_in] = g_k * delta
    c0 = g_k * (lam * c_in + mu) + c_const
    return dict(n_in=n_in, a=a, c_in=c_in, lam=lam, mu=mu, delta=delta,
                g_k=g_k, extra=extra, c_const=c_const, d=d, c0=c0, gen=gen)


def test_backward_sensitivity_recovers_g():
    for relax, _name, (lo, hi) in _OPS:
        for seed in (1, 2, 3):
            s = _scenario(relax, lo, hi, seed)
            g = backward_sensitivity(s['d'][s['n_in']], s['delta'])
            assert abs(float(g) - s['g_k']) < 1e-9, (
                f'{_name}: g_k {float(g)} != {s["g_k"]}')


def test_correction_matches_honest_repropagation():
    # The single load-bearing test: applying the correction must equal building
    # the objective from scratch with the child band — for both split sides,
    # every op, several alphas.
    for relax, name, (lo, hi) in _OPS:
        for seed in (10, 11, 12):
            s = _scenario(relax, lo, hi, seed)
            n_in = s['n_in']
            p = float(split_point(name, lo, hi))
            g = backward_sensitivity(s['d'][n_in], s['delta'])
            for (clo, chi), side in (((lo, p), 'left'), ((p, hi), 'right')):
                for alpha in (0.0, 0.5, 1.0):
                    lam_n, mu_n, delta_n = _band(relax, clo, chi, alpha)
                    # honest re-propagation: rebuild d, c0 with child band.
                    d_honest = s['d'].clone()
                    d_honest[:n_in] = s['g_k'] * lam_n * s['a'] + s['extra']
                    d_honest[n_in] = s['g_k'] * delta_n
                    c0_honest = s['g_k'] * (lam_n * s['c_in'] + mu_n) \
                        + s['c_const']
                    # correction path.
                    dcr, dce, c0c = band_change_correction(
                        g, s['c_in'], s['a'],
                        s['lam'], s['mu'], s['delta'],
                        lam_n, mu_n, delta_n)
                    d_corr = s['d'].clone()
                    d_corr[:n_in] = d_corr[:n_in] + dcr
                    d_corr[n_in] = d_corr[n_in] + dce
                    c0_corr = s['c0'] + float(c0c)
                    assert torch.allclose(d_corr, d_honest, atol=1e-9), (
                        f'{name} {side} α={alpha}: d mismatch '
                        f'{(d_corr - d_honest).abs().max():.2e}')
                    assert abs(c0_corr - c0_honest) < 1e-9, (
                        f'{name} {side} α={alpha}: c0 mismatch')


def test_corrected_objective_sound_lower_bound():
    # For sampled inputs in a split sub-domain, the relaxed objective (e_new
    # free in [-1,1]) must lower-bound the TRUE objective g_k·f(z)+extra·e+const.
    torch.manual_seed(0)
    for relax, name, (lo, hi) in _OPS:
        s = _scenario(relax, lo, hi, seed=21)
        n_in = s['n_in']
        p = float(split_point(name, lo, hi))
        g = backward_sensitivity(s['d'][n_in], s['delta'])
        for (clo, chi), side in (((lo, p), 'left'), ((p, hi), 'right')):
            lam_n, mu_n, delta_n = _band(relax, clo, chi, 0.5)
            dcr, dce, c0c = band_change_correction(
                g, s['c_in'], s['a'], s['lam'], s['mu'], s['delta'],
                lam_n, mu_n, delta_n)
            d_corr = s['d'].clone()
            d_corr[:n_in] += dcr
            d_corr[n_in] += dce
            c0_corr = s['c0'] + float(c0c)
            # sample e in box; keep only those whose z lands in [clo,chi].
            e = (torch.rand(20000, n_in, dtype=_F64) * 2 - 1)
            z = s['c_in'] + e @ s['a']
            keep = (z >= clo - 1e-12) & (z <= chi + 1e-12)
            e = e[keep]; z = z[keep]
            if e.shape[0] == 0:
                continue
            true_obj = (s['g_k'] * relax.func(z) + e @ s['extra']
                        + s['c_const'])
            # relaxed obj minimised over e_new in [-1,1]: linear in e_new with
            # coeff d_corr[n_in], so min at -sign(coeff).
            lin = c0_corr + e @ d_corr[:n_in]
            relaxed_min = lin - d_corr[n_in].abs()
            assert bool((relaxed_min <= true_obj + 1e-7).all()), (
                f'{name} {side}: relaxation NOT a lower bound, worst '
                f'{float((relaxed_min - true_obj).max()):.2e}')


def test_backward_sensitivity_guards_zero_delta():
    # δ→0 (band already exact): g_k is irrelevant (its band terms vanish), so
    # we return 0 rather than divide by ~0.
    g = backward_sensitivity(torch.tensor(0.7, dtype=_F64),
                             torch.tensor(1e-40, dtype=_F64))
    assert float(g) == 0.0


def test_split_halfspace_rejects_bad_side():
    import pytest
    with pytest.raises(ValueError):
        split_halfspace(0.0, torch.tensor([1.0], dtype=_F64), 0.5, 'middle')


def test_split_halfspace_pins_preactivation():
    # a·e ≤ p−c_in  ⟺  z ≤ p (left);  −a·e ≤ −(p−c_in)  ⟺  z ≥ p (right).
    s = _scenario(REGISTRY['Sigmoid'](), -3.0, 4.0, seed=5)
    n_in = s['n_in']
    p = 0.7
    e = (torch.rand(5000, n_in, dtype=_F64) * 2 - 1)
    z = s['c_in'] + e @ s['a']
    for side, want in (('left', z <= p), ('right', z >= p)):
        hs_a, hs_b = split_halfspace(s['c_in'], s['a'], p, side)
        satisfied = (e @ hs_a) <= float(hs_b) + 1e-12
        assert bool((satisfied == want).all() | (
            (z - p).abs() < 1e-9).all()), f'{side} halfspace mismatch'
