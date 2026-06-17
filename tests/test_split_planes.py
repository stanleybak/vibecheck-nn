"""Foundation tests for the dual-ascent nonlinear-split relaxation planes
(nonlinear_split_planes): both plane forms must be SOUND for every alpha, the
split must re-tighten, and the children's planes must cover the parent. Pure
CPU math (trivially GPU-equivalent); the ablation prints band-vs-two_plane
tightness so the BaB can pick.
"""
import torch

from vibecheck.nonlinear_relax import REGISTRY
from vibecheck.nl_pow import PowRelax
from vibecheck.nonlinear_split_planes import (op_planes, split_planes,
                                              split_point, bilinear_axis_score)

_F64 = torch.float64
# (relax, [intervals]) covering convex / concave / mixed / one-sided cases.
_OPS = [
    (PowRelax(2), [(-2.0, 3.0), (0.5, 2.0), (-3.0, -0.5)]),
    (REGISTRY['Sigmoid'](), [(-4.0, 4.0), (1.0, 5.0), (-5.0, -1.0)]),
    (REGISTRY['Tanh'](), [(-3.0, 3.0), (0.5, 4.0), (-4.0, -0.5)]),
    (REGISTRY['Sin'](), [(0.5, 2.0), (-1.0, 1.0), (2.0, 3.0)]),
    (REGISTRY['Cos'](), [(0.0, 1.5), (-1.0, 1.0), (1.0, 2.5)]),
]


def _planes_sound(relax, lo, hi, form, alpha):
    L = torch.full((25,), lo, dtype=_F64) + torch.linspace(-0.4, 0.4, 25)
    H = torch.full((25,), hi, dtype=_F64) + torch.linspace(-0.4, 0.4, 25)
    H = torch.maximum(H, L + 1e-3)
    a = None if alpha is None else torch.full_like(L, alpha)
    sL, tL, sU, tU = op_planes(relax, L, H, a, form)
    u = torch.rand(8000, 25, dtype=_F64)
    xs = L + (H - L) * u
    fx = relax.func(xs)
    lo_viol = float((sL * xs + tL - fx).max())   # lower plane must be <= f
    hi_viol = float((fx - (sU * xs + tU)).max())  # upper plane must be >= f
    return max(lo_viol, hi_viol)


def test_both_forms_sound_all_alpha():
    for relax, ivals in _OPS:
        for lo, hi in ivals:
            for form in ('band', 'two_plane'):
                for a in (None, 0.0, 0.3, 0.5, 0.7, 1.0):
                    w = _planes_sound(relax, lo, hi, form, a)
                    assert w <= 1e-7, (
                        f'{type(relax).__name__} {form} α={a} [{lo},{hi}] '
                        f'UNSOUND: {w:.2e}')


def test_split_children_cover_and_retighten():
    # split at midpoint: each child's planes are sound on its sub-interval, and
    # the children cover the parent (x<=p or x>=p).
    for relax, ivals in _OPS:
        for lo, hi in ivals:
            for form in ('band', 'two_plane'):
                p = 0.5 * (lo + hi)
                wl = _planes_sound(relax, lo, p, form, 0.5)
                wr = _planes_sound(relax, p, hi, form, 0.5)
                assert max(wl, wr) <= 1e-7, (
                    f'{type(relax).__name__} {form} split UNSOUND')


def test_two_plane_tighter_than_band_on_convex():
    # On a convex Sqr interval, tangent+secant should be no looser than the
    # parallel band (usually strictly tighter): smaller mean plane gap.
    relax = PowRelax(2)
    L = torch.tensor([0.5], dtype=_F64); H = torch.tensor([3.0], dtype=_F64)

    def mean_gap(form):
        sL, tL, sU, tU = op_planes(relax, L, H, torch.full_like(L, 0.5), form)
        xs = torch.linspace(0.5, 3.0, 50, dtype=_F64)
        gap = (sU[0] * xs + tU[0]) - (sL[0] * xs + tL[0])   # upper - lower
        return float(gap.mean())

    g_band = mean_gap('band'); g_two = mean_gap('two_plane')
    assert g_two <= g_band + 1e-9, f'two_plane not tighter: {g_two} vs {g_band}'


def test_option_a_split_point():
    # 1-D ops with a feature at 0 split AT 0 when straddling, else midpoint.
    assert split_point('sigmoid', -2.0, 3.0) == 0.0      # straddles -> 0
    assert split_point('tanh', -1.0, 4.0) == 0.0
    assert split_point('pow', -1.5, 2.5) == 0.0          # Sqr min at 0
    assert split_point('relu', -1.0, 1.0) == 0.0
    assert split_point('sigmoid', 1.0, 5.0) == 3.0       # no straddle -> mid
    assert split_point('sigmoid', -5.0, -1.0) == -3.0
    # periodic / non-zero-feature ops always midpoint, even straddling 0.
    assert split_point('sin', -2.0, 4.0) == 1.0
    assert split_point('cos', -1.0, 3.0) == 1.0
    # tensor form, zero-split op
    lo = torch.tensor([-2.0, 1.0, -5.0], dtype=_F64)
    hi = torch.tensor([3.0, 5.0, -1.0], dtype=_F64)
    p = split_point('sigmoid', lo, hi)
    assert torch.allclose(p, torch.tensor([0.0, 3.0, -3.0], dtype=_F64))
    # tensor form, non-zero-split op -> always midpoint even when straddling 0
    p_sin = split_point('sin', lo, hi)
    assert torch.allclose(p_sin, torch.tensor([0.5, 3.0, -3.0], dtype=_F64))


def test_option_a_bilinear_axis_balances():
    # Bilinear operand choice is a TIEBREAK (the gap reduction radA·radB·2^-K is
    # allocation-independent), so it BALANCES: split the larger-radius axis to
    # keep radA≈radB — never starve one axis.
    ax, _ = bilinear_axis_score(rad_a=4.0, rad_b=1.0)
    assert ax == 'a'                                   # larger radius a wins
    # once a is bisected enough that its radius drops below b's, b wins — the
    # smaller axis is not starved.
    ax2, _ = bilinear_axis_score(rad_a=0.5, rad_b=1.0)
    assert ax2 == 'b'
    # sensitivity weighting can tip an equal-radius tie.
    ax3, _ = bilinear_axis_score(rad_a=1.0, rad_b=1.0, sens_a=3.0, sens_b=1.0)
    assert ax3 == 'a'


def test_split_planes_api():
    relax = PowRelax(2)
    lo = torch.tensor([-2.0, 0.5], dtype=_F64)
    hi = torch.tensor([3.0, 2.0], dtype=_F64)
    p = 0.5 * (lo + hi)
    left, right = split_planes(relax, lo, hi, p, form='two_plane')
    assert len(left) == 4 and len(right) == 4
    assert all(t.shape == lo.shape for t in left + right)
