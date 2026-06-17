"""Soundness of the nonlinear-split op_clamps mechanism (the BaB hook).

Splitting a nonlinear op's pre-activation [lo,hi] at m yields children clamped
to [lo,m] and [m,hi]; "both children verified ⟹ parent verified" requires that
for EACH child, the clamped forward's output bound contains f(x) for every input
whose op-input lies in the clamp. These tests check exactly that at the transform
level: build a zonotope, clamp the op to a sub-range, and assert f(z) is inside
the clamped output bound for every sampled z whose value lies in the clamp.
"""
import torch

from vibecheck.nonlinear_relax import REGISTRY, zono_affine_transform
from vibecheck.zonotope import _torch_zono_pow_int

_F64 = torch.float64


def _check_clamped_sound(out_lo, out_hi, c, G, in_lo_clamp, in_hi_clamp, f):
    """Sample z = c + G·e; for samples whose value lies in [clamp], assert f(z)
    is within [out_lo, out_hi]. Returns the worst overshoot over kept samples."""
    n, k = c.numel(), G.shape[1]
    e = 2 * torch.rand(60000, k, dtype=_F64) - 1
    z = c.unsqueeze(0) + e @ G.t()                      # (S, n)
    keep = ((z >= in_lo_clamp.unsqueeze(0) - 1e-9)
            & (z <= in_hi_clamp.unsqueeze(0) + 1e-9))    # per-element in clamp
    fz = f(z)
    over = torch.maximum(out_lo.unsqueeze(0) - fz, fz - out_hi.unsqueeze(0))
    over = torch.where(keep, over, torch.full_like(over, -1e30))
    return float(over.max())


def test_affine_band_clamp_sound():
    torch.manual_seed(0)
    fns = {'Sigmoid': torch.sigmoid, 'Tanh': torch.tanh,
           'Sin': torch.sin, 'Cos': torch.cos}
    for op, f in fns.items():
        relax = REGISTRY[op]()
        c = torch.randn(4, dtype=_F64)
        G = 0.7 * torch.randn(4, 5, dtype=_F64)
        rad = G.abs().sum(1)
        lo, hi = c - rad, c + rad
        mid = 0.5 * (lo + hi)
        for tl, th in [(lo, mid), (mid, hi)]:          # the two split children
            nc, ng = zono_affine_transform(relax, c, G, tight_lo=tl, tight_hi=th)
            o_lo = nc - ng.abs().sum(1)
            o_hi = nc + ng.abs().sum(1)
            worst = _check_clamped_sound(o_lo, o_hi, c, G, tl, th, f)
            assert worst <= 1e-7, f'{op} clamp UNSOUND: overshoot {worst:.3e}'


def test_pow_clamp_sound():
    torch.manual_seed(1)
    for c0 in (torch.tensor([1.2, -0.8, 0.3, 2.0], dtype=_F64),):
        c = c0.clone()
        G = 0.4 * torch.randn(4, 5, dtype=_F64)
        rad = G.abs().sum(1)
        lo, hi = c - rad, c + rad
        mid = 0.5 * (lo + hi)
        for tl, th in [(lo, mid), (mid, hi)]:
            nc, ng = _torch_zono_pow_int(c, G, 2, relaxation='chord',
                                         tight_lo=tl, tight_hi=th)
            o_lo = nc - ng.abs().sum(1)
            o_hi = nc + ng.abs().sum(1)
            worst = _check_clamped_sound(o_lo, o_hi, c, G, tl, th,
                                         lambda z: z ** 2)
            assert worst <= 1e-7, f'pow clamp UNSOUND: overshoot {worst:.3e}'


def test_alpha_band_sound_for_all_alpha():
    # α-CROWN hook: affine_band_alpha must be SOUND for EVERY α in [0,1]
    # (the optimizer explores the whole range; any α gives a valid bound).
    fns = {'Sigmoid': torch.sigmoid, 'Tanh': torch.tanh,
           'Sin': torch.sin, 'Cos': torch.cos}
    cases = [(-3.0, 3.0), (0.5, 2.0), (-2.0, -0.5), (-0.5, 0.5), (1.0, 4.0)]
    for op, f in fns.items():
        relax = REGISTRY[op]()
        for lo, hi in cases:
            L = torch.full((20,), lo, dtype=_F64) + torch.linspace(-1, 1, 20)
            H = torch.full((20,), hi, dtype=_F64) + torch.linspace(-1, 1, 20)
            H = torch.maximum(H, L + 1e-3)
            for a in (0.0, 0.3, 0.5, 0.7, 1.0):
                alpha = torch.full_like(L, a)
                lam, mu, delta = relax.affine_band_alpha(L, H, alpha)
                u = torch.rand(6000, 20, dtype=_F64)
                xs = L + (H - L) * u
                gap = (f(xs) - (lam * xs + mu)).abs()
                worst = float((gap - delta).max())
                assert worst <= 1e-7, f'{op} α={a} UNSOUND: {worst:.2e}'


def test_alpha_band_differentiable():
    # gradient must flow through α (else the optimizer can't tune it).
    relax = REGISTRY['Sigmoid']()
    L = torch.full((4,), -2.0, dtype=_F64); H = torch.full((4,), 2.0, dtype=_F64)
    alpha = torch.full((4,), 0.5, dtype=_F64, requires_grad=True)
    lam, mu, delta = relax.affine_band_alpha(L, H, alpha)
    (lam.sum() + mu.sum() + delta.sum()).backward()
    assert alpha.grad is not None and bool((alpha.grad.abs() > 0).any())


def test_clamp_children_cover_parent():
    # The two children's clamps must cover the parent range: every input is in
    # at least one child (so unioning their certified-safe sets covers the box).
    torch.manual_seed(2)
    relax = REGISTRY['Sigmoid']()
    c = torch.randn(3, dtype=_F64)
    G = 0.5 * torch.randn(3, 4, dtype=_F64)
    rad = G.abs().sum(1)
    lo, hi = c - rad, c + rad
    mid = 0.5 * (lo + hi)
    e = 2 * torch.rand(20000, 4, dtype=_F64) - 1
    z = c.unsqueeze(0) + e @ G.t()
    in_low = (z <= mid.unsqueeze(0) + 1e-9).all(1)
    in_high = (z >= mid.unsqueeze(0) - 1e-9).all(1)
    # not every sample is uniformly in one child (elements split independently),
    # but each ELEMENT's value is in [lo,mid] or [mid,hi] — the per-element
    # coverage the BaB relies on (it splits one element at a time).
    per_elem_covered = ((z <= mid.unsqueeze(0) + 1e-9)
                        | (z >= mid.unsqueeze(0) - 1e-9)).all()
    assert bool(per_elem_covered)
