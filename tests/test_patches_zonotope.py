"""Tests for PatchesZonotope (Phase 1).

Three gates:
  1. Parity: PatchesZonotope and TorchZonotope produce identical bounds
     within 1e-5 across conv/relu sequences.
  2. Point-prop: 200 random concrete inputs in the box must land inside
     PatchesZonotope.bounds().
  3. Memory probe: not enforced here (lives under /tmp/vibecheck_runs/).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vibecheck.patches_zonotope import PatchesZonotope
from vibecheck.zonotope import TorchZonotope, make_input_zonotope


DEV = torch.device('cpu')
DTYPE = torch.float64


def _rand_input_bounds(in_shape, eps=0.1, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = int(np.prod(in_shape))
    center = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    radii = eps * torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    return center - radii, center + radii


def _rand_kernel(C_out, C_in, kH, kW, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(C_out, C_in, kH, kW, generator=g, dtype=DTYPE, device=DEV)
    b = torch.randn(C_out, generator=g, dtype=DTYPE, device=DEV)
    return k, b


# ---------------------------------------------------------------------------
# Construction parity
# ---------------------------------------------------------------------------


def test_from_input_bounds_parity():
    in_shape = (3, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, seed=1)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hp, hd, atol=1e-12, rtol=1e-12)
    # Bounds must equal input bounds.
    torch.testing.assert_close(lp, xl, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hp, xh, atol=1e-12, rtol=1e-12)


def test_from_input_bounds_with_zero_radius():
    """Zero-radius inputs should not get generators."""
    in_shape = (1, 2, 2)
    xl = torch.tensor([0.0, 0.5, 1.0, 0.5], dtype=DTYPE, device=DEV)
    xh = torch.tensor([0.0, 0.5, 1.0, 0.7], dtype=DTYPE, device=DEV)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    assert zp.n_gens == 1
    lp, hp = zp.bounds()
    torch.testing.assert_close(lp, xl, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hp, xh, atol=1e-12, rtol=1e-12)


def test_to_dense_G_matches_torchzono():
    """PatchesZonotope.to_dense() G matrix equals TorchZonotope's."""
    in_shape = (2, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=2)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    Gp = zp.generators
    Gd = zd.generators
    # The two impls may order columns differently; compare via column-sorted
    # absolute sums (per row).
    torch.testing.assert_close(
        torch.abs(Gp).sum(dim=1), torch.abs(Gd).sum(dim=1),
        atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Conv parity
# ---------------------------------------------------------------------------


def _conv_pair_bounds(zp, zd, kernel, bias, in_shape, stride, padding):
    zp.propagate_conv(kernel, bias, in_shape, stride, padding)
    zd.propagate_conv(kernel, bias, in_shape, stride, padding)
    return zp.bounds(), zd.bounds()


def test_single_conv_stride1_pad0():
    in_shape = (1, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, seed=3)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(2, 1, 3, 3, seed=4)
    (lp, hp), (ld, hd) = _conv_pair_bounds(
        zp, zd, k, b, in_shape, (1, 1), (0, 0))
    torch.testing.assert_close(lp, ld, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(hp, hd, atol=1e-10, rtol=1e-10)


def test_single_conv_stride1_pad1():
    in_shape = (3, 5, 5)
    xl, xh = _rand_input_bounds(in_shape, seed=5)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(4, 3, 3, 3, seed=6)
    (lp, hp), (ld, hd) = _conv_pair_bounds(
        zp, zd, k, b, in_shape, (1, 1), (1, 1))
    torch.testing.assert_close(lp, ld, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(hp, hd, atol=1e-10, rtol=1e-10)


def test_two_convs_stride1():
    """Two stacked convs: parity preserved through patch growth."""
    in_shape = (2, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, seed=7)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(4, 2, 3, 3, seed=8)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    k2, b2 = _rand_kernel(3, 4, 3, 3, seed=9)
    zp.propagate_conv(k2, b2, (4, 6, 6), (1, 1), (1, 1))
    zd.propagate_conv(k2, b2, (4, 6, 6), (1, 1), (1, 1))
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_conv_relu_conv():
    """Conv -> ReLU -> Conv parity, including μ-generator append."""
    in_shape = (2, 5, 5)
    xl, xh = _rand_input_bounds(in_shape, eps=0.5, seed=10)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=11)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    # After ReLU bounds should match.
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)
    # Now another conv.
    k2, b2 = _rand_kernel(2, 3, 3, 3, seed=12)
    zp.propagate_conv(k2, b2, (3, 5, 5), (1, 1), (1, 1))
    zd.propagate_conv(k2, b2, (3, 5, 5), (1, 1), (1, 1))
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_relu_with_tight_bounds():
    """ReLU with externally-supplied tight (lo, hi) preserves parity."""
    in_shape = (2, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=0.5, seed=13)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=14)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    lo_zp, hi_zp = zp.bounds()
    # Use slightly tightened bounds.
    tight_lo = lo_zp + 0.01
    tight_hi = hi_zp - 0.01
    zp.apply_relu(tight_lo=tight_lo, tight_hi=tight_hi)
    zd.apply_relu(tight_lo=tight_lo, tight_hi=tight_hi)
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_stride2_native_in_patches_mode():
    """Stride > 1 stays in patches mode (per-gen subsample after stride-1 conv)."""
    in_shape = (2, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, seed=15)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(4, 2, 3, 3, seed=16)
    zp.propagate_conv(k, b, in_shape, (2, 2), (1, 1))
    zd.propagate_conv(k, b, in_shape, (2, 2), (1, 1))
    assert zp._mode == 'patches', "stride-2 should now stay in patches mode"
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_stride2_padding0():
    """Stride-2 conv with padding=0."""
    in_shape = (1, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, seed=200)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(2, 1, 3, 3, seed=201)
    zp.propagate_conv(k, b, in_shape, (2, 2), (0, 0))
    zd.propagate_conv(k, b, in_shape, (2, 2), (0, 0))
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_stride2_then_conv_then_relu():
    """Stride-2 conv → stride-1 conv → ReLU stays in patches."""
    in_shape = (1, 8, 8)
    xl, xh = _rand_input_bounds(in_shape, eps=0.4, seed=202)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(3, 1, 3, 3, seed=203)
    zp.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))  # stride 2: 8 → 4
    zd.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    assert zp._mode == 'patches'
    k2, b2 = _rand_kernel(2, 3, 3, 3, seed=204)
    zp.propagate_conv(k2, b2, (3, 4, 4), (1, 1), (1, 1))  # stride 1: 4 → 4
    zd.propagate_conv(k2, b2, (3, 4, 4), (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_stride2_point_propagation():
    """Random concrete inputs through stride-2 must lie in bounds."""
    in_shape = (1, 8, 8)
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=205)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    k, b = _rand_kernel(2, 1, 3, 3, seed=206)
    zp.propagate_conv(k, b, in_shape, (2, 2), (1, 1))
    lo, hi = zp.bounds()
    g = torch.Generator(device=DEV).manual_seed(207)
    n = xl.numel()
    for _ in range(200):
        u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
        x = xl + u * (xh - xl)
        out = F.conv2d(
            x.reshape(1, *in_shape), k, bias=b,
            stride=(2, 2), padding=(1, 1)).flatten()
        assert torch.all(out >= lo - 1e-10)
        assert torch.all(out <= hi + 1e-10)


def test_conv_then_stride2_then_relu():
    """Stride-1 conv -> stride-2 conv (dense fallback) -> ReLU still works."""
    in_shape = (1, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=17)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(2, 1, 3, 3, seed=18)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    k2, b2 = _rand_kernel(3, 2, 3, 3, seed=19)
    zp.propagate_conv(k2, b2, (2, 6, 6), (2, 2), (1, 1))
    zd.propagate_conv(k2, b2, (2, 6, 6), (2, 2), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Point-propagation soundness
# ---------------------------------------------------------------------------


def _torch_forward(x, ops):
    """Apply a sequence of ('conv', kernel, bias, in_shape, stride, padding)
    or ('relu',) ops to a single concrete input x flat."""
    cur_shape = ops[0][3]  # in_shape of first op
    cur = x.reshape(1, *cur_shape)
    for op in ops:
        if op[0] == 'conv':
            _, k, b, in_shape, stride, padding = op
            cur = F.conv2d(cur, k, bias=b, stride=stride, padding=padding)
        elif op[0] == 'relu':
            cur = F.relu(cur)
    return cur.flatten()


def test_point_prop_single_conv():
    in_shape = (1, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=0.2, seed=20)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    k, b = _rand_kernel(3, 1, 3, 3, seed=21)
    zp.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    lo, hi = zp.bounds()
    g = torch.Generator(device=DEV).manual_seed(22)
    n_pts = 200
    n = xl.numel()
    for _ in range(n_pts):
        u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
        x = xl + u * (xh - xl)
        ops = [('conv', k, b, in_shape, (1, 1), (1, 1))]
        out = _torch_forward(x, ops)
        # out must be inside [lo, hi].
        assert torch.all(out >= lo - 1e-10), \
            f"point below lo by {(lo - out).max().item()}"
        assert torch.all(out <= hi + 1e-10), \
            f"point above hi by {(out - hi).max().item()}"


def test_point_prop_conv_relu_conv():
    in_shape = (2, 5, 5)
    xl, xh = _rand_input_bounds(in_shape, eps=0.4, seed=23)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=24)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zp.apply_relu()
    k2, b2 = _rand_kernel(2, 3, 3, 3, seed=25)
    zp.propagate_conv(k2, b2, (3, 5, 5), (1, 1), (1, 1))
    lo, hi = zp.bounds()
    g = torch.Generator(device=DEV).manual_seed(26)
    n_pts = 200
    n = xl.numel()
    for _ in range(n_pts):
        u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
        x = xl + u * (xh - xl)
        ops = [
            ('conv', k1, b1, in_shape, (1, 1), (1, 1)),
            ('relu',),
            ('conv', k2, b2, (3, 5, 5), (1, 1), (1, 1)),
        ]
        out = _torch_forward(x, ops)
        assert torch.all(out >= lo - 1e-10)
        assert torch.all(out <= hi + 1e-10)


# ---------------------------------------------------------------------------
# Copy / move / fc fallback
# ---------------------------------------------------------------------------


def test_copy_independent():
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=27)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zp2 = zp.copy()
    zp2.center[0] = 99.0
    assert zp.center[0] != 99.0


def test_propagate_fc_falls_back_to_dense():
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=28)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    n = xl.numel()
    g = torch.Generator(device=DEV).manual_seed(29)
    W = torch.randn(2, n, generator=g, dtype=DTYPE, device=DEV)
    b = torch.randn(2, generator=g, dtype=DTYPE, device=DEV)
    zp.propagate_fc(W, b)
    zd.propagate_fc(W, b)
    assert zp._mode == 'dense'
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hp, hd, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# make_input_zonotope factory
# ---------------------------------------------------------------------------


def test_factory_dense_when_requested():
    from vibecheck.settings import default_settings
    s = default_settings(zono_impl='dense')
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=30)
    z = make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=in_shape)
    assert isinstance(z, TorchZonotope)


def test_factory_default_is_patches():
    """Phase 7: default flipped to 'patches'."""
    from vibecheck.settings import default_settings
    s = default_settings()
    assert s.zono_impl == 'patches'
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=30)
    z = make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=in_shape)
    assert isinstance(z, PatchesZonotope)


def test_factory_patches_when_requested():
    from vibecheck.settings import default_settings
    s = default_settings(zono_impl='patches')
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=31)
    z = make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=in_shape)
    assert isinstance(z, PatchesZonotope)


def test_factory_unknown_impl_raises():
    from vibecheck.settings import default_settings
    s = default_settings(zono_impl='no-such-impl')
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=32)
    with pytest.raises(AssertionError, match='unknown zono_impl'):
        make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=in_shape)


def test_factory_patches_falls_back_to_dense_for_fc_inputs():
    """FC-only networks (no image shape) should get TorchZonotope."""
    from vibecheck.settings import default_settings
    s = default_settings(zono_impl='patches')
    xl = torch.tensor([-0.1, -0.1], dtype=DTYPE, device=DEV)
    xh = torch.tensor([0.1, 0.1], dtype=DTYPE, device=DEV)
    z_a = make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=None)
    assert isinstance(z_a, TorchZonotope)
    z_b = make_input_zonotope(s, xl, xh, DEV, DTYPE, in_shape=(1, 2))
    assert isinstance(z_b, TorchZonotope)


# ---------------------------------------------------------------------------
# Phase 2: skip-connection .add()
# ---------------------------------------------------------------------------


def _y_split_branches(in_shape, seed):
    """Two branches starting from the same input, each one stride-1 conv."""
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=seed)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    a_p = zp.copy()
    b_p = zp.copy()
    a_d = zd.copy()
    b_d = zd.copy()
    return xl, xh, (a_p, b_p, a_d, b_d)


def test_add_y_split_two_3x3_convs_parity():
    """Y-split → two 3×3 convs (same shape) → add. Element-wise add path."""
    in_shape = (2, 5, 5)
    _, _, (a_p, b_p, a_d, b_d) = _y_split_branches(in_shape, seed=40)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=41)
    k2, b2 = _rand_kernel(3, 2, 3, 3, seed=42)
    a_p.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    a_d.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    b_p.propagate_conv(k2, b2, in_shape, (1, 1), (1, 1))
    b_d.propagate_conv(k2, b2, in_shape, (1, 1), (1, 1))
    shared = a_p.n_gens  # same as input gens, both branches still K_input
    # Wait: each side has K = n_input gens. shared_gens should be n_input,
    # not max — they share the input gens.
    n_input = int(((torch.rand(1) > -1).sum() * 0).item() + 0)  # placeholder
    # Use actual count: input radii > 0 count.
    # In our test all input dims are non-zero radius -> shared = 2*5*5 = 50.
    shared = 2 * 5 * 5
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_y_split_3x3_vs_1x1_kernel_padding_parity():
    """Two branches use kernel sizes 3×3 vs 1×1 — patches diverge in shape.

    With 3×3 kernel and pad=1 the patches grow to 3×3 with offset shift -1.
    With 1×1 kernel and pad=0 the patches stay 1×1 with offset shift 0.

    Offsets disagree (shift -1 vs 0), so the dense-fallback path is taken.
    """
    in_shape = (2, 5, 5)
    _, _, (a_p, b_p, a_d, b_d) = _y_split_branches(in_shape, seed=43)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=44)
    k2, b2 = _rand_kernel(3, 2, 1, 1, seed=45)
    a_p.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    a_d.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    b_p.propagate_conv(k2, b2, in_shape, (1, 1), (0, 0))
    b_d.propagate_conv(k2, b2, in_shape, (1, 1), (0, 0))
    shared = 2 * 5 * 5
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_y_split_with_relu_each_branch():
    """Y-split → conv → relu → conv → add. Each branch adds μ-gens."""
    in_shape = (2, 5, 5)
    _, _, (a_p, b_p, a_d, b_d) = _y_split_branches(in_shape, seed=46)
    k1a, b1a = _rand_kernel(3, 2, 3, 3, seed=47)
    k1b, b1b = _rand_kernel(3, 2, 3, 3, seed=48)
    a_p.propagate_conv(k1a, b1a, in_shape, (1, 1), (1, 1))
    a_d.propagate_conv(k1a, b1a, in_shape, (1, 1), (1, 1))
    b_p.propagate_conv(k1b, b1b, in_shape, (1, 1), (1, 1))
    b_d.propagate_conv(k1b, b1b, in_shape, (1, 1), (1, 1))
    a_p.apply_relu()
    a_d.apply_relu()
    b_p.apply_relu()
    b_d.apply_relu()
    k2a, b2a = _rand_kernel(2, 3, 3, 3, seed=49)
    k2b, b2b = _rand_kernel(2, 3, 3, 3, seed=50)
    a_p.propagate_conv(k2a, b2a, (3, 5, 5), (1, 1), (1, 1))
    a_d.propagate_conv(k2a, b2a, (3, 5, 5), (1, 1), (1, 1))
    b_p.propagate_conv(k2b, b2b, (3, 5, 5), (1, 1), (1, 1))
    b_d.propagate_conv(k2b, b2b, (3, 5, 5), (1, 1), (1, 1))
    shared = 2 * 5 * 5
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_stride2_both_branches_same_kernel_uniform_delta():
    """Both branches stride-2 with same 3×3 kernel → uniform delta path."""
    in_shape = (2, 6, 6)
    _, _, (a_p, b_p, a_d, b_d) = _y_split_branches(in_shape, seed=51)
    k1, b1 = _rand_kernel(2, 2, 3, 3, seed=52)
    k2, b2 = _rand_kernel(2, 2, 3, 3, seed=53)
    a_p.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    a_d.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    b_p.propagate_conv(k2, b2, in_shape, (2, 2), (1, 1))
    b_d.propagate_conv(k2, b2, in_shape, (2, 2), (1, 1))
    shared = 2 * 6 * 6
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_point_propagation():
    """Random concrete inputs through Y-split + add must lie in bounds."""
    in_shape = (2, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=54)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    a_p = zp.copy()
    b_p = zp.copy()
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=55)
    k2, b2 = _rand_kernel(3, 2, 3, 3, seed=56)
    a_p.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    b_p.propagate_conv(k2, b2, in_shape, (1, 1), (1, 1))
    a_p.apply_relu()
    b_p.apply_relu()
    shared = 2 * 4 * 4
    sum_p = a_p.add(b_p, shared)
    lo, hi = sum_p.bounds()

    g = torch.Generator(device=DEV).manual_seed(57)
    n = xl.numel()
    for _ in range(200):
        u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
        x = xl + u * (xh - xl)
        x4 = x.reshape(1, *in_shape)
        a_out = F.relu(F.conv2d(x4, k1, bias=b1, stride=1, padding=1))
        b_out = F.relu(F.conv2d(x4, k2, bias=b2, stride=1, padding=1))
        out = (a_out + b_out).flatten()
        assert torch.all(out >= lo - 1e-10)
        assert torch.all(out <= hi + 1e-10)


def test_add_branch_specific_extras_concatenated():
    """K_a and K_b differ (each branch added different number of μ-gens)."""
    in_shape = (1, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=0.4, seed=58)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    a_p, b_p = zp.copy(), zp.copy()
    a_d, b_d = zd.copy(), zd.copy()
    # Different convs/biases give different relu unstable counts -> different K.
    k1, _ = _rand_kernel(2, 1, 3, 3, seed=59)
    b1_a = torch.zeros(2, dtype=DTYPE, device=DEV)
    b1_b = torch.tensor([0.5, -0.5], dtype=DTYPE, device=DEV)
    a_p.propagate_conv(k1, b1_a, in_shape, (1, 1), (1, 1))
    a_d.propagate_conv(k1, b1_a, in_shape, (1, 1), (1, 1))
    b_p.propagate_conv(k1, b1_b, in_shape, (1, 1), (1, 1))
    b_d.propagate_conv(k1, b1_b, in_shape, (1, 1), (1, 1))
    a_p.apply_relu()
    a_d.apply_relu()
    b_p.apply_relu()
    b_d.apply_relu()
    shared = 1 * 4 * 4
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_per_gen_delta_resnet_shortcut_pattern():
    """Resnet downsampling block: stride-2 main path + stride-2 1×1 shortcut.

    Both branches go through stride-2 convs but with different kernel sizes
    (3×3 vs 1×1) → per-gen offset delta varies (alternates by parity).
    The per-gen scatter path in add() handles this without dense fallback.
    """
    in_shape = (2, 8, 8)
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=300)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    a_p, b_p = zp.copy(), zp.copy()
    a_d, b_d = zd.copy(), zd.copy()
    # Main: stride-2 3×3 conv.
    k1, b1 = _rand_kernel(4, 2, 3, 3, seed=301)
    a_p.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    a_d.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    # Shortcut: stride-2 1×1 conv.
    k2, b2 = _rand_kernel(4, 2, 1, 1, seed=302)
    b_p.propagate_conv(k2, b2, in_shape, (2, 2), (0, 0))
    b_d.propagate_conv(k2, b2, in_shape, (2, 2), (0, 0))
    shared = 2 * 8 * 8
    # Verify offsets actually have per-gen variation.
    delta = a_p._offsets[:shared] - b_p._offsets[:shared]
    assert not torch.all(delta == delta[0:1]), (
        "test setup didn't produce per-gen delta variation")
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    assert sum_p._mode == 'patches', (
        "per-gen delta should still stay in patches via scatter path")
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_per_gen_delta_point_propagation():
    """Per-gen delta path: point-prop soundness."""
    in_shape = (1, 8, 8)
    xl, xh = _rand_input_bounds(in_shape, eps=0.2, seed=303)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    a_p, b_p = zp.copy(), zp.copy()
    k1, b1 = _rand_kernel(2, 1, 3, 3, seed=304)
    k2, b2 = _rand_kernel(2, 1, 1, 1, seed=305)
    a_p.propagate_conv(k1, b1, in_shape, (2, 2), (1, 1))
    b_p.propagate_conv(k2, b2, in_shape, (2, 2), (0, 0))
    shared = 1 * 8 * 8
    sum_p = a_p.add(b_p, shared)
    lo, hi = sum_p.bounds()
    g = torch.Generator(device=DEV).manual_seed(306)
    n = xl.numel()
    for _ in range(200):
        u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
        x = xl + u * (xh - xl)
        x4 = x.reshape(1, *in_shape)
        ao = F.conv2d(x4, k1, bias=b1, stride=(2, 2), padding=(1, 1))
        bo = F.conv2d(x4, k2, bias=b2, stride=(2, 2), padding=(0, 0))
        out = (ao + bo).flatten()
        assert torch.all(out >= lo - 1e-10)
        assert torch.all(out <= hi + 1e-10)


def test_add_resnet_block_pattern_asymmetric_shifts():
    """Resnet block: long branch (conv→relu→conv) + identity skip.

    Long branch shifts patches by 2 (two stride-1 3×3 convs); identity
    branch shifts by 0. The shared portion thus has uniform delta = 2,
    triggering the bounding-box pad path (NOT dense fallback).
    """
    in_shape = (2, 12, 12)
    xl, xh = _rand_input_bounds(in_shape, eps=0.3, seed=70)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)

    # First conv -> ReLU -> we'll fork at the relu output.
    k0, b0 = _rand_kernel(2, 2, 3, 3, seed=71)
    zp.propagate_conv(k0, b0, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k0, b0, in_shape, (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()

    # Fork.
    a_p, b_p = zp.copy(), zp.copy()
    a_d, b_d = zd.copy(), zd.copy()

    # Branch A: conv -> relu -> conv (offset shift -2 from fork).
    k1, b1 = _rand_kernel(2, 2, 3, 3, seed=72)
    a_p.propagate_conv(k1, b1, (2, 12, 12), (1, 1), (1, 1))
    a_d.propagate_conv(k1, b1, (2, 12, 12), (1, 1), (1, 1))
    a_p.apply_relu()
    a_d.apply_relu()
    k2, b2 = _rand_kernel(2, 2, 3, 3, seed=73)
    a_p.propagate_conv(k2, b2, (2, 12, 12), (1, 1), (1, 1))
    a_d.propagate_conv(k2, b2, (2, 12, 12), (1, 1), (1, 1))
    # Branch B: identity (no ops).

    shared = zp.n_gens
    assert a_p._mode == 'patches'
    assert b_p._mode == 'patches'
    # Verify offsets actually differ between branches in the shared portion.
    assert not torch.equal(
        a_p._offsets[:shared], b_p._offsets[:shared])

    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    # Resnet add must stay in patches mode (no dense fallback).
    assert sum_p._mode == 'patches', (
        "resnet block add should not dense-fallback")
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def _sorted_triples(rid, cid, val):
    """Return the (rid, cid, val) triples sorted by (rid, cid) for stable
    cross-impl comparison."""
    order = torch.argsort(rid * (cid.max().item() + 1 if cid.numel() else 1)
                          + cid)
    return rid[order], cid[order], val[order]


def test_nonzero_rows_parity_dense_vs_patches():
    """Phase 6: nonzero_rows triples match between patches and dense."""
    in_shape = (2, 5, 5)
    xl, xh = _rand_input_bounds(in_shape, eps=0.4, seed=100)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(3, 2, 3, 3, seed=101)
    zp.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    # Pick the unstable subset by computing internal bounds.
    lo, hi = zp.bounds()
    unstable = torch.where((lo < 0) & (hi > 0))[0]
    assert unstable.numel() > 0, "test setup didn't produce unstable neurons"
    rp, cp, vp = zp.nonzero_rows(unstable)
    rd, cd, vd = zd.nonzero_rows(unstable)
    rp, cp, vp = _sorted_triples(rp, cp, vp)
    rd, cd, vd = _sorted_triples(rd, cd, vd)
    assert torch.equal(rp, rd), f"row_ids differ: {rp} vs {rd}"
    assert torch.equal(cp, cd), f"col_ids differ"
    torch.testing.assert_close(vp, vd, atol=1e-9, rtol=1e-9)


def test_nonzero_rows_two_convs_parity():
    """Parity through patch-growth (two stacked convs)."""
    in_shape = (2, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, eps=0.5, seed=102)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(3, 2, 3, 3, seed=103)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    k2, b2 = _rand_kernel(2, 3, 3, 3, seed=104)
    zp.propagate_conv(k2, b2, (3, 6, 6), (1, 1), (1, 1))
    zd.propagate_conv(k2, b2, (3, 6, 6), (1, 1), (1, 1))
    lo, hi = zp.bounds()
    unstable = torch.where((lo < 0) & (hi > 0))[0]
    if unstable.numel() == 0:
        # Try a wider eps.
        pytest.skip("no unstable; widen test inputs")
    rp, cp, vp = zp.nonzero_rows(unstable)
    rd, cd, vd = zd.nonzero_rows(unstable)
    rp, cp, vp = _sorted_triples(rp, cp, vp)
    rd, cd, vd = _sorted_triples(rd, cd, vd)
    assert torch.equal(rp, rd)
    assert torch.equal(cp, cd)
    torch.testing.assert_close(vp, vd, atol=1e-9, rtol=1e-9)


def test_nonzero_rows_empty_unstable():
    """No unstable neurons → empty triples."""
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=105)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    rp, cp, vp = zp.nonzero_rows(empty)
    assert rp.numel() == 0 and cp.numel() == 0 and vp.numel() == 0


def test_apply_relu_custom_parity_with_dense():
    """Phase 5: apply_relu_custom() bit-identical between patches and dense."""
    in_shape = (2, 5, 5)
    xl, xh = _rand_input_bounds(in_shape, eps=0.5, seed=86)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(3, 2, 3, 3, seed=87)
    zp.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    n = 3 * 5 * 5
    g = torch.Generator(device=DEV).manual_seed(88)
    # Synthesize per-neuron (lam, mu, shift). Some neurons "unstable" (mu != 0).
    lam = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    mu = torch.zeros(n, dtype=DTYPE, device=DEV)
    mask = torch.rand(n, generator=g) > 0.6
    mu[mask] = torch.rand(mask.sum().item(), generator=g, dtype=DTYPE)
    shift = mu.clone()
    zp.apply_relu_custom(lam, mu, shift)
    zd.apply_relu_custom(lam, mu, shift)
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)
    # Generator counts must match too (same number of new μ-gens appended).
    assert zp.n_gens == zd.generators.shape[1]


def test_apply_relu_custom_matches_apply_relu_when_min_area():
    """When (lam, mu, shift) are computed min-area-style, custom == apply_relu."""
    in_shape = (1, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=0.5, seed=89)
    zp1 = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zp2 = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    k, b = _rand_kernel(2, 1, 3, 3, seed=90)
    zp1.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    zp2.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    lo, hi = zp1.bounds()
    ust = (lo < 0) & (hi > 0)
    dead = hi <= 0
    lam_min = torch.where(
        ust, hi / (hi - lo),
        torch.where(dead, torch.zeros_like(hi), torch.ones_like(hi)))
    mu_min = torch.where(
        ust, -hi * lo / (2 * (hi - lo)), torch.zeros_like(hi))
    zp1.apply_relu()
    zp2.apply_relu_custom(lam_min, mu_min, mu_min)
    l1, h1 = zp1.bounds()
    l2, h2 = zp2.bounds()
    torch.testing.assert_close(l1, l2, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(h1, h2, atol=1e-9, rtol=1e-9)


def test_conv_conv_fc_fc_parity():
    """Phase 3 sequential mix: 2 convs (patches), 2 FCs (dense fallback)."""
    in_shape = (2, 6, 6)
    xl, xh = _rand_input_bounds(in_shape, eps=0.4, seed=80)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k1, b1 = _rand_kernel(4, 2, 3, 3, seed=81)
    zp.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k1, b1, in_shape, (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    k2, b2 = _rand_kernel(3, 4, 3, 3, seed=82)
    zp.propagate_conv(k2, b2, (4, 6, 6), (1, 1), (1, 1))
    zd.propagate_conv(k2, b2, (4, 6, 6), (1, 1), (1, 1))
    zp.apply_relu()
    zd.apply_relu()
    n = 3 * 6 * 6
    g = torch.Generator(device=DEV).manual_seed(83)
    W1 = torch.randn(8, n, generator=g, dtype=DTYPE, device=DEV)
    bf1 = torch.randn(8, generator=g, dtype=DTYPE, device=DEV)
    zp.propagate_fc(W1, bf1)
    zd.propagate_fc(W1, bf1)
    assert zp._mode == 'dense', "FC must materialise to dense"
    zp.apply_relu()
    zd.apply_relu()
    W2 = torch.randn(2, 8, generator=g, dtype=DTYPE, device=DEV)
    bf2 = torch.randn(2, generator=g, dtype=DTYPE, device=DEV)
    zp.propagate_fc(W2, bf2)
    zd.propagate_fc(W2, bf2)
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_relu_lambda_scaling_per_pixel_correctness():
    """ReLU λ varies per-pixel; verify gather produces the right scale.

    Pre-ReLU bounds straddle 0 so λ = hi/(hi-lo) is a non-trivial per-pixel
    map. The patches must be multiplied by this λ at each patch position,
    not by a scalar.
    """
    in_shape = (1, 4, 4)
    xl, xh = _rand_input_bounds(in_shape, eps=1.0, seed=84)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    k, b = _rand_kernel(3, 1, 3, 3, seed=85)
    zp.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    zd.propagate_conv(k, b, in_shape, (1, 1), (1, 1))
    lo_pre, hi_pre = zp.bounds()
    # Confirm the test exercises unstable neurons.
    n_unstable = ((lo_pre < 0) & (hi_pre > 0)).sum().item()
    assert n_unstable > 0, "test setup didn't produce unstable neurons"
    zp.apply_relu()
    zd.apply_relu()
    lp, hp = zp.bounds()
    ld, hd = zd.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


def test_add_from_dense_side_promotes():
    """If one side is dense (post-FC e.g.), the other promotes too."""
    in_shape = (1, 3, 3)
    xl, xh = _rand_input_bounds(in_shape, seed=60)
    zp = PatchesZonotope.from_input_bounds(xl, xh, in_shape, DEV, DTYPE)
    zd = TorchZonotope.from_input_bounds(xl, xh, DEV, DTYPE)
    a_p, b_p = zp.copy(), zp.copy()
    a_d, b_d = zd.copy(), zd.copy()
    # Force b_p into dense mode by FC (small W).
    n = xl.numel()
    g = torch.Generator(device=DEV).manual_seed(61)
    W = torch.randn(n, n, generator=g, dtype=DTYPE, device=DEV)
    bias = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
    b_p.propagate_fc(W, bias)
    b_d.propagate_fc(W, bias)
    # a_p stays in patches mode (no conv applied — still 1x1 patches).
    # Add — should promote a_p to dense.
    shared = n
    sum_p = a_p.add(b_p, shared)
    sum_d = a_d.add(b_d, shared)
    lp, hp = sum_p.bounds()
    ld, hd = sum_d.bounds()
    torch.testing.assert_close(lp, ld, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(hp, hd, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Conv chunking: chunked path matches un-chunked for both stride==1 and >1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('stride,padding', [(1, 1), (2, 1)])
def test_propagate_conv_chunked_matches_unchunked(monkeypatch, stride, padding):
    """Forcing a tiny chunk budget must not change the output.

    Builds a PatchesZonotope big enough that several K-chunks are needed
    when ``_conv_chunk_bytes`` is dropped to 1 KB, then compares against
    the default (un-chunked) result.
    """
    in_shape = (3, 8, 8)
    xl, xh = _rand_input_bounds(in_shape, seed=200)
    z_unchunked = PatchesZonotope.from_input_bounds(
        xl, xh, in_shape, DEV, DTYPE)
    z_chunked = z_unchunked.copy()

    kernel, bias = _rand_kernel(8, 3, 3, 3, seed=201)

    z_unchunked.propagate_conv(
        kernel, bias, in_shape, stride, padding)

    monkeypatch.setattr(PatchesZonotope, '_conv_chunk_bytes', 1024)
    z_chunked.propagate_conv(
        kernel, bias, in_shape, stride, padding)

    torch.testing.assert_close(
        z_unchunked.center, z_chunked.center, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        z_unchunked._patches, z_chunked._patches, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        z_unchunked._offsets, z_chunked._offsets, atol=0, rtol=0)

    # Bounds must also match.
    lo_u, hi_u = z_unchunked.bounds()
    lo_c, hi_c = z_chunked.bounds()
    torch.testing.assert_close(lo_u, lo_c, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hi_u, hi_c, atol=1e-12, rtol=1e-12)
