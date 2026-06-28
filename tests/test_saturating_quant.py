"""Saturating (non-VNNI) quantized GEMM/Conv twins + the quant-oracle detector.

The numpy twins are validated BIT-EXACT against real AWS non-VNNI ORT (matmul 8/8, conv
20/20) during development; here we pin: (a) the canonical saturation case, (b) torch==numpy
on the forward (hard), (c) the soft-gradient path produces finite gradients, (d) odd-K and
bias/stride/pad branches, and (e) the detector returns a valid regime."""
import numpy as np
import torch

from vibecheck.saturating_quant import (
    sat_matmul, sat_conv, sat_matmul_torch, sat_conv_torch, INT16_MAX)
from vibecheck.surrogate_pgd import detect_quant_oracle, resolve_saturation
from vibecheck.settings import default_settings


def _exact_matmul(a, b, a_zp, b_zp, a_s, b_s, y_s, y_zp):
    acc = ((a.astype(np.int64) - a_zp) * 0 + a.astype(np.int64)) @ b.astype(np.int64)
    acc = acc - a_zp * b.astype(np.int64).sum(0)[None, :] - b_zp * a.astype(np.int64).sum(1)[:, None]
    acc = acc + a.shape[1] * a_zp * b_zp
    y = np.rint(acc.astype(np.float64) * (a_s * b_s / y_s)) + y_zp
    return np.clip(y, 0, 255).astype(np.uint8)


def test_canonical_saturation_case():
    # 64 products of 255*127: int16 pair-sums saturate -> non-VNNI != exact int32.
    a = np.full((1, 64), 255, np.uint8)
    b = np.full((64, 1), 127, np.int8)
    ys = float(64 * 255 * 127 * 0.1 * 0.1 / 250.0)
    sat = int(sat_matmul(a, b, 0, 0, 0.1, 0.1, ys, 0).ravel()[0])
    exact = int(_exact_matmul(a, b, 0, 0, 0.1, 0.1, ys, 0).ravel()[0])
    assert exact == 250 and sat == 126            # exact path would give 250; saturation -> 126


def test_no_saturation_matches_exact():
    # Small magnitudes never reach the int16 bound -> saturating twin == exact int32.
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(0, 16, (2, 31), np.uint8)        # odd K, small values
        b = rng.integers(-16, 16, (31, 3), np.int8)
        azp, bzp = 3, 0
        out = sat_matmul(a, b, azp, bzp, 0.05, 0.07, 7.0, 4)
        assert np.array_equal(out, _exact_matmul(a, b, azp, bzp, 0.05, 0.07, 7.0, 4))


def test_torch_matches_numpy_matmul():
    rng = np.random.default_rng(1)
    for _ in range(6):
        K = int(rng.integers(2, 120))
        a = rng.integers(0, 255, (2, K), np.uint8)
        b = rng.integers(-127, 127, (K, 3), np.int8)
        azp, bzp, ys, yzp = 7, 0, float(rng.uniform(1, 40)), 5
        npy = sat_matmul(a, b, azp, bzp, 0.05, 0.07, ys, yzp)
        tch = sat_matmul_torch(torch.tensor(a.astype(np.float64)), torch.tensor(b.astype(np.float64)),
                               azp, bzp, 0.05, 0.07, ys, yzp).numpy().astype(np.uint8)
        assert np.array_equal(npy, tch)


def test_torch_matches_numpy_conv():
    rng = np.random.default_rng(2)
    for _ in range(6):
        Cin, Cout = int(rng.integers(2, 5)), int(rng.integers(1, 4))
        KH, KW = int(rng.choice([1, 3])), int(rng.choice([1, 3]))
        Hi, Wi = KH + int(rng.integers(0, 4)), KW + int(rng.integers(0, 4))
        pad, stride = int(rng.choice([0, 1])), int(rng.choice([1, 2]))
        x = rng.integers(0, 255, (1, Cin, Hi, Wi), np.uint8)
        w = rng.integers(-127, 127, (Cout, Cin, KH, KW), np.int8)
        x_zp, x_s = 6, 0.05
        w_s = rng.uniform(0.02, 0.09, Cout).astype(np.float32)
        y_s, y_zp = float(rng.uniform(2, 40)), 4
        bias = rng.integers(-5000, 5000, Cout).astype(np.int32) if rng.random() > 0.5 else None
        npy = sat_conv(x, w, x_zp, 0, x_s, w_s, y_s, y_zp, bias, stride, pad)
        tch = sat_conv_torch(
            torch.tensor(x.astype(np.float64)), torch.tensor(w.astype(np.float64)),
            x_zp, 0, x_s, w_s, y_s, y_zp, stride, pad,
            None if bias is None else torch.tensor(bias.astype(np.float64))).numpy().astype(np.uint8)
        assert np.array_equal(npy, tch)


def test_soft_gradient_flows():
    # Differentiable path (hard=False): forward is continuous, gradient reaches the input even
    # when products saturate (soft-saturation backward), for both matmul and conv.
    a = (torch.rand(1, 96) * 200 + 40).detach().requires_grad_(True)
    b = torch.randint(-127, 127, (96, 1)).double()
    sat_matmul_torch(a, b, 5, 0, 0.2, 0.1, 1.0, 0, hard=False, out_clamp=False).sum().backward()
    assert torch.isfinite(a.grad).all() and (a.grad != 0).any()

    x = (torch.rand(1, 4, 6, 6) * 200 + 40).detach().requires_grad_(True)
    w = torch.randint(-127, 127, (2, 4, 3, 3)).double()
    sat_conv_torch(x, w, 5, 0, 0.2, np.array([0.1, 0.1], np.float32), 1.0, 0,
                   hard=False, out_clamp=False).sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad != 0).any()


def test_detect_and_resolve_oracle():
    assert detect_quant_oracle() in ('exact', 'saturating')
    s = default_settings()
    s.surrogate_saturation = 'on'
    assert resolve_saturation(s, log=lambda *_: None) is True
    s.surrogate_saturation = 'off'
    assert resolve_saturation(s, log=lambda *_: None) is False
    s.surrogate_saturation = 'auto'
    assert resolve_saturation(s, log=lambda *_: None) in (True, False)


def test_int16_bound_constant():
    assert INT16_MAX == 32767
