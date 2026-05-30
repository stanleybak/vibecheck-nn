"""1D ground-truth Div(a, b) bound check.

Validates `_torch_zono_div_bilinear` with `fallback='box'`:
- Sound over a 100x100 grid for various (a_range, b_range) pairs
  where b is sign-stable (entirely positive OR entirely negative).
- Reports gap vs true min/max so we can quantify the tightness loss
  from the 4-corner box bound.

Run: .venv/bin/python -m pytest tests/test_div_1d_bound.py -v
"""
import pytest
import torch
import numpy as np
from vibecheck.zonotope import _torch_zono_div_bilinear


# (a_lo, a_hi, b_lo, b_hi)
DIV_CASES = [
    (1.0, 3.0, 1.0, 2.0),            # all positive, narrow b
    (0.5, 1.5, 0.5, 0.7),            # small b range
    (-2.0, -1.0, 1.0, 2.0),          # a neg, b pos
    (-2.0, 2.0, 1.0, 2.0),           # a straddles 0, b pos
    (1.0, 10.0, 100.0, 200.0),       # pensieve-like: huge b
    (1.0, 10.0, 1e6, 2e6),           # very huge b (cb in millions)
    (-5.0, 5.0, -2.0, -1.0),         # b all negative
    (0.0, 1.0, 1.0, 10.0),           # tight a, wide b
]


def _zono(c, rad):
    c_t = torch.tensor([c], dtype=torch.float64)
    g_t = torch.tensor([[rad]], dtype=torch.float64) if rad > 0 \
        else torch.zeros(1, 0, dtype=torch.float64)
    return c_t, g_t


def _range_of(c_out, g_out):
    rad = g_out.abs().sum(dim=1)
    return float((c_out - rad)[0]), float((c_out + rad)[0])


@pytest.mark.parametrize('a_lo,a_hi,b_lo,b_hi', DIV_CASES,
    ids=[f'a={c[0]},{c[1]}_b={c[2]},{c[3]}' for c in DIV_CASES])
def test_div_box_sound(a_lo, a_hi, b_lo, b_hi):
    c_a = (a_lo + a_hi) / 2; rad_a = (a_hi - a_lo) / 2
    c_b = (b_lo + b_hi) / 2; rad_b = (b_hi - b_lo) / 2
    c_a_t, g_a_t = _zono(c_a, rad_a)
    c_b_t, g_b_t = _zono(c_b, rad_b)
    c_out, g_out = _torch_zono_div_bilinear(
        c_a_t, g_a_t, c_b_t, g_b_t, fallback='box',
        prefer_shared_when_scalar_b=False)
    out_lo, out_hi = _range_of(c_out, g_out)
    # Ground truth: 100x100 grid scan.
    A = np.linspace(a_lo, a_hi, 101)
    B = np.linspace(b_lo, b_hi, 101)
    AA, BB = np.meshgrid(A, B)
    Y = AA / BB
    true_lo = float(Y.min()); true_hi = float(Y.max())
    slack = max(1e-6 * max(abs(true_lo), abs(true_hi), 1.0), 1e-6)
    assert out_lo <= true_lo + slack, (
        f'LB violation: out_lo {out_lo} > true_lo {true_lo}')
    assert out_hi >= true_hi - slack, (
        f'UB violation: out_hi {out_hi} < true_hi {true_hi}')
    # Report gap for human eyes (4-corner is exact on 1D
    # bilinear / quotient when a, b are independent intervals).
    gap_lo = true_lo - out_lo
    gap_hi = out_hi - true_hi
    span = true_hi - true_lo
    rel = (gap_lo + gap_hi) / max(span, 1e-12)
    # 4-corner on 1×1D ranges should match exactly (monotonic in a
    # for fixed b; monotonic in b for fixed a; corners enclose).
    assert rel < 1e-6, f'unexpected slack on 1D: rel={rel}'
