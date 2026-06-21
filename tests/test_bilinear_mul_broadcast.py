"""Regression: _torch_zono_mul_bilinear broadcasting path with a point operand
that carries MORE (zero) generator columns than the varying operand.

In the augmented nonlinear net (adaptive_cruise_2026), a monomial Mul multiplies a
varying lane (K_a generator columns) by an effectively-constant lane stored with a
different (larger) column count K_b. The broadcasting path used K = max(K_a, K_b),
then reshaped g_a to K columns — which fails when K_b > K_a
(`shape '[1,168]' invalid for input of size 2`). For a point operand only the other
side's generators propagate, so the output column count must be that side's own
count (matching the simple element-wise path, line 812). These pin that.
"""
import torch

from vibecheck.zonotope import _torch_zono_mul_bilinear


def test_broadcast_b_point_more_gens():
    # a varies (K_a=2); b is a point but carries K_b=3 (zero) columns.
    c_a = torch.tensor([2.0], dtype=torch.float64)
    g_a = torch.tensor([[1.0, 0.5]], dtype=torch.float64)      # (1, 2)
    c_b = torch.tensor([3.0], dtype=torch.float64)
    g_b = torch.zeros(1, 3, dtype=torch.float64)               # (1, 3) point
    c_out, g_out = _torch_zono_mul_bilinear(
        c_a, g_a, c_b, g_b, shape_a=(1, 1), shape_b=(1, 1), shape_out=(1, 1))
    torch.testing.assert_close(c_out, torch.tensor([6.0], dtype=torch.float64))
    torch.testing.assert_close(g_out, torch.tensor([[3.0, 1.5]], dtype=torch.float64))


def test_broadcast_a_point_more_gens():
    # symmetric: a is the point with K_a=3 (zero) columns, b varies (K_b=2).
    c_a = torch.tensor([3.0], dtype=torch.float64)
    g_a = torch.zeros(1, 3, dtype=torch.float64)               # (1, 3) point
    c_b = torch.tensor([2.0], dtype=torch.float64)
    g_b = torch.tensor([[1.0, 0.5]], dtype=torch.float64)      # (1, 2)
    c_out, g_out = _torch_zono_mul_bilinear(
        c_a, g_a, c_b, g_b, shape_a=(1, 1), shape_b=(1, 1), shape_out=(1, 1))
    torch.testing.assert_close(c_out, torch.tensor([6.0], dtype=torch.float64))
    # g_out = g_b * c_a = [[1.0, 0.5]] * 3.0
    torch.testing.assert_close(g_out, torch.tensor([[3.0, 1.5]], dtype=torch.float64))
