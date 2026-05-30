"""Tests for CROWN backward helpers used by the ND/bilinear ops in
nn4sys mscn_*.

Adds tests for:
  - `_bias_dot_ew` broadcast: bias of shape (n_inner,) over an ew
    that represents an out_shape_nd = (..., n_inner) layer
  - `_mul_bilinear_backward` and `_div_bilinear_backward`: when one
    side is a point zonotope (zero gens), these reduce to linear
    scaling — same broadcast logic as forward
  - `_reduce_sum_backward`: linear adjoint = broadcast over reduced
    axes

Failing-test-first: each test fails with ImportError on the helper
that doesn't exist yet, then passes after the implementation lands.
"""
import numpy as np
import torch
import pytest


def test_bias_dot_ew_matching_size():
    """Legacy contract: matching-size bias dots ew directly."""
    from vibecheck.alpha_crown import _bias_dot_ew
    ew = torch.randn(4, 8, dtype=torch.float64)
    bias = np.random.randn(8)
    acc = _bias_dot_ew(ew, bias, torch.float64, torch.device('cpu'))
    expected = ew @ torch.as_tensor(bias, dtype=torch.float64)
    torch.testing.assert_close(acc, expected)


def test_bias_dot_ew_scalar_broadcast():
    """Scalar bias broadcasts to ew.sum(-1) * bias."""
    from vibecheck.alpha_crown import _bias_dot_ew
    ew = torch.randn(4, 8, dtype=torch.float64)
    bias = np.array([3.5])
    acc = _bias_dot_ew(ew, bias, torch.float64, torch.device('cpu'))
    expected = ew.sum(dim=-1) * 3.5
    torch.testing.assert_close(acc, expected)


def test_bias_dot_ew_nd_broadcast():
    """ND broadcast: bias (n_inner,) over ew that represents a layer
    of nd-shape (..., n_inner). Used by mscn's Add(MatMul_out=(6,128),
    bias=(128,)) → ew of shape (B, 768) when expanded.

    Backward through `y[..., j] = x[..., j] + bias[j]`:
      grad_x = grad_y          (identity)
      acc   += sum_{..., j} grad_y[..., j] * bias[j]
            = sum_outer (grad_y_inner @ bias)
    """
    from vibecheck.alpha_crown import _bias_dot_ew
    # Forward: y has shape (B, 6, 128) flat=(B, 768). bias=(128,).
    B = 4
    n_outer, n_inner = 6, 128
    ew = torch.randn(B, n_outer * n_inner, dtype=torch.float64)
    bias = np.random.randn(n_inner)
    # Need to pass out_shape so the helper knows how to broadcast.
    acc = _bias_dot_ew(ew, bias, torch.float64, torch.device('cpu'),
                       out_shape=(n_outer, n_inner))
    bt = torch.as_tensor(bias, dtype=torch.float64)
    ew_nd = ew.reshape(B, n_outer, n_inner)
    expected = (ew_nd @ bt).sum(dim=-1)
    torch.testing.assert_close(acc, expected)


def test_reduce_sum_backward_broadcast():
    """Adjoint of ReduceSum is broadcast (repeat).

    Forward: y = x.sum(dim=axis)  (no keepdim)
    Backward: grad_x[..., j, ...] = grad_y[...]  (broadcast over axis)
    """
    from vibecheck.alpha_crown import _reduce_sum_backward
    B = 2
    in_shape = (3, 7)   # x: (B, 3, 7)
    axes = (1,)          # reduce along axis 1 of x → output (3,)
    keepdims = False
    out_shape = (3,)
    # ew has shape (B, prod(out_shape)) = (B, 3)
    ew_out = torch.randn(B, 3, dtype=torch.float64)
    ew_in = _reduce_sum_backward(ew_out, in_shape, axes, keepdims, out_shape)
    # Adjoint: broadcast ew_out over the reduced axis (axis 1 in (B, 3, 7))
    # in batched form: dim shift by +1.
    assert ew_in.shape == (B, in_shape[0] * in_shape[1])
    expected = ew_out.unsqueeze(-1).expand(B, 3, 7).reshape(B, -1)
    torch.testing.assert_close(ew_in, expected)


def test_mul_bilinear_backward_point_b():
    """Mul(a, b) backward when b is a point. y = a * c_b, so
    grad_a = grad_y * c_b (and no grad to b since b is fixed)."""
    from vibecheck.alpha_crown import _mul_bilinear_backward
    B = 3
    sh_a = sh_b = sh_out = (5,)
    c_b = torch.tensor([2.0, -1.0, 0.5, 3.0, 1.0], dtype=torch.float64)
    g_b = torch.zeros(5, 2, dtype=torch.float64)  # point
    ew_out = torch.randn(B, 5, dtype=torch.float64)
    ew_a, ew_b = _mul_bilinear_backward(
        ew_out, c_a=None, g_a=None, c_b=c_b, g_b=g_b,
        sh_a=sh_a, sh_b=sh_b, sh_out=sh_out)
    expected_a = ew_out * c_b
    torch.testing.assert_close(ew_a, expected_a)
    # b's grad isn't propagated through the bilinear because b has
    # no varying generators; helper returns None or zeros for ew_b.
    assert ew_b is None or bool(ew_b.abs().sum() < 1e-12)


def test_mul_bilinear_backward_nd_broadcast():
    """Mul(features (3, 128), mask (3, 1)) → (3, 128). When mask is
    point, grad_features = grad_y * mask (broadcast). Used in mscn."""
    from vibecheck.alpha_crown import _mul_bilinear_backward
    B = 2
    sh_a = (3, 128); sh_b = (3, 1); sh_out = (3, 128)
    c_b_full = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)
    g_b = torch.zeros(3, 2, dtype=torch.float64)  # point
    ew_out = torch.randn(B, 3 * 128, dtype=torch.float64)
    ew_a, _ = _mul_bilinear_backward(
        ew_out, c_a=None, g_a=None, c_b=c_b_full.flatten(), g_b=g_b,
        sh_a=sh_a, sh_b=sh_b, sh_out=sh_out)
    expected_nd = ew_out.reshape(B, 3, 128) * c_b_full.reshape(3, 1)
    torch.testing.assert_close(ew_a, expected_nd.reshape(B, -1))


def test_div_bilinear_backward_point_denom():
    """Div(a, b) backward with point b: y = a / c_b, grad_a = grad_y / c_b."""
    from vibecheck.alpha_crown import _div_bilinear_backward
    B = 2
    sh_a = sh_b = sh_out = (4,)
    c_b = torch.tensor([2.0, 3.0, 4.0, -6.0], dtype=torch.float64)
    g_b = torch.zeros(4, 2, dtype=torch.float64)
    ew_out = torch.randn(B, 4, dtype=torch.float64)
    ew_a, ew_b = _div_bilinear_backward(
        ew_out, c_a=None, g_a=None, c_b=c_b, g_b=g_b,
        sh_a=sh_a, sh_b=sh_b, sh_out=sh_out)
    expected_a = ew_out / c_b
    torch.testing.assert_close(ew_a, expected_a)
    assert ew_b is None or bool(ew_b.abs().sum() < 1e-12)


def test_div_bilinear_backward_nonpoint_raises():
    """Div with non-point denom: raises (nonlinear adjoint)."""
    from vibecheck.alpha_crown import _div_bilinear_backward
    ew_out = torch.randn(2, 3, dtype=torch.float64)
    g_b = torch.tensor([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        dtype=torch.float64)
    c_b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    with pytest.raises(NotImplementedError, match='non-point'):
        _div_bilinear_backward(
            ew_out, c_a=None, g_a=None, c_b=c_b, g_b=g_b,
            sh_a=(3,), sh_b=(3,), sh_out=(3,))


def test_compute_point_centers_runs_forward():
    """`_compute_point_centers` runs a single forward at x_point and
    returns each op's center value as a tensor. Used to provide the
    'constant' side value for bilinear backwards in mscn."""
    import numpy as np
    from vibecheck.network import ComputeGraph
    from vibecheck.alpha_crown import _compute_point_centers
    from onnx import helper, TensorProto
    import onnx, tempfile, os
    # Build a tiny model: y = relu(x @ W + b) where x is (2,), y is (3,)
    # ONNX Gemm: Y = A @ B + C → W shape (K=2, N=3).
    x_inp = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
    y_out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    W_init = helper.make_tensor('W', TensorProto.FLOAT, [2, 3],
                                  [1, 0, 1, 0, 1, 1])
    b_init = helper.make_tensor('b', TensorProto.FLOAT, [3], [0, 0, 0])
    n1 = helper.make_node('Gemm', ['x', 'W', 'b'], ['z'])
    n2 = helper.make_node('Relu', ['z'], ['y'])
    g = helper.make_graph([n1, n2], 'test', [x_inp], [y_out], [W_init, b_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(m, f.name)
        path = f.name
    try:
        graph = ComputeGraph.from_onnx(path, dtype=np.float32)
        gg = graph.gpu_graph(torch.device('cpu'), torch.float32)
        x_pt = torch.tensor([1.0, 2.0], dtype=torch.float32)
        centers = _compute_point_centers(gg, x_pt, torch.device('cpu'), torch.float32)
        # W (transposed) = [[1,0,1],[0,1,1]] → x @ W = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
        gemm_op = gg['ops'][0]; relu_op = gg['ops'][1]
        torch.testing.assert_close(centers[gemm_op['name']],
                                   torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(centers[relu_op['name']],
                                   torch.tensor([1.0, 2.0, 3.0]))
    finally:
        os.unlink(path)
