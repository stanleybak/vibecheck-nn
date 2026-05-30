"""Tests for sub_bilinear gg op (Sub(a, b) with both inputs computed).

Catches the pre-fix bug where Sub silently dropped the second input
(took only inp[0] as the active operand and dropped inp[1]). On
nn4sys pensieve_*_parallel this turned the final `output = MatMul1 -
MatMul2` into `output = MatMul1`, producing forward values ~100×
larger than reality (101.5 vs correct 5.57) and an UNSOUND
"verified" verdict whose CROWN spec_lb of ~99 contradicted the actual
Y_0 range of [5.55, 5.59].
"""
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto
import pytest


def _build_sub_bilinear_onnx(tmp_path):
    """y = (x[0] + 1.5) - (x[1] - 2.0). Both inputs computed."""
    n_in = 2
    # x → split into (x0, x1) via Slice; each Add-biased; Sub.
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, n_in])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1])

    bias_a = helper.make_tensor('bias_a', TensorProto.FLOAT, [1], [1.5])
    bias_b = helper.make_tensor('bias_b', TensorProto.FLOAT, [1], [-2.0])
    # Use Gemm with weight = single column to extract scalar per branch.
    Wa = helper.make_tensor('Wa', TensorProto.FLOAT, [1, 2], [1.0, 0.0])
    Wb = helper.make_tensor('Wb', TensorProto.FLOAT, [1, 2], [0.0, 1.0])
    zero_b = helper.make_tensor('zero_b', TensorProto.FLOAT, [1], [0.0])

    nodes = [
        # branch a: pull x0, add 1.5
        helper.make_node('Gemm', ['x', 'Wa', 'bias_a'], ['a'],
                          alpha=1.0, beta=1.0, transB=1),
        # branch b: pull x1, add -2.0 (with -2 sign already baked in bias)
        helper.make_node('Gemm', ['x', 'Wb', 'zero_b'], ['b_pre'],
                          alpha=1.0, beta=1.0, transB=1),
        # b = b_pre + (-2.0)
        helper.make_node('Add', ['b_pre', 'bias_b'], ['b']),
        # y = a - b
        helper.make_node('Sub', ['a', 'b'], ['y']),
    ]

    graph = helper.make_graph(
        nodes, 'sub_test', [x], [y],
        initializer=[Wa, bias_a, Wb, zero_b, bias_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    onnx.checker.check_model(model)
    p = str(tmp_path / 'sub_bilinear_test.onnx')
    onnx.save(model, p)
    return p


def test_sub_bilinear_forward_matches_ort(tmp_path):
    """vibecheck gg point-forward of Sub(a, b) must match ORT."""
    p = _build_sub_bilinear_onnx(tmp_path)
    from vibecheck.network import ComputeGraph
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    g = ComputeGraph.from_onnx(p, dtype=np.float32)
    gg = g.gpu_graph(torch.device('cpu'), torch.float32)

    # Test that the gg has a sub_bilinear op (the fix), not a single-input sub.
    sub_ops = [op for op in gg['ops'] if op['type'] == 'sub_bilinear']
    assert len(sub_ops) == 1, (
        f'Expected one sub_bilinear op, got {[op["type"] for op in gg["ops"]]}')
    assert len(sub_ops[0]['inputs']) == 2

    # ORT vs vibecheck forward at several points.
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip('onnxruntime not installed')
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    for _ in range(10):
        x_np = rng.uniform(-3, 3, size=(1, 2)).astype(np.float32)
        ort_y = sess.run(None, {in_name: x_np})[0].flatten()
        vc_y = _forward_batch_graph(
            torch.as_tensor(x_np.flatten()).reshape(1, -1), gg
        ).flatten().numpy()
        np.testing.assert_allclose(vc_y, ort_y, atol=1e-5,
            err_msg=f'vibecheck {vc_y} != ORT {ort_y} for x={x_np}')


def test_sub_bilinear_crown_lb_is_sound(tmp_path):
    """CROWN backward through sub_bilinear must give SOUND lb on y.

    Pre-fix: Sub dropped its second input, so the network being
    bounded was `M1`, not `M1 - M2`. CROWN reported lb based on M1
    only — UNSOUND for the spec evaluated on the real (Sub) network.
    """
    p = _build_sub_bilinear_onnx(tmp_path)
    from vibecheck.network import ComputeGraph
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    g = ComputeGraph.from_onnx(p, dtype=np.float32)
    dev, dt = torch.device('cpu'), torch.float32
    gg = g.gpu_graph(dev, dt)
    # Box [-1, 1]^2
    xl = torch.tensor([-1.0, -1.0])
    xh = torch.tensor([1.0, 1.0])
    sb, z = _forward_zonotope_graph(xl, xh, gg, dev, dt)
    lo_y, hi_y = z.bounds()
    lo_y = float(lo_y.flatten()[0])
    hi_y = float(hi_y.flatten()[0])

    # Sample real y over the box and confirm [lo_y, hi_y] contains it.
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip('onnxruntime not installed')
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    y_min, y_max = float('inf'), -float('inf')
    for _ in range(2000):
        x = (xl + (xh - xl) * torch.as_tensor(
            rng.random(2).astype(np.float32))).numpy().reshape(1, 2)
        y = float(sess.run(None, {in_name: x})[0].flatten()[0])
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    assert lo_y - 1e-4 <= y_min, (
        f'UNSOUND: zono lb {lo_y} > sample y_min {y_min}')
    assert y_max <= hi_y + 1e-4, (
        f'UNSOUND: zono ub {hi_y} < sample y_max {y_max}')
