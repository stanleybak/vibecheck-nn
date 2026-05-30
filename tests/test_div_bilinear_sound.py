"""Soundness tests for Div(a, b) backward when b has nonzero radius.

Pre-fix bug: `_crown_backward_matrix` linearised a/b at center via
ew_a = ew/c_b, ew_b = -ew·c_a/c_b² with NO slack term. That's
unsound for non-point b — the residual R(a, b) = a/b - L(a, b) is
±-signed (~10% relative on a 30% wide box), so the CROWN lb can
exceed the actual minimum of a/b.

The fix: when forward zono sets `op['_div_decoupled']=True` (non-point
b), backward uses the same linearisation plus per-element corner-bound
slack `ep · R_min + en · R_max` accumulated into acc.
"""
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto
import pytest


def _build_div_onnx(tmp_path):
    """y = (x[0] + 10) / (x[1] + 10). Both a and b vary, b > 0."""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1])
    Wa = helper.make_tensor('Wa', TensorProto.FLOAT, [1, 2], [1.0, 0.0])
    Wb = helper.make_tensor('Wb', TensorProto.FLOAT, [1, 2], [0.0, 1.0])
    bias_a = helper.make_tensor('bias_a', TensorProto.FLOAT, [1], [10.0])
    bias_b = helper.make_tensor('bias_b', TensorProto.FLOAT, [1], [10.0])
    nodes = [
        helper.make_node('Gemm', ['x', 'Wa', 'bias_a'], ['a'],
                          alpha=1.0, beta=1.0, transB=1),
        helper.make_node('Gemm', ['x', 'Wb', 'bias_b'], ['b'],
                          alpha=1.0, beta=1.0, transB=1),
        helper.make_node('Div', ['a', 'b'], ['y']),
    ]
    graph = helper.make_graph(
        nodes, 'div_test', [x], [y],
        initializer=[Wa, Wb, bias_a, bias_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    onnx.checker.check_model(model)
    p = str(tmp_path / 'div_test.onnx')
    onnx.save(model, p)
    return p


def test_div_bilinear_crown_lb_is_sound(tmp_path):
    """Forward + CROWN-style backward on Div(a, b) must give sound lb.

    Pin: when b's range crosses an interesting span, the linearised
    backward must add corner-bounded slack — otherwise the CROWN lb
    can exceed the sample-min and produce a false-verified verdict.
    """
    p = _build_div_onnx(tmp_path)
    from vibecheck.network import ComputeGraph
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    from vibecheck.settings import default_settings
    g = ComputeGraph.from_onnx(p, dtype=np.float32)
    dev, dt = torch.device('cpu'), torch.float32
    gg = g.gpu_graph(dev, dt)
    # Box x ∈ [-2, 2]^2 → a ∈ [8, 12], b ∈ [8, 12], y = a/b ∈ [8/12, 12/8] = [0.667, 1.5]
    xl = torch.tensor([-2.0, -2.0])
    xh = torch.tensor([2.0, 2.0])
    s = default_settings(device='cpu', bits=32)
    sb, z = _forward_zonotope_graph(xl, xh, gg, dev, dt, settings=s)
    lo_y, hi_y = z.bounds()
    lo_y = float(lo_y.flatten()[0])
    hi_y = float(hi_y.flatten()[0])

    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip('onnxruntime not installed')
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    y_min, y_max = float('inf'), -float('inf')
    for _ in range(5000):
        x = (xl + (xh - xl) * torch.as_tensor(
            rng.random(2).astype(np.float32))).numpy().reshape(1, 2)
        y = float(sess.run(None, {in_name: x})[0].flatten()[0])
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    assert lo_y - 1e-4 <= y_min, (
        f'UNSOUND zono lb: {lo_y} > sample min {y_min}')
    assert y_max <= hi_y + 1e-4, (
        f'UNSOUND zono ub: {hi_y} < sample max {y_max}')


def test_div_bilinear_spec_lb_is_sound(tmp_path):
    """End-to-end CROWN spec_lb on Div network must respect sampled bound."""
    p = _build_div_onnx(tmp_path)
    from vibecheck.network import ComputeGraph
    from vibecheck.spec import VNNSpec, Constraint, Conjunct
    from vibecheck.settings import default_settings
    from vibecheck.verify_graph import verify_graph
    g = ComputeGraph.from_onnx(p, dtype=np.float32)
    # Spec: Y_0 <= 0 (which is false — Y_0 is always in [0.667, 1.5])
    conj = Conjunct([Constraint(0, '<=', 0.0)])
    spec = VNNSpec(
        x_lo=np.array([-2.0, -2.0], dtype=np.float32),
        x_hi=np.array([2.0, 2.0], dtype=np.float32),
        disjuncts=[conj])
    s = default_settings(device='cpu', bits=32, total_timeout=10,
                          print_progress=False,
                          verified_validation_samples=0)
    v, det = verify_graph(g, spec, s)
    spec_lbs = det.get('spec_lbs') or {}
    if spec_lbs:
        # Sound: spec_lb ≤ true min Y_0 = 0.667
        assert max(spec_lbs.values()) <= 0.667 + 1e-3, (
            f'UNSOUND spec_lb: {spec_lbs} exceeds sample-min 0.667')
