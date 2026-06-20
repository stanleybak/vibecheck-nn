"""The output-count probe in verify_graph._run_pipeline must not OOM.

When a net's final op isn't fc/conv (e.g. a residual Add, or ml4acopf's nonlinear
Concat), n_output is obtained by a forward-zonotope probe. It used to propagate the
REAL input box, which on a wide-ReLU net creates one error generator per unstable
neuron — the dense generator matrix then OOMs (soundnessbench model_residual: ~16 GB)
BEFORE PGD runs. The fix propagates a DEGENERATE (point) box: every ReLU is stable,
so NO generators are created, the count (center.numel()) is identical, and memory
stays flat. These tests pin that mechanism + the count's correctness.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

from vibecheck.network import ComputeGraph
from vibecheck.verify_zono_bnb import _forward_zonotope_graph


def _wide_relu_residual_onnx(path, d=8, wide=400):
    """X[1,d] -> Gemm(d->wide) -> ReLU -> Gemm(wide->d) -> Add(X) -> Y[1,d].
    Ends in Add (NOT fc/conv) so it takes the probe branch; the `wide` ReLU is the
    generator-explosion driver under a real input box."""
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((d, wide)).astype(np.float32)
    b1 = rng.standard_normal(wide).astype(np.float32)
    W2 = rng.standard_normal((wide, d)).astype(np.float32)
    b2 = rng.standard_normal(d).astype(np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['h1']),
        helper.make_node('Add', ['h1', 'b1'], ['h1b']),
        helper.make_node('Relu', ['h1b'], ['a1']),
        helper.make_node('MatMul', ['a1', 'W2'], ['h2']),
        helper.make_node('Add', ['h2', 'b2'], ['h2b']),
        helper.make_node('Add', ['h2b', 'X'], ['Y']),     # residual; final op = Add
    ]
    inits = [numpy_helper.from_array(a, n) for a, n in
             [(W1, 'W1'), (b1, 'b1'), (W2, 'W2'), (b2, 'b2')]]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, d])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, d])
    m = helper.make_model(helper.make_graph(nodes, 'wr', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.checker.check_model(m)
    onnx.save(m, path)


def test_degenerate_probe_no_generator_explosion(tmp_path):
    d, wide = 8, 400
    p = str(tmp_path / 'wr.onnx'); _wide_relu_residual_onnx(p, d, wide)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    # final op is not fc/conv -> the n_output branch that probes a forward zonotope
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    assert gg['ops'][-1]['type'] not in ('fc', 'conv')

    xl = torch.full((d,), -1.0, dtype=torch.float64)
    xh = torch.full((d,), 1.0, dtype=torch.float64)
    xm = (xl + xh) / 2.0

    # real box: one generator per unstable ReLU -> matrix blows up
    _, zf_box = _forward_zonotope_graph(xl, xh, gg, 'cpu', torch.float64, settings=None)
    # degenerate (point) box: all ReLUs stable -> zero generators (the fix)
    _, zf_pt = _forward_zonotope_graph(xm, xm, gg, 'cpu', torch.float64, settings=None)

    n_box = zf_box.generators.shape[1]
    n_pt = zf_pt.generators.shape[1]
    assert n_pt == 0, f"degenerate probe must create no generators, got {n_pt}"
    assert n_box >= wide, f"real-box probe should explode (>= {wide} gens), got {n_box}"
    # the OUTPUT COUNT (what the probe is for) is identical either way
    assert zf_pt.center.numel() == zf_box.center.numel() == d


def test_verify_residual_net_no_error(tmp_path):
    """End-to-end: a non-fc/conv-final net verifies (returns a verdict, not an OOM
    'error') — exercises the real n_output branch in _run_pipeline."""
    from vibecheck.verify_graph import verify_graph
    from vibecheck.spec import VNNSpec, Conjunct, Constraint
    from vibecheck.settings import default_settings
    p = str(tmp_path / 'wr.onnx'); _wide_relu_residual_onnx(p, d=8, wide=400)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    spec = VNNSpec(
        x_lo=np.full(8, -1.0), x_hi=np.full(8, 1.0),
        disjuncts=[Conjunct([Constraint(0, '<=', 1e9)])])   # trivially-holds output bound
    s = default_settings(device='cpu', total_timeout=20)
    result, _ = verify_graph(g, spec, s)
    assert result in ('verified', 'unknown', 'sat', 'timeout')   # NOT an OOM crash
