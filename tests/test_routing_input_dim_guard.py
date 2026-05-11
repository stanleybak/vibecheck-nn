"""Regression test: `verify_graph` must NOT auto-route conv nets with
small input dim to `milp_verify`. Otherwise cifar_biasfield (input_dim=16,
Conv, no forks) goes into the milp pipeline where the joint α-CROWN
tightening OOMs the 10 GB GPU instead of the input-split BaB / fast-leaf
path that handles it cleanly.

The test patches `milp_verify` so it raises if reached, then exercises a
small Conv graph with input_dim=4. The current routing must skip
`milp_verify` and fall through to the input-split / standard graph
pipeline.
"""
import numpy as np
import onnx
import onnx.helper as oh
import pytest

from vibecheck.onnx_loader import load_onnx
from vibecheck.spec import Conjunct, PairwiseConstraint, VNNSpec
from vibecheck.settings import default_settings
from vibecheck import verify_graph as vg


def _tiny_conv_onnx(tmp_path):
    """Tiny 1×2×2 input Conv → ReLU → Conv → Flatten → output. 4 input
    elements total — small enough that input_split would fire.

    Layer shapes:
      Conv1: in (1, 2, 2) → out (2, 2, 2)  (kernel 1×1, in_c=1, out_c=2)
      ReLU
      Conv2: in (2, 2, 2) → out (2, 1, 1)  (kernel 2×2, in_c=2, out_c=2)
      Flatten → (2,)
    """
    K1 = np.array([[[[1.0]]], [[[-1.0]]]], dtype=np.float32)  # (2, 1, 1, 1)
    b1 = np.zeros(2, dtype=np.float32)
    K2 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    K2[0, 0, 0, 0] = 1.0; K2[0, 1, 1, 1] = 1.0
    K2[1, 0, 1, 0] = 1.0; K2[1, 1, 0, 1] = 1.0
    b2 = np.zeros(2, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])
    inits = [
        oh.make_tensor('K1', onnx.TensorProto.FLOAT, K1.shape, K1.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, b1.shape, b1),
        oh.make_tensor('K2', onnx.TensorProto.FLOAT, K2.shape, K2.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, b2.shape, b2),
    ]
    nodes = [
        oh.make_node('Conv', ['x', 'K1', 'B1'], ['z1'],
                     kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Conv', ['a1', 'K2', 'B2'], ['z2'],
                     kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Flatten', ['z2'], ['y']),
    ]
    graph = oh.make_graph(nodes, 'g', [inp], [out], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    model.ir_version = 7
    p = tmp_path / 'tiny_conv.onnx'
    onnx.save(model, str(p))
    return str(p)


def test_small_input_conv_skips_milp_route(monkeypatch, tmp_path):
    """Conv graph with input_dim=4 (≤ input_split_max_dims=20) must NOT
    auto-route to milp_verify. With pre-fix code (no input-dim guard) this
    test fails because the route would fire on (has_conv ∧ no_forks)
    alone. The fix at verify_graph.py:5370 added an input-dim guard."""
    g = load_onnx(_tiny_conv_onnx(tmp_path))
    spec = VNNSpec(
        x_lo=np.zeros((1, 1, 2, 2), dtype=np.float32),
        x_hi=np.ones((1, 1, 2, 2), dtype=np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),),
    )

    called = {'milp_verify': False}

    def _stub_milp(*a, **kw):
        called['milp_verify'] = True
        raise AssertionError(
            'milp_verify must not be reached on a small-input conv net')
    monkeypatch.setattr('vibecheck.verify_milp.milp_verify', _stub_milp)

    s = default_settings(device='cpu', bits=64, total_timeout=10,
                         print_progress=False)
    g.optimize(s)
    # If the routing guard works, this returns a verdict (verified /
    # unknown / sat) without ever calling _stub_milp.
    result, _ = vg.verify_graph(g, spec, s)
    assert result in ('verified', 'unknown', 'sat')
    assert not called['milp_verify']


def test_large_input_conv_still_auto_routes(monkeypatch, tmp_path):
    """Inverse guard: when input_dim > input_split_max_dims, the conv
    auto-route must still fire. We verify by patching milp_verify to
    record the call and check it was hit."""
    # Reuse tiny_conv but pretend input_dim is 50 by setting
    # `input_split_max_dims=2` so the guard treats 4 as "large".
    g = load_onnx(_tiny_conv_onnx(tmp_path))
    spec = VNNSpec(
        x_lo=np.zeros((1, 1, 2, 2), dtype=np.float32),
        x_hi=np.ones((1, 1, 2, 2), dtype=np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),),
    )

    called = {'milp_verify': False}

    def _stub_milp(graph, sp, settings):
        called['milp_verify'] = True
        return 'unknown', {'phase': 'stub'}
    monkeypatch.setattr('vibecheck.verify_milp.milp_verify', _stub_milp)

    s = default_settings(device='cpu', bits=64, total_timeout=10,
                         print_progress=False, input_split_max_dims=2)
    g.optimize(s)
    vg.verify_graph(g, spec, s)
    assert called['milp_verify']
