"""Tests for the cgan branching fix and its observability:

  * `_crown_intermediate_batched` is sparse (unstable-only) + chunked — the
    tightening is INVARIANT to the chunk size, and a layer with no unstable
    neuron is skipped (m == 0).
  * The batched input-split records its branching heuristic in the verbose
    trace (`[branch] input-split heuristic=...`), so a silently-disabled `sb`
    is visible instead of needing a probe.
  * `input_split_sb_margin_score` selects the margin-augmented worst-query sb
    score; without it `sb` uses the legacy sum; without
    `input_split_batched_branch_sb` it falls to widest-axis and SAYS so.
"""
import io
import contextlib

import numpy as np
import onnx
import onnx.helper as oh
import torch

from vibecheck.onnx_loader import load_onnx
from vibecheck.spec import VNNSpec, Conjunct, PairwiseConstraint
from vibecheck.settings import default_settings
from vibecheck import verify_graph as vg


# --- tiny 3-in / 3-out net: y = relu(x + b1) (identity W1, W2) ---------------
def _relu_net(tmp_path, b1):
    W = np.eye(3, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3])
    inits = [
        oh.make_tensor('W1', onnx.TensorProto.FLOAT, [3, 3], W.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, [3], np.array(b1, np.float32)),
        oh.make_tensor('W2', onnx.TensorProto.FLOAT, [3, 3], W.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, [3], np.zeros(3, np.float32)),
    ]
    nodes = [
        oh.make_node('Gemm', ['x', 'W1', 'B1'], ['z1'], transB=1),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Gemm', ['a1', 'W2', 'B2'], ['y'], transB=1),
    ]
    graph = oh.make_graph(nodes, 'g', [inp], [out], inits)
    m = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    m.ir_version = 7
    p = tmp_path / 'relu_net.onnx'
    onnx.save(m, str(p))
    return str(p)


def _gg_for(tmp_path, b1):
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g = load_onnx(_relu_net(tmp_path, b1))
    g.optimize(s)
    return g.gpu_graph(torch.device('cpu'), torch.float64)


# --- crown_intermediate sparse + chunk ---------------------------------------
def test_crown_intermediate_chunk_invariant(tmp_path):
    """The sparse backward tightening is identical whether every unstable neuron
    is seeded in one shot (large chunk) or one-at-a-time (chunk=1) — exercising
    the multi-iteration chunk loop."""
    from vibecheck.verify_zono_bnb import _crown_intermediate_batched
    gg = _gg_for(tmp_path, b1=[0.0, 0.0, 0.0])
    dev, dt = torch.device('cpu'), torch.float64
    xl, xh = -torch.ones((1, 3), dtype=dt), torch.ones((1, 3), dtype=dt)
    big = _crown_intermediate_batched(gg, xl, xh, dev, dt, chunk=4096)
    one = _crown_intermediate_batched(gg, xl, xh, dev, dt, chunk=1)
    assert big.keys() == one.keys() and big, 'expected a ReLU layer'
    for L in big:
        assert torch.allclose(big[L][0], one[L][0], atol=1e-9)
        assert torch.allclose(big[L][1], one[L][1], atol=1e-9)


def test_crown_intermediate_all_stable_skipped(tmp_path):
    """A ReLU layer with NO unstable neuron (m == 0) is left at its forward-zono
    bound — the sparse loop's empty-mask branch."""
    from vibecheck.verify_zono_bnb import (
        _crown_intermediate_batched, _forward_zonotope_graph_batched)
    # Large +bias -> every pre-activation > 0 on x in [0,1] -> all stable.
    gg = _gg_for(tmp_path, b1=[10.0, 10.0, 10.0])
    dev, dt = torch.device('cpu'), torch.float64
    xl, xh = torch.zeros((1, 3), dtype=dt), torch.ones((1, 3), dtype=dt)
    tight = _crown_intermediate_batched(gg, xl, xh, dev, dt)
    sb, _ = _forward_zonotope_graph_batched(xl, xh, gg, dev, dt)
    assert tight, 'expected a ReLU layer'
    for L in tight:
        lo, hi = sb[L]
        assert torch.all(lo > 0)        # confirm the layer is all-stable
        assert torch.allclose(tight[L][0], lo) and torch.allclose(tight[L][1], hi)


# --- branch-method trace + scoring selection ---------------------------------
# Conv net + spec lifted from test_input_split_alpha_threshold — input_dim=4
# routes through the input-split BaB and the spec leaves the root open, so the
# batched BaB splits at least once and the `[branch]` trace fires.
def _conv_net(tmp_path):
    K1 = np.array([[[[1.0]]], [[[-1.0]]]], dtype=np.float32)
    K2 = np.array([[[[1.0]], [[-1.0]]], [[[0.5]], [[1.0]]]], dtype=np.float32)
    K3 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    K3[0, 0, 0, 0] = 1.0; K3[0, 1, 1, 1] = 1.0
    K3[1, 0, 1, 0] = 1.0; K3[1, 1, 0, 1] = 1.0
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])
    inits = [
        oh.make_tensor('K1', onnx.TensorProto.FLOAT, K1.shape, K1.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, [2], np.zeros(2, np.float32)),
        oh.make_tensor('K2', onnx.TensorProto.FLOAT, K2.shape, K2.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, [2], np.zeros(2, np.float32)),
        oh.make_tensor('K3', onnx.TensorProto.FLOAT, K3.shape, K3.flatten()),
        oh.make_tensor('B3', onnx.TensorProto.FLOAT, [2], np.zeros(2, np.float32)),
    ]
    nodes = [
        oh.make_node('Conv', ['x', 'K1', 'B1'], ['z1'], kernel_shape=[1, 1],
                     strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Conv', ['a1', 'K2', 'B2'], ['z2'], kernel_shape=[1, 1],
                     strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z2'], ['a2']),
        oh.make_node('Conv', ['a2', 'K3', 'B3'], ['z3'], kernel_shape=[2, 2],
                     strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Flatten', ['z3'], ['y']),
    ]
    graph = oh.make_graph(nodes, 'g', [inp], [out], inits)
    m = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    m.ir_version = 7
    p = tmp_path / 'conv.onnx'
    onnx.save(m, str(p))
    return str(p)


def _spec():
    return VNNSpec(
        x_lo=np.zeros((1, 1, 2, 2), np.float32),
        x_hi=np.ones((1, 1, 2, 2), np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),))


def _run_capture(tmp_path, **over):
    g = load_onnx(_conv_net(tmp_path))
    # Disable the root/leaf PGD short-circuit so the BaB actually splits (and
    # the `[branch]` trace fires) instead of PGD finding a witness at the root.
    s = default_settings(device='cpu', bits=64, total_timeout=20,
                         print_progress=True, input_split_batched_enabled=True,
                         pgd_phase0_enabled=False, disable_sat_finding=True,
                         **over)
    g.optimize(s)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result, _ = vg.verify_graph(g, _spec(), s)
    return result, buf.getvalue()


def test_branch_trace_sb_margin(tmp_path):
    """sb on + margin score → trace says `heuristic=sb-margin` + per-iter splits."""
    result, out = _run_capture(
        tmp_path, input_split_batched_branch_sb=True,
        input_split_sb_margin_score=True)
    assert '[branch] input-split heuristic=sb-margin' in out
    assert '[branch] iter=' in out
    assert result in ('verified', 'sat', 'unknown')


def test_branch_trace_sb_sum(tmp_path):
    """sb on, margin off → legacy sum score, trace says `heuristic=sb-sum`."""
    _, out = _run_capture(tmp_path, input_split_batched_branch_sb=True,
                          input_split_sb_margin_score=False)
    assert '[branch] input-split heuristic=sb-sum' in out


def test_branch_trace_widest_axis_disabled(tmp_path):
    """sb off (default) → widest-axis, and the trace NAMES the reason."""
    _, out = _run_capture(tmp_path, input_split_batched_branch_sb=False)
    assert 'heuristic=widest-axis' in out
    assert 'sb disabled' in out
