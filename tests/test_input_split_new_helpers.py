"""Unit tests for the input-split BaB helpers added for acasxu:
  - `_simple_pgd_batched` (verify_hybrid_acasxu): batched leaf-PGD that samples
    inside MANY distinct boxes at once — finds a narrow SAT witness a wide-box
    PGD misses.
  - `_crown_intermediate_batched` (verify_zono_bnb): backward-CROWN intermediate
    pre-ReLU bounds, intersected with the forward zonotope (both sound).

Synthetic FC ReLU net (no vnncomp benchmark), CPU, so it runs in the unit suite.
The end-to-end input-split path (vectorized split + the leaf-PGD wire-in) is
covered by the acasxu integration tests.
"""
import numpy as np
import onnx
import onnx.helper as oh
import torch

from vibecheck.onnx_loader import load_onnx
from vibecheck.spec import Conjunct, PairwiseConstraint, VNNSpec
from vibecheck.settings import default_settings


def _relu_identity_onnx(tmp_path):
    """y = relu(x) for x in R^2: Gemm(I) -> ReLU -> Gemm(I). So
    out[0]-out[1] = relu(x0)-relu(x1), which is controllable by the input box."""
    I = np.eye(2, dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 2])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])
    inits = [
        oh.make_tensor('W1', onnx.TensorProto.FLOAT, [2, 2], I.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, [2], z),
        oh.make_tensor('W2', onnx.TensorProto.FLOAT, [2, 2], I.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, [2], z),
    ]
    nodes = [
        oh.make_node('Gemm', ['x', 'W1', 'B1'], ['z1'], transB=1),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Gemm', ['a1', 'W2', 'B2'], ['y'], transB=1),
    ]
    g = oh.make_graph(nodes, 'g', [inp], [out], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid('', 14)])
    m.ir_version = 7
    p = tmp_path / 'relu_id.onnx'
    onnx.save(m, str(p))
    return str(p)


def _gg(tmp_path):
    s = default_settings(device='cpu', bits=64, print_progress=False)
    graph = load_onnx(_relu_identity_onnx(tmp_path))
    graph.optimize(s)
    return graph.gpu_graph(torch.device('cpu'), torch.float64)


def test_simple_pgd_batched_finds_witness_in_one_leaf(tmp_path):
    """The unsafe set {out[0] <= out[1]} = {x0 <= x1} is empty in box A
    ([0.6,1]x[0,0.4]) but non-empty in box B ([0,0.4]x[0.6,1]). Batched-PGD over
    BOTH boxes at once must find the witness in B and return it."""
    from vibecheck.verify_hybrid_acasxu import _simple_pgd_batched
    gg = _gg(tmp_path)
    spec = VNNSpec(
        x_lo=np.zeros((1, 2), dtype=np.float32),
        x_hi=np.ones((1, 2), dtype=np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),),
    )
    dev, dt = torch.device('cpu'), torch.float64
    xl = torch.tensor([[0.6, 0.0], [0.0, 0.6]], dtype=dt)   # box A (safe), box B (sat)
    xh = torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=dt)
    sat, w = _simple_pgd_batched(xl, xh, spec, gg, 2, dev, dt,
                                  n_restarts=64, n_iter=30)
    assert sat is True
    w = np.asarray(w)
    assert w.shape == (2,)
    # witness is a real in-box point of B that violates out[0] <= out[1]
    assert w[0] <= w[1] + 1e-5


def test_simple_pgd_batched_no_witness_when_safe(tmp_path):
    """Both boxes have x0 > x1 (safe) -> no witness -> (False, None)."""
    from vibecheck.verify_hybrid_acasxu import _simple_pgd_batched
    gg = _gg(tmp_path)
    spec = VNNSpec(
        x_lo=np.zeros((1, 2), dtype=np.float32),
        x_hi=np.ones((1, 2), dtype=np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),),
    )
    dev, dt = torch.device('cpu'), torch.float64
    xl = torch.tensor([[0.6, 0.0], [0.7, 0.1]], dtype=dt)
    xh = torch.tensor([[1.0, 0.4], [0.9, 0.3]], dtype=dt)
    sat, w = _simple_pgd_batched(xl, xh, spec, gg, 2, dev, dt,
                                  n_restarts=64, n_iter=20)
    assert sat is False
    assert w is None


def test_crown_intermediate_batched_sound_and_tighter(tmp_path):
    """`_crown_intermediate_batched` returns per-ReLU (lo, hi) that (a) bracket
    the true pre-ReLU range and (b) are no looser than the forward zonotope's."""
    from vibecheck.verify_zono_bnb import (
        _crown_intermediate_batched, _forward_zonotope_graph_batched)
    gg = _gg(tmp_path)
    dev, dt = torch.device('cpu'), torch.float64
    xl = torch.tensor([[0.0, 0.0]], dtype=dt)
    xh = torch.tensor([[1.0, 1.0]], dtype=dt)
    tight = _crown_intermediate_batched(gg, xl, xh, dev, dt)
    sb, _ = _forward_zonotope_graph_batched(xl, xh, gg, dev, dt)
    assert tight, 'expected at least one ReLU layer'
    for L, (lo, hi) in tight.items():
        zlo, zhi = sb[L]
        # intersection is no looser than the forward zono
        assert torch.all(lo >= zlo - 1e-9)
        assert torch.all(hi <= zhi + 1e-9)
        assert torch.all(hi >= lo - 1e-9)
        # for this net the pre-ReLU value is x in [0,1] -> bounds must contain it
        assert torch.all(lo <= 1e-9) and torch.all(hi >= 1.0 - 1e-9)
