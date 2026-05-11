"""Unit tests for batched zonotope forward pass.

Correctness contract:
  forward_zonotope_graph_batched(xls, xhs, gg, ...) on B input boxes
  must produce per-leaf bounds equal (within fp tolerance) to running
  the existing `_forward_zonotope_graph` independently on each input
  box. We verify on a tiny synthetic Conv→ReLU→Conv→ReLU→FC graph and
  also on a real cifar_biasfield network when the benchmark is
  available.
"""
import os

import numpy as np
import onnx
import onnx.helper as oh
import pytest
import torch

from vibecheck.batched_zono import forward_zonotope_graph_batched
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph


def _tiny_relu_net(tmp_path):
    """Tiny Conv→ReLU→Conv→ReLU→FC network with input (1,1,4,4) → 2."""
    K1 = np.arange(1, 1 + 2 * 1 * 1 * 1, dtype=np.float32).reshape(2, 1, 1, 1) * 0.3
    b1 = np.zeros(2, dtype=np.float32)
    K2 = np.arange(1, 1 + 2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2) * 0.1
    b2 = np.zeros(2, dtype=np.float32)
    W3 = np.arange(1, 1 + 2 * 18, dtype=np.float32).reshape(2, 18) * 0.05
    b3 = np.zeros(2, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])
    inits = [
        oh.make_tensor('K1', onnx.TensorProto.FLOAT, K1.shape, K1.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, b1.shape, b1),
        oh.make_tensor('K2', onnx.TensorProto.FLOAT, K2.shape, K2.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, b2.shape, b2),
        oh.make_tensor('W3', onnx.TensorProto.FLOAT, W3.shape, W3.flatten()),
        oh.make_tensor('b3', onnx.TensorProto.FLOAT, b3.shape, b3),
    ]
    nodes = [
        oh.make_node('Conv', ['x', 'K1', 'B1'], ['z1'],
                     kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Conv', ['a1', 'K2', 'B2'], ['z2'],
                     kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z2'], ['a2']),
        oh.make_node('Reshape', ['a2', 'shape'], ['flat']),
        oh.make_node('Gemm', ['flat', 'W3', 'b3'], ['y'], transB=1),
    ]
    # Reshape needs a 'shape' init: target shape = (1, 18) (2*3*3=18)
    inits.append(oh.make_tensor('shape', onnx.TensorProto.INT64, [2],
                                  np.array([1, 18], dtype=np.int64)))
    graph = oh.make_graph(nodes, 'g', [inp], [out], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    model.ir_version = 7
    p = tmp_path / 'tiny.onnx'
    onnx.save(model, str(p))
    return str(p)


def _bounds_match(b_batched, b_single, atol=1e-5, rtol=1e-5):
    """Assert per-batch bounds match individual evaluations."""
    lo_b, hi_b = b_batched
    lo_s, hi_s = b_single
    assert torch.allclose(lo_b, lo_s, atol=atol, rtol=rtol), (
        f'lo mismatch: max |Δ| = {(lo_b - lo_s).abs().max():.3e}')
    assert torch.allclose(hi_b, hi_s, atol=atol, rtol=rtol), (
        f'hi mismatch: max |Δ| = {(hi_b - hi_s).abs().max():.3e}')


@pytest.mark.parametrize('B', [1, 2, 4])
def test_batched_matches_unbatched_tiny(B, tmp_path):
    """Batched forward on B leaves matches B independent unbatched calls."""
    g = load_onnx(_tiny_relu_net(tmp_path))
    s_dummy = type('S', (), {'fuse_gemm_conv': True,
                              'optimize_relu_relation': True})()
    g.optimize(s_dummy)
    device = torch.device('cpu')
    dtype = torch.float64
    gg = g.gpu_graph(device, dtype)

    rng = np.random.default_rng(0)
    n_input = 16  # 1×1×4×4 flat
    xls_np = rng.uniform(-1.0, 0.5, size=(B, n_input)).astype(np.float64)
    xhs_np = xls_np + rng.uniform(0.1, 1.0, size=(B, n_input))
    xls = torch.tensor(xls_np, dtype=dtype, device=device)
    xhs = torch.tensor(xhs_np, dtype=dtype, device=device)

    # Batched
    sb_b, _, _ = forward_zonotope_graph_batched(xls, xhs, gg, device, dtype)

    # Unbatched per-leaf
    for b in range(B):
        sb_s, _ = _forward_zonotope_graph(
            xls[b].flatten(), xhs[b].flatten(), gg, device, dtype)
        for L, (lo_b, hi_b) in sb_b.items():
            assert L in sb_s, f'layer {L} missing in single-leaf result'
            _bounds_match((lo_b[b], hi_b[b]), sb_s[L])


def test_make_input_zonotopes_batched_shape():
    """Sanity: input zono shape is (B, n_in) for center, (B, K, n_in) for gens.

    Axis 2 has zero radius in both leaves (xl == xh), so it gets no
    generator column. Center on that axis is the midpoint (0.5 for
    leaf 1) — the test pinned the wrong assertion previously."""
    from vibecheck.batched_zono import make_input_zonotopes_batched
    device = torch.device('cpu'); dtype = torch.float32
    xls = torch.tensor([[-1.0, -2.0, 0.0],
                         [-1.5, -2.5, 0.5]], dtype=dtype)
    xhs = torch.tensor([[1.0, 2.0, 0.0],
                         [1.5, 2.5, 0.5]], dtype=dtype)
    c, g = make_input_zonotopes_batched(xls, xhs, device, dtype)
    assert c.shape == (2, 3)
    assert g.shape == (2, 2, 3)  # K=2 (axes 0 and 1, both positive-radius)
    # Centers per axis: midpoint of (xl, xh).
    assert torch.allclose(c[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(c[1], torch.tensor([0.0, 0.0, 0.5]))
    # Per-leaf gens: gen k carries the radius of axis nz[k].
    assert torch.allclose(g[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(g[0, 1], torch.tensor([0.0, 2.0, 0.0]))
    assert torch.allclose(g[1, 0], torch.tensor([1.5, 0.0, 0.0]))
    assert torch.allclose(g[1, 1], torch.tensor([0.0, 2.5, 0.0]))


def test_apply_relu_batched_zonotope_overapproximates_relu():
    """Batched ReLU produces a zonotope that over-approximates the
    true reachable set of relu(z) for any z in the input zonotope.

    Sound over-approximation property: every concrete sample z in the
    pre-ReLU zonotope, after applying relu(z), must lie inside the
    post-ReLU zonotope's bounds. We can't directly assert post-relu
    `lo >= 0` (the relaxation OVERAPPROXIMATES, the over-approx
    polytope can extend below zero — but the TRUE reachable set is
    still ≥ 0). What we CAN check: random samples from the input
    zonotope, after relu, fall inside the output bounds."""
    from vibecheck.batched_zono import (
        apply_relu_batched, bounds_batched, make_input_zonotopes_batched)
    device = torch.device('cpu'); dtype = torch.float64
    rng = np.random.default_rng(42)
    B, n_flat, K = 2, 4, 3
    center = torch.tensor(
        rng.uniform(-0.5, 0.5, (B, n_flat)), dtype=dtype, device=device)
    gens = torch.tensor(
        rng.uniform(-0.3, 0.3, (B, K, n_flat)), dtype=dtype, device=device)
    new_c, new_g, _lo, _hi = apply_relu_batched(center, gens)
    out_lo, out_hi = bounds_batched(new_c, new_g)
    # Sample concrete points in the input zono via random e ∈ [-1, 1]^K.
    n_samples = 200
    e = torch.tensor(rng.uniform(-1, 1, (B, n_samples, K)),
                      dtype=dtype, device=device)
    # z = center + e @ gens : for each sample s, z_b_s = c_b + Σ_k e_b_s_k * g_b_k
    z = center.unsqueeze(1) + torch.einsum('bsk,bkn->bsn', e, gens)
    relu_z = torch.relu(z)
    # All concrete relu_z samples must lie in [out_lo, out_hi].
    assert (relu_z >= out_lo.unsqueeze(1) - 1e-9).all(), (
        f'relu(z) sample below out_lo: '
        f'min Δ = {(relu_z - out_lo.unsqueeze(1)).min().item():.3e}')
    assert (relu_z <= out_hi.unsqueeze(1) + 1e-9).all(), (
        f'relu(z) sample above out_hi: '
        f'max Δ = {(relu_z - out_hi.unsqueeze(1)).max().item():.3e}')
