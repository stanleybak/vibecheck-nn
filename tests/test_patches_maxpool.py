"""Patches-native ``slice`` + ``sub`` keep the maxpool->relu decomposition in
the PatchesZonotope representation (no dense G materialisation).

The decomposition emits, per pool window, contiguous channel-block Slices of a
phase-extraction Conv plus a binary-max tree of Sub/ReLU/Add. On a full-image
input the dense (n_flat, K) generator matrix is terabytes, so these ops MUST
stay patches-native. These tests pin (a) the two new primitives against their
dense equivalents and (b) the whole forward through a real maxpool net, dense
vs patches, bit-for-bit.
"""
import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import maxpool_to_relu
from vibecheck.patches_zonotope import PatchesZonotope
from vibecheck.settings import default_settings
from vibecheck.zonotope import TorchZonotope
from vibecheck.verify_zono_bnb import _forward_zonotope_graph


def _grown_patches(C=4, H=6, W=6, seed=0):
    """A PatchesZonotope after one conv (patches grown past 1x1), float64."""
    torch.manual_seed(seed)
    x_lo = torch.rand(C * H * W, dtype=torch.float64) - 1.0
    x_hi = x_lo + torch.rand(C * H * W, dtype=torch.float64) * 0.5 + 0.01
    z = PatchesZonotope.from_input_bounds(
        x_lo, x_hi, (C, H, W), 'cpu', torch.float64)
    kernel = torch.randn(C, C, 3, 3, dtype=torch.float64)
    bias = torch.randn(C, dtype=torch.float64)
    z.propagate_conv(kernel, bias, (C, H, W), (1, 1), (1, 1))
    return z


def test_slice_channel_block_matches_dense():
    z = _grown_patches(C=4, H=6, W=6)
    assert z._mode == 'patches'
    C, H, W = z.out_shape
    lo, hi = z.bounds()
    # Every contiguous channel block must slice exactly (patches-native).
    for c0, c1 in [(0, 1), (1, 3), (0, 4), (2, 4)]:
        idx = torch.arange(c0 * H * W, c1 * H * W)
        zs = z.slice(idx)
        assert zs._mode == 'patches', 'channel block must stay patches-native'
        lo_s, hi_s = zs.bounds()
        torch.testing.assert_close(lo_s, lo[c0 * H * W:c1 * H * W], atol=1e-12,
                                   rtol=0)
        torch.testing.assert_close(hi_s, hi[c0 * H * W:c1 * H * W], atol=1e-12,
                                   rtol=0)


def test_slice_non_channel_block_falls_back_dense():
    z = _grown_patches(C=4, H=6, W=6)
    lo, hi = z.bounds()
    # A scattered (non-contiguous) index can't stay patches-native.
    idx = torch.tensor([0, 5, 17, 100, 3])
    zs = z.slice(idx)
    assert zs._mode == 'dense'
    lo_s, hi_s = zs.bounds()
    torch.testing.assert_close(lo_s, lo[idx], atol=1e-12, rtol=0)
    torch.testing.assert_close(hi_s, hi[idx], atol=1e-12, rtol=0)


def test_sub_equal_gens_matches_dense():
    """b - a where both are channel-slices of the same conv (same K, offsets)."""
    z = _grown_patches(C=4, H=6, W=6)
    C, H, W = z.out_shape
    a = z.slice(torch.arange(0, 2 * H * W))           # channels 0:2
    b = z.slice(torch.arange(2 * H * W, 4 * H * W))   # channels 2:4
    K = a.n_gens
    assert a.n_gens == b.n_gens == K
    diff = b.sub(a, K)
    # Dense reference.
    ad, bd = a.to_dense(), b.to_dense()
    ref = TorchZonotope(bd.center - ad.center, bd.generators - ad.generators)
    lo_r, hi_r = ref.bounds()
    lo_d, hi_d = diff.bounds()
    torch.testing.assert_close(lo_d, lo_r, atol=1e-12, rtol=0)
    torch.testing.assert_close(hi_d, hi_r, atol=1e-12, rtol=0)


def test_sub_unequal_gens_matches_dense():
    """c - u where u has extra gens (post-ReLU) that c lacks: shared-prefix."""
    z = _grown_patches(C=4, H=6, W=6)
    C, H, W = z.out_shape
    c = z.slice(torch.arange(0, 2 * H * W))
    u = z.slice(torch.arange(2 * H * W, 4 * H * W))
    K = c.n_gens
    # Grow u's generator count via a ReLU (adds one gen per unstable neuron).
    u.apply_relu()
    assert u.n_gens > K
    diff = c.sub(u, K)            # shared = the K conv columns
    # Dense reference: [c_shared - u_shared | -u_extra].
    cd, ud = c.to_dense(), u.to_dense()
    g_ref = torch.cat([cd.generators - ud.generators[:, :K],
                       -ud.generators[:, K:]], dim=1)
    ref = TorchZonotope(cd.center - ud.center, g_ref)
    lo_r, hi_r = ref.bounds()
    lo_d, hi_d = diff.bounds()
    torch.testing.assert_close(lo_d, lo_r, atol=1e-12, rtol=0)
    torch.testing.assert_close(hi_d, hi_r, atol=1e-12, rtol=0)


def _maxpool_net_onnx(path, C=3, H=8, W=8):
    """Conv(C->4) -> Relu -> MaxPool(2x2,s2) -> Conv(4->2) -> Relu."""
    rng = np.random.RandomState(0)
    w1 = rng.randn(4, C, 3, 3).astype(np.float32)
    b1 = rng.randn(4).astype(np.float32)
    w2 = rng.randn(2, 4, 3, 3).astype(np.float32)
    b2 = rng.randn(2).astype(np.float32)
    init = [
        helper.make_tensor('w1', TensorProto.FLOAT, w1.shape, w1.flatten()),
        helper.make_tensor('b1', TensorProto.FLOAT, b1.shape, b1.flatten()),
        helper.make_tensor('w2', TensorProto.FLOAT, w2.shape, w2.flatten()),
        helper.make_tensor('b2', TensorProto.FLOAT, b2.shape, b2.flatten()),
    ]
    nodes = [
        helper.make_node('Conv', ['x', 'w1', 'b1'], ['c1'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('MaxPool', ['r1'], ['m1'], kernel_shape=[2, 2],
                         strides=[2, 2], pads=[0, 0, 0, 0]),
        helper.make_node('Conv', ['m1', 'w2', 'b2'], ['c2'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[1, 1]),
        helper.make_node('Relu', ['c2'], ['y']),
    ]
    g = helper.make_graph(
        nodes, 'mpnet',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info(
            'y', TensorProto.FLOAT, [1, 2, H // 2, W // 2])], init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.save(m, path)


def test_forward_dense_equals_patches_through_maxpool(tmp_path):
    """The whole forward (slice + sub_bilinear + add + relu through the
    maxpool decomposition) must give identical bounds in dense and patches
    mode — and patches mode must never need the dense G."""
    C, H, W = 3, 8, 8
    p = str(tmp_path / 'mpnet.onnx')
    _maxpool_net_onnx(p, C, H, W)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert maxpool_to_relu(g) is True
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)

    torch.manual_seed(0)
    xl = (torch.rand(C * H * W, dtype=torch.float64) - 1.0)
    xh = xl + torch.rand(C * H * W, dtype=torch.float64) * 0.3 + 0.01

    s_dense = default_settings(); s_dense.zono_impl = 'dense'
    s_patch = default_settings(); s_patch.zono_impl = 'patches'

    _, z_dense = _forward_zonotope_graph(
        xl, xh, gg, 'cpu', torch.float64, settings=s_dense)
    _, z_patch = _forward_zonotope_graph(
        xl, xh, gg, 'cpu', torch.float64, settings=s_patch)

    assert isinstance(z_patch, PatchesZonotope)
    lo_d, hi_d = z_dense.bounds()
    lo_p, hi_p = z_patch.bounds()
    torch.testing.assert_close(lo_p, lo_d, atol=1e-10, rtol=0)
    torch.testing.assert_close(hi_p, hi_d, atol=1e-10, rtol=0)
