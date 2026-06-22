"""forward-LiRPA on 2D conv (Conv with a 3D (C,H,W) in_shape_nd).

Regression (policy: implemented = tested): `batched_forward_linear_bounds`'
conv handler asserted `len(in_shape_nd) == 2`, which blocked 2D conv even
though the handler body already supports the 3D (C,H,W) case. That blocked the
memory-efficient forward-LiRPA path on any 2D-conv net (e.g. soundnessbench,
whose dense zonotope OOMs). The assert was relaxed to `in (2, 3)`.

Pins: forward-LiRPA RUNS on a Gemm-free 2D-conv ReLU net (no AssertionError),
and its output box soundly contains every sampled output.
"""
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())


def _conv2d_net(path):
    rng = np.random.default_rng(0)
    Wc1 = (rng.standard_normal((3, 1, 3, 3)) * 0.4).astype(np.float32)
    bc1 = (rng.standard_normal(3) * 0.1).astype(np.float32)
    Wc2 = (rng.standard_normal((2, 3, 1, 1)) * 0.4).astype(np.float32)
    bc2 = (rng.standard_normal(2) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node('Conv', ['X', 'Wc1', 'bc1'], ['c1'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('Conv', ['r1', 'Wc2', 'bc2'], ['Y'], kernel_shape=[1, 1]),
    ]
    g = helper.make_graph(
        nodes, 'conv2d',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 4, 4])],
        [_init('Wc1', Wc1), _init('bc1', bc1), _init('Wc2', Wc2), _init('bc2', bc2)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return Wc1, bc1, Wc2, bc2


def test_forward_lirpa_2d_conv_sound(tmp_path):
    from vibecheck.network import ComputeGraph
    from vibecheck.forward_lirpa import batched_forward_linear_bounds
    p = str(tmp_path / 'conv2d.onnx')
    Wc1, bc1, Wc2, bc2 = _conv2d_net(p)
    g = ComputeGraph.from_onnx(p)
    gg = g.gpu_graph(device='cpu', dtype=torch.float32)
    lo = -0.5 * torch.ones(16, dtype=torch.float32)
    hi = 0.5 * torch.ones(16, dtype=torch.float32)
    # The fix: this used to raise AssertionError on the 2D conv's 3D in_shape_nd.
    state = batched_forward_linear_bounds(gg, lo.reshape(1, 16), hi.reshape(1, 16),
                                          'cpu', torch.float32)
    bb = state[gg['ops'][-1]['name']]
    blo = bb.lo_box.flatten().numpy()
    bhi = bb.hi_box.flatten().numpy()

    # Soundness: every sampled output must lie within the LiRPA box.
    def t(a):
        return torch.from_numpy(np.asarray(a))
    xs = torch.rand(2000, 1, 4, 4, dtype=torch.float32) - 0.5
    with torch.no_grad():
        y = torch.relu(F.conv2d(xs, t(Wc1), t(bc1), padding=1))
        y = F.conv2d(y, t(Wc2), t(bc2)).reshape(2000, -1).numpy()
    assert (y.min(0) >= blo - 1e-4).all(), 'forward-LiRPA lower bound unsound'
    assert (y.max(0) <= bhi + 1e-4).all(), 'forward-LiRPA upper bound unsound'
