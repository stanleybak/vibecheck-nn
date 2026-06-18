"""maxpool_to_relu: EXACT MaxPool -> ReLU decomposition.

max(a,b) = a + ReLU(b-a). The pass extracts the kH*kW window phases with a
1-hot phase-extraction Conv + contiguous-channel Slices, then a binary-max tree
of Sub/ReLU/Add. It must reproduce F.max_pool2d EXACTLY (it is not a relaxation)
and leave no MaxPool node. Validated through the production gpu_graph point
forward, which is the path conv nets actually run."""
import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import maxpool_to_relu


def _maxpool_onnx(path, C, H, W, k, s):
    node = helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=[k, k],
                            strides=[s, s], pads=[0, 0, 0, 0])
    g = helper.make_graph(
        [node], 'mp',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info(
            'y', TensorProto.FLOAT, [1, C, (H - k) // s + 1, (W - k) // s + 1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.save(m, path)


@pytest.mark.parametrize('C,H,W,k,s', [
    (3, 4, 4, 2, 2),    # the vggnet16 case (2x2 / stride 2)
    (5, 8, 8, 2, 2),
    (2, 6, 6, 3, 3),    # odd window count P=9 -> exercises the carry path
])
def test_maxpool_to_relu_exact(tmp_path, C, H, W, k, s):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    p = str(tmp_path / 'mp.onnx')
    _maxpool_onnx(p, C, H, W, k, s)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert any(n.op_type == 'MaxPool' for n in g.nodes.values())

    assert maxpool_to_relu(g) is True
    assert not any(n.op_type == 'MaxPool' for n in g.nodes.values()), \
        'MaxPool must be fully decomposed'
    # second run is a no-op (no MaxPool left)
    assert maxpool_to_relu(g) is False

    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    for seed in range(4):
        torch.manual_seed(seed)
        x = torch.randn(1, C, H, W, dtype=torch.float64)
        ref = F.max_pool2d(x, k, s).reshape(-1)
        out = _forward_batch_graph(x.reshape(1, -1), gg).reshape(-1).double()
        torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)


def test_maxpool_to_relu_padded_raises(tmp_path):
    """Padded MaxPool pads with -inf — the ReLU decomposition can't represent
    it, so the pass must raise (loud), never silently produce a wrong bound."""
    p = str(tmp_path / 'mp_pad.onnx')
    node = helper.make_node('MaxPool', ['x'], ['y'], kernel_shape=[2, 2],
                            strides=[2, 2], pads=[1, 1, 1, 1])
    g = helper.make_graph(
        [node], 'mp',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 3, 3])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.save(m, p)
    graph = ComputeGraph.from_onnx(p, dtype=np.float64)
    with pytest.raises(NotImplementedError, match='padded MaxPool'):
        maxpool_to_relu(graph)
