"""drop_identity_pads: all-zero Pad nodes are exact identities (TinyYOLO carries them in
front of its AveragePools) and must be spliced out at optimize() time; non-zero pads must
be KEPT so the gpu_graph serializer still raises loudly instead of silently mis-binding."""
import numpy as np
import onnx
from onnx import helper, TensorProto

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import drop_identity_pads
from vibecheck.settings import default_settings
from tests.test_cgan_op_bounds import _ensure_from_onnx_model

_ensure_from_onnx_model()


def _pad_model(pads, pad_is_output=False):
    """input -> Pad -> (Relu ->) output, opset-9-style Pad attributes."""
    inp = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 4, 4])
    pad = helper.make_node('Pad', ['X'], ['P'], name='pad0',
                           mode='constant', pads=list(pads), value=0.0)
    h = 4 + pads[2] + pads[6] if len(pads) == 8 else 4
    w = 4 + pads[3] + pads[7] if len(pads) == 8 else 4
    if pad_is_output:
        out = helper.make_tensor_value_info('P', TensorProto.FLOAT, [1, 2, h, w])
        g = helper.make_graph([pad], 'test', [inp], [out])
    else:
        relu = helper.make_node('Relu', ['P'], ['Y'], name='relu0')
        out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, h, w])
        g = helper.make_graph([pad, relu], 'test', [inp], [out])
    return helper.make_model(g, opset_imports=[helper.make_opsetid('', 9)])


def test_zero_pad_spliced_out():
    g = ComputeGraph.from_onnx_model(_pad_model([0] * 8))
    assert any(n.op_type == 'Pad' for n in g.nodes.values())
    g.optimize(default_settings())
    assert not any(n.op_type == 'Pad' for n in g.nodes.values())
    relu = next(n for n in g.nodes.values() if n.op_type == 'Relu')
    pad_inputs = [i for i in relu.inputs if 'pad' in i.lower()]
    assert pad_inputs == [], f'Relu still consumes the pad: {relu.inputs}'


def test_nonzero_pad_kept():
    g = ComputeGraph.from_onnx_model(_pad_model([0, 0, 1, 1, 0, 0, 1, 1]))
    drop_identity_pads(g)
    assert any(n.op_type == 'Pad' for n in g.nodes.values()), \
        'non-zero pad must be kept (and keep raising loudly downstream)'


def test_zero_pad_as_graph_output():
    g = ComputeGraph.from_onnx_model(_pad_model([0] * 8, pad_is_output=True))
    pad_name = next(n.name for n in g.nodes.values() if n.op_type == 'Pad')
    assert g.output_name == pad_name
    src = g.nodes[pad_name].inputs[0]
    drop_identity_pads(g)
    assert g.output_name == src
    assert pad_name not in g.nodes
