"""Unit tests for onnx_torch_runner op coverage.

Focused on the ops added for ml4acopf_2024 (Sin/Cos/Pow/Floor/Equal/Where/
Expand/ConstantOfShape/Slice). Each is exercised through `_torch_op` and
checked against the direct torch computation, so PGD / point-prop witness
validation can run on the AC-OPF models.
"""
import numpy as np
import torch
from onnx import numpy_helper

from vibecheck.onnx_torch_runner import _torch_op


def test_runner_sin_cos_pow_floor():
    x = torch.tensor([0.0, 0.5, 1.5, -2.0])
    assert torch.allclose(_torch_op('Sin', [x], {}), torch.sin(x))
    assert torch.allclose(_torch_op('Cos', [x], {}), torch.cos(x))
    assert torch.allclose(_torch_op('Pow', [x.abs(), torch.tensor(2.0)], {}),
                          x.abs() ** 2)
    assert torch.allclose(_torch_op('Floor', [x], {}), torch.floor(x))


def test_runner_equal_where():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 9.0, 3.0])
    eq = _torch_op('Equal', [a, b], {})
    assert eq.dtype == torch.bool
    assert eq.tolist() == [True, False, True]
    w = _torch_op('Where', [eq, a, b], {})
    assert w.tolist() == [1.0, 9.0, 3.0]


def test_runner_expand():
    x = torch.tensor([[1.0], [2.0]])          # (2, 1)
    shape = torch.tensor([2, 3])
    out = _torch_op('Expand', [x, shape], {})
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1.0, 1.0, 1.0] and out[1].tolist() == [2.0, 2.0, 2.0]


def test_runner_constant_of_shape():
    shape = torch.tensor([2, 3])
    # default fill 0.0
    z = _torch_op('ConstantOfShape', [shape], {})
    assert z.shape == (2, 3) and float(z.sum()) == 0.0
    # explicit fill value via an onnx TensorProto attr
    val = numpy_helper.from_array(np.array([7.0], dtype=np.float32), name='value')
    f = _torch_op('ConstantOfShape', [shape], {'value': val})
    assert f.shape == (2, 3) and float(f.flatten()[0]) == 7.0


def test_runner_slice_tensor_inputs():
    data = torch.arange(20).reshape(4, 5).float()
    # data[1:3, 0:5:2] on axes [0,1]
    out = _torch_op('Slice', [data, torch.tensor([1, 0]), torch.tensor([3, 5]),
                              torch.tensor([0, 1]), torch.tensor([1, 2])], {})
    assert torch.equal(out, data[1:3, 0:5:2])


def test_runner_slice_negative_axis_and_attr_form():
    data = torch.arange(12).reshape(3, 4).float()
    # negative axis (-1), default steps
    out = _torch_op('Slice', [data, torch.tensor([1]), torch.tensor([3]),
                              torch.tensor([-1])], {})
    assert torch.equal(out, data[:, 1:3])
    # opset<10 attribute form (no tensor inputs)
    out2 = _torch_op('Slice', [data],
                     {'starts': [0], 'ends': [2], 'axes': [0]})
    assert torch.equal(out2, data[0:2, :])
