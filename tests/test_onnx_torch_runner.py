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


def test_runner_min_max():
    a = torch.tensor([1.0, 5.0, -2.0])
    b = torch.tensor([3.0, 2.0, -1.0])
    assert _torch_op('Min', [a, b], {}).tolist() == [1.0, 2.0, -2.0]
    assert _torch_op('Max', [a, b], {}).tolist() == [3.0, 5.0, -1.0]
    # variadic + broadcast against a scalar (the clamp form)
    c = torch.tensor(0.0)
    assert _torch_op('Max', [a, c], {}).tolist() == [1.0, 5.0, 0.0]
    assert _torch_op('Min', [a, b, c], {}).tolist() == [0.0, 0.0, -2.0]


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


def test_pgd_no_false_sat_on_near_miss(tmp_path):
    """pgd_via_onnx must accept a witness only for a REAL violation (margin<=0),
    not a near-miss 1e-6 outside the unsafe region (the ml4acopf prop3 false-sat
    class). Identity net Y=X, unsafe if Y_0 >= 1.0."""
    import onnx
    from onnx import helper, TensorProto
    from vibecheck.onnx_torch_runner import pgd_via_onnx
    from vibecheck.spec import VNNSpec, Conjunct, Constraint

    W = helper.make_tensor('W', TensorProto.FLOAT, [1, 1], [1.0])
    b = helper.make_tensor('b', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'])
    graph = helper.make_graph(
        [node], 'identity',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    p = tmp_path / 'identity.onnx'
    onnx.save(model, str(p))
    dev = torch.device('cpu')

    # Near-miss box: max Y = 1 - 5e-7 -> worst margin (1.0 - Y) is in (0, 1e-6].
    # Old `<= 1e-6` falsely returned sat here; `<= 0` must return no-sat.
    near = VNNSpec(np.array([0.0], np.float32), np.array([np.float32(1.0 - 5e-7)]),
                   [Conjunct([Constraint(0, '>=', 1.0)])])
    sat, wit = pgd_via_onnx(str(p), near, n_restarts=16, n_iter=80,
                            device=dev, dtype=torch.float32, simplify=False)
    assert sat is False and wit is None, 'near-miss must NOT be a false-sat'

    # Real violation reachable (box up to 2.0): margin <= 0 -> sat.
    real = VNNSpec(np.array([0.0], np.float32), np.array([2.0], np.float32),
                   [Conjunct([Constraint(0, '>=', 1.0)])])
    sat2, wit2 = pgd_via_onnx(str(p), real, n_restarts=16, n_iter=80,
                              device=dev, dtype=torch.float32, simplify=False)
    assert sat2 is True and wit2 is not None, 'real violation must be sat'


def test_pgd_accept_margin_demands_clear_ce(tmp_path):
    """`accept_margin` controls how deep into the unsafe region PGD must reach.
    Identity net Y=X, unsafe Y_0 <= 0 (closure). The default (0.0) accepts any
    point in the closure (incl. the Y_0=0 boundary); a NEGATIVE accept_margin
    forces PGD to keep pushing until it finds a CLEAR violation (Y_0 <= margin).
    This is the network-pair diagonal-upgrade lever."""
    import onnx
    from onnx import helper, TensorProto
    from vibecheck.onnx_torch_runner import pgd_via_onnx
    from vibecheck.spec import VNNSpec, Conjunct, Constraint

    W = helper.make_tensor('W', TensorProto.FLOAT, [1, 1], [1.0])
    b = helper.make_tensor('b', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'])
    graph = helper.make_graph(
        [node], 'identity',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    p = tmp_path / 'identity.onnx'
    onnx.save(model, str(p))
    dev = torch.device('cpu')
    # Box [-1, 1] -> clear CEs (Y_0 down to -1) exist for unsafe Y_0 <= 0.
    spec = VNNSpec(np.array([-1.0], np.float32), np.array([1.0], np.float32),
                   [Conjunct([Constraint(0, '<=', 0.0)])])
    sat, wit = pgd_via_onnx(str(p), spec, n_restarts=16, n_iter=80,
                            device=dev, dtype=torch.float32, simplify=False)
    assert sat is True and wit is not None
    # Demand a CLEAR CE strictly inside the unsafe region.
    sat2, wit2 = pgd_via_onnx(str(p), spec, n_restarts=16, n_iter=80,
                              accept_margin=-0.5, device=dev,
                              dtype=torch.float32, simplify=False)
    assert sat2 is True and float(wit2.flatten()[0]) <= -0.5 + 1e-5


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
