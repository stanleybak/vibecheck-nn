"""Tier-0 soundness tests for the backward propagator (design 4.2):
forward/backward containment, alpha/beta monotonicity, clamp soundness,
McCormick bracketing. Sampling validates, never defines."""
import numpy as np
import pytest
import torch

from vibecheck2.core import backward, forward, graph as g2

_rng = np.random.default_rng(7)
torch.manual_seed(7)


def _fc_relu_net(tmp_path, sizes=(6, 8, 8, 4), residual=False):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    nodes, inits = [], []
    prev = 'X'
    for i in range(len(sizes) - 1):
        W = numpy_helper.from_array(
            (_rng.normal(size=(sizes[i], sizes[i + 1])) / np.sqrt(sizes[i]))
            .astype(np.float32), f'W{i}')
        inits.append(W)
        nodes.append(helper.make_node('MatMul', [prev, f'W{i}'], [f'h{i}']))
        if i < len(sizes) - 2:
            nodes.append(helper.make_node('Relu', [f'h{i}'], [f'r{i}']))
            prev = f'r{i}'
        else:
            prev = f'h{i}'
    out = prev
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, sizes[0]])
    Y = helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, sizes[-1]])
    g = helper.make_graph(nodes, 'g', [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'fc.onnx')
    onnx.save(m, p)
    return g2.load(p)


def _mul_net(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W = numpy_helper.from_array(
        _rng.normal(size=(4, 4)).astype(np.float32), 'W')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W'], ['h']),
         helper.make_node('Mul', ['h', 'X'], ['Y'])],       # bilinear h*x
        'g', [X], [Y], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'mul.onnx')
    onnx.save(m, p)
    return g2.load(p)


def _box(n, w=1.0):
    c = torch.tensor(_rng.normal(size=(1, n)).astype(np.float32))
    return c - w / 2, c + w / 2


def test_crown_bounds_contain_samples(tmp_path):
    net = _fc_relu_net(tmp_path)
    lo, hi = _box(net.n_in)
    W = torch.eye(net.n_out)
    lb = backward.crown(net, lo, hi, W)
    xs = torch.rand(512, net.n_in) * (hi - lo) + lo
    ys = forward.point(net, xs)
    assert (ys >= lb - 1e-4).all()


def test_alpha_only_tightens(tmp_path):
    net = _fc_relu_net(tmp_path)
    lo, hi = _box(net.n_in, w=2.0)
    W = torch.eye(net.n_out)
    inter = backward.intermediates(net, lo, hi)
    lb0 = backward.crown(net, lo, hi, W, inter)
    lba = backward.alpha_crown(net, lo, hi, W, inter, iters=15)
    assert (lba >= lb0 - 1e-5).all()
    xs = torch.rand(512, net.n_in) * (hi - lo) + lo
    ys = forward.point(net, xs)
    assert (ys >= lba - 1e-4).all()


def test_clamps_sound_and_beta_monotone(tmp_path):
    """Clamped domains bound their own samples; beta never loosens; zero
    beta reproduces the plain clamped bound exactly."""
    net = _fc_relu_net(tmp_path)
    lo, hi = _box(net.n_in, w=2.0)
    W = torch.eye(net.n_out)
    inter = backward.intermediates(net, lo, hi)
    relu0 = next(nm for nm in net.order
                 if net.ops[nm].kind == 'nonlin')
    n0 = net.ops[relu0].n
    l0, h0 = inter[relu0]
    j = int(((l0 < 0) & (h0 > 0)).float()[0].argmax())      # an unstable one
    for sgn in (1, -1):
        clamp = torch.zeros(1, n0, dtype=torch.int8)
        clamp[0, j] = sgn
        clamps = {relu0: clamp}
        lb_c = backward.crown(net, lo, hi, W, inter, clamps=clamps)
        beta0 = {relu0: torch.zeros(1, net.n_out, n0)}
        lb_b0 = backward.crown(net, lo, hi, W, inter, clamps=clamps,
                               beta=beta0)
        assert torch.allclose(lb_c, lb_b0, atol=1e-5)       # beta=0 == plain
        lb_ab = backward.alpha_beta_crown(net, lo, hi, W, inter, clamps,
                                          iters=10)
        assert (lb_ab >= lb_c - 1e-4).all()                 # never looser
        # samples living in the clamped half-domain stay above the bound
        xs = torch.rand(2048, net.n_in) * (hi - lo) + lo
        pre = None
        state = {net.input_name: xs}
        for nm in net.order:                                 # find pre-act
            op = net.ops[nm]
            if op.kind == 'linmap':
                state[nm] = op.lm.point(state[op.inputs[0]])
            elif op.kind == 'nonlin':
                if nm == relu0:
                    pre = state[op.inputs[0]]
                state[nm] = torch.relu(state[op.inputs[0]])
        keep = (pre[:, j] >= 0) if sgn > 0 else (pre[:, j] <= 0)
        ys = forward.point(net, xs[keep])
        assert (ys >= lb_ab - 1e-4).all()
        # child bounds cover the parent's: min over both signs <= parent samples
    # both children together must contain everything the parent did


def test_mccormick_mul_sound(tmp_path):
    net = _mul_net(tmp_path)
    lo, hi = _box(net.n_in, w=1.5)
    W = torch.eye(net.n_out)
    lb = backward.crown(net, lo, hi, W)
    xs = torch.rand(2048, net.n_in) * (hi - lo) + lo
    ys = forward.point(net, xs)
    assert (ys >= lb - 1e-4).all()
    ilo, ihi = forward.interval(net, lo, hi)
    zlo, zhi = forward.zono(net, lo, hi)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all()
    assert (ys >= ilo - 1e-4).all() and (ys <= ihi + 1e-4).all()


def test_sigmoid_tanh_planes_bracket():
    from vibecheck2.core.relax import REL
    for fn in ('sigmoid', 'tanh'):
        rel = REL[fn]
        lo = torch.tensor([[-5.0, -0.5, 0.0, 1.0, -8.0]])
        hi = torch.tensor([[5.0, 0.5, 0.0, 3.0, -7.5]])
        al, bl, au, bu = rel.planes(lo, hi)
        for t in torch.linspace(0, 1, 201):
            x = lo + t * (hi - lo)
            y = rel.point(x)
            assert (al * x + bl <= y + 1e-5).all(), fn
            assert (y <= au * x + bu + 1e-5).all(), fn


def test_reciprocal_planes_bracket():
    from vibecheck2.core.relax import REL
    rel = REL['reciprocal']
    lo = torch.tensor([[0.5, 2.0, -3.0, -0.7]])
    hi = torch.tensor([[3.0, 2.0001, -0.5, -0.6]])
    al, bl, au, bu = rel.planes(lo, hi)
    for t in torch.linspace(0, 1, 301):
        x = lo + t * (hi - lo)
        y = 1.0 / x
        assert (al * x + bl <= y + 1e-4).all()
        assert (y <= au * x + bu + 1e-4).all()
    lam, mu, delta = rel.band(lo, hi)
    for t in torch.linspace(0, 1, 301):
        x = lo + t * (hi - lo)
        y = 1.0 / x
        assert ((y - (lam * x + mu)).abs() <= delta + 1e-4).all()
    import pytest as _pytest
    with _pytest.raises(NotImplementedError):
        rel.planes(torch.tensor([[-1.0]]), torch.tensor([[1.0]]))


def test_leaky_relu_planes_bracket():
    from vibecheck2.core.relax import REL
    rel = REL['leaky_relu']
    p = {'alpha': 0.1}
    lo = torch.tensor([[-2.0, -1.0, 0.5, -3.0]])
    hi = torch.tensor([[1.0, 2.0, 2.0, -0.5]])
    al, bl, au, bu = rel.planes(lo, hi, p)
    lam, mu, delta = rel.band(lo, hi, p)
    for t in torch.linspace(0, 1, 201):
        x = lo + t * (hi - lo)
        y = rel.point(x, p)
        assert (al * x + bl <= y + 1e-5).all()
        assert (y <= au * x + bu + 1e-5).all()
        assert ((y - (lam * x + mu)).abs() <= delta + 1e-5).all()


def test_bmm_interval_sound(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 24])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16])
    ini = [onnx.numpy_helper.from_array(np.array(v, np.int64), k)
           for k, v in [('s0', [0]), ('e0', [8]), ('s1', [8]), ('e1', [24]),
                        ('ax', [1]), ('sh_a', [4, 2]), ('sh_b', [2, 8])]]
    g = helper.make_graph(
        [helper.make_node('Slice', ['X', 's0', 'e0', 'ax'], ['a']),
         helper.make_node('Slice', ['X', 's1', 'e1', 'ax'], ['b']),
         helper.make_node('Reshape', ['a', 'sh_a'], ['a2']),
         helper.make_node('Reshape', ['b', 'sh_b'], ['b2']),
         helper.make_node('MatMul', ['a2', 'b2'], ['Y'])],
        'g', [X], [Y], ini)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'bmm.onnx')
    onnx.save(m, p)
    net = g2.load(p)
    lo, hi = _box(net.n_in, w=1.0)
    ilo, ihi = forward.interval(net, lo, hi)
    xs = torch.rand(1024, net.n_in) * (hi - lo) + lo
    ys = forward.point(net, xs)
    assert (ys >= ilo - 1e-4).all() and (ys <= ihi + 1e-4).all()
