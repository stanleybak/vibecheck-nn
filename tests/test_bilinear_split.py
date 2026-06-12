"""Bilinear-input splits in the zono BnB (policy: implemented = tested).

ABC's splittable-Mul analog: when a BnB leaf has no unstable ReLU left
but its residual gap is bilinear/softmax slack, split a pre-softmax
score coordinate's interval at its midpoint. The clamp flows into the
forward's exp parallelogram (op_clamps intersection) and the recorded
op_bounds, so the backward planes tighten identically.

Toy: a RELU-FREE softmax net y = softmax(x)_1 over a box (the spec
reads a NON-shift coordinate: under the softmax max-shift the k-th
coordinate is exactly linear and slack-free). The exp/recip/
mul_bilinear relaxation slack leaves the root open for a threshold the
true minimum clears, so ONLY exp-input splits can close it. Pins:
  1. op_clamps forward soundness: a midpoint-clamped forward still
     bounds every sampled output whose TRUE score is in the clamped
     half (the value-split subdomain).
  2. root open + _zono_relu_split_close closes via exp splits
     (nodes > 1, reason 'closed').
  3. closed verdict is consistent with sampling (true min clears the
     threshold).
"""
import os

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                              arr.flatten())


def _softmax_net(tmpdir, n=2):
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    W = np.eye(n, dtype=np.float32)
    Wo = np.zeros((n, 1), dtype=np.float32)
    # spec reads coordinate min(1, n-1): under the softmax max-shift the
    # k-th coordinate's exp is the constant 1 (no bilinear slack), so
    # specs on coordinate 0 are exactly bounded by the plain forward —
    # coordinate 1 keeps the both-vary product the split pins need
    Wo[min(1, n - 1), 0] = 1.0
    nodes = [
        helper.make_node('MatMul', ['X', 'W'], ['S']),
        helper.make_node('Softmax', ['S'], ['A'], axis=-1),
        helper.make_node('MatMul', ['A', 'Wo'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'toy_softmax',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [_init('W', W), _init('Wo', Wo)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = os.path.join(tmpdir, 'softmax.onnx')
    onnx.save(model, path)
    graph = load_onnx(path, dtype=np.float64, simplify=False)
    graph.optimize(default_settings())
    return graph


def _box():
    xl = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    xh = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return xl, xh


def test_op_clamps_forward_sound(tmp_path):
    """Clamping the LIVE exp coordinate (1; coordinate 0 is the
    max-shift pivot, identically 0) to the upper half [m, u] of its
    input range must still soundly bound every input whose TRUE shifted
    score z_1 = x_1 - x_0 is >= m (the value-split subdomain)."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    graph = _softmax_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    exp_ops = [op for op in gg['ops'] if op['type'] == 'exp']
    assert exp_ops, 'softmax decomposition did not emit an exp op'
    nm = exp_ops[0]['name']
    xl, xh = _box()
    ob = {}
    _, zf0 = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                     torch.float64, op_bounds=ob)
    lo0, hi0 = ob[nm]
    m = float(0.5 * (lo0[1] + hi0[1]))
    cl = torch.full((lo0.numel(),), -np.inf, dtype=torch.float64)
    ch = torch.full((lo0.numel(),), np.inf, dtype=torch.float64)
    cl[1] = m
    ob_c = {}
    _, zf = _forward_zonotope_graph(
        xl, xh, gg, torch.device('cpu'), torch.float64,
        op_bounds=ob_c, op_clamps={nm: (cl, ch)})
    assert float(ob_c[nm][0][1]) == pytest.approx(m)
    lo_y, hi_y = zf.bounds()
    # subdomain samples: true z_1 = x_1 - x_0 >= m
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, (4000, 2))
    xs = xs[(xs[:, 1] - xs[:, 0]) >= m]
    assert len(xs) > 100
    ys = _forward_batch_graph(torch.tensor(xs, dtype=torch.float64), gg)
    assert float(ys.min()) >= float(lo_y[0]) - 1e-9
    assert float(ys.max()) <= float(hi_y[0]) + 1e-9
    # and the clamped zono is strictly tighter than the unclamped one
    lo_y0, hi_y0 = zf0.bounds()
    assert float(hi_y[0] - lo_y[0]) < float(hi_y0[0] - lo_y0[0]) - 1e-6


def test_beta_uses_subdomain_constraint(tmp_path):
    """The ('beta', name) Lagrangian must (a) IMPROVE the bound on a
    value-split subdomain for hand-set beta > 0 (concave lb(beta), so a
    grid finds it — optimizer-free to keep the pin deterministic), and
    (b) stay SOUND vs subdomain sampling for EVERY beta >= 0 (the sign
    and constant conventions are the safety-critical part). Spec reads
    sigma_1; subdomain: live shifted score z_1 = x_1 - x_0 >= 0 over
    x in [-2,2]^2 (subdomain min 0.5 vs unrestricted ~0.018)."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_crown_lb
    graph = _softmax_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    xl = torch.full((2,), -2.0, dtype=torch.float64)
    xh = torch.full((2,), 2.0, dtype=torch.float64)
    nm = [op['name'] for op in gg['ops'] if op['type'] == 'exp'][0]
    inf = float('inf')
    oc = {nm: (torch.tensor([-inf, 0.0], dtype=torch.float64),
               torch.tensor([inf, inf], dtype=torch.float64))}
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob,
                                    op_clamps=oc)
    w = torch.tensor([1.0], dtype=torch.float64)
    rng = np.random.default_rng(7)
    xs = rng.uniform(-2.0, 2.0, (4000, 2))
    xs = xs[(xs[:, 1] - xs[:, 0]) >= 0.0]
    ys = _forward_batch_graph(torch.tensor(xs, dtype=torch.float64), gg)
    sub_min = float(ys.min())
    lbs = {}
    for b in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0):
        params = {('beta', nm): torch.tensor(
            [[0.0, b], [0.0, 0.0]], dtype=torch.float64)}
        with torch.no_grad():
            lb = float(attn_crown_lb(gg, xl, xh, sb, ob, w, 0.0,
                                     params, op_clamps=oc))
        assert lb <= sub_min + 1e-9, f'UNSOUND at beta={b}: {lb} > {sub_min}'
        lbs[b] = lb
    assert max(lbs.values()) > lbs[0.0] + 0.1, (
        f'beta never improved the subdomain bound: {lbs}')


def _relu_fork_net(tmpdir):
    """y = relu(x0+x1) - 0.6*(x0+x1) + 0.1 over x in [-1,1]^2."""
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    W1 = np.array([[1.0], [1.0]], dtype=np.float32)
    Wl = np.array([[-0.6], [-0.6]], dtype=np.float32)
    bl = np.array([0.1], dtype=np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['s']),
        helper.make_node('Relu', ['s'], ['r']),
        helper.make_node('MatMul', ['X', 'Wl'], ['m']),
        helper.make_node('Add', ['m', 'bl'], ['a']),
        helper.make_node('Add', ['r', 'a'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'relu_fork',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [_init('W1', W1), _init('Wl', Wl), _init('bl', bl)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = os.path.join(tmpdir, 'relu_fork.onnx')
    onnx.save(model, path)
    graph = load_onnx(path, dtype=np.float64, simplify=False)
    graph.optimize(default_settings())
    return graph


def test_rbeta_uses_relu_split_constraint(tmp_path):
    """Clamp-only relu splits keep spurious traces (inputs with true
    s > 0 routed through the clamped relaxation): on the subdomain
    s >= 0 the net is y = 0.4 s + 0.1 with true min 0.1, but the
    clamp-only backward concretizes over the FULL box (s in [-2, 2])
    to -0.7. The ('rbeta', L) Lagrangian must close it; clamp-only
    must not (the beta-CROWN mechanism, pinned)."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_crown_alpha
    graph = _relu_fork_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    L = [op['layer_idx'] for op in gg['ops'] if op['type'] == 'relu'][0]
    xl = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    xh = torch.tensor([1.0, 1.0], dtype=torch.float64)
    tb = {L: (np.array([0.0]), np.array([2.0]))}    # split side: s >= 0
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, tight_bounds=tb,
                                    op_bounds=ob)
    w = torch.tensor([1.0], dtype=torch.float64)
    lb_clamp, _ = attn_crown_alpha(gg, xl, xh, sb, ob, w, 0.0,
                                   n_iters=100, lr=0.2)
    assert lb_clamp < 0, ('clamp-only closed it — pin is vacuous '
                          f'({lb_clamp})')
    inf = float('inf')
    rc = {L: (torch.tensor([0.0], dtype=torch.float64),
              torch.tensor([inf], dtype=torch.float64))}
    lb_beta, params = attn_crown_alpha(gg, xl, xh, sb, ob, w, 0.0,
                                       n_iters=200, lr=0.2,
                                       relu_clamps=rc)
    assert lb_beta > 0, f'rbeta failed to close ({lb_beta})'
    assert float(params[('rbeta', L)].abs().max()) > 0
    # soundness vs subdomain samples (true s = x0+x1 >= 0)
    rng = np.random.default_rng(9)
    xs = rng.uniform(-1.0, 1.0, (4000, 2))
    xs = xs[xs.sum(axis=1) >= 0]
    ys = _forward_batch_graph(torch.tensor(xs, dtype=torch.float64), gg)
    assert lb_beta <= float(ys.min()) + 1e-9, 'UNSOUND rbeta bound'


def test_exp_split_closes_relufree_query(tmp_path):
    """Root open (relaxation slack) but true min clears the threshold:
    only exp-input splits exist (no relus), and they must close it."""
    from vibecheck.verify_graph import _zono_relu_split_close
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _spec_backward_graph,
                                           _forward_batch_graph)
    # 3 coords: TWO live shifted scores interact through the recip, so
    # the relaxation has genuine slack (2 coords are exactly bounded
    # under the max-shift — measured)
    graph = _softmax_net(str(tmp_path), n=3)
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    xl = torch.full((3,), -1.0, dtype=torch.float64)
    xh = torch.full((3,), 1.0, dtype=torch.float64)
    w = np.array([1.0])
    # spec reads sigma_1; true min at x1=-1, x0=x2=+1
    true_min = np.exp(-2.0) / (2.0 + np.exp(-2.0))
    b_q = -0.5 * true_min       # TRUE, but inside root relaxation slack
    ob = {}
    sb, zf = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                     torch.float64, op_bounds=ob)
    w_t = torch.tensor(w, dtype=torch.float64)
    lb_fwd = float(w_t @ zf.center + b_q
                   - (w_t @ zf.generators).abs().sum())
    bw, _ = _spec_backward_graph(
        sb, xl, xh, gg, {0: (w_t, b_q)}, {0}, len(sb),
        torch.device('cpu'), torch.float64, op_bounds=ob)
    lb_root = max(lb_fwd, float(bw[0]))
    assert lb_root < 0, f'toy mis-designed: root already closed {lb_root}'
    ok, nodes, reason = _zono_relu_split_close(
        gg, xl, xh, w, b_q, torch.device('cpu'), torch.float64,
        None, lambda: 100.0, max_nodes=256)
    assert ok, f'exp splits failed to close: {reason} after {nodes}'
    assert nodes > 1, 'root closed without splitting — pin is vacuous'
    # consistency: the verified claim holds on samples
    rng = np.random.default_rng(1)
    xs = rng.uniform(-1.0, 1.0, (500, 3))
    ys = _forward_batch_graph(torch.tensor(xs, dtype=torch.float64), gg)
    assert float(ys.min()) + b_q > 0
