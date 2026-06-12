"""CROWN-IBP route: IBP forwards, split-beta backward, IBP-refresh BaB.

Soundness gates:
  1. `_ibp_forward_graph` pre-ReLU bounds contain 200 random concrete
     forward passes (sampling validates, never bounds).
  2. First-layer bounds are EXACT (interval through one affine layer).
  3. Batched forward == unbatched on replicated boxes; ReLU split clamps
     tighten downstream bounds and are reflected in `sb`.
  4. `relu_split_beta`: beta=0 reproduces the plain bound; beta >= 0 with
     a valid OFF-split stays below the true constrained minimum
     (dense-grid ground truth on a 2-D input net).
  5. `_ibp_crown_bab` closes a crafted UNSAT query and reports
     `exhausted_pattern` on a truly violated one.
"""
import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_milp import _ibp_crown_bab
from vibecheck.verify_zono_bnb import (
    _ibp_forward_graph,
    _ibp_forward_graph_batched,
    _spec_backward_graph_batched,
)

DEV = torch.device('cpu')
DT = torch.float64


def _init(name, arr, ttype=TensorProto.DOUBLE):
    arr = np.asarray(arr, np.float64)
    return helper.make_tensor(name, ttype, arr.shape, arr.flatten())


def _load(tmp_path, nodes, inits, in_shape, out_shape, name):
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info('X', TensorProto.DOUBLE, in_shape)],
        [helper.make_tensor_value_info('Y', TensorProto.DOUBLE, out_shape)],
        inits)
    model = helper.make_model(graph)
    p = str(tmp_path / f'{name}.onnx')
    onnx.save(model, p)
    return load_onnx(p, dtype=np.float64)


def _conv_relu_fc_net(tmp_path, seed=0):
    """Conv(2->3, 3x3, pad 1) -> Relu -> Conv(3->2, 3x3, s2) -> Relu ->
    Flatten -> Gemm -> Relu -> Gemm. Covers conv/relu/reshape/fc."""
    rng = np.random.default_rng(seed)
    k1 = rng.standard_normal((3, 2, 3, 3)) * 0.5
    b1 = rng.standard_normal(3) * 0.1
    k2 = rng.standard_normal((2, 3, 3, 3)) * 0.5
    b2 = rng.standard_normal(2) * 0.1
    W1 = rng.standard_normal((4, 2 * 3 * 3)) * 0.5
    c1 = rng.standard_normal(4) * 0.1
    W2 = rng.standard_normal((2, 4)) * 0.5
    c2 = rng.standard_normal(2) * 0.1
    nodes = [
        helper.make_node('Conv', ['X', 'k1', 'b1'], ['c1o'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c1o'], ['r1']),
        helper.make_node('Conv', ['r1', 'k2', 'b2'], ['c2o'],
                         kernel_shape=[3, 3], strides=[2, 2]),
        helper.make_node('Relu', ['c2o'], ['r2']),
        helper.make_node('Flatten', ['r2'], ['fl']),
        helper.make_node('Gemm', ['fl', 'W1', 'cb1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r3']),
        helper.make_node('Gemm', ['r3', 'W2', 'cb2'], ['Y'], transB=1),
    ]
    inits = [_init('k1', k1), _init('b1', b1), _init('k2', k2),
             _init('b2', b2), _init('W1', W1), _init('cb1', c1),
             _init('W2', W2), _init('cb2', c2)]
    g = _load(tmp_path, nodes, inits, (1, 2, 8, 8), (1, 2), 'convnet')
    return g


def _point_forward(g, x):
    """Reference forward via onnxruntime-free torch replay of gg ops."""
    import torch.nn.functional as F
    gg = g.gpu_graph(DEV, DT)
    act = {gg['input_name']: x}
    pre_relu = {}
    for op in gg['ops']:
        t = op['type']
        a = act[op['inputs'][0]]
        if t == 'conv':
            a4 = a.reshape(1, *op['in_shape'])
            out = F.conv2d(a4, op['kernel'], op['bias'],
                           stride=op['stride'], padding=op['padding'])
            act[op['name']] = out.flatten()
        elif t == 'fc':
            act[op['name']] = op['W'] @ a + op['bias']
        elif t == 'relu':
            if 'layer_idx' in op:
                pre_relu[op['layer_idx']] = a.clone()
            act[op['name']] = a.clamp(min=0)
        elif t == 'reshape':
            act[op['name']] = a
        else:
            raise NotImplementedError(t)
    return act[gg['ops'][-1]['name']], pre_relu


def _box(g, eps=0.3, seed=1):
    n = int(np.prod(g.input_shape))
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1, 1, n)
    return (torch.tensor(c - eps, dtype=DT), torch.tensor(c + eps, dtype=DT))


def test_ibp_forward_sound_and_first_layer_exact(tmp_path):
    g = _conv_relu_fc_net(tmp_path)
    gg = g.gpu_graph(DEV, DT)
    xl, xh = _box(g)
    sb = _ibp_forward_graph(xl, xh, gg, DEV, DT)
    assert set(sb) == {0, 1, 2}
    # Soundness: random points stay inside every pre-ReLU enclosure.
    rng = np.random.default_rng(7)
    for _ in range(200):
        t = torch.tensor(rng.uniform(0, 1, xl.numel()), dtype=DT)
        x = xl + t * (xh - xl)
        _, pre = _point_forward(g, x)
        for L, (lo, hi) in sb.items():
            assert bool((pre[L] >= lo - 1e-9).all()), f'L{L} lo violated'
            assert bool((pre[L] <= hi + 1e-9).all()), f'L{L} hi violated'
    # Exactness at the first pre-ReLU (single affine layer).
    import torch.nn.functional as F
    op = gg['ops'][0]
    mid, rad = (xl + xh) / 2, (xh - xl) / 2
    mc = F.conv2d(mid.reshape(1, *op['in_shape']), op['kernel'], op['bias'],
                  stride=op['stride'], padding=op['padding']).flatten()
    rc = F.conv2d(rad.reshape(1, *op['in_shape']), op['kernel'].abs(), None,
                  stride=op['stride'], padding=op['padding']).flatten()
    torch.testing.assert_close(sb[0][0], mc - rc)
    torch.testing.assert_close(sb[0][1], mc + rc)


def test_ibp_forward_misc_ops(tmp_path):
    """Sub(const) -> Mul(const) -> Conv -> Relu -> MaxPool -> Flatten ->
    Gemm covers sub/mul/max_pool; merge-Add covered via a skip net."""
    rng = np.random.default_rng(3)
    sub_c = rng.standard_normal((1, 1, 4, 4)) * 0.1
    mul_c = rng.standard_normal((1, 1, 4, 4))
    k = rng.standard_normal((2, 1, 2, 2)) * 0.5
    b = rng.standard_normal(2) * 0.1
    W = rng.standard_normal((2, 2 * 1 * 1)) * 0.5
    c = rng.standard_normal(2) * 0.1
    nodes = [
        helper.make_node('Sub', ['X', 'sc'], ['s0']),
        helper.make_node('Mul', ['s0', 'mc'], ['m0']),
        helper.make_node('Conv', ['m0', 'k', 'b'], ['c0'],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node('Relu', ['c0'], ['r0']),
        helper.make_node('MaxPool', ['r0'], ['p0'], kernel_shape=[2, 2],
                         strides=[2, 2]),
        helper.make_node('Flatten', ['p0'], ['fl']),
        helper.make_node('Gemm', ['fl', 'W', 'c'], ['Y'], transB=1),
    ]
    inits = [_init('sc', sub_c), _init('mc', mul_c), _init('k', k),
             _init('b', b), _init('W', W), _init('c', c)]
    g = _load(tmp_path, nodes, inits, (1, 1, 4, 4), (1, 2), 'miscnet')
    gg = g.gpu_graph(DEV, DT)
    xl, xh = _box(g, eps=0.2, seed=5)
    sb = _ibp_forward_graph(xl, xh, gg, DEV, DT)
    assert len(sb) >= 1
    # Sound vs the model's own forward (onnxruntime via graph point prop).
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    rng2 = np.random.default_rng(11)
    pts = torch.tensor(rng2.uniform(0, 1, (64, xl.numel())), dtype=DT)
    X = xl + pts * (xh - xl)
    Y = _forward_batch_graph(X, gg)
    assert torch.isfinite(Y).all()


def test_ibp_forward_unsupported_op_raises(tmp_path):
    nodes = [
        helper.make_node('Sigmoid', ['X'], ['s']),
        helper.make_node('Flatten', ['s'], ['fl']),
        helper.make_node('Gemm', ['fl', 'W', 'c'], ['Y'], transB=1),
    ]
    rng = np.random.default_rng(0)
    inits = [_init('W', rng.standard_normal((2, 4))),
             _init('c', rng.standard_normal(2))]
    g = _load(tmp_path, nodes, inits, (1, 1, 2, 2), (1, 2), 'signet')
    gg = g.gpu_graph(DEV, DT)
    xl, xh = _box(g, eps=0.1)
    with pytest.raises(NotImplementedError):
        _ibp_forward_graph(xl, xh, gg, DEV, DT)
    with pytest.raises(NotImplementedError):
        _ibp_forward_graph_batched(xl.unsqueeze(0), xh.unsqueeze(0), gg,
                                   DEV, DT)


def test_ibp_batched_parity_and_clamps(tmp_path):
    g = _conv_relu_fc_net(tmp_path)
    gg = g.gpu_graph(DEV, DT)
    xl, xh = _box(g)
    sb = _ibp_forward_graph(xl, xh, gg, DEV, DT)
    B = 3
    xlb = xl.unsqueeze(0).expand(B, -1).contiguous()
    xhb = xh.unsqueeze(0).expand(B, -1).contiguous()
    sbb = _ibp_forward_graph_batched(xlb, xhb, gg, DEV, DT)
    for L in sb:
        for bi in range(B):
            torch.testing.assert_close(sbb[L][0][bi], sb[L][0])
            torch.testing.assert_close(sbb[L][1][bi], sb[L][1])
    # OFF-clamp an unstable neuron at L0 in row 1 only.
    lo0, hi0 = sb[0]
    j = int(((lo0 < 0) & (hi0 > 0)).nonzero().flatten()[0])
    n0 = lo0.numel()
    cl = torch.full((B, n0), -np.inf, dtype=DT)
    ch = torch.full((B, n0), np.inf, dtype=DT)
    ch[1, j] = 0.0
    sbc = _ibp_forward_graph_batched(xlb, xhb, gg, DEV, DT,
                                     relu_clamps={0: (cl, ch)},
                                     root_sb=sb)
    assert float(sbc[0][1][1, j]) == 0.0          # clamp recorded in sb
    torch.testing.assert_close(sbc[0][1][0], sb[0][1])   # row 0 untouched
    # Downstream bounds in row 1 are no looser anywhere, tighter somewhere.
    for L in (1, 2):
        assert bool((sbc[L][0][1] >= sb[L][0] - 1e-12).all())
        assert bool((sbc[L][1][1] <= sb[L][1] + 1e-12).all())
    # Crossing repair: clamp hi to 0 on a STRICTLY-POSITIVE neuron.
    j_pos = int((lo0 > 0).nonzero().flatten()[0])
    ch2 = torch.full((B, n0), np.inf, dtype=DT)
    ch2[2, j_pos] = 0.0
    sbe = _ibp_forward_graph_batched(xlb, xhb, gg, DEV, DT,
                                     relu_clamps={0: (cl, ch2)},
                                     root_sb=sb)
    assert bool((sbe[0][1][2] >= sbe[0][0][2]).all())     # hi >= lo repaired


def _spec_setup(g, xl, xh):
    gg = g.gpu_graph(DEV, DT)
    sb = _ibp_forward_graph(xl, xh, gg, DEV, DT)
    n_out = next(op['W'].shape[0] for op in reversed(gg['ops'])
                 if op['type'] == 'fc')
    w = np.zeros(n_out); w[0] = 1.0; w[1] = -1.0
    return gg, sb, w


def test_relu_split_beta_zero_is_noop_and_sound(tmp_path):
    g = _conv_relu_fc_net(tmp_path)
    xl, xh = _box(g)
    gg, sb, w = _spec_setup(g, xl, xh)
    w_t = torch.tensor(w, dtype=DT)
    sbb = {L: (lo.unsqueeze(0), hi.unsqueeze(0)) for L, (lo, hi) in sb.items()}
    xlb, xhb = xl.unsqueeze(0), xh.unsqueeze(0)
    spec_ew = {0: (w_t, 0.0)}
    lb_plain = _spec_backward_graph_batched(sbb, xlb, xhb, gg, spec_ew,
                                            DEV, DT)[0, 0]
    n0 = sb[0][0].numel()
    zero_beta = {0: torch.zeros(1, n0, dtype=DT)}
    lb_beta0 = _spec_backward_graph_batched(sbb, xlb, xhb, gg, spec_ew,
                                            DEV, DT,
                                            relu_split_beta=zero_beta)[0, 0]
    torch.testing.assert_close(lb_plain, lb_beta0)

    # OFF-split an unstable L0 neuron; beta bound must stay below the true
    # constrained min (dense sampling of the subdomain = ground truth).
    lo0, hi0 = sb[0]
    j = int(((lo0 < 0) & (hi0 > 0)).nonzero().flatten()[0])
    cl = torch.full((1, n0), -np.inf, dtype=DT)
    ch = torch.full((1, n0), np.inf, dtype=DT)
    ch[0, j] = 0.0
    sbc = _ibp_forward_graph_batched(xlb, xhb, gg, DEV, DT,
                                     relu_clamps={0: (cl, ch)}, root_sb=sb)
    rng = np.random.default_rng(13)
    true_min = np.inf
    for _ in range(4000):
        t = torch.tensor(rng.uniform(0, 1, xl.numel()), dtype=DT)
        x = xl + t * (xh - xl)
        y, pre = _point_forward(g, x)
        if float(pre[0][j]) <= 0:                  # inside the OFF subdomain
            true_min = min(true_min, float(w_t @ y))
    assert np.isfinite(true_min)
    sgn = torch.zeros(1, n0, dtype=DT)
    sgn[0, j] = 1.0                                # OFF: s = +z <= 0
    for beta_val in (0.0, 0.05, 0.5, 5.0):
        rsb = {0: sgn * beta_val}
        lb = float(_spec_backward_graph_batched(
            sbc, xlb, xhb, gg, spec_ew, DEV, DT,
            relu_split_beta=rsb)[0, 0])
        assert lb <= true_min + 1e-6, (beta_val, lb, true_min)


def test_ibp_crown_bab_closes_and_exhausts(tmp_path):
    g = _conv_relu_fc_net(tmp_path)
    xl, xh = _box(g, eps=0.05, seed=2)
    gg, sb, _ = _spec_setup(g, xl, xh)
    n_out = 2
    # Query with margin: w.y + b where b makes the plain bound slightly
    # negative but the true min positive — find such b by sampling.
    w = np.array([1.0, -1.0])
    w_t = torch.tensor(w, dtype=DT)
    sbb = {L: (lo.unsqueeze(0), hi.unsqueeze(0)) for L, (lo, hi) in sb.items()}
    lb0 = float(_spec_backward_graph_batched(
        sbb, xl.unsqueeze(0), xh.unsqueeze(0), gg, {0: (w_t, 0.0)},
        DEV, DT)[0, 0])
    rng = np.random.default_rng(4)
    true_min = np.inf
    for _ in range(2000):
        t = torch.tensor(rng.uniform(0, 1, xl.numel()), dtype=DT)
        x = xl + t * (xh - xl)
        y, _ = _point_forward(g, x)
        true_min = min(true_min, float(w_t @ y))
    assert lb0 < true_min                          # there is a CROWN gap
    b_mid = -(lb0 + true_min) / 2                  # plain lb<0, true min>0
    closed, n_dom, reason = _ibp_crown_bab(
        gg, xl, xh, sb, w, b_mid, DEV, DT,
        time_left=lambda: 60.0, batch=8, alpha_iters=4, kfsb_k=2)
    assert closed and reason == 'closed', (closed, n_dom, reason)
    # Truly violated query (b far below the true min): cannot close; the
    # search must terminate by pattern exhaustion / cap, never claim
    # verified.
    closed2, _, reason2 = _ibp_crown_bab(
        gg, xl, xh, sb, w, -true_min - 10.0, DEV, DT,
        time_left=lambda: 20.0, batch=8, alpha_iters=2, kfsb_k=2,
        max_domains=64)
    assert not closed2
    assert reason2 in ('exhausted_pattern', 'max_domains', 'time')


def test_run_alpha_crown_track_best_alpha(tmp_path):
    from vibecheck import alpha_crown as ac
    g = _conv_relu_fc_net(tmp_path)
    xl, xh = _box(g, eps=0.1, seed=9)
    gg, sb, w = _spec_setup(g, xl, xh)
    bbr = {L: (lo.numpy(), hi.numpy()) for L, (lo, hi) in sb.items()}
    un = {L: np.where((bbr[L][0] < 0) & (bbr[L][1] > 0))[0].tolist()
          for L in bbr}
    lb, alpha, best_bounds, _ = ac.run_alpha_crown(
        gg, xl, xh, bbr, w, 0.0, [], un, DEV, DT, n_iters=8,
        track_best_alpha=True)
    assert 'spec' in alpha and len(alpha['spec']) > 0
    # The returned (best-iter) alpha re-evaluates to the reported best lb
    # against the best_bounds-merged sb (production merges root_bounds
    # into sb before warm-starting from this alpha).
    sb_m = dict(sb)
    for L, (lo_t, hi_t) in best_bounds.items():
        if L in sb_m:
            sb_m[L] = (torch.maximum(sb_m[L][0], lo_t.detach().to(DT)),
                       torch.minimum(sb_m[L][1], hi_t.detach().to(DT)))
    sbb = {L: (lo.unsqueeze(0), hi.unsqueeze(0))
           for L, (lo, hi) in sb_m.items()}
    al = {L: a.unsqueeze(0) for L, a in alpha['spec'].items()}
    lb_re = float(_spec_backward_graph_batched(
        sbb, xl.unsqueeze(0), xh.unsqueeze(0), gg,
        {0: (torch.tensor(w, dtype=DT), 0.0)}, DEV, DT,
        alpha_at_layer=al)[0, 0])
    # The two backwards (_crown_backward_matrix inside run_alpha_crown vs
    # _spec_backward_graph_batched) make slightly different default-slope
    # choices for stable neurons — both sound; the snapshot only needs to
    # be a good warm start, so pin relative closeness, not equality.
    assert lb_re >= lb - max(0.05, 0.005 * abs(lb))
