"""Correctness of the backward alpha_zono state builder vs a dense forward.

`gen_state_backward.build_alpha_zono_state_backward` computes each unstable
neuron's generator row by a chunked backward pass, to avoid materializing
the dense generator matrix (which OOMs on large-input conv nets). This test
cross-checks it against an INDEPENDENT dense-forward ground truth that
applies the identical zonotope (parallelogram) relaxation, on a net small
enough to materialize G. Rows, lam/mu, c_in, and the column layout must
match to fp tolerance — anything else would mean the dual-ascent reads
different coefficients than the bounds imply (a soundness risk).
"""
import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from vibecheck.gen_state_backward import (
    build_alpha_zono_state_backward, _relu_lam_mu)
from vibecheck.onnx_loader import load_onnx

DEV = torch.device('cpu')
DT = torch.float64


def _init(name, arr):
    arr = np.asarray(arr, np.float64)
    return helper.make_tensor(name, TensorProto.DOUBLE, arr.shape,
                              arr.flatten())


def _small_net(tmp_path, seed=0):
    """Conv(3->4,3x3,pad1) -> Relu -> Conv(4->3,3x3,stride2) -> Relu ->
    Flatten -> Gemm -> Relu -> Gemm. Input 3x6x6 = 108 (G fits densely)."""
    rng = np.random.default_rng(seed)
    k1 = rng.standard_normal((4, 3, 3, 3)) * 0.4
    b1 = rng.standard_normal(4) * 0.1
    k2 = rng.standard_normal((3, 4, 3, 3)) * 0.4
    b2 = rng.standard_normal(3) * 0.1
    # conv2: 6x6 -> stride2 3x3 floor((6-3)/2)+1=2 -> 3*2*2 = 12 flat
    W1 = rng.standard_normal((5, 12)) * 0.4
    c1 = rng.standard_normal(5) * 0.1
    W2 = rng.standard_normal((3, 5)) * 0.4
    c2 = rng.standard_normal(3) * 0.1
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
    graph = helper.make_graph(
        nodes, 'small',
        [helper.make_tensor_value_info('X', TensorProto.DOUBLE, (1, 3, 6, 6))],
        [helper.make_tensor_value_info('Y', TensorProto.DOUBLE, (1, 3))],
        inits)
    p = str(tmp_path / 'small.onnx')
    onnx.save(helper.make_model(graph), p)
    return load_onnx(p, dtype=np.float64)


def _ibp_bounds(gg, xl, xh):
    """Plain interval bounds at each pre-ReLU (the bbr both builders use)."""
    lo = {gg['input_name']: xl.clone()}
    hi = {gg['input_name']: xh.clone()}
    bbr = {}
    for op in gg['ops']:
        t, name = op['type'], op['name']
        l, h = lo[op['inputs'][0]], hi[op['inputs'][0]]
        if t == 'conv':
            mid, rad = (l + h) / 2, (h - l) / 2
            mc = torch.nn.functional.conv2d(
                mid.reshape(1, *op['in_shape']), op['kernel'], op['bias'],
                stride=op['stride'], padding=op['padding'])
            rc = torch.nn.functional.conv2d(
                rad.reshape(1, *op['in_shape']), op['kernel'].abs(), None,
                stride=op['stride'], padding=op['padding'])
            lo[name], hi[name] = (mc - rc).flatten(), (mc + rc).flatten()
        elif t == 'fc':
            mid, rad = (l + h) / 2, (h - l) / 2
            mc = op['W'] @ mid + (op['bias'] if op['bias'] is not None else 0)
            rc = op['W'].abs() @ rad
            lo[name], hi[name] = mc - rc, mc + rc
        elif t == 'relu':
            if 'layer_idx' in op:
                bbr[op['layer_idx']] = (l.cpu().numpy(), h.cpu().numpy())
            lo[name], hi[name] = l.clamp(min=0), h.clamp(min=0)
        elif t == 'reshape':
            lo[name], hi[name] = l, h
        else:
            raise NotImplementedError(t)
    return bbr


def _dense_forward_state(gg, xl, xh, bbr, alpha_per_layer):
    """Ground truth: propagate the dense generator matrix with the SAME
    parallelogram relaxation and read each unstable neuron's row off it."""
    relu_layers = sorted(L for L in bbr)
    n_in = xl.numel()
    center = {gg['input_name']: (xl + xh) / 2}
    G = {gg['input_name']: torch.diag((xh - xl) / 2)}      # (n_gens × n_in)
    # column model
    new_col_start, running, un_by_L = {}, n_in, {}
    for L in relu_layers:
        lo, hi = bbr[L]
        un = np.where((lo < 0) & (hi > 0))[0]
        un_by_L[L] = un
        new_col_start[L] = running
        running += len(un)
    n_gens = running
    # pad helper: keep G as (n_gens_so_far × n_acts); grow cols lazily
    state = {'unstable': []}

    def _grow(Gm, target_cols):
        if Gm.shape[0] < target_cols:
            pad = torch.zeros(target_cols - Gm.shape[0], Gm.shape[1],
                              dtype=DT)
            return torch.cat([Gm, pad], dim=0)
        return Gm

    for op in gg['ops']:
        t, name = op['type'], op['name']
        c = center[op['inputs'][0]]
        Gm = G[op['inputs'][0]]
        if t == 'conv':
            cc = torch.nn.functional.conv2d(
                c.reshape(1, *op['in_shape']), op['kernel'], op['bias'],
                stride=op['stride'], padding=op['padding']).flatten()
            # propagate each generator (row) through the conv (no bias)
            ng = Gm.shape[0]
            Gimg = Gm.reshape(ng, *op['in_shape'])
            Go = torch.nn.functional.conv2d(
                Gimg, op['kernel'], None, stride=op['stride'],
                padding=op['padding']).reshape(ng, -1)
            center[name], G[name] = cc, Go
        elif t == 'fc':
            cc = op['W'] @ c + (op['bias'] if op['bias'] is not None else 0)
            center[name], G[name] = cc, Gm @ op['W'].T
        elif t == 'relu':
            L = op.get('layer_idx')
            if L is None:
                center[name] = c.clamp(min=0)
                G[name] = Gm * (c > 0).to(DT)       # not used for cct
                continue
            lo, hi = bbr[L]
            lo_t = torch.as_tensor(lo, dtype=DT)
            hi_t = torch.as_tensor(hi, dtype=DT)
            alpha = alpha_per_layer[L]
            lam, mu, unstable = _relu_lam_mu(lo, hi, alpha)
            lam_t = torch.as_tensor(lam, dtype=DT)
            mu_t = torch.as_tensor(mu, dtype=DT)
            dead = torch.as_tensor(hi <= 0, dtype=torch.bool)
            # record unstable rows (pre-ReLU z) BEFORE transforming
            Gm_full = _grow(Gm, n_gens)
            for local, j in enumerate(un_by_L[L]):
                j = int(j)
                state['unstable'].append({
                    'layer_idx': L, 'neuron_idx': j,
                    'c_in': float(c[j]), 'lo': float(lo[j]),
                    'hi': float(hi[j]),
                    'e_new_col': new_col_start[L] + local,
                    'row': Gm_full[:, j].clone(),
                    'lam': float(lam[j]), 'mu': float(mu[j]),
                })
            # transform: y = lam*z + mu*(1+e_new); center = lam*c + mu
            cc = lam_t * c + mu_t
            cc = torch.where(dead, torch.zeros_like(cc), cc)
            Gm2 = _grow(Gm, n_gens) * lam_t.unsqueeze(0)
            for local, j in enumerate(un_by_L[L]):
                Gm2[new_col_start[L] + local, int(j)] = mu_t[int(j)]
            center[name], G[name] = cc, Gm2
        elif t == 'reshape':
            center[name], G[name] = c, Gm
        else:
            raise NotImplementedError(t)
    return state['unstable'], n_gens, n_in


@pytest.mark.parametrize('net_seed,in_seed', [(1, 3), (2, 7), (5, 11)])
def test_backward_matches_dense_forward(tmp_path, net_seed, in_seed):
    gg_graph = _small_net(tmp_path, seed=net_seed)
    gg = gg_graph.gpu_graph(DEV, DT)
    n = int(np.prod(gg_graph.input_shape))
    rng = np.random.default_rng(in_seed)
    c = torch.tensor(rng.uniform(-1, 1, n), dtype=DT)
    eps = 0.25
    xl, xh = c - eps, c + eps
    bbr = _ibp_bounds(gg, xl, xh)
    # full-size α per layer (random in [0,1]) — both builders must use it
    alpha = {L: rng.uniform(0.0, 1.0, len(bbr[L][0])) for L in bbr}

    gt, n_gens_gt, n_in_gt = _dense_forward_state(gg, xl, xh, bbr, alpha)
    st = build_alpha_zono_state_backward(
        gg, xl.numpy(), xh.numpy(), bbr,
        {L: torch.tensor(a, dtype=DT) for L, a in alpha.items()},
        device=DEV, dtype=DT, chunk=32)

    assert st['n_input'] == n_in_gt
    assert st['n_gens'] == n_gens_gt
    assert len(st['unstable_list']) == len(gt)
    gt_by_key = {(e['layer_idx'], e['neuron_idx']): e for e in gt}
    assert len(gt_by_key) == len(gt)
    for e in st['unstable_list']:
        g = gt_by_key[(e['layer_idx'], e['neuron_idx'])]
        assert e['e_new_col'] == g['e_new_col']
        assert abs(e['lam'] - g['lam']) < 1e-12
        assert abs(e['mu'] - g['mu']) < 1e-12
        assert abs(e['c_in'] - g['c_in']) < 1e-9
        # reconstruct dense row from sparse and compare to ground truth
        row = np.zeros(n_gens_gt)
        row[e['row_indices']] = e['row_values']
        gt_row = g['row'].cpu().numpy()
        assert np.allclose(row, gt_row, atol=1e-9), (
            f"row mismatch at {(e['layer_idx'], e['neuron_idx'])}: "
            f"max |Δ|={np.abs(row - gt_row).max():.2e}")


if __name__ == '__main__':
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        for ns, is_ in [(1, 3), (2, 7), (5, 11)]:
            test_backward_matches_dense_forward(pathlib.Path(d), ns, is_)
    print('PASS: backward state matches dense forward (3 seeds)')
