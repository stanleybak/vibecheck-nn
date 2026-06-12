"""Batched beta-CROWN (attn_crown_lb_batch / attn_beta_bab) pins.

1. Row-equality: the batched walk must reproduce single-domain
   attn_crown_lb exactly for every domain in the batch, across random
   plane params, random relu/exp clamps and betas — on both the
   attention toy (exp/recip/mc arms) and a relu-fork toy (relu/rbeta).
2. e2e: attn_beta_bab closes the relu-free softmax query that needs
   exp-input splits + beta (the same calibration as
   test_bilinear_split's unbatched pin).
"""
import os

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto

from tests.test_bilinear_split import _softmax_net, _relu_fork_net, _box
from tests.test_attention_backward import _attn_net


def _rand_params_like(params0, rng, betas):
    out = {}
    for k, v in params0.items():
        out[k] = torch.tensor(rng.uniform(0, 1, v.shape),
                              dtype=torch.float64)
    for k, shape in betas.items():
        out[k] = torch.tensor(rng.uniform(0, 0.5, shape),
                              dtype=torch.float64)
    return out


def _stack(dicts):
    keys = dicts[0].keys()
    return {k: torch.stack([d[k] for d in dicts]) for k in keys}


def test_batch_matches_single_attention(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    from vibecheck.attn_crown import (attn_crown_lb, attn_crown_lb_batch,
                                      init_params)
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(11)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.08, dtype=torch.float64)
    xh = torch.tensor(xc + 0.08, dtype=torch.float64)
    w = np.array([1.0, -2.0, 0.5])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    p0 = init_params(gg, sb, ob, torch.device('cpu'), torch.float64)
    nm = [op['name'] for op in gg['ops'] if op['type'] == 'exp'][0]
    n_exp = ob[nm][0].numel()
    B = 3
    doms = []
    inf = float('inf')
    for i in range(B):
        params = _rand_params_like(p0, rng,
                                   {('beta', nm): (2, n_exp)})
        cl = torch.full((n_exp,), -inf, dtype=torch.float64)
        ch = torch.full((n_exp,), inf, dtype=torch.float64)
        j = int(rng.integers(0, n_exp))
        m = float(0.5 * (ob[nm][0][j] + ob[nm][1][j]))
        if i % 2 == 0:
            ch[j] = m
        else:
            cl[j] = m
        doms.append((params, {nm: (cl, ch)}))
    singles = []
    for params, oc in doms:
        with torch.no_grad():
            singles.append(float(attn_crown_lb(
                gg, xl, xh, sb, ob, w, 0.25, params, op_clamps=oc)))
    sb_b = {L: (lo.unsqueeze(0).expand(B, -1),
                hi.unsqueeze(0).expand(B, -1)) for L, (lo, hi) in sb.items()}
    ob_b = {}
    for k, v in ob.items():
        if isinstance(v[0], tuple):
            ob_b[k] = ((v[0][0].unsqueeze(0).expand(B, -1),
                        v[0][1].unsqueeze(0).expand(B, -1)),
                       (v[1][0].unsqueeze(0).expand(B, -1),
                        v[1][1].unsqueeze(0).expand(B, -1)))
        else:
            ob_b[k] = (v[0].unsqueeze(0).expand(B, -1),
                       v[1].unsqueeze(0).expand(B, -1))
    params_b = _stack([d[0] for d in doms])
    oc_b = {nm: (torch.stack([d[1][nm][0] for d in doms]),
                 torch.stack([d[1][nm][1] for d in doms]))}
    with torch.no_grad():
        lb_b = attn_crown_lb_batch(gg, xl, xh, sb_b, ob_b, w, 0.25,
                                   params_b, op_clamps_b=oc_b)
    for i in range(B):
        assert float(lb_b[i]) == pytest.approx(singles[i], abs=1e-9), \
            (i, float(lb_b[i]), singles[i])


def test_batch_matches_single_relu(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    from vibecheck.attn_crown import (attn_crown_lb, attn_crown_lb_batch,
                                      init_params)
    graph = _relu_fork_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    L = [op['layer_idx'] for op in gg['ops'] if op['type'] == 'relu'][0]
    xl = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    xh = torch.tensor([1.0, 1.0], dtype=torch.float64)
    rng = np.random.default_rng(13)
    inf = float('inf')
    B = 3
    singles = []
    sb_rows = []
    rc_rows = []
    p_rows = []
    for i in range(B):
        side = i % 2
        tb = {L: (np.array([0.0]), np.array([2.0]))} if side else \
             {L: (np.array([-2.0]), np.array([0.0]))}
        ob = {}
        sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                        torch.float64, tight_bounds=tb,
                                        op_bounds=ob)
        p0 = init_params(gg, sb, ob, torch.device('cpu'), torch.float64)
        params = _rand_params_like(p0, rng, {('rbeta', L): (2, 1)})
        rc = ({L: (torch.tensor([0.0], dtype=torch.float64),
                   torch.tensor([inf], dtype=torch.float64))} if side else
              {L: (torch.tensor([-inf], dtype=torch.float64),
                   torch.tensor([0.0], dtype=torch.float64))})
        with torch.no_grad():
            singles.append(float(attn_crown_lb(
                gg, xl, xh, sb, ob, np.array([1.0]), 0.0, params,
                relu_clamps=rc)))
        sb_rows.append(sb[L])
        rc_rows.append(rc[L])
        p_rows.append(params)
    sb_b = {L: (torch.stack([r[0] for r in sb_rows]),
                torch.stack([r[1] for r in sb_rows]))}
    rc_b = {L: (torch.stack([r[0] for r in rc_rows]),
                torch.stack([r[1] for r in rc_rows]))}
    params_b = _stack(p_rows)
    with torch.no_grad():
        lb_b = attn_crown_lb_batch(gg, xl, xh, sb_b, {},
                                   np.array([1.0]), 0.0, params_b,
                                   relu_clamps_b=rc_b)
    for i in range(B):
        assert float(lb_b[i]) == pytest.approx(singles[i], abs=1e-9)


def test_beta_bab_closes_softmax_query(tmp_path):
    """e2e: the batched driver closes the exp-split + beta calibration
    (root open, true min clears the threshold)."""
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    from vibecheck.attn_crown import attn_beta_bab, attn_crown_alpha
    # 3 coords (two LIVE shifted scores under the max-shift) so the
    # relaxation has genuine slack; budgeted 3-iter root alpha keeps
    # the root open and the BaB must close — exercising the driver
    # mechanics (batch assembly, heap, warm-start, pruning, closure;
    # measured: root -0.018, bab closes in ~5 domains).
    graph = _softmax_net(str(tmp_path), n=3)
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    xl = torch.full((3,), -1.0, dtype=torch.float64)
    xh = torch.full((3,), 1.0, dtype=torch.float64)
    w = np.array([1.0])
    true_min = np.exp(-2.0) / (2.0 + np.exp(-2.0))
    b_q = -0.5 * true_min
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    w_t = torch.tensor(w, dtype=torch.float64)
    root_lb, root_params = attn_crown_alpha(
        gg, xl, xh, sb, ob, w_t, b_q, n_iters=3, lr=0.2)
    assert root_lb < 0, f'toy mis-designed: root closed {root_lb}'
    ok, n_dom, reason = attn_beta_bab(
        gg, xl, xh, sb, ob, w_t, b_q, root_params,
        time_left=lambda: 100.0, batch=4, n_iters=15, lr=0.2)
    assert ok, f'beta-bab failed: {reason} after {n_dom}'
    assert n_dom > 1


def test_intermediate_refine_tightens_and_sound(tmp_path):
    """attn_refine_op_bounds must (a) only shrink the recorded ranges,
    (b) keep the spec backward SOUND vs sampling, (c) not worsen the
    root bound (best-of tighter planes)."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import (attn_crown_lb, attn_refine_op_bounds)
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(17)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.15, dtype=torch.float64)
    xh = torch.tensor(xc + 0.15, dtype=torch.float64)
    w = np.array([1.0, -2.0, 0.5])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    import copy
    ob_raw = {k: (copy.deepcopy(v) if isinstance(v[0], tuple)
                  else (v[0].clone(), v[1].clone()))
              for k, v in ob.items()}
    with torch.no_grad():
        lb_raw = float(attn_crown_lb(gg, xl, xh, sb, ob_raw, w, 0.0, {}))
    n_t = attn_refine_op_bounds(gg, xl, xh, sb, ob)
    assert n_t > 0, 'refinement tightened nothing on a 0.15-eps box'
    for k, v in ob.items():
        if isinstance(v[0], tuple):
            pairs = [(ob_raw[k][0], v[0]), (ob_raw[k][1], v[1])]
        else:
            pairs = [(ob_raw[k], v)]
        for (lo0, hi0), (lo1, hi1) in pairs:
            assert bool((lo1 >= lo0 - 1e-12).all())
            assert bool((hi1 <= hi0 + 1e-12).all())
    with torch.no_grad():
        lb_ref = float(attn_crown_lb(gg, xl, xh, sb, ob, w, 0.0, {}))
    assert lb_ref >= lb_raw - 1e-9, (lb_ref, lb_raw)
    xs = torch.tensor(rng.uniform(xl.numpy(), xh.numpy(), (500, 4)),
                      dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    true_min = float((ys @ torch.tensor(w, dtype=torch.float64)).min())
    assert lb_ref <= true_min + 1e-9, f'UNSOUND refined: {lb_ref} > {true_min}'


def test_alpha_joint_sound_and_not_worse(tmp_path):
    """attn_alpha_joint (differentiable intermediate bounds + joint
    spec matrix): per-query results must be SOUND vs sampling and not
    worse than the plain per-query alpha (best-of argument; the
    dynamic bounds only intersect the frozen enclosure)."""
    import copy
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_crown_alpha, attn_alpha_joint
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(4)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.12, dtype=torch.float64)
    xh = torch.tensor(xc + 0.12, dtype=torch.float64)
    W = np.array([[1.0, -2.0, 0.5], [-1.0, 0.3, 1.2]])
    bv = np.array([0.0, 0.1])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    base = []
    for i in range(2):
        ob_i = {k: (copy.deepcopy(v) if isinstance(v[0], tuple)
                    else (v[0].clone(), v[1].clone()))
                for k, v in ob.items()}
        lb, _ = attn_crown_alpha(gg, xl, xh, sb, ob_i,
                                 torch.tensor(W[i]), float(bv[i]),
                                 n_iters=80, lr=0.2)
        base.append(lb)
    lb_j, p_j, ob_c, sb_c = attn_alpha_joint(
        gg, xl, xh, sb, ob, W, bv, n_iters=60, lr=0.2)
    xs = torch.tensor(rng.uniform(xl.numpy(), xh.numpy(), (3000, 4)),
                      dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    for i in range(2):
        tm = float((ys @ torch.tensor(W[i])).min()) + bv[i]
        assert lb_j[i] <= tm + 1e-9, f'UNSOUND joint q{i}: {lb_j[i]} > {tm}'
        assert lb_j[i] >= base[i] - 0.02, (
            f'joint q{i} materially worse than per-query: '
            f'{lb_j[i]} vs {base[i]}')
    # certified dynamic bounds are valid enclosures of true intermediate
    # values: spot-check via the relu pre-activation rows
    for L, (lo_c, hi_c) in sb_c.items():
        lo0, hi0 = sb[L]
        assert bool((lo_c >= lo0 - 1e-9).all())
        assert bool((hi_c <= hi0 + 1e-9).all())


def test_alpha_joint_mem_row_cap(tmp_path, capsys):
    """Memory-adaptive differentiable-row cap (A10G ibp_3_3_8 #253
    OOM): injected memory readers reporting allocation growth + no
    free memory must floor the cap at min_rows and announce it;
    readers reporting no growth derive no cap. Both runs stay SOUND
    vs sampling."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_alpha_joint, _mem_row_cap
    # pure budget math: no growth -> None; floor; proportional cap
    assert _mem_row_cap(10 * 10 ** 9, 0.0) is None
    assert _mem_row_cap(0, 1.0e6) == 256
    assert _mem_row_cap(10 * 10 ** 9, 1.0e6, safety=0.6) == 6000
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(4)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.12, dtype=torch.float64)
    xh = torch.tensor(xc + 0.12, dtype=torch.float64)
    W = np.array([[1.0, -2.0, 0.5]])
    bv = np.array([0.0])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    # (a) probe sees 50 MB growth, zero free memory: cap floors at
    # 256 (< default max_rows=4096) and is announced
    grow = iter([0, 50 * 10 ** 6])
    lb_a, _, _, _ = attn_alpha_joint(
        gg, xl, xh, sb, ob, W, bv, n_iters=4, lr=0.2,
        mem_fns=(lambda: 0, lambda: next(grow)))
    out = capsys.readouterr().out
    assert 'memory cap' in out and '256 differentiable rows' in out
    # (b) flat readers (no measurable growth): no cap derivable
    lb_b, _, _, _ = attn_alpha_joint(
        gg, xl, xh, sb, ob, W, bv, n_iters=4, lr=0.2,
        mem_fns=(lambda: 0, lambda: 0))
    assert 'memory cap' not in capsys.readouterr().out
    xs = torch.tensor(rng.uniform(xl.numpy(), xh.numpy(), (2000, 4)),
                      dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    tm = float((ys @ torch.tensor(W[0])).min())
    assert lb_a[0] <= tm + 1e-9, f'UNSOUND capped joint: {lb_a[0]}'
    assert lb_b[0] <= tm + 1e-9, f'UNSOUND uncapped joint: {lb_b[0]}'


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='default mem_fns need the CUDA readers')
def test_alpha_joint_mem_cap_cuda_defaults(tmp_path):
    """On a CUDA device the memory readers default to the real
    torch.cuda ones: the probe runs, the budget on an idle GPU
    exceeds the toy's rows (no trim) and the result is sound."""
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_alpha_joint, _cuda_free_bytes
    dev = torch.device('cuda')
    assert _cuda_free_bytes(dev) > 0
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(dev, torch.float64)
    rng = np.random.default_rng(4)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.12, dtype=torch.float64, device=dev)
    xh = torch.tensor(xc + 0.12, dtype=torch.float64, device=dev)
    W = np.array([[1.0, -2.0, 0.5]])
    bv = np.array([0.0])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, dev, torch.float64,
                                    op_bounds=ob)
    lb, _, _, _ = attn_alpha_joint(gg, xl, xh, sb, ob, W, bv,
                                   n_iters=3, lr=0.2)
    xs = torch.tensor(rng.uniform(xl.cpu().numpy(), xh.cpu().numpy(),
                                  (2000, 4)),
                      dtype=torch.float64, device=dev)
    ys = _forward_batch_graph(xs, gg)
    tm = float((ys @ torch.tensor(W[0], dtype=torch.float64,
                                  device=dev)).min())
    assert lb[0] <= tm + 1e-9, f'UNSOUND cuda joint: {lb[0]}'


def test_alpha_joint_freezing_and_refresh(tmp_path, capsys):
    """Adaptive target freezing + refresh interval: with a loose freeze
    tolerance the targets freeze early (frozen raw bounds reused as
    constants — sound enclosures), refresh_every=2 skips non-per-row
    re-derivation on odd iterations, and the certified results stay
    SOUND vs sampling with enclosures that only shrink."""
    import re
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    from vibecheck.attn_crown import attn_alpha_joint
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(7)
    xc = rng.uniform(-0.5, 0.5, 4)
    xl = torch.tensor(xc - 0.12, dtype=torch.float64)
    xh = torch.tensor(xc + 0.12, dtype=torch.float64)
    W = np.array([[1.0, -2.0, 0.5], [-1.0, 0.3, 1.2]])
    bv = np.array([0.0, 0.1])
    ob = {}
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64, op_bounds=ob)
    lb_j, p_j, ob_c, sb_c = attn_alpha_joint(
        gg, xl, xh, sb, ob, W, bv, n_iters=25, lr=0.2,
        per_row_rows=0, refresh_every=2,
        freeze_tol=1e-2, freeze_patience=1)
    m = re.search(r'(\d+)/(\d+) targets frozen', capsys.readouterr().out)
    assert m is not None, 'freeze summary line missing'
    assert int(m.group(1)) > 0, 'freezing never engaged at tol 1e-2'
    xs = torch.tensor(rng.uniform(xl.numpy(), xh.numpy(), (3000, 4)),
                      dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    for i in range(2):
        tm = float((ys @ torch.tensor(W[i])).min()) + bv[i]
        assert lb_j[i] <= tm + 1e-9, \
            f'UNSOUND frozen joint q{i}: {lb_j[i]} > {tm}'
    for L, (lo_c, hi_c) in sb_c.items():
        lo0, hi0 = sb[L]
        assert bool((lo_c >= lo0 - 1e-9).all())
        assert bool((hi_c <= hi0 + 1e-9).all())
    # relu pre-activation ('sb'-kind) target: its ±I walk is linear in
    # the input, so its width is constant and it must freeze; the
    # frozen-reuse branch for sb consumers stays sound (true min 0.1).
    graph_r = _relu_fork_net(str(tmp_path))
    gg_r = graph_r.gpu_graph(torch.device('cpu'), torch.float64)
    xl_r = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    xh_r = torch.tensor([1.0, 1.0], dtype=torch.float64)
    ob_r = {}
    sb_r, _ = _forward_zonotope_graph(
        xl_r, xh_r, gg_r, torch.device('cpu'), torch.float64,
        op_bounds=ob_r)
    lb_r, _, _, sb_rc = attn_alpha_joint(
        gg_r, xl_r, xh_r, sb_r, ob_r, np.array([[1.0]]),
        np.array([0.0]), n_iters=12, lr=0.2,
        freeze_tol=1e-2, freeze_patience=1)
    m = re.search(r'(\d+)/(\d+) targets frozen', capsys.readouterr().out)
    assert m is not None and int(m.group(1)) == 1, \
        'linear relu target failed to freeze'
    assert lb_r[0] <= 0.1 + 1e-9, f'UNSOUND relu joint: {lb_r[0]}'
    for L, (lo_c, hi_c) in sb_rc.items():
        lo0, hi0 = sb_r[L]
        assert bool((lo_c >= lo0 - 1e-9).all())
        assert bool((hi_c <= hi0 + 1e-9).all())
