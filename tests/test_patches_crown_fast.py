"""Coverage + soundness for the patches-CROWN backward fast paths.

``_FAST_MAXPOOL_BWD`` (merge-branch clone elision + disjoint phase-block in-place
accumulation) and ``_FAST_PHASE_CONV`` (the 1-hot maxpool phase-extraction conv
backward as a strided scatter rather than a winograd ``conv_transpose2d``) are
DATA-MOVEMENT refactors of the maxpool->relu decomposition's backward. On
CPU/float64 (deterministic conv_transpose) they must be BIT-IDENTICAL to the
naive path, and both must match the dense ``_crown_backward_matrix`` (the trusted
reference) — that is the soundness gate.
"""
import numpy as np
import torch

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import maxpool_to_relu
from vibecheck.settings import default_settings
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck.alpha_crown import _crown_backward_matrix
import vibecheck.patches_crown as pc
from tests.test_patches_maxpool import _maxpool_net_onnx


def _setup(tmp_path):
    C, H, W = 3, 8, 8
    p = str(tmp_path / 'mpnet.onnx')
    _maxpool_net_onnx(p, C, H, W)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert maxpool_to_relu(g) is True
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    torch.manual_seed(0)
    xl = torch.rand(C * H * W, dtype=torch.float64) - 1.0
    xh = xl + torch.rand(C * H * W, dtype=torch.float64) * 0.3 + 0.01
    s = default_settings(); s.zono_impl = 'patches'
    sb, _ = _forward_zonotope_graph(xl, xh, gg, 'cpu', torch.float64, settings=s)
    ops = gg['ops']
    relu_ops = sorted((o for o in ops if o['type'] == 'relu' and 'layer_idx' in o),
                      key=lambda o: o['layer_idx'])
    rop = relu_ops[-1]                                  # last relu (after conv2)
    pre_op = rop['inputs'][0]
    pre_obj = next(o for o in ops if o['name'] == pre_op)
    out_shape = tuple(pre_obj['out_shape_nd'])
    lo, hi = sb[rop['layer_idx']]
    uns = ((lo < 0) & (hi > 0)).nonzero().flatten()
    sel = uns[:8] if uns.numel() >= 8 else torch.arange(min(8, lo.numel()))
    return gg, xl, xh, sb, pre_op, out_shape, sel


def _bounds(gg, xl, xh, sb, pre_op, out_shape, sel, fast):
    pc._FAST_MAXPOOL_BWD[0] = fast
    pc._FAST_PHASE_CONV[0] = fast
    pc._RELU_COMPILE[0] = False                          # eager, deterministic
    return pc.patches_bounds(gg, xl, xh, sb, pre_op, out_shape, sel,
                             'cpu', torch.float64)


def test_detect_phase_conv_identifies_the_1hot_conv(tmp_path):
    gg, *_ = _setup(tmp_path)
    ops = gg['ops']
    phase = next(o for o in ops if o['name'].endswith('__mp2relu_conv'))
    assert pc._detect_phase_conv(phase) == (2, 2, 2, 2)
    # A normal (non-phase) conv must NOT be taken as a phase conv.
    normal = next(o for o in ops if o['type'] == 'conv'
                  and not o['name'].endswith('__mp2relu_conv'))
    assert pc._detect_phase_conv(normal) is False


def test_detect_phase_conv_rejects_non_phase_variants():
    C, kH, kW = 2, 2, 2
    P = kH * kW
    k = torch.zeros(P * C, C, kH, kW)
    for p in range(P):
        for c in range(C):
            k[p * C + c, c, p // kW, p % kW] = 1.0
    base = {'kernel': k, 'stride': (2, 2), 'padding': (0, 0),
            'bias': torch.zeros(P * C)}
    assert pc._detect_phase_conv(dict(base)) == (2, 2, 2, 2)
    assert pc._detect_phase_conv({**base, 'padding': (1, 1)}) is False
    assert pc._detect_phase_conv({**base, 'bias': torch.ones(P * C)}) is False
    assert pc._detect_phase_conv({**base, 'stride': (1, 1)}) is False     # stride!=kernel
    bad = k.clone(); bad[0, 0, 0, 0] = 0.5                                # not 1-hot
    assert pc._detect_phase_conv({**base, 'kernel': bad}) is False


def test_fast_paths_bit_identical_to_naive(tmp_path):
    gg, xl, xh, sb, pre_op, out_shape, sel = _setup(tmp_path)
    lb_f, ub_f = _bounds(gg, xl, xh, sb, pre_op, out_shape, sel, True)
    lb_n, ub_n = _bounds(gg, xl, xh, sb, pre_op, out_shape, sel, False)
    # CPU/float64 conv_transpose is deterministic -> EXACTLY equal.
    torch.testing.assert_close(lb_f, lb_n, atol=0.0, rtol=0.0)
    torch.testing.assert_close(ub_f, ub_n, atol=0.0, rtol=0.0)
    pc._FAST_MAXPOOL_BWD[0] = True; pc._FAST_PHASE_CONV[0] = True


def test_fast_matches_dense_backward(tmp_path):
    gg, xl, xh, sb, pre_op, out_shape, sel = _setup(tmp_path)
    lb, ub = _bounds(gg, xl, xh, sb, pre_op, out_shape, sel, True)
    n_at = sb[next(o['layer_idx'] for o in gg['ops']
                   if o['type'] == 'relu' and o['inputs'][0] == pre_op)][0].numel()
    Bn = sel.numel()
    ew = torch.zeros(Bn, n_at, dtype=torch.float64)
    ew[torch.arange(Bn), sel] = 1.0
    d_lb, _ = _crown_backward_matrix(gg, xl, xh, {}, sb, pre_op, ew, 'cpu',
                                     torch.float64)
    d_nlb, _ = _crown_backward_matrix(gg, xl, xh, {}, sb, pre_op, -ew, 'cpu',
                                      torch.float64)
    torch.testing.assert_close(lb, d_lb, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(ub, -d_nlb, atol=1e-9, rtol=0.0)
