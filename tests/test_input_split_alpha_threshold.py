"""Regression tests for the α-CROWN dispatch in `_input_split_fast_leaf`.

Pre-2026-05-10 the leaf wrapped the joint α-CROWN call in
`except torch.cuda.OutOfMemoryError: pass`, masking real memory
regressions. The new contract:

  1. Decide which α-CROWN variant to run BEFORE allocating, based on
     `total_unstable` vs `alpha_crown_impl_auto_switch_threshold`.
  2. Joint α (run_alpha_crown_batched) is used when below the threshold.
  3. Lightweight v2 (run_alpha_crown_fixed_intermediate_batched, sparse)
     is used when above.
  4. Either variant's OOM propagates — we DO NOT swallow it.
"""
import io
import sys
import contextlib

import numpy as np
import onnx
import onnx.helper as oh
import pytest
import torch

from vibecheck.onnx_loader import load_onnx
from vibecheck.spec import Conjunct, PairwiseConstraint, VNNSpec
from vibecheck.settings import default_settings
from vibecheck import verify_graph as vg


def _tiny_conv_input_split(tmp_path):
    """Tiny conv graph with input_dim=4 (≤ input_split_max_dims=20) so
    `_input_split_verify` (and therefore `_input_split_fast_leaf`)
    actually runs.

    Two hidden ReLU layers so the α-CROWN dispatch in fast_leaf has at
    least one L>0 layer with unstable neurons (it iterates over
    `Lk > 0`; a single hidden layer produces an empty `isn` and skips
    α entirely)."""
    K1 = np.array([[[[1.0]]], [[[-1.0]]]], dtype=np.float32)  # (2,1,1,1)
    b1 = np.zeros(2, dtype=np.float32)
    K2 = np.array([[[[1.0]], [[-1.0]]],
                    [[[0.5]], [[1.0]]]], dtype=np.float32)  # (2,2,1,1)
    b2 = np.zeros(2, dtype=np.float32)
    K3 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    K3[0, 0, 0, 0] = 1.0; K3[0, 1, 1, 1] = 1.0
    K3[1, 0, 1, 0] = 1.0; K3[1, 1, 0, 1] = 1.0
    b3 = np.zeros(2, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])
    inits = [
        oh.make_tensor('K1', onnx.TensorProto.FLOAT, K1.shape, K1.flatten()),
        oh.make_tensor('B1', onnx.TensorProto.FLOAT, b1.shape, b1),
        oh.make_tensor('K2', onnx.TensorProto.FLOAT, K2.shape, K2.flatten()),
        oh.make_tensor('B2', onnx.TensorProto.FLOAT, b2.shape, b2),
        oh.make_tensor('K3', onnx.TensorProto.FLOAT, K3.shape, K3.flatten()),
        oh.make_tensor('B3', onnx.TensorProto.FLOAT, b3.shape, b3),
    ]
    nodes = [
        oh.make_node('Conv', ['x', 'K1', 'B1'], ['z1'],
                     kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z1'], ['a1']),
        oh.make_node('Conv', ['a1', 'K2', 'B2'], ['z2'],
                     kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Relu', ['z2'], ['a2']),
        oh.make_node('Conv', ['a2', 'K3', 'B3'], ['z3'],
                     kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node('Flatten', ['z3'], ['y']),
    ]
    graph = oh.make_graph(nodes, 'g', [inp], [out], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    model.ir_version = 7
    p = tmp_path / 'tiny_conv.onnx'
    onnx.save(model, str(p))
    return str(p)


def _spec_with_open_disjunct():
    """Simple Y[0] >= Y[1] spec — guaranteed to leave at least one
    leaf 'open' so the α-CROWN code path actually fires."""
    return VNNSpec(
        x_lo=np.zeros((1, 1, 2, 2), dtype=np.float32),
        x_hi=np.ones((1, 1, 2, 2), dtype=np.float32),
        disjuncts=(Conjunct(constraints=(PairwiseConstraint(0, 1),)),),
    )


def _force_open_specs(monkeypatch):
    """Make the in-leaf spec_backward report all-negative lbs so the
    `if isn and any(un.values())` α-CROWN block actually fires.

    Also fake unstable bounds via `_forward_zonotope_graph` so that
    `un = {L: indices}` is non-empty (otherwise α-CROWN is skipped).
    """
    import torch as _torch
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph as orig_fwd
    from vibecheck.verify_zono_bnb import _spec_backward_graph as orig_bw

    def fake_fwd(xl, xh, gg, device, dtype):
        sb, x = orig_fwd(xl, xh, gg, device, dtype)
        # Force unstable: rewrite each layer's bounds so lo<0 and hi>0.
        for L, (lo, hi) in list(sb.items()):
            sb[L] = (-_torch.ones_like(lo), _torch.ones_like(hi))
        return sb, x

    def fake_bw(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                return_ew=False):
        if return_ew:
            return orig_bw(sb, xl, xh, gg, spec_ew, qids, nh,
                            device, dtype, return_ew=True)
        return {qi: -1.0 for qi in qids}, None

    # `_input_split_fast_leaf` does
    #   `from .verify_zono_bnb import _forward_zonotope_graph, _spec_backward_graph`
    # at call time, so we patch the source module (not verify_graph's
    # re-export, which doesn't even exist for the fast-leaf path).
    monkeypatch.setattr('vibecheck.verify_zono_bnb._forward_zonotope_graph',
                         fake_fwd)
    monkeypatch.setattr('vibecheck.verify_zono_bnb._spec_backward_graph',
                         fake_bw)
    # Disable PGD short-circuit (verify_graph still uses its own ref).
    monkeypatch.setattr('vibecheck.verify_graph._pgd_attack_general',
                         lambda *a, **k: (False, None))


def test_oom_propagates_under_threshold(monkeypatch, tmp_path):
    """When total_unstable ≤ threshold, the joint α path runs. If it
    OOMs (synthetic), the error must propagate (not be swallowed)."""
    g = load_onnx(_tiny_conv_input_split(tmp_path))
    spec = _spec_with_open_disjunct()
    s = default_settings(device='cpu', bits=64, total_timeout=10,
                         print_progress=False)
    g.optimize(s)
    _force_open_specs(monkeypatch)

    def _raise_oom(*a, **kw):
        raise torch.cuda.OutOfMemoryError('synthetic OOM joint α')
    monkeypatch.setattr('vibecheck.alpha_crown.run_alpha_crown_batched',
                         _raise_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        vg.verify_graph(g, spec, s)


def test_lightweight_path_used_above_threshold(monkeypatch, tmp_path):
    """Setting `alpha_crown_impl_auto_switch_threshold=0` forces the
    lightweight v2 path on every leaf. Joint α must NOT be called;
    lightweight α MUST be called at least once."""
    g = load_onnx(_tiny_conv_input_split(tmp_path))
    spec = _spec_with_open_disjunct()
    s = default_settings(device='cpu', bits=64, total_timeout=10,
                         print_progress=False,
                         alpha_crown_impl_auto_switch_threshold=0)
    g.optimize(s)
    _force_open_specs(monkeypatch)

    joint_called = {'n': 0}
    light_called = {'n': 0}
    orig_light = (vg.alpha_crown.run_alpha_crown_fixed_intermediate_batched
                  if hasattr(vg, 'alpha_crown')
                  else None)
    if orig_light is None:
        from vibecheck import alpha_crown as ac
        orig_light = ac.run_alpha_crown_fixed_intermediate_batched

    def stub_joint(*a, **kw):
        joint_called['n'] += 1
        raise AssertionError('joint α must not be called above threshold')

    def stub_light(*a, **kw):
        light_called['n'] += 1
        return orig_light(*a, **kw)

    monkeypatch.setattr('vibecheck.alpha_crown.run_alpha_crown_batched',
                         stub_joint)
    monkeypatch.setattr(
        'vibecheck.alpha_crown.run_alpha_crown_fixed_intermediate_batched',
        stub_light)

    vg.verify_graph(g, spec, s)
    assert joint_called['n'] == 0
    assert light_called['n'] >= 1


def test_oom_propagates_above_threshold(monkeypatch, tmp_path):
    """Even when the lightweight path is selected, an OOM there must
    also propagate — not silently fall through."""
    g = load_onnx(_tiny_conv_input_split(tmp_path))
    spec = _spec_with_open_disjunct()
    s = default_settings(device='cpu', bits=64, total_timeout=10,
                         print_progress=False,
                         alpha_crown_impl_auto_switch_threshold=0)
    g.optimize(s)
    _force_open_specs(monkeypatch)

    def _raise(*a, **kw):
        raise torch.cuda.OutOfMemoryError('synthetic OOM lightweight α')
    monkeypatch.setattr(
        'vibecheck.alpha_crown.run_alpha_crown_fixed_intermediate_batched',
        _raise)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        vg.verify_graph(g, spec, s)
