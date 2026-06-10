"""Pins for the vit-v1 fallback mechanisms (policy: implemented = tested).

Toy: f(x) = relu(x) + (-0.4 x + 0.05) over x in [-1, 1] via a fork
(Relu branch + affine branch merged by Add). True min = 0.05 > 0, but the
min-area zono ReLU (lambda* = 0.5) gives lb = -|0.5 - 0.4| + 0.05 = -0.05:
NOT closed. Two independent mechanisms must close it:
  1. _zono_alpha_close: lambda = 0.4 gives lb = +0.05 (the optimizer must
     find it)
  2. _zono_input_split_close: splitting at x = 0 makes the ReLU stable on
     both halves (exact arms, lb >= 0.05 each)
Plus the end-to-end pipeline with phase1_method='plain' +
zono_input_split_enabled (the vit config shape) must return 'verified'.
"""
import os
import tempfile

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                              arr.flatten())


def _toy_graph(tmpdir):
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    W1 = np.array([[1.0]], dtype=np.float32)      # x -> pre-relu
    Wl = np.array([[-0.4]], dtype=np.float32)     # x -> affine branch
    bl = np.array([0.05], dtype=np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['m1']),
        helper.make_node('Relu', ['m1'], ['r1']),
        helper.make_node('MatMul', ['X', 'Wl'], ['m2']),
        helper.make_node('Add', ['m2', 'bl'], ['a2']),
        helper.make_node('Add', ['r1', 'a2'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'toy_fork',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [_init('W1', W1), _init('Wl', Wl), _init('bl', bl)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = os.path.join(tmpdir, 'toy.onnx')
    onnx.save(model, path)
    graph = load_onnx(path, dtype=np.float64)
    graph.optimize(default_settings())
    return graph


def _gg(graph):
    return graph.gpu_graph(torch.device('cpu'), torch.float64)


def _baseline_lb(gg):
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    xl = torch.tensor([-1.0], dtype=torch.float64)
    xh = torch.tensor([1.0], dtype=torch.float64)
    _, zf = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                    torch.float64)
    w = torch.tensor([1.0], dtype=torch.float64)
    return float(w @ zf.center - (w @ zf.generators).abs().sum())


def test_min_area_baseline_is_open(tmp_path):
    gg = _gg(_toy_graph(str(tmp_path)))
    lb = _baseline_lb(gg)
    assert lb < 0, f'toy mis-designed: min-area lb {lb} already closed'
    assert lb == pytest.approx(-0.05, abs=1e-6)


def test_zono_alpha_close_finds_lambda(tmp_path):
    from vibecheck.verify_graph import _zono_alpha_close
    gg = _gg(_toy_graph(str(tmp_path)))
    xl = torch.tensor([-1.0], dtype=torch.float64)
    xh = torch.tensor([1.0], dtype=torch.float64)
    ok = _zono_alpha_close(
        gg, xl, xh, np.array([1.0]), 0.0, torch.device('cpu'),
        torch.float64, None, lambda: 100.0, n_iters=200, lr=0.05)
    assert ok, 'alpha-zono failed to close a query closable at lambda=0.4'


def test_zono_input_split_close(tmp_path):
    from vibecheck.verify_graph import _zono_input_split_close
    gg = _gg(_toy_graph(str(tmp_path)))
    xl = torch.tensor([-1.0], dtype=torch.float64)
    xh = torch.tensor([1.0], dtype=torch.float64)
    ok, nodes = _zono_input_split_close(
        gg, xl, xh, np.array([1.0]), 0.0, torch.device('cpu'),
        torch.float64, None, lambda: 100.0, max_nodes=64)
    assert ok, 'input-split failed to close (stable arms after x=0 split)'
    assert nodes <= 16, f'used {nodes} boxes for a depth-1 problem'


def test_pipeline_plain_phase1_verifies(tmp_path):
    """End-to-end: phase1_method='plain' + forward-zono spec margins +
    the alpha/input-split fallback returns 'verified' on the toy."""
    from vibecheck.verify_graph import verify_graph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.settings import default_settings
    graph = _toy_graph(str(tmp_path))
    spec_path = os.path.join(str(tmp_path), 'toy.vnnlib')
    with open(spec_path, 'w') as f:
        f.write("""
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (<= Y_0 0.0))
""")
    spec = load_vnnlib(spec_path, dtype=np.float64)
    s = default_settings(
        phase1_method='plain', phase2_crown_enabled=False,
        zono_lift_enabled=False, max_tighten_layer=0,
        bab_refine_passes=0, skip_phase8_milp=True,
        zono_input_split_enabled=True, parallel_pgd_enabled=False,
        input_split_enabled=False,
        total_timeout=60.0, device='cpu')
    result, details = verify_graph(graph, spec, s)
    assert result == 'verified', f'got {result}'
