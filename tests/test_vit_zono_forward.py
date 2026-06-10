"""Sound forward zono bounds through vit attention (matmul_bilinear+softmax).

The v1 handlers produce interval (box) results for the bilinear matmuls and
softmax — sound but correlation-dropping. Pins:
  1. point box (lo == hi): the zono forward must reproduce the exact network
     output (all interval products/softmax collapse to points)
  2. eps box: 200 random points inside the box must have outputs INSIDE the
     zono output bounds (sampling-based soundness check — the test, not the
     bound)
Skipped when the vnncomp vit_2023 checkout is absent.
"""
import os

import numpy as np
import pytest
import torch
import yaml


def _vit_dir():
    here = os.path.dirname(__file__)
    paths = os.path.join(here, 'paths.yaml')
    if not os.path.exists(paths):
        return None
    with open(paths) as f:
        base = yaml.safe_load(f).get('vnncomp_benchmarks')
    if not base:
        return None
    d = os.path.join(base, 'benchmarks', 'vit_2023')
    return d if os.path.isdir(d) else None


_VIT = _vit_dir()


def _load(net):
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    p = os.path.join(_VIT, 'onnx', f'{net}.onnx')
    g = load_onnx(p)
    g.optimize(default_settings())
    return g.gpu_graph(torch.device('cpu'), torch.float64)


@pytest.mark.vnncomp
@pytest.mark.skipif(_VIT is None, reason='vnncomp vit_2023 not available')
@pytest.mark.parametrize('net', ['ibp_3_3_8', 'pgd_2_3_16'])
def test_point_box_exact(net):
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    gg = _load(net)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 3 * 32 * 32)
    xt = torch.tensor(x, dtype=torch.float64)
    sb, z_final = _forward_zonotope_graph(
        xt.clone(), xt.clone(), gg, torch.device('cpu'), torch.float64)
    lo, hi = z_final.bounds()
    y = _forward_batch_graph(xt.unsqueeze(0), gg).flatten()
    np.testing.assert_allclose(lo.numpy(), y.numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hi.numpy(), y.numpy(), rtol=1e-8, atol=1e-8)


@pytest.mark.vnncomp
@pytest.mark.skipif(_VIT is None, reason='vnncomp vit_2023 not available')
@pytest.mark.parametrize('net', ['ibp_3_3_8', 'pgd_2_3_16'])
def test_eps_box_sound_by_sampling(net):
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _forward_batch_graph)
    gg = _load(net)
    rng = np.random.default_rng(1)
    xc = rng.uniform(0.3, 0.7, 3 * 32 * 32)
    eps = 1.0 / 255.0
    xl = torch.tensor(np.clip(xc - eps, 0, 1), dtype=torch.float64)
    xh = torch.tensor(np.clip(xc + eps, 0, 1), dtype=torch.float64)
    sb, z_final = _forward_zonotope_graph(
        xl, xh, gg, torch.device('cpu'), torch.float64)
    lo, hi = z_final.bounds()
    xs = torch.tensor(
        rng.uniform(xl.numpy(), xh.numpy(), (200, xl.numel())),
        dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    viol_lo = (ys < lo.unsqueeze(0) - 1e-9).any().item()
    viol_hi = (ys > hi.unsqueeze(0) + 1e-9).any().item()
    assert not viol_lo and not viol_hi, (
        f'UNSOUND: sampled outputs escape zono bounds '
        f'(lo viol={viol_lo}, hi viol={viol_hi})')
