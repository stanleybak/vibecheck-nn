"""vit_2023 gpu_graph point-prop exactness vs onnxruntime.

The vit groundwork (bench/vit_2023) serializes the transformer into
canonical gg ops: patch-Conv, CLS-token Concat → placement-fc, Transpose →
slice (flat permutation), per-token MatMul → kron(I_T, W) fc, BatchNorm →
mul+add, ReduceMean → averaging-fc, plus the existing softmax /
matmul_bilinear forward ops. This test pins that the composed gg forward
computes EXACTLY the network function (the basic per-node DenseZonotope
path does NOT yet — vit stays in test_vnncomp_nets._HARD_EXTENDED).

Skipped when the vnncomp benchmark checkout (tests/paths.yaml) is absent.
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


@pytest.mark.vnncomp
@pytest.mark.skipif(_VIT is None, reason='vnncomp vit_2023 benchmark not available')
@pytest.mark.parametrize('net', ['ibp_3_3_8', 'pgd_2_3_16'])
def test_vit_gg_forward_matches_onnxruntime(net):
    import onnxruntime as ort
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    from vibecheck.verify_zono_bnb import _forward_batch_graph

    onnx_path = os.path.join(_VIT, 'onnx', f'{net}.onnx')
    if not os.path.exists(onnx_path):
        onnx_path += '.gz'
    g = load_onnx(onnx_path)
    g.optimize(default_settings())
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (4, 3 * 32 * 32))
    y = _forward_batch_graph(torch.tensor(x, dtype=torch.float64), gg)

    sess = ort.InferenceSession(onnx_path if not onnx_path.endswith('.gz')
                                else onnx_path[:-3],
                                providers=['CPUExecutionProvider'])
    yo = np.stack([
        sess.run(None, {sess.get_inputs()[0].name:
                        x[i].reshape(1, 3, 32, 32).astype(np.float32)}
                 )[0].flatten()
        for i in range(4)])
    err = np.abs(y.numpy() - yo).max()
    assert err < 1e-4, f'{net}: gg forward differs from onnxruntime by {err}'
