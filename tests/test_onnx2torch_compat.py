"""`convert_onnx_to_torch`: onnx2torch with an opset upgrade so OLD-opset conv
nets (e.g. vgg16-7.onnx, opset 8, Flatten-v1) load instead of raising
NotImplementedError. The contract: the resulting torch forward matches ORT (the
authoritative model) — the converted module is only the attack's gradient oracle.
"""
import os
import tempfile

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx2torch")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper


def _flatten_gemm_onnx(opset):
    """x[1,2,2] -> Flatten(axis=1) -> [1,4] -> Gemm(W[3,4],transB) -> y[1,3].
    At opset 8 the Flatten is version-1 (what onnx2torch lacks a raw converter for)."""
    W = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4) / 11, name='W')
    b = numpy_helper.from_array(np.array([0.1, -0.2, 0.3], np.float32), name='b')
    inp = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 2])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    nodes = [helper.make_node('Flatten', ['x'], ['f'], axis=1),
             helper.make_node('Gemm', ['f', 'W', 'b'], ['y'], transB=1)]
    g = helper.make_graph(nodes, 'm', [inp], [out], initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', opset)])
    m.ir_version = 4
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    f.write(m.SerializeToString()); f.close()
    return f.name


def _ort(path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0].ravel()


def test_old_opset_flatten_converts_and_matches_ort():
    import torch
    from vibecheck.surrogate_pgd import convert_onnx_to_torch
    p = _flatten_gemm_onnx(opset=8)
    model = convert_onnx_to_torch(p).eval()          # must NOT raise on opset-8 Flatten-v1
    x = np.random.RandomState(0).rand(1, 2, 2).astype(np.float32)
    with torch.no_grad():
        yt = model(torch.tensor(x)).numpy().ravel()
    yo = _ort(p, x)
    assert np.allclose(yt, yo, atol=1e-5), (yt, yo)
    os.unlink(p)


def test_modern_opset_passthrough_matches_ort():
    """A model already >= opset 13 is converted unchanged (upgrade branch skipped)
    and still matches ORT."""
    import torch
    from vibecheck.surrogate_pgd import convert_onnx_to_torch
    p = _flatten_gemm_onnx(opset=13)
    model = convert_onnx_to_torch(p).eval()
    x = np.random.RandomState(1).rand(1, 2, 2).astype(np.float32)
    with torch.no_grad():
        yt = model(torch.tensor(x)).numpy().ravel()
    assert np.allclose(yt, _ort(p, x), atol=1e-5)
    os.unlink(p)
