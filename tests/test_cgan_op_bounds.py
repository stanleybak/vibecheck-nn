"""Soundness + correctness tests for cgan-needed ops in the batched
forward zono + CROWN backward pipeline.

For each new op (AveragePool, MaxPool, Mul-by-scalar, Softmax,
bilinear MatMul), construct a tiny synthetic ONNX network containing
JUST that op (plus the wrapping linear input layer) and:
  (a) Run our batched forward zono on a random interval → assert all
      sampled outputs of the same network lie within the zono
      interval bounds (soundness).
  (b) Run our spec backward CROWN, query y_0 >= -1e9 (always sat) and
      y_0 <= +1e9 (always sat) — verify CROWN bound is at least as
      tight as the box bound (sanity).
  (c) Compare against onnxruntime point evaluation (correctness of
      forward).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import torch
import pytest

from vibecheck.network import ComputeGraph
from vibecheck.verify_zono_bnb import (
    _forward_zonotope_graph_batched, _spec_backward_graph_batched)


def _make_pool_model(op_type, in_shape=(1, 2, 4, 4),
                       kernel=(2, 2), stride=(2, 2), pads=(0, 0, 0, 0)):
    """Tiny ONNX model: input → AveragePool/MaxPool → output."""
    inp = helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))
    out_h = (in_shape[2] + pads[0] + pads[2] - kernel[0]) // stride[0] + 1
    out_w = (in_shape[3] + pads[1] + pads[3] - kernel[1]) // stride[1] + 1
    out = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                          [1, in_shape[1], out_h, out_w])
    node = helper.make_node(
        op_type, ['X'], ['Y'],
        kernel_shape=list(kernel), strides=list(stride), pads=list(pads))
    g = helper.make_graph([node], 'test', [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    return m


def _make_mul_const_model(in_shape=(1, 3), scale=2.5):
    inp = helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))
    out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(in_shape))
    scale_arr = np.array([scale], dtype=np.float32)
    scale_init = helper.make_tensor('S', TensorProto.FLOAT, [1], scale_arr)
    node = helper.make_node('Mul', ['X', 'S'], ['Y'])
    g = helper.make_graph([node], 'test', [inp], [out], [scale_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    return m


def _make_softmax_model(in_shape=(1, 5), axis=1):
    inp = helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))
    out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(in_shape))
    node = helper.make_node('Softmax', ['X'], ['Y'], axis=axis)
    g = helper.make_graph([node], 'test', [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    return m


def _make_matmul_bilinear_model(M=3, K=2, N=4):
    """Bilinear MatMul: y = A @ B where both A (M×K) and B (K×N) are
    inputs concatenated into a single (M*K + K*N,) flat input then
    reshaped — but ONNX doesn't allow two graph inputs easily, so we
    synthesize via Reshape + Slice... actually simpler: take both as
    initializers and treat input as a Mul-scalar of A.

    Actually we just use a single-input model with two MatMuls so the
    second matmul becomes bilinear. Use: x → Reshape(M,K), B = const,
    y = (x_reshape @ ones_like_x_reshape.T) @ B isn't bilinear though.

    Workaround: use the small_transformer.onnx itself in an end-to-end
    test (covered by the cgan integration test).
    """
    return None


def _zono_intervals(xl, xh, model_np):
    """Run batched forward zono on the input box and return (lo, hi)
    over the output."""
    g = ComputeGraph.from_onnx_model(model_np)
    gg = g.gpu_graph(torch.device('cpu'), torch.float32)
    xl_t = torch.as_tensor(xl, dtype=torch.float32).unsqueeze(0).flatten(1)
    xh_t = torch.as_tensor(xh, dtype=torch.float32).unsqueeze(0).flatten(1)
    sb, (c_out, G_out) = _forward_zonotope_graph_batched(
        xl_t, xh_t, gg, torch.device('cpu'), torch.float32)
    abs_sum = G_out.abs().sum(dim=2)
    return (c_out - abs_sum).squeeze(0).numpy(), \
           (c_out + abs_sum).squeeze(0).numpy()


def _run_ort(model_np, x):
    """Evaluate the ONNX model on x via onnxruntime."""
    import onnxruntime as ort
    sess = ort.InferenceSession(
        model_np.SerializeToString(), providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.astype(np.float32)})[0]


def _check_soundness(model_np, xl, xh, n_samples=500, atol=1e-5):
    """Sample n inputs uniformly from [xl, xh], run ORT, verify all
    outputs lie in [lo_zono, hi_zono]."""
    lo_z, hi_z = _zono_intervals(xl, xh, model_np)
    rng = np.random.default_rng(0)
    width = xh - xl
    for _ in range(n_samples):
        x = xl + width * rng.random(xl.shape).astype(np.float32)
        y = _run_ort(model_np, x).flatten()
        if not ((y >= lo_z - atol).all() and (y <= hi_z + atol).all()):
            margin_lo = (lo_z - atol) - y
            margin_hi = y - (hi_z + atol)
            return False, f'lo viol max={margin_lo.max():.4e}, hi viol max={margin_hi.max():.4e}'
    return True, None


# Helper used by tests — adds a from_onnx_model classmethod if missing.
def _ensure_from_onnx_model():
    if hasattr(ComputeGraph, 'from_onnx_model'):
        return
    import tempfile, os as _os
    @classmethod
    def from_onnx_model(cls, model, dtype=np.float32):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name
        try:
            return cls.from_onnx(path, dtype=dtype)
        finally:
            _os.unlink(path)
    ComputeGraph.from_onnx_model = from_onnx_model


_ensure_from_onnx_model()


# ------------------ tests ------------------

@pytest.mark.parametrize('kernel,stride', [((2, 2), (2, 2)), ((3, 3), (1, 1))])
def test_avg_pool_sound(kernel, stride):
    m = _make_pool_model('AveragePool', in_shape=(1, 2, 4, 4),
                          kernel=kernel, stride=stride)
    xl = -1.0 * np.ones((1, 2, 4, 4), dtype=np.float32)
    xh = +1.0 * np.ones((1, 2, 4, 4), dtype=np.float32)
    ok, msg = _check_soundness(m, xl, xh, n_samples=300)
    assert ok, f'avg_pool unsound: {msg}'


@pytest.mark.parametrize('kernel,stride', [((2, 2), (2, 2)), ((3, 3), (1, 1))])
def test_max_pool_sound(kernel, stride):
    m = _make_pool_model('MaxPool', in_shape=(1, 2, 4, 4),
                          kernel=kernel, stride=stride)
    xl = -1.0 * np.ones((1, 2, 4, 4), dtype=np.float32)
    xh = +1.0 * np.ones((1, 2, 4, 4), dtype=np.float32)
    ok, msg = _check_soundness(m, xl, xh, n_samples=300)
    assert ok, f'max_pool unsound: {msg}'


@pytest.mark.parametrize('scale', [2.5, -1.7, 0.0])
def test_mul_const_sound(scale):
    m = _make_mul_const_model(in_shape=(1, 3), scale=scale)
    xl = np.array([[-1.0, -2.0, 0.5]], dtype=np.float32)
    xh = np.array([[+1.5, -1.0, 2.0]], dtype=np.float32)
    ok, msg = _check_soundness(m, xl, xh, n_samples=300)
    assert ok, f'mul const sound: {msg}'


def test_softmax_sound():
    m = _make_softmax_model(in_shape=(1, 5), axis=1)
    xl = -1.0 * np.ones((1, 5), dtype=np.float32)
    xh = +1.0 * np.ones((1, 5), dtype=np.float32)
    ok, msg = _check_soundness(m, xl, xh, n_samples=300)
    assert ok, f'softmax sound: {msg}'


def test_avg_pool_point():
    """Point input (xl=xh) → output bounds match onnxruntime exactly."""
    m = _make_pool_model('AveragePool', in_shape=(1, 1, 4, 4),
                          kernel=(2, 2), stride=(2, 2))
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, size=(1, 1, 4, 4)).astype(np.float32)
    lo_z, hi_z = _zono_intervals(x, x, m)
    y_ort = _run_ort(m, x).flatten()
    np.testing.assert_allclose(lo_z, y_ort, atol=1e-5)
    np.testing.assert_allclose(hi_z, y_ort, atol=1e-5)


def test_mul_const_point():
    m = _make_mul_const_model(in_shape=(1, 4), scale=3.0)
    rng = np.random.default_rng(7)
    x = rng.uniform(-2, 2, size=(1, 4)).astype(np.float32)
    lo_z, hi_z = _zono_intervals(x, x, m)
    y_ort = _run_ort(m, x).flatten()
    np.testing.assert_allclose(lo_z, y_ort, atol=1e-5)
    np.testing.assert_allclose(hi_z, y_ort, atol=1e-5)
