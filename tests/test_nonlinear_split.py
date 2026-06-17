"""End-to-end soundness of the nonlinear-split BaB (_verify_trig_nonlinear_split).

Builds a tiny net that routes to the ACOPF trig verifier (has a Pow → enters the
non-LP block, and a Sin → _route_acopf) and FORCES the nonlinear-split path via
trig_bab_max_var=0. Asserts the verdict is always SOUND:
  - a SAT spec (counterexample exists) must NOT return 'verified';
  - a clearly-safe spec must NOT return 'sat'.
The clamp mechanism itself is unit-tested in test_op_clamp_soundness.py.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.spec import VNNSpec, Conjunct, Constraint
from vibecheck.verify_graph import verify_graph

NIN = 4


def _build(path):
    rng = np.random.default_rng(0)
    W1 = (0.5 * rng.standard_normal((3, NIN))).astype(np.float32)
    b1 = (0.2 * rng.standard_normal(3)).astype(np.float32)
    W3 = (0.5 * rng.standard_normal((1, 6))).astype(np.float32)
    b3 = np.zeros(1, np.float32)
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['h'], alpha=1.0,
                         beta=1.0, transB=1),
        helper.make_node('Pow', ['h', 'two'], ['p']),
        helper.make_node('Sin', ['h'], ['s']),
        helper.make_node('Concat', ['p', 's'], ['ps'], axis=1),
        helper.make_node('Gemm', ['ps', 'W3', 'b3'], ['Y'], alpha=1.0,
                         beta=1.0, transB=1),
    ]
    inits = [
        helper.make_tensor('W1', TensorProto.FLOAT, W1.shape, W1.flatten()),
        helper.make_tensor('b1', TensorProto.FLOAT, b1.shape, b1.flatten()),
        helper.make_tensor('two', TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor('W3', TensorProto.FLOAT, W3.shape, W3.flatten()),
        helper.make_tensor('b3', TensorProto.FLOAT, b3.shape, b3.flatten()),
    ]
    g = helper.make_graph(
        nodes, 'nlsplit_tiny',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, NIN])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)


def _run(path, T, tmp_path):
    g = ComputeGraph.from_onnx(path, dtype=np.float64)
    s = default_settings(device='cpu', bits=64)
    s.total_timeout = 30.0
    s.trig_bab_max_var = 0      # force the nonlinear-split path (not input-split)
    g.optimize(s)
    xl = np.full(NIN, -1.0, np.float64); xh = np.full(NIN, 1.0, np.float64)
    spec = VNNSpec(xl, xh, [Conjunct([Constraint(0, '>=', float(T))])])
    return verify_graph(g, spec, s)


def _true_max(path):
    import onnxruntime as ort
    m = onnx.load(path)
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'N'
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(7)
    xs = (-1.0 + 2.0 * rng.random((100000, NIN))).astype(np.float32)
    return float(sess.run(None, {sess.get_inputs()[0].name: xs})[0].max())


@pytest.mark.parametrize('seed_path', ['nlsplit_tiny.onnx'])
def test_nonlinear_split_verdicts_sound(tmp_path, seed_path):
    path = str(tmp_path / seed_path)
    _build(path)
    ymax = _true_max(path)
    # SAT: threshold well below a sampled attainable value -> NOT verified.
    v_sat, _ = _run(path, ymax - 0.5, tmp_path)
    assert v_sat != 'verified', f'false verify on SAT spec (got {v_sat})'
    # clearly-safe: threshold far above any attainable output -> NOT sat.
    v_safe, _ = _run(path, ymax + 1e4, tmp_path)
    assert v_safe != 'sat', f'false sat on safe spec (got {v_safe})'
