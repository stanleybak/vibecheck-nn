"""Tests for the gated PGD pre-phase on the nonlinear route (verify_graph).

On big nonlinear nets the backward-CROWN / α-CROWN bound work can consume the
whole budget before the CE search (which otherwise only runs *after* it inside
_verify_nonlinear_graph) ever starts, so short-budget SAT cases time out even
though plain PGD cracks them in seconds. `nonlinear_pre_pgd` attacks FIRST.

These build a tiny bilinear net Y_0 = X_0*(1-X_1) (routes through the nonlinear
path via the both-vary Mul) whose only counterexample sits at the mixed corner
(1,0) — NOT the box center or the two axis-aligned corners the cheap nominal
probe checks — so the SAT is found by PGD, letting us assert which phase wins.
Synthetic ONNX/VNNLIB (no benchmark clone needed).
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_graph import verify_graph


def _bilinear_net(path):
    """Y_0 = X_0 * (1 - X_1) via two Gemms feeding a both-vary Mul."""
    A0 = np.array([[1.0], [0.0]], np.float32); b0 = np.array([0.0], np.float32)
    A1 = np.array([[0.0], [-1.0]], np.float32); b1 = np.array([1.0], np.float32)
    nodes = [helper.make_node('Gemm', ['X', 'A0', 'b0'], ['h0']),
             helper.make_node('Gemm', ['X', 'A1', 'b1'], ['h1']),
             helper.make_node('Mul', ['h0', 'h1'], ['Y'])]
    inits = [numpy_helper.from_array(A0, 'A0'), numpy_helper.from_array(b0, 'b0'),
             numpy_helper.from_array(A1, 'A1'), numpy_helper.from_array(b1, 'b1')]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
    m = helper.make_model(helper.make_graph(nodes, 'bil', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9; onnx.checker.check_model(m); onnx.save(m, path)


def _spec(path):
    # X in [0,1]^2; unsafe iff Y_0 >= 0.5. max Y_0 = 1 at (1,0); center/axis
    # corners give <= 0.25 so the nominal probe misses it -> PGD must find it.
    open(path, 'w').write('\n'.join([
        '(declare-const X_0 Real)', '(declare-const X_1 Real)',
        '(declare-const Y_0 Real)',
        '(assert (<= X_0 1.0))', '(assert (>= X_0 0.0))',
        '(assert (<= X_1 1.0))', '(assert (>= X_1 0.0))',
        '(assert (>= Y_0 0.5))']) + '\n')


def _run(tmp_path, pre_pgd, disable_sat=False, print_progress=False):
    net = str(tmp_path / 'bil.onnx'); _bilinear_net(net)
    sp = str(tmp_path / 'bil.vnnlib'); _spec(sp)
    g = ComputeGraph.from_onnx(net, dtype=np.float32)
    s = default_settings(nonlinear_pre_pgd=pre_pgd, disable_sat_finding=disable_sat,
                         total_timeout=30.0, print_progress=print_progress)
    g.optimize(s)
    spec = load_vnnlib(sp, dtype=np.float32)
    return net, spec, verify_graph(g, spec, s)


def test_default_off():
    assert default_settings().nonlinear_pre_pgd is False


def test_pre_pgd_fires_and_finds_sat(tmp_path):
    """ON -> sat via the pre-PGD phase, and the witness is a REAL CE on ORT."""
    net, spec, (v, d) = _run(tmp_path, pre_pgd=True, print_progress=True)
    assert v == 'sat' and d['phase'] == 'nonlinear_pre_pgd'
    w = np.asarray(d['witness'], np.float32).reshape(1, 2)
    # in-box and output-spec violated within tolerance (mirror the scorer)
    assert (w.flatten() >= spec.x_lo - 1e-4).all()
    assert (w.flatten() <= spec.x_hi + 1e-4).all()
    sess = ort.InferenceSession(net, providers=['CPUExecutionProvider'])
    y = sess.run(None, {'X': w})[0].flatten()
    assert y[0] >= 0.5 - 1e-4


def test_pre_pgd_off_is_inert(tmp_path):
    """OFF -> the pre-PGD phase never wins (a later path still finds the SAT)."""
    _, _, (v, d) = _run(tmp_path, pre_pgd=False)
    assert v == 'sat' and d.get('phase') != 'nonlinear_pre_pgd'


def test_pre_pgd_respects_disable_sat_finding(tmp_path):
    """ON + disable_sat_finding -> no CE search at all (no false pre-PGD sat)."""
    _, _, (v, d) = _run(tmp_path, pre_pgd=True, disable_sat=True)
    assert v != 'sat' and d.get('phase') != 'nonlinear_pre_pgd'
