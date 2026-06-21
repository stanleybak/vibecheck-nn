"""Unit coverage for the adaptive_cruise integration pieces:
  * verify_graph._resolve_device — device follows settings.device (GPU default,
    ml4acopf CPU override); the trig SAT phase + PGD honor it
  * main._maybe_nonlinear_augment / _counterexample_sexpr_orig — the CLI hook +
    original-net counterexample emission
  * pgd.pgd_attack_general — deterministic seeding + the verbose gap/restart log
all on a tiny synthetic 2-in/1-out net + a nonlinear v2 spec.
"""
import types

import numpy as np
import onnx
from onnx import helper, TensorProto
import torch

from vibecheck import nonlinear_augment as nla
from vibecheck import main as vmain
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_graph import _resolve_device
from vibecheck.pgd import pgd_attack_general
from vibecheck.settings import default_settings


def _tiny_net(path):
    W = helper.make_tensor('W', TensorProto.FLOAT, [1, 2], [1.0, 1.0])
    b = helper.make_tensor('b', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)
    g = helper.make_graph(
        [node], 'f',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, path)


# X-box [0,1]^2; clause0: X0*X1 <= 0.5 trivially satisfiable (X=[0,0]) -> a CE
# exists, so PGD must find it.
_V2 = """
(vnnlib-version <2.0>)
(declare-network f (declare-input X float32 [1,2]) (declare-output Y float32 [1,1]))
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))
(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))
(assert (or (<= (* X[0,0] X[0,1]) 0.5) (< Y[0,0] -100.0)))
"""


def _aug(tmp_path):
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    sp = str(tmp_path / 's.vnnlib')
    open(sp, 'w').write(_V2)
    aug_onnx, aug_vnnlib = nla.build_augmented_instance(net, sp)
    spec = load_vnnlib(aug_vnnlib)
    graph = ComputeGraph.from_onnx(aug_onnx, dtype=np.float32)
    gg = graph.gpu_graph(torch.device('cpu'), torch.float32)
    xl = torch.tensor(spec.x_lo.astype(np.float32))
    xh = torch.tensor(spec.x_hi.astype(np.float32))
    return net, aug_onnx, aug_vnnlib, spec, gg, xl, xh


def test_resolve_device():
    # explicit CPU (ml4acopf override) -> cpu
    assert _resolve_device(default_settings(device='cpu')) == 'cpu'
    # GPU default -> cuda iff available, else cpu
    want = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert _resolve_device(default_settings(device='gpu')) == want
    # None settings -> defaults to the gpu preference
    assert _resolve_device(None) == want


def test_pgd_seed_deterministic(tmp_path):
    """Same seed => identical witness; the attack is reproducible for tuning."""
    _net, _ao, _av, spec, gg, xl, xh = _aug(tmp_path)
    s = default_settings(device='cpu', bits=32)
    s.pgd_restarts = 16
    s.pgd_iter = 30
    sat1, w1 = pgd_attack_general(xl, xh, spec, gg, s, seed=0)
    sat2, w2 = pgd_attack_general(xl, xh, spec, gg, s, seed=0)
    assert sat1 and sat2
    np.testing.assert_array_equal(np.asarray(w1), np.asarray(w2))


def test_maybe_nonlinear_augment_hook(tmp_path):
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    sp = str(tmp_path / 's.vnnlib')
    open(sp, 'w').write(_V2)
    args = types.SimpleNamespace(net=net, spec=sp)
    vmain._maybe_nonlinear_augment(args)
    assert args.net != net and args.net.endswith('.onnx')      # swapped
    assert getattr(args, 'orig_net_for_cex') == net            # original stashed

    # no-op on a linear v1 spec
    sp1 = str(tmp_path / 'v1.vnnlib')
    open(sp1, 'w').write('(declare-const X_0 Real)\n(declare-const Y_0 Real)\n'
                         '(assert (<= X_0 1.0))\n(assert (<= Y_0 0.0))\n')
    args1 = types.SimpleNamespace(net=net, spec=sp1)
    vmain._maybe_nonlinear_augment(args1)
    assert args1.net == net and not hasattr(args1, 'orig_net_for_cex')

    # no-op on a missing spec file (OSError caught)
    args2 = types.SimpleNamespace(net=net, spec=str(tmp_path / 'nope.vnnlib'))
    vmain._maybe_nonlinear_augment(args2)
    assert args2.net == net


def test_counterexample_sexpr_orig(tmp_path):
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    ce = vmain._counterexample_sexpr_orig(net, np.array([0.3, 0.7]), '.9g')
    # f(X)=X0+X1=1.0; cex carries original X (2) + original Y (1)
    assert '(X_0 0.3)' in ce and '(X_1 0.7)' in ce and '(Y_0 1)' in ce


def test_emit_result_uses_original_net_cex(tmp_path):
    """_emit_result with orig_net_for_cex set writes a cex with the ORIGINAL net's
    output (Y_0), not the augmented net's polynomial outputs."""
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    rf = str(tmp_path / 'res.txt')
    args = types.SimpleNamespace(net='unused.onnx', results_file=rf,
                                 orig_net_for_cex=net)
    sat_state = {'emitted': False}
    vmain._emit_result(args, None, 'sat', np.array([0.3, 0.7]), sat_state, '.9g')
    txt = open(rf).read()
    assert txt.splitlines()[0] == 'sat'
    assert '(X_0 0.3)' in txt and '(Y_0 1)' in txt
    assert sat_state['emitted'] is True


def test_pgd_time_budget_break(tmp_path):
    """A tiny time_budget aborts the iter loop early (covers the budget break)."""
    _net, _ao, _av, spec, gg, xl, xh = _aug(tmp_path)
    s = default_settings(device='cpu', bits=32)
    s.pgd_restarts = 4
    s.pgd_iter = 100000          # would run forever without the budget
    sat, _w = pgd_attack_general(xl, xh, spec, gg, s, time_budget=1e-6)
    assert sat in (True, False)  # returns promptly, no hang


def test_pgd_verbose_gap_log(tmp_path, capsys):
    """The no-CE verbose branch prints the restart/iter/gap diagnostic."""
    _net, _ao, _av, _spec, gg, xl, xh = _aug(tmp_path)
    # a trivially-unsatisfiable spec (both atoms need a value <= -1000, never
    # reachable on f=X0+X1 with X in [0,1]) -> PGD finds no CE -> verbose branch
    sp = str(tmp_path / 'u.vnnlib')
    open(sp, 'w').write(
        "(vnnlib-version <2.0>)\n(declare-network f "
        "(declare-input X float32 [1,2]) (declare-output Y float32 [1,1]))\n"
        "(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))\n"
        "(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))\n"
        "(assert (or (<= (* X[0,0] X[0,1]) -1000.0) (< Y[0,0] -1000.0)))\n")
    aug_onnx, aug_vnnlib = nla.build_augmented_instance(_net, sp)
    uspec = load_vnnlib(aug_vnnlib)
    g2 = ComputeGraph.from_onnx(aug_onnx, dtype=np.float32)
    gg2 = g2.gpu_graph(torch.device('cpu'), torch.float32)
    s = default_settings(device='cpu', bits=32)
    s.print_progress = True
    s.pgd_restarts = 8
    s.pgd_iter = 5
    sat, w = pgd_attack_general(xl, xh, uspec, gg2, s)
    assert sat is False and w is None
    out = capsys.readouterr().out
    assert '[pgd] no CE' in out and 'gap(best_margin)=' in out
