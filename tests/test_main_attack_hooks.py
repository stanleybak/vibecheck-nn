"""Coverage for the incomplete-attack hooks in main.py: _maybe_sign_attack (Sign-BNN) and
_maybe_torch_attack (generic onnx2torch PGD). Reuses the tiny synthetic ONNX builders from
the per-module unit tests."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling test helpers
from vibecheck import main as vbmain
from test_sign_attack import _bnn_onnx, _plain_onnx, _vnnlib
from test_torch_attack import _net


def _args(**kw):
    d = dict(net=None, spec=None, config=None, results_file=None, timeout=30,
             verbose=False, dtype='float32')
    d.update(kw)
    return argparse.Namespace(**d)


def _cfg(tmp_path, body):
    p = tmp_path / 'a.yaml'
    p.write_text(body)
    return str(p)


# ---- _maybe_sign_attack ----

def test_sign_hook_no_config():
    assert vbmain._maybe_sign_attack(_args(config=None), {'emitted': False}) is None


def test_sign_hook_flag_off(tmp_path):
    assert vbmain._maybe_sign_attack(
        _args(config=_cfg(tmp_path, 'sign_attack: false\n'), net='x'), {'emitted': False}) is None


def test_sign_hook_non_sign_net(tmp_path):
    q = _plain_onnx(str(tmp_path / 'p.onnx'))
    assert vbmain._maybe_sign_attack(
        _args(config=_cfg(tmp_path, 'sign_attack: true\n'), net=q), {'emitted': False}) is None


def test_sign_hook_sat(tmp_path):
    q = _bnn_onnx(str(tmp_path / 'b.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    rf = str(tmp_path / 'r.txt')
    cfg = _cfg(tmp_path, 'sign_attack: true\nsign_attack_restarts: 4\nsign_attack_steps: 40\n'
                         'device: cpu\n')
    code = vbmain._maybe_sign_attack(_args(config=cfg, net=q, spec=v, results_file=rf),
                                     {'emitted': False})
    assert code == 1 and open(rf).read().startswith('sat')


# ---- _maybe_torch_attack ----

def test_torch_hook_no_config():
    assert vbmain._maybe_torch_attack(_args(config=None), {'emitted': False}) is None


def test_torch_hook_flag_off(tmp_path):
    assert vbmain._maybe_torch_attack(
        _args(config=_cfg(tmp_path, 'torch_attack: false\n'), net='x'), {'emitted': False}) is None


def test_torch_hook_sat(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    rf = str(tmp_path / 'r.txt')
    cfg = _cfg(tmp_path, 'torch_attack: true\ntorch_attack_restarts: 6\ntorch_attack_steps: 40\n'
                         'device: cpu\n')
    code = vbmain._maybe_torch_attack(_args(config=cfg, net=q, spec=v, results_file=rf),
                                      {'emitted': False})
    assert code == 1 and open(rf).read().startswith('sat')
