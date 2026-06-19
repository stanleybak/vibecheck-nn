"""Coverage for the surrogate-attack hooks in main.py (_maybe_surrogate_attack,
_surrogate_path, _emit_surrogate_result, the _verify hook, and the --build-surrogate
CLI handler). Reuses the tiny synthetic quantized ONNX from test_surrogate_pgd."""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling test helpers
from vibecheck import main as vbmain
from test_surrogate_pgd import _plain_onnx, _quant_onnx, _v1_spec


def _args(**kw):
    d = dict(net=None, spec=None, config=None, results_file=None, timeout=30,
             verbose=False, dtype='float32')
    d.update(kw)
    return argparse.Namespace(**d)


def _cfg(tmp_path, on=True):
    p = tmp_path / 'st.yaml'
    p.write_text(f'surrogate_attack: {"true" if on else "false"}\n'
                 'surrogate_attack_restarts: 2\nsurrogate_attack_steps: 25\n')
    return str(p)


def test_maybe_surrogate_no_config():
    assert vbmain._maybe_surrogate_attack(_args(config=None), {'emitted': False}) is None


def test_maybe_surrogate_flag_off(tmp_path):
    assert vbmain._maybe_surrogate_attack(
        _args(config=_cfg(tmp_path, on=False), net='x'), {'emitted': False}) is None


def test_maybe_surrogate_non_quantized(tmp_path):
    q = _plain_onnx(str(tmp_path / 'p.onnx'))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'))
    assert vbmain._maybe_surrogate_attack(
        _args(config=_cfg(tmp_path), net=q, spec=v), {'emitted': False}) is None


def test_surrogate_path_deterministic():
    a = vbmain._surrogate_path('/x/y.onnx')
    assert a == vbmain._surrogate_path('/x/y.onnx')
    assert a.endswith('.onnx') and 'surrogate' in a


def test_emit_surrogate_unknown_and_never_downgrade(tmp_path):
    rf = str(tmp_path / 'r.txt')
    vbmain._emit_surrogate_result(_args(net='x', results_file=rf), 'unknown', None,
                                  {'emitted': False})
    assert open(rf).read().strip() == 'unknown'
    # never-downgrade: a later non-sat verdict with emitted=True must NOT overwrite
    vbmain._emit_surrogate_result(_args(net='x', results_file=rf), 'timeout', None,
                                  {'emitted': True})
    assert open(rf).read().strip() == 'unknown'


def test_verify_surrogate_hook_sat(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.5)
    rf = str(tmp_path / 'r.txt')
    args = _args(config=_cfg(tmp_path), net=q, spec=v, results_file=rf)
    with pytest.raises(SystemExit) as e:
        vbmain._verify(args)
    assert e.value.code == 1
    assert open(rf).read().startswith('sat')


def test_build_surrogate_cli(tmp_path, monkeypatch):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    monkeypatch.setattr(sys, 'argv',
                        ['vibecheck', '--net', q, '--spec', 'x', '--build-surrogate'])
    with pytest.raises(SystemExit) as e:
        vbmain.main()
    assert e.value.code == 0
    assert os.path.exists(vbmain._surrogate_path(q))


def test_build_surrogate_cli_nonquant(tmp_path, monkeypatch):
    p = _plain_onnx(str(tmp_path / 'p.onnx'))
    monkeypatch.setattr(sys, 'argv',
                        ['vibecheck', '--net', p, '--spec', 'x', '--build-surrogate'])
    with pytest.raises(SystemExit) as e:
        vbmain.main()
    assert e.value.code == 0
