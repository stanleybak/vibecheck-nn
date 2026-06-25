"""Coverage for the surrogate-attack hooks in main.py (_maybe_surrogate_attack,
_surrogate_path, _emit_surrogate_result, the _verify hook, and the --prepare-pkl
CLI handler — quantized builds surrogates, non-quantized writes the pre-parse cache).
Reuses the tiny synthetic quantized ONNX from test_surrogate_pgd."""
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


def test_maybe_surrogate_missing_spec_raises(tmp_path):
    # A missing spec file must raise LOUDLY (the surrogate path does not exist to silently
    # swallow it as timeout/unknown) -> main's crash handler records 'error', exit 2.
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    with pytest.raises(FileNotFoundError):
        vbmain._maybe_surrogate_attack(
            _args(config=_cfg(tmp_path), net=q, spec=str(tmp_path / 'nope.vnnlib'), timeout=10),
            {'emitted': False})


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


def test_resolve_cex_version(tmp_path):
    assert vbmain._resolve_cex_version('1', 'x') == '1.0'
    assert vbmain._resolve_cex_version('v2', 'x') == '2.0'
    v1 = _v1_spec(str(tmp_path / 'v1.vnnlib'))
    v2 = str(tmp_path / 'v2.vnnlib')
    open(v2, 'w').write('(vnnlib-version <2.0>)\n(declare-network f '
                        '(declare-input X real [1,2]) (declare-output Y real [1,1]))\n'
                        '(assert (> Y[0,0] 0.5))\n')
    assert vbmain._resolve_cex_version('auto', v1) == '1.0'   # detect v1
    assert vbmain._resolve_cex_version('auto', v2) == '2.0'   # detect v2
    # only a .gz on disk, referenced by the plain instances.csv name -> still detected
    import gzip
    with gzip.open(v2 + '.gz', 'wt') as f:
        f.write(open(v2).read())
    os.remove(v2)
    assert vbmain._resolve_cex_version('auto', v2) == '2.0'


def test_format_cex_v1_v2(tmp_path):
    import numpy as np
    q = _quant_onnx(str(tmp_path / 'q.onnx'))                 # input 'X' [1,2], output 'Y' [1,1]
    x, y = np.array([0.7, 0.3]), np.array([0.6])
    v1 = vbmain._format_cex('1.0', q, x, y, '.6g')
    assert v1.startswith('((X_0 ') and '(Y_0 ' in v1
    v2 = vbmain._format_cex('2.0', q, x, y, '.6g').splitlines()
    assert v2[0] == 'X float32 [1,2]' and v2[1:3] == ['0.7', '0.3']
    assert v2[3] == 'Y float32 [1,1]' and v2[4] == '0.6'


def test_emit_surrogate_result_v2(tmp_path):
    import numpy as np
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    rf = str(tmp_path / 'r.txt')
    wit = [np.array([[0.5, 0.5]], np.float32)]               # in-box witness
    vbmain._emit_surrogate_result(_args(net=q, results_file=rf, cex_version='2.0'),
                                  'sat', wit, {'emitted': False}, cex_fmt='.6g')
    txt = open(rf).read()
    assert txt.startswith('sat\n')
    assert 'X float32 [1,2]' in txt and 'Y float32 [1,1]' in txt


def test_verify_surrogate_hook_sat(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.5)
    rf = str(tmp_path / 'r.txt')
    args = _args(config=_cfg(tmp_path), net=q, spec=v, results_file=rf)
    with pytest.raises(SystemExit) as e:
        vbmain._verify(args)
    assert e.value.code == 1
    assert open(rf).read().startswith('sat')


def test_prepare_pkl_cli_quant(tmp_path, monkeypatch):
    # --prepare-pkl on a QUANTIZED net folds the float (STE) + fake-quant surrogates
    # (the graph pre-parse is skipped — that net uses the surrogate-attack path).
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    monkeypatch.setattr(sys, 'argv',
                        ['vibecheck', '--net', q, '--spec', 'x', '--prepare-pkl'])
    with pytest.raises(SystemExit) as e:
        vbmain.main()
    assert e.value.code == 0
    p = vbmain._surrogate_path(q)
    assert os.path.exists(p)                       # float (STE) surrogate
    assert os.path.exists(p[:-5] + '_fq.onnx')     # fake-quant eval surrogate (Path B)


def test_prepare_pkl_cli_nonquant(tmp_path, monkeypatch):
    # --prepare-pkl on a normal net writes the pre-parse .pkl cache. Needs a
    # graph-parseable spec (non-strict >= on the output; the surrogate-path
    # _v1_spec's strict > is not accepted by the graph load_vnnlib).
    p = _plain_onnx(str(tmp_path / 'p.onnx'))
    v = str(tmp_path / 'v.vnnlib')
    open(v, 'w').write(
        '(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n'
        '(assert (<= X_0 1.0))\n(assert (>= X_0 0.0))\n'
        '(assert (<= X_1 1.0))\n(assert (>= X_1 0.0))\n'
        '(assert (>= Y_0 0.5))\n')
    monkeypatch.setattr(sys, 'argv',
                        ['vibecheck', '--net', p, '--spec', v, '--prepare-pkl'])
    with pytest.raises(SystemExit) as e:
        vbmain.main()
    assert e.value.code == 0
