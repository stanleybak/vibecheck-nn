"""Unit tests for auto-config detection (config_detect), the config_loader meta-key
support, and main._auto_select_config.

Synthetic tiny ONNX (fc / conv / transformer / nonlinear) + synthetic vnnlib text drive
the fingerprint + rule paths; the pure detect_config tree is exercised on hand-built
fingerprints so every leaf is covered without loading a model.
"""
import gzip
import os
import types

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from vibecheck import config_detect as cd
from vibecheck.config_detect import DetectFingerprint, detect_config
from vibecheck import config_loader as cl


# --------------------------------------------------------------------------- #
# tiny ONNX builders (from_onnx_and_spec only reads op types / initializer dims /
# input shape, so these need not be runnable — just parseable by onnx.load).
# --------------------------------------------------------------------------- #

def _save(nodes, inits, in_shape, path, out='Y'):
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, in_shape)
    Y = helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 2])
    g = helper.make_graph(nodes, 'g', [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, path)


def _fc(path, in_shape=(1, 5)):
    W = numpy_helper.from_array(np.zeros((in_shape[-1], 2), np.float32), 'W')
    _save([helper.make_node('MatMul', ['X', 'W'], ['h']),
           helper.make_node('Relu', ['h'], ['Y'])], [W], list(in_shape), path)


def _conv(path, in_shape=(1, 3, 8, 8)):
    K = numpy_helper.from_array(np.zeros((2, in_shape[1], 3, 3), np.float32), 'K')
    _save([helper.make_node('Conv', ['X', 'K'], ['c']),
           helper.make_node('Relu', ['c'], ['Y'])], [K], list(in_shape), path)


def _transformer(path, in_shape=(1, 5)):
    W1 = numpy_helper.from_array(np.zeros((in_shape[-1], 4), np.float32), 'W1')
    W2 = numpy_helper.from_array(np.zeros((4, 2), np.float32), 'W2')
    _save([helper.make_node('MatMul', ['X', 'W1'], ['a']),
           helper.make_node('Softmax', ['a'], ['s']),
           helper.make_node('MatMul', ['s', 'W2'], ['Y'])], [W1, W2], list(in_shape), path)


def _nonlinear(path, in_shape=(1, 5)):
    W = numpy_helper.from_array(np.zeros((in_shape[-1], 2), np.float32), 'W')
    _save([helper.make_node('MatMul', ['X', 'W'], ['h']),
           helper.make_node('Sin', ['h'], ['Y'])], [W], list(in_shape), path)


# --------------------------------------------------------------------------- #
# detect_config: pure tree, every leaf
# --------------------------------------------------------------------------- #

def _fp(**kw):
    base = dict(is_pair=False, pair_kind=None, has_conv=False, is_transformer=False,
                smooth_nonlinear=False, spec_nonlinear=False, params=0, in_dim=100, n_relu=0)
    base.update(kw)
    return DetectFingerprint(**base)


@pytest.mark.parametrize('fp,expected', [
    (_fp(is_pair=True, pair_kind='mono'), 'monotonic_acasxu_2026.yaml'),
    (_fp(is_pair=True, pair_kind='iso'), 'isomorphic_acasxu_2026.yaml'),
    (_fp(is_pair=True, pair_kind=None, in_dim=100), 'cora_2024.yaml'),   # pair w/o kind -> falls through
    (_fp(spec_nonlinear=True), 'adaptive_cruise_control_non_linear_2026.yaml'),
    (_fp(smooth_nonlinear=True, has_conv=False), 'ml4acopf_2024.yaml'),
    (_fp(in_dim=5, has_conv=True), 'cgan2026.yaml'),
    (_fp(in_dim=5, has_conv=False), 'acasxu_2023.yaml'),
    (_fp(is_transformer=True, in_dim=64000), 'smart_turn_multimodal_2026.yaml'),
    (_fp(is_transformer=True, in_dim=3072), 'vit_2023.yaml'),
    (_fp(has_conv=True, in_dim=150528), 'vggnet16_2022.yaml'),
    (_fp(has_conv=True, in_dim=3072), 'cifar100_2024.yaml'),
    # smooth_nonlinear AND conv-huge -> NOT ml4acopf (the `not has_conv` guard), falls to vgg
    (_fp(smooth_nonlinear=True, has_conv=True, in_dim=1_200_000), 'vggnet16_2022.yaml'),
    (_fp(has_conv=False, in_dim=784), 'cora_2024.yaml'),
])
def test_detect_config_leaves(fp, expected):
    name, rule = detect_config(fp)
    assert name == expected
    assert rule and rule[0].isdigit()


# --------------------------------------------------------------------------- #
# from_onnx_and_spec: fingerprint extraction per architecture
# --------------------------------------------------------------------------- #

def test_fingerprint_fc(tmp_path):
    p = str(tmp_path / 'fc.onnx'); _fc(p, (1, 5))
    fp = DetectFingerprint.from_onnx_and_spec(p)
    assert fp.has_conv is False and fp.is_transformer is False and fp.smooth_nonlinear is False
    assert fp.in_dim == 5 and fp.n_relu == 1 and fp.params == 10   # 5*2 weight


def test_fingerprint_conv(tmp_path):
    p = str(tmp_path / 'c.onnx'); _conv(p)
    fp = DetectFingerprint.from_onnx_and_spec(p)
    assert fp.has_conv is True and fp.in_dim == 3 * 8 * 8


def test_fingerprint_transformer(tmp_path):
    p = str(tmp_path / 't.onnx'); _transformer(p)
    fp = DetectFingerprint.from_onnx_and_spec(p)
    assert fp.is_transformer is True          # Softmax + >=2 MatMul


def test_fingerprint_nonlinear_and_spec_flag(tmp_path):
    p = str(tmp_path / 'n.onnx'); _nonlinear(p)
    fp = DetectFingerprint.from_onnx_and_spec(p, spec_nonlinear=True)
    assert fp.smooth_nonlinear is True and fp.spec_nonlinear is True


def test_fingerprint_dynamic_input_dim(tmp_path):
    # symbolic leading batch dim (dim_value 0) -> counted as 1; concrete dims multiply
    p = str(tmp_path / 'dyn.onnx'); _fc(p, ('N', 5))
    fp = DetectFingerprint.from_onnx_and_spec(p)
    assert fp.in_dim == 5


# --------------------------------------------------------------------------- #
# _load_onnx: plain, .gz, and the missing-plain -> .gz fallback
# --------------------------------------------------------------------------- #

def test_load_onnx_plain_gz_and_fallback(tmp_path):
    p = str(tmp_path / 'm.onnx'); _fc(p)
    assert cd._load_onnx(p) is not None                       # plain
    with open(p, 'rb') as f, gzip.open(p + '.gz', 'wb') as gz:
        gz.write(f.read())
    assert cd._load_onnx(p + '.gz') is not None               # explicit .gz
    os.remove(p)
    assert cd._load_onnx(p) is not None                       # missing plain -> .gz fallback


# --------------------------------------------------------------------------- #
# detect_from_field: pair short-circuit + single-net + base_dir join + spec_nonlinear
# --------------------------------------------------------------------------- #

def _spec(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return path


def test_detect_from_field_pair_mono_iso(tmp_path):
    mono = _spec(str(tmp_path / 'm.vnnlib'),
                 '(declare-network g (equal-to f))\n(assert (< Y_f[3] Y_g[3]))\n')
    iso = _spec(str(tmp_path / 'i.vnnlib'),
                '(declare-network g (isomorphic-to f))\n(assert (or (> Y_g[0] Y_f[0])))\n')
    field = "[('f', 'a.onnx'), ('g', 'b.onnx')]"    # onnx never loaded for a pair
    _, name_m, _ = cd.detect_from_field(field, mono)
    _, name_i, _ = cd.detect_from_field(field, iso)
    assert name_m == 'monotonic_acasxu_2026.yaml'
    assert name_i == 'isomorphic_acasxu_2026.yaml'


def test_detect_from_field_single_and_basedir(tmp_path):
    net = tmp_path / 'onnx'; net.mkdir()
    _fc(str(net / 'x.onnx'), (1, 5))
    spec = _spec(str(tmp_path / 's.vnnlib'), '(assert (<= X_0 1.0))\n')
    # absolute path
    _, name, rule = cd.detect_from_field(str(net / 'x.onnx'), spec)
    assert name == 'acasxu_2023.yaml' and rule.startswith('5')
    # relative path resolved against base_dir (covers the join branch)
    _, name2, _ = cd.detect_from_field('onnx/x.onnx', spec, base_dir=str(tmp_path))
    assert name2 == 'acasxu_2023.yaml'


def test_detect_from_field_spec_nonlinear(tmp_path):
    p = str(tmp_path / 'fc.onnx'); _fc(p, (1, 5))
    spec = _spec(str(tmp_path / 'nl.vnnlib'),
                 '(assert (<= (* X[0,1] X[0,1]) 1.0))\n')      # var*var -> spec_nonlinear
    fp, name, _ = cd.detect_from_field(p, spec)
    assert fp.spec_nonlinear is True
    assert name == 'adaptive_cruise_control_non_linear_2026.yaml'


# --------------------------------------------------------------------------- #
# config_loader: description meta-key + helpers
# --------------------------------------------------------------------------- #

def test_load_config_strips_description(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text('description: "hello"\npgd_restarts: 7\n')
    ov = cl.load_config(str(p))
    assert 'description' not in ov and ov['pgd_restarts'] == 7
    assert cl.config_description(str(p)) == 'hello'


def test_load_config_unknown_key_still_raises(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('description: "ok"\nnot_a_real_setting: 1\n')
    with pytest.raises(AssertionError):
        cl.load_config(str(p))


def test_config_description_absent_and_missing(tmp_path):
    p = tmp_path / 'nodesc.yaml'
    p.write_text('pgd_restarts: 1\n')
    assert cl.config_description(str(p)) is None            # no description key
    assert cl.config_description(str(tmp_path / 'nope.yaml')) is None  # missing file


def test_config_path_and_bundled_descriptions():
    p = cl.config_path('acasxu_2023.yaml')
    assert p.endswith(os.path.join('configs', 'acasxu_2023.yaml'))
    assert cl.config_description(p)                          # bundled config has one


# --------------------------------------------------------------------------- #
# main._auto_select_config: sets args.config + records/logs; no-op guards
# --------------------------------------------------------------------------- #

def _args(**kw):
    base = dict(config=None, mode='graph', net=None, spec=None, verbose=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_auto_select_sets_config(tmp_path, capsys):
    from vibecheck import main
    p = str(tmp_path / 'fc.onnx'); _fc(p, (1, 5))
    spec = _spec(str(tmp_path / 's.vnnlib'), '(assert (<= X_0 1.0))\n')
    a = _args(net=p, spec=spec, verbose=True)
    main._auto_select_config(a)
    assert a.config.endswith('acasxu_2023.yaml')
    assert a.auto_selected[1] == 'acasxu_2023.yaml'
    out = capsys.readouterr().out
    assert 'Auto-config: acasxu_2023.yaml' in out and 'fingerprint' in out


def test_auto_select_noops(tmp_path):
    from vibecheck import main
    # explicit --config -> no-op
    a = _args(config='x.yaml', net='n', spec='s')
    main._auto_select_config(a); assert a.config == 'x.yaml'
    # non-graph mode -> no-op
    a = _args(mode='milp', net='n', spec='s')
    main._auto_select_config(a); assert a.config is None
    # missing spec -> no-op
    a = _args(net='n', spec=str(tmp_path / 'absent.vnnlib'))
    main._auto_select_config(a); assert a.config is None


def test_detect_only_cli(tmp_path, monkeypatch, capsys):
    # `--detect-only` prints the routed yaml + description + fingerprint and exits 0,
    # WITHOUT verifying (covers the main() dry-run block).
    from vibecheck import main
    p = str(tmp_path / 'fc.onnx'); _fc(p, (1, 5))
    spec = _spec(str(tmp_path / 's.vnnlib'), '(assert (<= X_0 1.0))\n')
    monkeypatch.setattr('sys.argv',
                        ['vibecheck', '--mode', 'graph', '--detect-only',
                         '--net', p, '--spec', spec])
    rc = main.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert 'AUTO-DETECT acasxu_2023.yaml' in out
    assert 'description:' in out and 'fingerprint:' in out
