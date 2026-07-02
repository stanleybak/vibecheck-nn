"""Tests for the VNN-LIB standard CLI (cli_standard.py + the main() dispatch).

Covers `--name`/`--version`, every `supports` capability, the `--network`
NAME=PATH -> legacy `--net` mapping (single / pair / equal-to / error cases),
and end-to-end `verify` runs on a tiny synthetic net: strict stdout (verdict
first line, no progress leakage), the assignment block format, TensorProto
serialisation, verdict-style spelling, and the crash path.
"""
import os
import re
import runpy
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from vibecheck import cli_standard as cs
from vibecheck import main as vbmain
from vibecheck.network import OP_REGISTRY


# --------------------------------------------------------------------------- #
# tiny runnable instance: identity net (Y = relu(X @ I)), 2-D input box [0,1]^2
# --------------------------------------------------------------------------- #

def _identity_net(path):
    W = numpy_helper.from_array(np.eye(2, dtype=np.float32), 'W')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['h']),
                           helper.make_node('Relu', ['h'], ['Y'])],
                          'g', [X], [Y], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, path)


_V1_BOX = ('(declare-const X_0 Real)\n(declare-const X_1 Real)\n'
           '(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n'
           '(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n'
           '(assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n')

_V2_HEAD = ('(vnnlib-version <2.0>)\n'
            '(declare-network f (declare-input X float32 [1,2]) '
            '(declare-output Y float32 [1,2]))\n'
            '(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n'
            '(assert (>= X[0,1] 0.0))\n(assert (<= X[0,1] 1.0))\n')


@pytest.fixture()
def instance(tmp_path):
    """dict of paths: tiny net + v1/v2 sat + v1 unsat specs."""
    net = str(tmp_path / 'net.onnx')
    _identity_net(net)
    p = {'net': net}
    for name, text in [
            ('sat_v1', _V1_BOX + '(assert (>= Y_0 0.5))\n'),
            ('unsat_v1', _V1_BOX + '(assert (>= Y_0 2.0))\n'),
            ('sat_v2', _V2_HEAD + '(assert (>= Y[0,0] 0.5))\n')]:
        fp = tmp_path / f'{name}.vnnlib'
        fp.write_text(text)
        p[name] = str(fp)
    return p


def _verify_args(spec, net, extra=()):
    return [spec, '--network', f'f={net}', '--timeout', '30',
            '--device', 'cpu'] + list(extra)


# --------------------------------------------------------------------------- #
# --name / --version / dispatch
# --------------------------------------------------------------------------- #

def test_name(capsys):
    assert cs.dispatch(['--name']) == 0
    assert capsys.readouterr().out == 'vibecheck\n'


def test_version_from_metadata(monkeypatch, capsys):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: {'vibecheck': '9.9.9'}[name])
    assert cs.dispatch(['--version']) == 0
    assert capsys.readouterr().out == '9.9.9\n'


def test_version_fallback_reads_pyproject(monkeypatch, capsys):
    """No installed package metadata (dev checkout via PYTHONPATH): the version
    comes from pyproject.toml and must be a plain semver string."""
    import importlib.metadata

    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', _raise)
    assert cs.dispatch(['--version']) == 0
    out = capsys.readouterr().out.strip()
    assert re.fullmatch(r'\d+(\.\d+)+', out)
    pyproject = os.path.join(os.path.dirname(cs.__file__), '..', '..',
                             'pyproject.toml')
    with open(pyproject) as f:
        assert out == re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)


def test_main_module_dispatches_standard_cli(monkeypatch, capsys):
    """`python -m vibecheck.main --name` goes through the __main__ block and the
    main() dispatcher into the standard CLI."""
    monkeypatch.setattr(sys, 'argv', ['vibecheck', '--name'])
    with pytest.raises(SystemExit) as e:
        runpy.run_module('vibecheck.main', run_name='__main__')
    assert e.value.code == 0
    assert capsys.readouterr().out == 'vibecheck\n'


def test_main_dispatches_supports(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['vibecheck', 'supports', '--vnnlib-versions'])
    assert vbmain.main() == 0
    assert capsys.readouterr().out == '1.0\n2.0\n'


# --------------------------------------------------------------------------- #
# supports
# --------------------------------------------------------------------------- #

_ALL_CAPS = sorted(set(cs._SUPPORTS_TABLE) | {'--onnx-operators'})


@pytest.mark.parametrize('cap', _ALL_CAPS)
def test_supports_first_tokens_are_identifiers(cap, capsys):
    assert cs.run_supports([cap]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines
    for line in lines:
        ident = line.split(' * ')[0].strip()
        # machine-parseable first token: no spaces, only ident/version chars
        assert re.fullmatch(r'[A-Za-z0-9._\-]+', ident), line
        if ' * ' in line:                       # partial marker carries a note
            assert line.split(' * ', 1)[1].strip()


def test_supports_booleans_parse(capsys):
    for cap in ('--optimised-disjunctive-reasoning', '--serialise-assignments'):
        assert cs.run_supports([cap]) == 0
        assert capsys.readouterr().out.strip() in ('true', 'false')


def test_supports_onnx_operators_match_registry(capsys):
    assert cs.run_supports(['--onnx-operators']) == 0
    assert capsys.readouterr().out.splitlines() == sorted(OP_REGISTRY)


def test_supports_vnnlib_versions(capsys):
    assert cs.run_supports(['--vnnlib-versions']) == 0
    assert capsys.readouterr().out.splitlines() == ['1.0', '2.0']


def test_supports_multiple_capabilities_one_call(capsys):
    assert cs.run_supports(['--hidden-node-theories',
                            '--multiple-input-output-theories']) == 0
    assert capsys.readouterr().out.splitlines() == ['NH', 'SIO']


def test_supports_no_args_errors(capsys):
    assert cs.run_supports([]) == 2
    err = capsys.readouterr().err
    assert 'requires a capability' in err and '--onnx-operators' in err


def test_supports_unknown_capability_errors(capsys):
    assert cs.run_supports(['--bogus']) == 2
    assert 'unknown capability' in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --network mapping (query declarations -> legacy --net field)
# --------------------------------------------------------------------------- #

def _write_spec(tmp_path, text, name='q.vnnlib'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_map_v1_single_network(tmp_path):
    q = _write_spec(tmp_path, _V1_BOX)
    assert cs._resolve_net_field(q, ['whatever=/a/net.onnx']) == '/a/net.onnx'


@pytest.mark.parametrize('flags', [[], ['f=a.onnx', 'g=b.onnx']])
def test_map_v1_wrong_network_count_errors(tmp_path, capsys, flags):
    q = _write_spec(tmp_path, _V1_BOX)
    with pytest.raises(SystemExit) as e:
        cs._resolve_net_field(q, flags)
    assert e.value.code == 2
    assert 'expected exactly 1' in capsys.readouterr().err


def test_map_malformed_network_value_errors(tmp_path, capsys):
    q = _write_spec(tmp_path, _V1_BOX)
    for bad in ('nopath', '=x.onnx', 'f='):
        with pytest.raises(SystemExit) as e:
            cs._resolve_net_field(q, [bad])
        assert e.value.code == 2
        assert 'malformed --network' in capsys.readouterr().err


def test_map_v2_single_network(tmp_path):
    q = _write_spec(tmp_path, _V2_HEAD + '(assert (>= Y[0,0] 0.5))\n')
    assert cs._resolve_net_field(q, ['f=m.onnx']) == 'm.onnx'


def test_map_v2_isomorphic_pair_two_files(tmp_path):
    q = _write_spec(tmp_path,
                    '(declare-network f (declare-input X_f real [5]))\n'
                    '(declare-network g (declare-input X_g real [5]) '
                    '(isomorphic-to f))\n')
    field = cs._resolve_net_field(q, ['f=a.onnx', 'g=b.onnx'])
    assert field == "[('f', 'a.onnx'), ('g', 'b.onnx')]"
    # order-insensitive on the CLI, declaration-ordered in the field
    assert cs._resolve_net_field(q, ['g=b.onnx', 'f=a.onnx']) == field


def test_map_v2_equal_to_reuses_source_file(tmp_path):
    """Per the standard, an `equal-to` network takes NO --network mapping: it
    reuses the source network's ONNX (monotonic_acasxu shape)."""
    q = _write_spec(tmp_path,
                    '(declare-network f (declare-input X_f real [5]))\n'
                    '(declare-network g (declare-input X_g real [5]) '
                    '(equal-to f))\n')
    assert cs._resolve_net_field(q, ['f=a.onnx']) == \
        "[('f', 'a.onnx'), ('g', 'a.onnx')]"


def test_map_v2_name_mismatch_errors(tmp_path, capsys):
    q = _write_spec(tmp_path, '(declare-network f (declare-input X real [5]))\n')
    with pytest.raises(SystemExit) as e:
        cs._resolve_net_field(q, ['wrong=a.onnx'])
    assert e.value.code == 2
    assert 'do not match' in capsys.readouterr().err


def test_map_v2_equal_to_unknown_source_errors(tmp_path, capsys):
    q = _write_spec(tmp_path,
                    '(declare-network f (declare-input X_f real [5]))\n'
                    '(declare-network g (equal-to h))\n')
    with pytest.raises(SystemExit) as e:
        cs._resolve_net_field(q, ['f=a.onnx'])
    assert e.value.code == 2
    assert 'no resolved file mapping' in capsys.readouterr().err


def test_map_more_than_two_networks_errors(tmp_path, capsys):
    q = _write_spec(tmp_path, ''.join(
        f'(declare-network n{i} (declare-input X{i} real [2]))\n'
        for i in range(3)))
    with pytest.raises(SystemExit) as e:
        cs._resolve_net_field(q, ['n0=a.onnx', 'n1=b.onnx', 'n2=c.onnx'])
    assert e.value.code == 2
    assert 'two-network pairs only' in capsys.readouterr().err


def test_spec_head_gz_and_missing(tmp_path):
    gz = tmp_path / 'q.vnnlib.gz'
    import gzip
    with gzip.open(gz, 'wt') as f:
        f.write(_V1_BOX)
    assert '(declare-const X_0 Real)' in cs._spec_head(str(tmp_path / 'q.vnnlib'))
    assert cs._spec_head(str(tmp_path / 'nope.vnnlib')) == ''


# --------------------------------------------------------------------------- #
# verify: end-to-end on the tiny instance
# --------------------------------------------------------------------------- #

def test_verify_unsat_strict_stdout(instance, capsys):
    """stdout carries ONLY the verdict line; all progress goes to stderr."""
    rc = cs.dispatch(['verify'] + _verify_args(instance['unsat_v1'],
                                               instance['net']))
    out, err = capsys.readouterr().out, capsys.readouterr().err
    assert rc == 0
    assert out == 'unsat\n'


def test_verify_sat_v1_assignment(instance, capsys):
    rc = cs.run_verify(_verify_args(instance['sat_v1'], instance['net']))
    out = capsys.readouterr().out
    assert rc == 1                       # legacy convention: 1 = not-verified
    lines = out.splitlines()
    assert lines[0] == 'sat'
    body = '\n'.join(lines[1:])
    # v1 assignment: one (X_i v)/(Y_j v) atom per variable
    assert len(re.findall(r'\((X|Y)_\d+ \S+\)', body)) == 4


def test_verify_sat_v2_assignment_format(instance, capsys):
    """v2 spec -> assignment blocks are `NAME dtype [dims]` + one value/line."""
    rc = cs.run_verify(_verify_args(instance['sat_v2'], instance['net']))
    out = capsys.readouterr().out
    assert rc == 1
    lines = out.splitlines()
    assert lines[0] == 'sat'
    assert lines[1] == 'X float32 [1,2]'
    assert lines[4] == 'Y float32 [1,2]'
    x = [float(v) for v in lines[2:4]]
    y = [float(v) for v in lines[5:7]]
    assert 0.0 <= x[0] <= 1.0 and y[0] >= 0.5


def test_verify_serialise_assignments(instance, capsys, tmp_path):
    out_dir = str(tmp_path / 'assign')
    rc = cs.run_verify(_verify_args(
        instance['sat_v2'], instance['net'],
        extra=['--serialise-assignments', out_dir]))
    out = capsys.readouterr().out
    assert rc == 1
    assert out == 'sat\n'                # values go to the .pb files, not stdout
    got = {}
    for name in ('X', 'Y'):
        t = TensorProto()
        with open(os.path.join(out_dir, f'{name}.pb'), 'rb') as f:
            t.ParseFromString(f.read())
        got[name] = numpy_helper.to_array(t)
        assert got[name].dtype == np.float32     # spec-declared dtype
        assert got[name].shape == (1, 2)         # spec-declared shape
    assert 0.0 <= got['X'][0, 0] <= 1.0 and got['Y'][0, 0] >= 0.5


def test_verify_results_file_and_style(instance, capsys, tmp_path, monkeypatch):
    """A run that exhausts its budget: standard style spells `timed-out` on both
    stdout and the results file; vnncomp style spells `timeout`."""
    # _verify that decides nothing and exits: the pre-seeded verdict survives.
    monkeypatch.setattr(vbmain, '_verify',
                        lambda args, sat_state=None: sys.exit(1))
    for style, word in (('standard', 'timed-out'), ('vnncomp', 'timeout')):
        rf = str(tmp_path / f'res_{style}.txt')
        rc = cs.run_verify(_verify_args(
            instance['unsat_v1'], instance['net'],
            extra=['--results-file', rf, '--verdict-style', style]))
        assert rc == 1
        assert capsys.readouterr().out == f'{word}\n'
        with open(rf) as f:
            assert f.read() == f'{word}\n'


def test_verify_default_style_is_standard_legacy_default_is_vnncomp(
        instance, tmp_path, capsys, monkeypatch):
    """The legacy flat CLI keeps the byte-identical `timeout` spelling by
    default; `verify` defaults to the standard's `timed-out` (previous test).
    Here: same aborted run through the LEGACY CLI -> `timeout` in the file."""
    monkeypatch.setattr(vbmain, '_verify',
                        lambda args, sat_state=None: sys.exit(1))
    rf = str(tmp_path / 'legacy.txt')
    with pytest.raises(SystemExit):
        vbmain._legacy_main(['--net', instance['net'],
                             '--spec', instance['unsat_v1'],
                             '--timeout', '30', '--results-file', rf])
    capsys.readouterr()
    with open(rf) as f:
        assert f.read() == 'timeout\n'


def test_verify_passthrough_flags(instance, capsys, tmp_path, monkeypatch):
    """--config/--set pass through to the legacy pipeline."""
    seen = {}

    def _fake_verify(args, sat_state=None):
        seen['config'] = args.config
        seen['set_kv'] = args.set_kv
        sys.exit(1)

    monkeypatch.setattr(vbmain, '_verify', _fake_verify)
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text('total_timeout: 5\n')
    rc = cs.run_verify(_verify_args(
        instance['unsat_v1'], instance['net'],
        extra=['--config', str(cfg), '--set', 'pgd_restarts=1']))
    capsys.readouterr()
    assert rc == 1
    assert seen['config'] == str(cfg)
    assert seen['set_kv'] == ['pgd_restarts=1']


def test_verify_crash_reports_error(instance, capsys, monkeypatch):
    """A crash inside verification -> exit 2, cause on stderr, clean stdout."""
    def _boom(args, sat_state=None):
        raise RuntimeError('synthetic crash')

    monkeypatch.setattr(vbmain, '_verify', _boom)
    rc = cs.run_verify(_verify_args(instance['unsat_v1'], instance['net']))
    cap = capsys.readouterr()
    assert rc == 2
    assert cap.out == ''
    assert 'RuntimeError: synthetic crash' in cap.err


# --------------------------------------------------------------------------- #
# results/assignment parsing + serialisation units
# --------------------------------------------------------------------------- #

def test_read_results_empty_file(tmp_path):
    rf = tmp_path / 'r.txt'
    rf.write_text('')
    assert cs._read_results(str(rf)) == ('unknown', '')


def test_parse_assignment_v1():
    got = cs._parse_assignment('((X_0 0.25)\n(X_1 1)\n(Y_0 0.25))')
    assert got == [('X', 'real', (2,), [0.25, 1.0]),
                   ('Y', 'real', (1,), [0.25])]


def test_parse_assignment_v2_multi_tensor():
    text = 'X_f float32 [1,2]\n0.5\n1\n\nY_f real [2]\n-3.0\n2e-3\n'
    got = cs._parse_assignment(text)
    assert got == [('X_f', 'float32', (1, 2), [0.5, 1.0]),
                   ('Y_f', 'real', (2,), [-3.0, 0.002])]


@pytest.mark.parametrize('bad', ['()', '0.5\n1.0\n', ''])
def test_parse_assignment_unparseable_raises(bad):
    with pytest.raises(ValueError, match='unparseable'):
        cs._parse_assignment(bad)


def test_serialise_assignment_unknown_dtype_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cs._serialise_assignment('X int64 [2]\n1\n2\n', str(tmp_path / 'd'))
    assert e.value.code == 2
    assert 'unsupported declared dtype' in capsys.readouterr().err
