"""The pre-parse cache (`preparse.write_cache`) is a PERF optimization. For a net or
spec it can't pre-parse it must LOG + SKIP that part (the timed run parses it from
source) rather than crash — a crash was previously masked by run_instance.sh's
`|| WARNING` into a false `status=ok`.

Cases seen in the 2026 sweep:
  * adaptive_cruise: vnnlib_loader raises NotImplementedError on the degree-2 (X^2)
    nonlinear spec (the run handles it via the augment, not load_vnnlib).
  * cctsdb_yolo: onnx_loader raises IndexError/ValueError on the patch nets (the run
    handles them via the custom YOLO handler).

Skipping is sound: any benchmark whose RUN actually uses load_onnx/load_vnnlib runs
the same parse at verify time, so a genuine loader bug still surfaces loudly there —
only the optional cache is skipped. This test also pins that we DON'T swallow
unexpected exception types (no broad `except Exception`).
"""
import gzip
import hashlib

import pytest

import vibecheck.preparse as pp


def test_file_sha1_resolves_gz_sibling(tmp_path):
    """Benchmarks ship gzipped; instances.csv names the un-gz file. `_file_sha1`
    must key on the `.gz` bytes when only the `.gz` exists (else the whole
    pre-parse cache silently FileNotFoundError'd for every gzipped benchmark).
    Regression for the cgan/nn4sys prepare-cache crash masked as status=ok."""
    payload = b'fake onnx/vnnlib bytes'
    gz = tmp_path / 'model.onnx.gz'
    with gzip.open(gz, 'wb') as f:
        f.write(payload)
    bare = str(tmp_path / 'model.onnx')          # does NOT exist; only the .gz does
    expect = hashlib.sha1(gz.read_bytes()).hexdigest()   # sha1 of the .gz bytes
    assert pp._file_sha1(bare) == expect          # resolves the .gz sibling, no crash


def test_skips_uncacheable_vnnlib_keeps_onnx(monkeypatch, capsys):
    monkeypatch.setattr(pp, 'write_onnx_cache', lambda p, d: '/cache/net.onnx.pkl')

    def _nonlinear(_p):
        raise NotImplementedError('degree>=2 monomial: 1.0*X_1*X_1 <= 0')

    monkeypatch.setattr(pp, 'write_vnnlib_cache', _nonlinear)
    onnx_pkl, vnnlib_pkl = pp.write_cache('net.onnx', 'spec.vnnlib', 'float32')
    assert onnx_pkl == '/cache/net.onnx.pkl'   # the cacheable part still cached
    assert vnnlib_pkl is None                   # the nonlinear spec skipped
    out = capsys.readouterr().out
    assert 'vnnlib pre-parse cache SKIPPED' in out and 'spec.vnnlib' in out


def test_skips_uncacheable_onnx_keeps_vnnlib(monkeypatch, capsys):
    def _bad_onnx(_p, _d):
        raise IndexError('tuple index out of range')

    monkeypatch.setattr(pp, 'write_onnx_cache', _bad_onnx)
    monkeypatch.setattr(pp, 'write_vnnlib_cache', lambda p: '/cache/spec.vnnlib.pkl')
    onnx_pkl, vnnlib_pkl = pp.write_cache('net.onnx', 'spec.vnnlib', 'float32')
    assert onnx_pkl is None
    assert vnnlib_pkl == '/cache/spec.vnnlib.pkl'
    assert 'onnx pre-parse cache SKIPPED' in capsys.readouterr().out


def test_does_not_swallow_unexpected_exception(monkeypatch):
    """An unexpected error class (a REAL bug, not an uncacheable net) must propagate,
    not be silently turned into a cache miss."""
    def _real_bug(_p, _d):
        raise KeyError('unexpected internal error')

    monkeypatch.setattr(pp, 'write_onnx_cache', _real_bug)
    monkeypatch.setattr(pp, 'write_vnnlib_cache', lambda p: '/cache/spec.vnnlib.pkl')
    with pytest.raises(KeyError):
        pp.write_cache('net.onnx', 'spec.vnnlib', 'float32')


def test_load_or_parse_box_selfheals_on_corrupt_cache(monkeypatch, capsys, tmp_path):
    """A 0-byte/corrupt surrogate box cache (e.g. a prepare interrupted mid-write)
    must self-heal: log a warning and re-parse the spec, not crash with EOFError.
    Regression for smart_turn instance_41 (run errored `EOFError: Ran out of input`
    loading a truncated box pkl)."""
    import os
    import types
    import vibecheck.main as m
    import vibecheck.surrogate_pgd as sp

    spec = str(tmp_path / 'x.vnnlib')
    open(spec, 'w').write('(declare-const X_0 Real)')
    cp = m._box_cache_path(spec)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    open(cp, 'wb').close()                                   # 0-byte cache
    monkeypatch.setattr(sp, 'parse_box_and_output', lambda p: 'REPARSED_BOX')

    args = types.SimpleNamespace(allow_unsafe_pkl_loading=True, spec=spec)
    out = m._load_or_parse_box(args)
    assert out == 'REPARSED_BOX'                             # fell back to parse
    assert 'unreadable' in capsys.readouterr().out          # logged the anomaly
