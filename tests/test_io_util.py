"""Unit tests for ensure_decompressed — gzip-aware path resolution."""

import gzip
import os
import stat

from vibecheck.io_util import ensure_decompressed


def _write_gz(path, data):
    with gzip.open(path, 'wb') as f:
        f.write(data)


def test_plain_exists_no_gz(tmp_path):
    p = tmp_path / 'a.onnx'
    p.write_bytes(b'x')
    assert ensure_decompressed(str(p)) == str(p)


def test_plain_fresh_reused_without_inflating(tmp_path, capsys):
    plain = tmp_path / 'a.onnx'
    gz = tmp_path / 'a.onnx.gz'
    _write_gz(gz, b'gzdata')
    plain.write_bytes(b'plain')
    os.utime(gz, (1, 1))
    os.utime(plain, (100, 100))  # plain newer than gz
    assert ensure_decompressed(str(plain)) == str(plain)
    assert plain.read_bytes() == b'plain'  # not overwritten
    assert 'Decompressing' not in capsys.readouterr().out


def test_gz_path_decompresses_to_sibling(tmp_path, capsys):
    gz = tmp_path / 'a.vnnlib.gz'
    _write_gz(gz, b'spec text')
    out = ensure_decompressed(str(gz))
    assert out == str(tmp_path / 'a.vnnlib')
    assert (tmp_path / 'a.vnnlib').read_bytes() == b'spec text'
    assert 'Decompressing' in capsys.readouterr().out


def test_plain_missing_gz_exists(tmp_path):
    gz = tmp_path / 'b.onnx.gz'
    _write_gz(gz, b'modeldata')
    out = ensure_decompressed(str(tmp_path / 'b.onnx'))
    assert out == str(tmp_path / 'b.onnx')
    assert (tmp_path / 'b.onnx').read_bytes() == b'modeldata'


def test_stale_plain_redecompressed(tmp_path):
    plain = tmp_path / 'c.onnx'
    gz = tmp_path / 'c.onnx.gz'
    plain.write_bytes(b'OLD')
    _write_gz(gz, b'NEW')
    os.utime(plain, (1, 1))
    os.utime(gz, (100, 100))  # gz newer -> plain is stale
    out = ensure_decompressed(str(plain))
    assert out == str(plain)
    assert plain.read_bytes() == b'NEW'


def test_no_gz_plain_missing_returns_path(tmp_path):
    p = str(tmp_path / 'missing.onnx')
    assert ensure_decompressed(p) == p


def test_readonly_dir_falls_back_to_gz(tmp_path):
    d = tmp_path / 'ro'
    d.mkdir()
    gz = d / 'd.onnx.gz'
    _write_gz(gz, b'data')
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)  # read-only: write of plain fails
    try:
        out = ensure_decompressed(str(d / 'd.onnx'))
        assert out == str(gz)
    finally:
        os.chmod(d, stat.S_IRWXU)
