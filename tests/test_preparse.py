"""Tests for the pre-parse .pkl cache (vibecheck.preparse).

prepare_instance.sh parses the ONNX and VNNLIB once and pickles each separately
(<onnx>.pkl, <vnnlib>.pkl, keyed by content sha1); the timed run loads them back
(when `--allow-unsafe-pkl-loading` is set) and skips the parse. These tests pin:
  - cache files are named after their source and live under the cache dir,
  - write->load round-trips to an equivalent graph + spec,
  - a cached load yields the SAME verdict as a fresh parse,
  - the key is CONTENT (mtime change still loads; content change invalidates),
  - dtype mismatch / version skew / missing files miss cleanly,
  - a shared ONNX graph is reused across different specs,
so the optimization can never silently change a verdict.
"""
import os

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import pytest

from vibecheck import preparse
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib


def _make_onnx(path, w=((1.0,), (-1.0,))):
    W = numpy_helper.from_array(np.array(w, np.float32), 'W')
    b = numpy_helper.from_array(np.zeros(len(w), np.float32), 'b')
    g = helper.make_graph(
        [helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)], 'm',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, len(w)])],
        initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, path)
    return path


def _make_spec(path, thr=0.5):
    with open(path, 'w') as f:
        f.write('(declare-const X_0 Real)\n(declare-const Y_0 Real)\n'
                '(declare-const Y_1 Real)\n(assert (>= X_0 0.0))\n'
                f'(assert (<= X_0 1.0))\n(assert (>= Y_0 {thr}))\n')
    return path


def _make_instance(tmp_path):
    return (_make_onnx(str(tmp_path / 'net.onnx')),
            _make_spec(str(tmp_path / 'spec.vnnlib')))


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    cdir = tmp_path / 'pkl_cache'
    monkeypatch.setenv('VIBECHECK_PKL_CACHE_DIR', str(cdir))
    return cdir


def test_paths_named_after_source(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    op = preparse.onnx_pkl_path(onnx_path)
    vp = preparse.vnnlib_pkl_path(vnnlib_path)
    assert os.path.basename(op) == 'net.onnx.pkl'
    assert os.path.basename(vp) == 'spec.vnnlib.pkl'
    assert op.startswith(str(isolated_cache)) and vp.startswith(str(isolated_cache))


def test_load_missing_returns_none_pair(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) == (None, None)


def test_write_then_load_roundtrips(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    op, vp = preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    assert os.path.isfile(op) and os.path.isfile(vp)

    graph, spec = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    assert isinstance(graph, ComputeGraph) and spec is not None

    fresh_graph = ComputeGraph.from_onnx(onnx_path, dtype=np.float32)
    fresh_spec = load_vnnlib(vnnlib_path)
    assert len(graph.nodes) == len(fresh_graph.nodes)
    assert graph.input_shape == fresh_graph.input_shape
    assert spec.n_constraints == fresh_spec.n_constraints
    np.testing.assert_array_equal(spec.x_lo, fresh_spec.x_lo)


def test_cached_verdict_matches_fresh(tmp_path, isolated_cache):
    from vibecheck.settings import default_settings
    from vibecheck.verify_graph import verify_graph
    onnx_path, vnnlib_path = _make_instance(tmp_path)

    def _verdict(graph, spec):
        settings = default_settings(device='cpu', bits=32, total_timeout=10)
        graph.optimize(settings)
        return verify_graph(graph, spec, settings)[0]

    fresh = _verdict(ComputeGraph.from_onnx(onnx_path, dtype=np.float32),
                     load_vnnlib(vnnlib_path))
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    g, s = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    assert _verdict(g, s) == fresh == 'sat'


def test_mtime_change_still_loads_content_keyed(tmp_path, isolated_cache):
    # Keyed by CONTENT sha1, not mtime: touching without changing bytes is fine.
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    st = os.stat(onnx_path)
    os.utime(onnx_path, (st.st_atime, st.st_mtime + 100))
    g, s = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    assert g is not None and s is not None


def test_content_change_invalidates_only_that_cache(tmp_path, isolated_cache):
    # Changing the onnx bytes invalidates the GRAPH cache (different sha1 -> the
    # stored meta no longer matches), but the unchanged spec cache still loads.
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    _make_onnx(onnx_path, w=((2.0,), (-3.0,)))          # different weights -> new sha1
    g, s = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    assert g is None and s is not None


def test_dtype_mismatch_misses(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    # Spec is dtype-independent (still loads); graph keyed on dtype (misses).
    g, s = preparse.load_cache(onnx_path, vnnlib_path, np.float64)
    assert g is None and s is not None


def test_write_idempotent_skips_rebuild(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    op1, vp1 = preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    m1 = (os.path.getmtime(op1), os.path.getmtime(vp1))
    op2, vp2 = preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    assert (op1, vp1) == (op2, vp2)
    assert (os.path.getmtime(op2), os.path.getmtime(vp2)) == m1   # not rebuilt


def test_shared_onnx_graph_reused_across_specs(tmp_path, isolated_cache):
    # One ONNX, two different specs: the graph cache (named after the onnx) is
    # written once and reused for both specs — the whole point of splitting.
    onnx_path = _make_onnx(str(tmp_path / 'shared.onnx'))
    spec1 = _make_spec(str(tmp_path / 's1.vnnlib'), thr=0.5)
    spec2 = _make_spec(str(tmp_path / 's2.vnnlib'), thr=0.7)
    op1, _ = preparse.write_cache(onnx_path, spec1, np.float32)
    op2, _ = preparse.write_cache(onnx_path, spec2, np.float32)
    assert op1 == op2                                  # same graph cache file
    g1, _ = preparse.load_cache(onnx_path, spec1, np.float32)
    g2, _ = preparse.load_cache(onnx_path, spec2, np.float32)
    assert g1 is not None and g2 is not None


def test_version_skew_ignored(tmp_path, isolated_cache, monkeypatch):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    monkeypatch.setattr(preparse, 'CACHE_FORMAT_VERSION',
                        preparse.CACHE_FORMAT_VERSION + 1)
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) == (None, None)
