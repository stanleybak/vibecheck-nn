"""Tests for the pre-parse .pkl cache (vibecheck.preparse).

prepare_instance.sh parses the ONNX+VNNLIB once and pickles the
`(ComputeGraph, VNNSpec)` pair; the timed run loads it back (when
`--allow-unsafe-pkl-loading` is set) and skips the parse. These tests pin:
  - the cache path is deterministic in (onnx, vnnlib, dtype),
  - write→load round-trips to an equivalent graph + spec,
  - a cached load yields the SAME verdict as a fresh parse,
  - stale / missing / version-skewed caches are ignored (fall back to parse),
so the optimization can never silently change a verdict.
"""
import os

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto, numpy_helper

from vibecheck import preparse
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib


def _make_instance(tmp_path):
    """Tiny Gemm net (1->2) + a SAT spec. Returns (onnx_path, vnnlib_path)."""
    W = numpy_helper.from_array(np.array([[1.0], [-1.0]], np.float32), 'W')
    b = numpy_helper.from_array(np.array([0.0, 0.0], np.float32), 'b')
    g = helper.make_graph(
        [helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)], 'm',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx_path = str(tmp_path / 'net.onnx')
    onnx.save(m, onnx_path)
    vnnlib_path = str(tmp_path / 'spec.vnnlib')
    with open(vnnlib_path, 'w') as f:
        f.write('(declare-const X_0 Real)\n(declare-const Y_0 Real)\n'
                '(declare-const Y_1 Real)\n(assert (>= X_0 0.0))\n'
                '(assert (<= X_0 1.0))\n(assert (>= Y_0 0.5))\n')
    return onnx_path, vnnlib_path


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a fresh tmp dir so tests don't collide."""
    cdir = tmp_path / 'pkl_cache'
    monkeypatch.setenv('VIBECHECK_PKL_CACHE_DIR', str(cdir))
    return cdir


def test_cache_path_deterministic_and_dtype_keyed(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    p1 = preparse.pkl_cache_path(onnx_path, vnnlib_path, np.float32)
    p2 = preparse.pkl_cache_path(onnx_path, vnnlib_path, np.float32)
    assert p1 == p2  # deterministic
    # dtype is part of the key
    assert preparse.pkl_cache_path(onnx_path, vnnlib_path, np.float64) != p1
    # lives under the (overridden) cache dir
    assert p1.startswith(str(isolated_cache))


def test_load_missing_returns_none(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    # No write_cache call → nothing on disk.
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) is None


def test_write_then_load_roundtrips(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    out = preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    assert os.path.isfile(out)

    loaded = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    assert loaded is not None
    graph, spec = loaded
    assert isinstance(graph, ComputeGraph)

    # Equivalent to a fresh parse: same op count + ReLU count + input shape,
    # same number of spec disjuncts/constraints.
    fresh_graph = ComputeGraph.from_onnx(onnx_path, dtype=np.float32)
    fresh_spec = load_vnnlib(vnnlib_path)
    assert len(graph.nodes) == len(fresh_graph.nodes)
    assert len(graph.relu_nodes()) == len(fresh_graph.relu_nodes())
    assert graph.input_shape == fresh_graph.input_shape
    assert len(spec.disjuncts) == len(fresh_spec.disjuncts)
    assert spec.n_constraints == fresh_spec.n_constraints
    np.testing.assert_array_equal(spec.x_lo, fresh_spec.x_lo)
    np.testing.assert_array_equal(spec.x_hi, fresh_spec.x_hi)


def test_cached_verdict_matches_fresh(tmp_path, isolated_cache):
    """The whole point: a cached load must not change the verdict."""
    from vibecheck.settings import default_settings
    from vibecheck.verify_graph import verify_graph
    onnx_path, vnnlib_path = _make_instance(tmp_path)

    def _verdict(graph, spec):
        settings = default_settings(device='cpu', bits=32, total_timeout=10)
        graph.optimize(settings)
        result, _ = verify_graph(graph, spec, settings)
        return result

    fresh = _verdict(ComputeGraph.from_onnx(onnx_path, dtype=np.float32),
                     load_vnnlib(vnnlib_path))
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    g_cached, s_cached = preparse.load_cache(onnx_path, vnnlib_path, np.float32)
    cached = _verdict(g_cached, s_cached)
    assert cached == fresh == 'sat'


def test_stale_cache_ignored_on_mtime_change(tmp_path, isolated_cache):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) is not None
    # Touch the onnx forward in time → stamp mtime mismatch → ignored.
    st = os.stat(onnx_path)
    os.utime(onnx_path, (st.st_atime, st.st_mtime + 100))
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) is None


def test_version_skew_ignored(tmp_path, isolated_cache, monkeypatch):
    onnx_path, vnnlib_path = _make_instance(tmp_path)
    preparse.write_cache(onnx_path, vnnlib_path, np.float32)
    # Simulate a format bump after the cache was written: the path is keyed by
    # version (so the new path is empty) AND the stamp records the old version.
    monkeypatch.setattr(preparse, 'CACHE_FORMAT_VERSION',
                        preparse.CACHE_FORMAT_VERSION + 1)
    assert preparse.load_cache(onnx_path, vnnlib_path, np.float32) is None
