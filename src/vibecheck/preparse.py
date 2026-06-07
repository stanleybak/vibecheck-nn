"""Pre-parse cache for VNNCOMP runs.

Parsing a large ONNX (shape inference, constant folding, BatchNorm folding,
topo sort) plus the VNNLIB spec can take a noticeable slice of a tight
per-instance budget. `prepare_instance.sh` runs once per instance *before* the
timed run, so it is the natural place to do that work and stash the result.

This module pickles the parsed `(ComputeGraph, VNNSpec)` pair to a deterministic
sidecar path keyed by the (onnx, vnnlib, dtype) triple. `run_instance.sh` then
loads it back (when `--allow-unsafe-pkl-loading` is passed) and skips the parse.

SECURITY: pickle executes arbitrary code on load. The cache is therefore only
read when the caller explicitly opts in via `--allow-unsafe-pkl-loading`, and
only for caches *this* tool wrote (validated by a stamped format version + the
recorded source paths/mtimes). Never point it at an untrusted .pkl.

The cached graph is the PRE-`optimize()` form (optimize is settings-dependent,
so it stays a per-run step) — the expensive `from_onnx` parse is what we skip.
"""
import hashlib
import os
import pickle

import numpy as np

# Bump when the pickled object layout changes so stale caches are ignored
# rather than silently mis-loaded.
CACHE_FORMAT_VERSION = 2

_DEFAULT_CACHE_DIR = '/tmp/vibecheck_pkl'


def cache_dir():
    """Directory for pre-parse caches (override via VIBECHECK_PKL_CACHE_DIR)."""
    return os.environ.get('VIBECHECK_PKL_CACHE_DIR', _DEFAULT_CACHE_DIR)


def pkl_cache_path(onnx_path, vnnlib_path, dtype):
    """Deterministic cache path for an (onnx, vnnlib, dtype) instance.

    Keyed by the realpaths + dtype so `prepare_instance.sh` and
    `run_instance.sh` independently derive the same path for the same instance.
    """
    key = '|'.join([
        os.path.realpath(onnx_path),
        os.path.realpath(vnnlib_path),
        np.dtype(dtype).name,
        f'v{CACHE_FORMAT_VERSION}',
    ])
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]
    return os.path.join(cache_dir(), f'{digest}.pkl')


def _source_stamp(onnx_path, vnnlib_path, dtype):
    """Identity stamp recorded in the cache to detect staleness on load."""
    return {
        'version': CACHE_FORMAT_VERSION,
        'onnx': os.path.realpath(onnx_path),
        'vnnlib': os.path.realpath(vnnlib_path),
        'dtype': np.dtype(dtype).name,
        'onnx_mtime': os.path.getmtime(onnx_path),
        'vnnlib_mtime': os.path.getmtime(vnnlib_path),
    }


def write_cache(onnx_path, vnnlib_path, dtype):
    """Parse the instance and pickle (graph, spec, stamp) to its cache path.

    Returns the cache path. Called by prepare_instance.sh (via `--write-pkl`).
    """
    from .network import ComputeGraph
    from .vnnlib_loader import load_vnnlib

    graph = ComputeGraph.from_onnx(onnx_path, dtype=dtype)
    spec = load_vnnlib(vnnlib_path)

    out_path = pkl_cache_path(onnx_path, vnnlib_path, dtype)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'stamp': _source_stamp(onnx_path, vnnlib_path, dtype),
        'graph': graph,
        'spec': spec,
    }
    # Write to a temp file + atomic rename so a concurrent/aborted prepare
    # never leaves a half-written cache that a run would load.
    tmp_path = out_path + f'.tmp{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)
    return out_path


def load_cache(onnx_path, vnnlib_path, dtype):
    """Load a pre-parsed (graph, spec) for this instance, or None.

    Returns None (caller parses normally) if no cache exists, the cache is
    stale (version / source path / mtime mismatch), or it fails to load. Only
    call this when the user passed --allow-unsafe-pkl-loading.
    """
    path = pkl_cache_path(onnx_path, vnnlib_path, dtype)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError,
            ImportError, ValueError) as e:
        # Corrupt / version-skewed cache → fall back to a normal parse. Narrow
        # set: these are the failure modes of loading a stale-but-present pkl.
        print(f'  [pkl] ignoring unreadable cache {path}: '
              f'{type(e).__name__}: {e}')
        return None
    stamp = payload.get('stamp', {})
    expected = _source_stamp(onnx_path, vnnlib_path, dtype)
    if stamp != expected:
        # Source files changed (or different instance hashed to this path):
        # don't trust it.
        print(f'  [pkl] cache {path} is stale (source changed); reparsing')
        return None
    return payload['graph'], payload['spec']
