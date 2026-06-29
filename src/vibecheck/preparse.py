"""Pre-parse cache for VNNCOMP runs.

Parsing a large ONNX (shape inference, constant folding, BatchNorm folding,
topo sort) plus the VNNLIB spec can take a noticeable slice of a tight
per-instance budget. `prepare_instance.sh` runs once per instance *before* the
timed run, so it is the natural place to do that work and stash the result.

We cache the ONNX graph and the VNNLIB spec SEPARATELY, each pickled to a file
named after its source so it is obvious what it corresponds to:

    <cache_dir>/<onnx-basename>.pkl     e.g. vgg16-7.onnx.pkl   -> ComputeGraph
    <cache_dir>/<vnnlib-basename>.pkl   e.g. instance_0.vnnlib.pkl -> VNNSpec

Splitting them means an ONNX shared across many specs (e.g. smart_turn's 50
instances, or acasxu nets reused across properties) is parsed ONCE and reused
for every spec, instead of rebuilding per (onnx, vnnlib) pair.

Each cache file stores a small METADATA record (the content sha1 of its source,
the format version, and for the graph the dtype) as the FIRST pickle object,
followed by the parsed object. The metadata is read first to decide whether the
cache is up to date for the current source bytes — so prepare can skip a rebuild
and a run can skip the graph load on a content mismatch, all without
materializing the big object.

SAFETY: pickle executes arbitrary code on load. The cache is therefore only READ
when the caller explicitly opts in via `--allow-unsafe-pkl-loading`, and it is
only ever WRITTEN by `--prepare-pkl-unsafe` (named to make the trust requirement
obvious). The content sha1 here is a STALENESS/correctness check (is this the
cache for these exact bytes?), NOT a security boundary — a recomputable hash
cannot stop someone who can write the cache directory from planting a malicious
pkl. Security comes from the cache directory being trusted; never point this at
an untrusted location.

The cached graph is the PRE-`optimize()` form (optimize is settings-dependent,
so it stays a per-run step) — the expensive `from_onnx` parse is what we skip.
"""
import hashlib
import os
import pickle

import numpy as np

# Bump when the pickled object layout OR the cache scheme changes so stale caches
# are ignored rather than silently mis-loaded.
CACHE_FORMAT_VERSION = 4

_DEFAULT_CACHE_DIR = '/tmp/vibecheck_pkl'

_PKL_ERRORS = (pickle.UnpicklingError, EOFError, OSError, AttributeError,
               ImportError, ValueError, IndexError)


def cache_dir():
    """Directory for pre-parse caches (override via VIBECHECK_PKL_CACHE_DIR)."""
    return os.environ.get('VIBECHECK_PKL_CACHE_DIR', _DEFAULT_CACHE_DIR)


def _file_sha1(path):
    """Streaming sha1 of a file's raw bytes (chunked so a 500 MB ONNX doesn't
    land in memory all at once)."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def onnx_pkl_path(onnx_path):
    """Graph cache path: <cache_dir>/<onnx-basename>.pkl (e.g. vgg16-7.onnx.pkl)."""
    return os.path.join(cache_dir(), os.path.basename(onnx_path) + '.pkl')


def vnnlib_pkl_path(vnnlib_path):
    """Spec cache path: <cache_dir>/<vnnlib-basename>.pkl (e.g. prop_1.vnnlib.pkl)."""
    return os.path.join(cache_dir(), os.path.basename(vnnlib_path) + '.pkl')


def _read_meta(path):
    """Read ONLY the metadata record (first pickle object) of a cache file, or
    None if missing/unreadable/old-format. Does not load the big object."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            meta = pickle.load(f)
    except _PKL_ERRORS:
        return None
    if not isinstance(meta, dict) or meta.get('version') != CACHE_FORMAT_VERSION:
        return None
    return meta


def _write(path, meta, obj):
    """Atomically write (meta, obj) as two sequential pickle objects."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# --------------------------------------------------------------------- ONNX graph
def write_onnx_cache(onnx_path, dtype):
    """Parse + cache the ONNX graph, keyed by content sha1 (+ dtype). Skips the
    parse if an up-to-date cache for these exact bytes already exists. Returns
    the cache path."""
    sha1 = _file_sha1(onnx_path)
    dt = np.dtype(dtype).name
    path = onnx_pkl_path(onnx_path)
    meta = _read_meta(path)
    if meta and meta.get('kind') == 'onnx' and meta.get('sha1') == sha1 \
            and meta.get('dtype') == dt:
        return path                                   # up to date, no rebuild
    from .network import ComputeGraph
    graph = ComputeGraph.from_onnx(onnx_path, dtype=dtype)
    _write(path, {'kind': 'onnx', 'version': CACHE_FORMAT_VERSION, 'sha1': sha1,
                  'dtype': dt, 'source': os.path.realpath(onnx_path)}, graph)
    return path


def load_onnx_cache(onnx_path, dtype):
    """Return the cached ComputeGraph for this ONNX (content sha1 + dtype match),
    or None. Only call when --allow-unsafe-pkl-loading was passed."""
    path = onnx_pkl_path(onnx_path)
    if not os.path.isfile(path):
        return None
    sha1 = _file_sha1(onnx_path)
    dt = np.dtype(dtype).name
    try:
        with open(path, 'rb') as f:
            meta = pickle.load(f)
            if not (isinstance(meta, dict) and meta.get('kind') == 'onnx'
                    and meta.get('version') == CACHE_FORMAT_VERSION
                    and meta.get('sha1') == sha1 and meta.get('dtype') == dt):
                return None                           # stale/wrong: don't load graph
            return pickle.load(f)                     # the ComputeGraph
    except _PKL_ERRORS as e:
        print(f'  [pkl] ignoring unreadable graph cache {path}: '
              f'{type(e).__name__}: {e}')
        return None


# --------------------------------------------------------------------- VNNLIB spec
def write_vnnlib_cache(vnnlib_path):
    """Parse + cache the VNNLIB spec, keyed by content sha1. Skips the parse if an
    up-to-date cache already exists. Returns the cache path."""
    sha1 = _file_sha1(vnnlib_path)
    path = vnnlib_pkl_path(vnnlib_path)
    meta = _read_meta(path)
    if meta and meta.get('kind') == 'vnnlib' and meta.get('sha1') == sha1:
        return path
    from .vnnlib_loader import load_vnnlib
    spec = load_vnnlib(vnnlib_path)
    _write(path, {'kind': 'vnnlib', 'version': CACHE_FORMAT_VERSION, 'sha1': sha1,
                  'source': os.path.realpath(vnnlib_path)}, spec)
    return path


def load_vnnlib_cache(vnnlib_path):
    """Return the cached VNNSpec for this VNNLIB (content sha1 match), or None.
    Only call when --allow-unsafe-pkl-loading was passed."""
    path = vnnlib_pkl_path(vnnlib_path)
    if not os.path.isfile(path):
        return None
    sha1 = _file_sha1(vnnlib_path)
    try:
        with open(path, 'rb') as f:
            meta = pickle.load(f)
            if not (isinstance(meta, dict) and meta.get('kind') == 'vnnlib'
                    and meta.get('version') == CACHE_FORMAT_VERSION
                    and meta.get('sha1') == sha1):
                return None
            return pickle.load(f)                     # the VNNSpec
    except _PKL_ERRORS as e:
        print(f'  [pkl] ignoring unreadable spec cache {path}: '
              f'{type(e).__name__}: {e}')
        return None


# A net/spec the pre-parse cache cannot build. NOT a bug and NOT swallowed-silently:
# the cache is a pure PERF optimization, and the timed run re-parses from source
# (the authoritative path) — nonlinear-v2 specs go through the augment, cctsdb_yolo
# nets through the custom YOLO handler, neither of which uses these caches. So we
# LOG the skip explicitly and continue. This catch is safe vs the no-swallow rule:
# for any benchmark whose RUN does use load_onnx/load_vnnlib, the same parse runs at
# verify time too, so a genuine loader bug still surfaces LOUDLY there (recorded as
# 'error') — it is never hidden, only the (optional) cache for it is skipped.
#   NotImplementedError: vnnlib_loader rejects a degree>=2 (nonlinear) spec atom.
#   ValueError / IndexError: onnx_loader shape inference on an unsupported net.
_UNCACHEABLE = (NotImplementedError, ValueError, IndexError)


# ------------------------------------------------------------------- combined API
def write_cache(onnx_path, vnnlib_path, dtype):
    """Cache both the ONNX graph and the VNNLIB spec. Returns (onnx_pkl, vnnlib_pkl);
    either element is None when that part could not be pre-parsed (logged + skipped —
    the timed run parses it from source). A failure to cache one never aborts the
    other."""
    try:
        onnx_pkl = write_onnx_cache(onnx_path, dtype)
    except _UNCACHEABLE as e:
        print(f'  [prepare] onnx pre-parse cache SKIPPED for '
              f'{os.path.basename(onnx_path)} ({type(e).__name__}: {e}); the timed '
              f'run will parse this net from source.', flush=True)
        onnx_pkl = None
    try:
        vnnlib_pkl = write_vnnlib_cache(vnnlib_path)
    except _UNCACHEABLE as e:
        print(f'  [prepare] vnnlib pre-parse cache SKIPPED for '
              f'{os.path.basename(vnnlib_path)} ({type(e).__name__}: {e}); the timed '
              f'run will parse this spec from source (e.g. nonlinear-v2 -> augment).',
              flush=True)
        vnnlib_pkl = None
    return onnx_pkl, vnnlib_pkl


def load_cache(onnx_path, vnnlib_path, dtype):
    """Load both caches independently. Returns (graph_or_None, spec_or_None) — each
    is None on a miss so the caller parses just that part."""
    return load_onnx_cache(onnx_path, dtype), load_vnnlib_cache(vnnlib_path)
