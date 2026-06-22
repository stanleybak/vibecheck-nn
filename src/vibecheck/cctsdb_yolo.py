"""cctsdb_yolo custom handler — COMPLETE verification by discrete patch-position enumeration.

The CCTSDB YOLO "patch" benchmarks bake the patch placement (ScatterND/Range/Where) + YOLO
detector + post-processing (ArgMax/dynamic-Min/Max) into one ONNX that neither vibecheck's
graph loader nor onnx2torch can ingest. But the property is NOT a continuous-input robustness
question: the vnnlib fixes the entire image (and the ground-truth box/label) and only varies
two INTEGER patch-position coordinates over a small range. So the unsafe set is a FINITE grid
of patch placements, and we can verify COMPLETELY by enumeration:

  for every integer patch position: run the ORIGINAL net via onnxruntime (which handles all the
  control-flow ops) -> detection score Y; if any position makes the output spec satisfiable
  (a CE) -> sat; if all positions are safe -> unsat.

This is exactly how alpha,beta-CROWN solves it (a custom model loader + enumeration); it bounds
nothing. We go one step simpler than ABC (which extracts the conv backbone and re-implements
the post-processing in torch for batched GPU inference): unbatched ORT on the *full original
net* is ~1.6s for 3844 positions and uses the authoritative graph directly, so there is no
re-implemented post-processing to trust — the sat/unsat verdict is decided only by the original
model on onnxruntime (the scoring engine).

SOUNDNESS: the verdict is complete only under the benchmark's discrete semantics — the patch
position is an INTEGER pixel offset. We assert the free input dims are integer-valued and
enumerate every integer in [lo, hi) (exclusive hi, matching the benchmark/ABC); a non-integer
free range or an enumeration larger than `cctsdb_max_positions` raises rather than guessing.
"""
import itertools
import time

import numpy as np

from .surrogate_pgd import _model_input_shapes
from .sign_attack import _worst_margin_np


def has_cctsdb_structure(onnx_path, vnnlib_path):
    """True if this looks like a discrete-patch instance: a single-input net whose vnnlib leaves
    only a FEW integer-valued input dims free (the patch positions), the rest fixed."""
    from .io_util import ensure_decompressed
    from .vnnlib_loader import load_vnnlib
    if len(_model_input_shapes(onnx_path)) != 1:
        return False
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    lo = np.asarray(spec.x_lo, np.float64); hi = np.asarray(spec.x_hi, np.float64)
    free = [d for d in range(lo.size) if hi[d] - lo[d] > 1e-6]
    if not free or len(free) > 4:
        return False
    return all(abs(lo[d] - round(lo[d])) < 1e-6 and abs(hi[d] - round(hi[d])) < 1e-6
               for d in free)


def cctsdb_yolo_verify(onnx_path, vnnlib_path, settings, timeout, log=print):
    """Enumerate the integer patch-position grid through the ORIGINAL net on ORT-CPU. Returns
    (verdict, witness): verdict in {'unsat','sat','timeout'}; witness is [input np.ndarray] for
    sat (the violating placement), else None. Complete: 'unsat' = every placement is safe."""
    import onnxruntime as ort
    from .io_util import ensure_decompressed
    from .vnnlib_loader import load_vnnlib

    t0 = time.time()
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    lo = np.asarray(spec.x_lo, np.float64); hi = np.asarray(spec.x_hi, np.float64)
    atol = float(getattr(settings, 'sat_validate_atol', 1e-4))
    max_pos = int(getattr(settings, 'cctsdb_max_positions', 1_000_000))
    free = [d for d in range(lo.size) if hi[d] - lo[d] > 1e-6]
    for d in free:
        if abs(lo[d] - round(lo[d])) > 1e-6 or abs(hi[d] - round(hi[d])) > 1e-6:
            raise NotImplementedError(
                f'cctsdb_yolo: free input dim {d} range [{lo[d]},{hi[d]}] is not integer-valued '
                f'— this is not a discrete-patch instance')
    ranges = [range(int(round(lo[d])), int(round(hi[d]))) for d in free]   # exclusive hi (= ABC)
    total = int(np.prod([len(r) for r in ranges])) if ranges else 0
    if total <= 0 or total > max_pos:
        raise NotImplementedError(
            f'cctsdb_yolo: {total} positions to enumerate over free dims {free} '
            f'(cap {max_pos}) — not a discrete-patch instance?')

    in_shape = _model_input_shapes(onnx_path)[0]
    sess = ort.InferenceSession(ensure_decompressed(onnx_path),
                                providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    base = lo.copy()
    log(f'[cctsdb] enumerating {total} integer patch positions over free dims {free}')

    within_tol = [None]
    n = 0
    for combo in itertools.product(*ranges):
        if time.time() - t0 > timeout:
            if within_tol[0] is not None:
                log(f'[cctsdb] timeout after {n}/{total} — emitting within-tol CE')
                return 'sat', within_tol[0]
            log(f'[cctsdb] timeout after {n}/{total} positions')
            return 'timeout', None
        x = base.copy()
        for d, v in zip(free, combo):
            x[d] = v
        feed = x.reshape(in_shape).astype(np.float32)
        y = np.asarray(sess.run([oname], {iname: feed})[0]).ravel()
        n += 1
        m = _worst_margin_np(y, spec.disjuncts)
        if m < 0.0:
            log(f'[cctsdb] CLEAR SAT at position {tuple(combo)} (worst_margin={m:.3e})')
            return 'sat', [feed]
        if m <= atol and within_tol[0] is None:
            within_tol[0] = [feed]
            log(f'[cctsdb] within-tol CE at {tuple(combo)} (worst_margin={m:.3e}, atol={atol:g})')
    if within_tol[0] is not None:
        log(f'[cctsdb] no clear CE — emitting within-tol CE (t={time.time()-t0:.1f}s)')
        return 'sat', within_tol[0]
    log(f'[cctsdb] all {n} positions safe -> unsat (complete) (t={time.time()-t0:.1f}s)')
    return 'unsat', None
