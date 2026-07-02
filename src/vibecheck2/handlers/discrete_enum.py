"""Discrete integer-grid enumeration (cctsdb_yolo): thin wrapper over the
self-contained v1 implementation. Complete: 'unsat' means every placement
was evaluated safe on ORT-CPU. Triggered as the fallback when the graph
loader cannot model the net; raises NotImplementedError for instances that
are not discrete-patch, so the caller re-raises the original load error."""
from __future__ import annotations

import time

import numpy as np


def try_discrete_enum(onnx_path, vnnlib_path, timeout, log=print):
    """Returns (verdict, details). Raises NotImplementedError when the
    instance is not a discrete-patch enumeration case."""
    from vibecheck.cctsdb_yolo import cctsdb_yolo_verify
    from vibecheck.settings import default_settings
    t0 = time.time()
    verdict, witness = cctsdb_yolo_verify(
        onnx_path, vnnlib_path, default_settings(), timeout, log=log)
    details = {'time': time.time() - t0, 'handler': 'discrete_enum'}
    if verdict == 'sat' and witness:
        details['witness'] = np.asarray(witness[0]).ravel().astype(np.float64)
    return verdict, details
