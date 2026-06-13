"""Phase heartbeat — diagnose where a verify run spends or *stalls* time.

The pipeline already prints once per phase, at the phase's END. That is
useless for a hang: a phase that never returns never prints, so a run that
overruns its deadline (e.g. challenging_certified_training wide-eps2 idx3310
ran to a 611 s shell-kill past its 550 s timeout) gives no clue which phase
is stuck.

This module runs a daemon thread that, every `interval_s` seconds, prints the
CURRENT phase marker with its in-phase elapsed time and GPU memory. A stalled
phase keeps reprinting the same marker with a growing in-phase time, which
pinpoints the hang. Phases call `set_phase(name)` at their boundaries.

Enabled via `settings.heartbeat_s` (0 = off) and `--heartbeat N` on the CLI.
Off by default and zero-overhead when off (no thread started).
"""
from __future__ import annotations

import sys
import threading
import time

_lock = threading.Lock()
_phase = "init"
_phase_t: float | None = None
_run_t: float | None = None
_thread: threading.Thread | None = None
_stop_evt: threading.Event | None = None


def _gpu_mem_gb() -> float | None:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return None


def set_phase(name: str) -> None:
    """Mark the phase the run is entering. Cheap; safe to call when off."""
    global _phase, _phase_t
    with _lock:
        _phase = str(name)
        _phase_t = time.perf_counter()


def start(interval_s: float | None, *, stream=None) -> None:
    """Start the heartbeat daemon. `interval_s` <= 0 / None is a no-op."""
    global _thread, _stop_evt, _run_t, _phase_t
    if not interval_s or interval_s <= 0 or _thread is not None:
        return
    out = stream if stream is not None else sys.stderr
    now = time.perf_counter()
    with _lock:
        _run_t = now
        if _phase_t is None:
            _phase_t = now
    evt = threading.Event()
    _stop_evt = evt

    def _loop() -> None:
        while not evt.wait(interval_s):
            with _lock:
                ph, pt, rt = _phase, _phase_t, _run_t
            t = time.perf_counter()
            mem = _gpu_mem_gb()
            mems = f"  gpu={mem:.2f}GB" if mem is not None else ""
            print(f"[heartbeat] phase={ph}  in-phase={t - pt:.0f}s  "
                  f"total={t - rt:.0f}s{mems}", file=out, flush=True)

    _thread = threading.Thread(target=_loop, name="vc-heartbeat", daemon=True)
    _thread.start()


def stop() -> None:
    """Stop the heartbeat daemon (idempotent)."""
    global _thread, _stop_evt
    if _stop_evt is not None:
        _stop_evt.set()
    _thread = None
    _stop_evt = None
