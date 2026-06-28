"""Backstop hard-timeout watchdog (main.py).

A phase that ignores the cooperative ``--timeout`` would otherwise leave the
process running until the competition harness SIGKILLs it — logged as a
penalized ``run_instance_timeout``. The watchdog forces a clean self-exit a
short grace past the budget so the harness records our pre-seeded ``timeout``
verdict instead. It must exit from a Timer THREAD (the main thread may be stuck
in a C extension), hence ``os._exit``.
"""
import threading

import pytest

import vibecheck.main as m


def test_hard_timeout_fire_calls_os_exit(monkeypatch, capsys):
    """`_hard_timeout_fire` writes a diagnostic and forces os._exit(1)."""
    calls = {}

    def _fake_exit(code):
        calls['code'] = code
        raise SystemExit(code)        # stand in for the real abrupt exit

    monkeypatch.setattr(m.os, '_exit', _fake_exit)
    with pytest.raises(SystemExit):
        m._hard_timeout_fire(30, 8)
    assert calls['code'] == 1
    err = capsys.readouterr().err
    assert 'hard-timeout' in err and '--timeout=30s' in err


def test_arm_hard_timeout_returns_started_daemon_timer():
    """`_arm_hard_timeout` arms a daemon Timer at timeout+grace; cancel works."""
    t = m._arm_hard_timeout(100, 8)
    try:
        assert isinstance(t, threading.Timer)
        assert t.daemon is True
        assert t.interval == 108.0
        assert t.is_alive()           # started, not yet fired
    finally:
        t.cancel()
    t.join(timeout=1)
    assert not t.is_alive()


def test_arm_hard_timeout_actually_fires(monkeypatch):
    """End-to-end: a tiny interval really invokes the fire path in its thread."""
    fired = threading.Event()
    monkeypatch.setattr(m, '_hard_timeout_fire',
                        lambda *_a: fired.set())
    # timeout 0 + grace 0 -> fires almost immediately.
    t = m._arm_hard_timeout(0, 0)
    assert fired.wait(timeout=2), 'watchdog Timer did not fire'
    t.cancel()


# --- OOM classifier for the gated graceful-OOM path (vggnet spec17) ----------
def test_is_oom_exception_classifies():
    """Host MemoryError and CUDA-style 'out of memory' RuntimeErrors are OOMs;
    an ordinary RuntimeError (a real bug) is NOT — so it still surfaces as
    'error', never masked."""
    assert m._is_oom_exception(MemoryError('host'))
    assert m._is_oom_exception(RuntimeError('CUDA out of memory. Tried to ...'))
    assert m._is_oom_exception(RuntimeError('GPU ran Out Of Memory'))  # case
    assert not m._is_oom_exception(RuntimeError('shape mismatch at layer 3'))
    assert not m._is_oom_exception(ValueError('bad spec'))
