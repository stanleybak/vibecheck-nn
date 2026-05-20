"""Tests for the ``raise_on_oom`` setting.

Default: any ``torch.cuda.OutOfMemoryError`` (or ``RuntimeError`` wrapping
one) that bubbles up from the zono / CROWN / α-CROWN path must re-raise to
the caller. Silent CPU fallback or other silent degradation must require an
explicit two-knob opt-in (``allow_cpu_fallback=True`` AND
``raise_on_oom=False``) so the user always sees memory regressions.
"""
import numpy as np
import pytest
import torch

from vibecheck.settings import default_settings


def test_default_raises_on_oom():
    """Default settings: raise_on_oom=True, allow_cpu_fallback=False."""
    s = default_settings()
    assert s.raise_on_oom is True
    assert s.allow_cpu_fallback is False


def test_fallback_requires_both_knobs():
    """Changing only one knob doesn't enable fallback."""
    s1 = default_settings(allow_cpu_fallback=True)
    assert s1.allow_cpu_fallback is True
    assert s1.raise_on_oom is True  # still default, so still raises

    s2 = default_settings(raise_on_oom=False)
    assert s2.raise_on_oom is False
    assert s2.allow_cpu_fallback is False  # still default, no fallback path

    s3 = default_settings(allow_cpu_fallback=True, raise_on_oom=False)
    assert s3.allow_cpu_fallback is True
    assert s3.raise_on_oom is False


def test_verify_milp_honors_raise_on_oom(monkeypatch, tmp_path):
    """verify_milp._milp_verify_graph must re-raise OOM unless BOTH
    ``allow_cpu_fallback=True`` AND ``raise_on_oom=False``."""
    # We don't need to actually trigger an OOM — just verify the boolean
    # gate in the except block by patching _forward_zonotope_graph to raise
    # OutOfMemoryError, and checking that the surrounding try/except
    # re-raises under default settings.
    from vibecheck import verify_milp

    # Replace _forward_zonotope_graph so the first call raises; record how
    # many times it was called. Under default settings we expect exactly 1
    # call (then re-raise). Under opt-in we'd expect 2 (then CPU retry).
    calls = {'n': 0}
    orig = verify_milp._forward_zonotope_graph

    def raising(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise torch.cuda.OutOfMemoryError('synthetic OOM for test')
        return orig(*args, **kwargs)

    monkeypatch.setattr(verify_milp, '_forward_zonotope_graph', raising)

    # Fake minimal graph + spec; we only reach the OOM point.
    # Simplest: use a trivial ONNX we can construct inline.
    from vibecheck.network import ComputeGraph
    from vibecheck.spec import VNNSpec, Conjunct, Constraint
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    W = np.eye(2, dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    nodes = [
        helper.make_node('Gemm', ['X', 'W', 'b'], ['Y']),
    ]
    graph_ = helper.make_graph(
        nodes, 'tiny',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(W, 'W'),
         numpy_helper.from_array(b, 'b')])
    model = helper.make_model(
        graph_, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / 'tiny.onnx')
    onnx.save(model, path)
    g = ComputeGraph.from_onnx(path)
    # Synthetic fork to force graph path.
    g.fork_points = lambda: {g.input_name}
    spec = VNNSpec(
        x_lo=np.zeros(2, dtype=np.float32),
        x_hi=0.01 * np.ones(2, dtype=np.float32),
        disjuncts=[Conjunct(
            [Constraint(index=0, op='>=', value=100.0)])])  # unreachable

    # 1. Default (raise_on_oom=True, allow_cpu_fallback=False): raise.
    calls['n'] = 0
    s = default_settings(device='cpu', total_timeout=5,
                         print_progress=False)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='synthetic'):
        verify_milp.milp_verify(g, spec, s)
    assert calls['n'] == 1, 'should NOT retry on default settings'

    # 2. allow_cpu_fallback=True alone: still raise (raise_on_oom still True).
    calls['n'] = 0
    s = default_settings(device='cpu', total_timeout=5,
                         print_progress=False, allow_cpu_fallback=True)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='synthetic'):
        verify_milp.milp_verify(g, spec, s)
    assert calls['n'] == 1

    # 3. raise_on_oom=False alone: still raise (fallback requires both).
    calls['n'] = 0
    s = default_settings(device='cpu', total_timeout=5,
                         print_progress=False, raise_on_oom=False)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='synthetic'):
        verify_milp.milp_verify(g, spec, s)
    assert calls['n'] == 1

    # 4. Both knobs set: fallback should kick in (call count == 2).
    # (But device.type == 'cpu' here, so the gate also requires
    # device.type != 'cpu' — which is false on CPU. Instead we just
    # confirm that on CPU the device-gate correctly re-raises.)
    calls['n'] = 0
    s = default_settings(device='cpu', total_timeout=5,
                         print_progress=False, allow_cpu_fallback=True,
                         raise_on_oom=False)
    with pytest.raises(torch.cuda.OutOfMemoryError, match='synthetic'):
        verify_milp.milp_verify(g, spec, s)
    # On a CPU device the `device.type != 'cpu'` gate itself blocks the
    # fallback — we can't retry "to CPU" if we're already on CPU.
    assert calls['n'] == 1
