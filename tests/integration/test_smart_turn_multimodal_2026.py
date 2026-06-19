"""Integration pin for smart_turn_multimodal_2026 — the surrogate-attack mode.

The model is INT8-quantized (DequantizeLinear/QuantizeLinear), so vibecheck can't build a
sound graph and the standard _runner path (ComputeGraph.from_onnx + load_vnnlib) doesn't
apply. This pins src/vibecheck/surrogate_pgd: fold a float surrogate, PGD via onnx2torch
on the GPU, and confirm the counterexample violates the spec on the ORIGINAL quantized
model via CPU onnxruntime. Pins one trivial-at-center instance and one boundary instance
that only the STE gradient cracks. Needs the 2026 benchmark clone + a CUDA GPU +
onnx2torch; skips otherwise. See docs/benchmarks/smart_turn_multimodal_2026.md.
"""
import os

import pytest

_CANDIDATES = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'smart_turn_multimodal_2026/2.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'smart_turn_multimodal_2026/2.0'),
]


def _bench_dir():
    return next((d for d in _CANDIDATES if os.path.isdir(d)), None)


@pytest.mark.integration
@pytest.mark.parametrize('inst', [0, 5])   # 0 = boundary (STE-PGD), 5 = trivial-at-center
def test_smart_turn_surrogate_sat(inst):
    bench = _bench_dir()
    if bench is None:
        pytest.skip('smart_turn_multimodal_2026 2.0 benchmark not found locally')
    pytest.importorskip('onnx2torch')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('surrogate-attack needs a CUDA GPU')
    onnx = f'{bench}/onnx/smart-turn-multimodal-cpu.onnx'
    vnnlib = f'{bench}/vnnlib/instance_{inst}.vnnlib.gz'
    if not (os.path.exists(onnx) and os.path.exists(vnnlib)):
        pytest.skip('smart_turn files missing')

    from vibecheck import surrogate_pgd as sp
    from vibecheck.settings import default_settings
    assert sp.has_quantized_ops(onnx)
    settings = default_settings(surrogate_attack=True, surrogate_attack_restarts=2,
                                surrogate_attack_steps=40)
    verdict, wit = sp.surrogate_attack(onnx, vnnlib, settings, timeout=120, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    # the verdict is decided by the ORIGINAL quantized model on ORT-CPU (scoring engine)
    y = sp._ort_eval(onnx, wit)
    assert y[0] > 0.5
