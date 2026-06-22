"""Integration pin for traffic_signs_recognition_2023 — the Sign-BNN STE-PGD attack mode.

The GTSRB nets use `Sign` activations (binarized "QConv" layers) that neither vibecheck nor
ABC can bound soundly through onnx2pytorch, so the standard _runner path (ComputeGraph +
verify_graph) doesn't apply. This pins src/vibecheck/sign_attack: PGD on a per-layer
adaptive clipped-STE surrogate of `Sign` finds the adversarial CE, validated on the ORIGINAL
(true-Sign) model via CPU onnxruntime. All instances are sat or unknown (attack-only mode
never proves unsat), so the pins are sat cases — including idx_11379 eps_3, which stalled at
margin -1.0 until the per-layer adaptive eps fix (a fixed eps zeroed the huge first-conv
gradient). The tiny net runs fast on CPU. Needs the 2026 clone + onnx2torch; skips otherwise.
See docs/benchmarks/traffic_signs_recognition_2023.md.
"""
import os

import pytest

_CANDIDATES = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'traffic_signs_recognition_2023/1.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'traffic_signs_recognition_2023/1.0'),
]
_ONNX = 'onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx'

# All sat; idx_11379 eps_3 is the per-layer-adaptive-eps regression pin (was unknown).
CASES = [
    'model_30_idx_11379_eps_3.00000',
    'model_30_idx_7573_eps_3.00000',
    'model_30_idx_12375_eps_3.00000',
]


def _bench_dir():
    return next((d for d in _CANDIDATES if os.path.isdir(d)), None)


@pytest.mark.integration
@pytest.mark.parametrize('stem', CASES)
def test_traffic_signs_sign_attack_sat(stem):
    bench = _bench_dir()
    if bench is None:
        pytest.skip('traffic_signs_recognition_2023 1.0 benchmark not found locally')
    pytest.importorskip('onnx2torch')
    onnx = f'{bench}/{_ONNX}'
    vnnlib = f'{bench}/vnnlib/{stem}.vnnlib'
    if not (os.path.exists(onnx) and os.path.exists(vnnlib)):
        pytest.skip('traffic_signs files missing')

    from vibecheck import sign_attack as sa
    from vibecheck.settings import default_settings
    assert sa.has_sign_ops(onnx)
    settings = default_settings(sign_attack=True, sign_attack_restarts=4,
                                sign_attack_steps=60, device='cpu')
    verdict, wit = sa.sign_attack(onnx, vnnlib, settings, timeout=120, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    # the verdict is decided by the ORIGINAL (true-Sign) model on ORT-CPU (the scoring engine);
    # accepted iff the spec is violated within COUNTEREXAMPLE_ATOL.
    from vibecheck.vnnlib_loader import load_vnnlib
    y = sa._ort_eval(onnx, wit)
    margin = sa._worst_margin_np(y, load_vnnlib(vnnlib).disjuncts)
    assert margin <= float(settings.sat_validate_atol), f'witness ORT margin {margin} not a CE'
