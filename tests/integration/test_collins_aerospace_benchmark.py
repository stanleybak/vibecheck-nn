"""Integration pin for collins_aerospace_benchmark — the generic onnx2torch PGD attack mode.

The model is a YOLOv5-nano (640x640, 60 Conv + LeakyRelu + Sigmoid) and the spec is a YOLO
detection robustness property with a tiny perturbed input patch (~405 of 1.23M dims) and a
19-way output disjunction. vibecheck can't bound it cheaply, so this pins src/vibecheck/
torch_attack: PGD over the perturbed dims finds the CE, validated on the ORIGINAL model via
CPU onnxruntime. All 6 instances are sat (ABC ~95s each); vibecheck finds a clear CE almost
immediately. Needs the 2026 clone (127MB specs) + a CUDA GPU + onnx2torch; skips otherwise.
See docs/benchmarks/collins_aerospace_benchmark.md.
"""
import os

import pytest

_CANDIDATES = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'collins_aerospace_benchmark/1.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'collins_aerospace_benchmark/1.0'),
]
_ONNX = 'onnx/yolov5nano_LRelu_640.onnx'
CASES = ['img_14421_perturbed_bbox_3_delta_0.001', 'img_12761_perturbed_bbox_0_delta_0.1']


def _bench_dir():
    return next((d for d in _CANDIDATES if os.path.isdir(d)), None)


@pytest.mark.integration
@pytest.mark.parametrize('stem', CASES)
def test_collins_torch_attack_sat(stem):
    bench = _bench_dir()
    if bench is None:
        pytest.skip('collins_aerospace_benchmark 1.0 not found locally')
    pytest.importorskip('onnx2torch')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('torch-attack on a 640x640 YOLOv5 needs a CUDA GPU')
    onnx = f'{bench}/{_ONNX}'
    vnnlib = f'{bench}/vnnlib/{stem}.vnnlib'
    if not (os.path.exists(onnx) and os.path.exists(vnnlib)):
        pytest.skip('collins files missing')

    from vibecheck import torch_attack as ta
    from vibecheck.settings import default_settings
    settings = default_settings(torch_attack=True, torch_attack_restarts=2,
                                torch_attack_steps=4, device='gpu')
    verdict, wit = ta.torch_attack(onnx, vnnlib, settings, timeout=300, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    from vibecheck.vnnlib_loader import load_vnnlib
    y = ta._ort_eval(onnx, wit)
    margin = ta._worst_margin_np(y, load_vnnlib(vnnlib).disjuncts)
    assert margin <= float(settings.sat_validate_atol), f'witness ORT margin {margin} not a CE'
