"""Integration tests for vggnet16_2022 — ImageNet VGG16 (3x224x224, 528 MB:
13 Conv + 5 MaxPool + 15 ReLU + 3 FC).

Runs the PRODUCTION path (`verify_graph` + `configs/vggnet16_2022.yaml`).

What this pins (the sparse-perturbation regime, spec0-14, which VC solves 15/15):
  - The exact `maxpool_to_relu` decomposition + patches-native slice/sub keep the
    forward zono in memory (no 1709 GiB dense maxpool materialisation), and
    input-split + forward-zono + CROWN verifies the few-pixel-perturbation specs.
  - spec0 is a real SAT (nominal misclassification), ORT-validated.
  - `auto_route_milp_for_conv: false` keeps conv graphs on the zono+CROWN path
    (milp/gpu_layers has no MaxPool handler).

NOT pinned: the 3 full-image L-inf cases (spec15/16/17, all 150528 px perturbed).
Those are an open gap — the box-reduced forward zono explodes and a sound bound
needs retightening ~all unstable deep-layer ReLUs via backward CROWN, ~2.7x over
the 1200 s budget at the current engine speed (see docs/benchmarks/vggnet16_2022.md).
A WIP patches-mode backward-CROWN engine exists (sound, bit-equivalent, gated OFF).

VGG16 is a 528 MB net; these cases take ~40-130 s each (loose max_wall_s — these
are regression catchers, not perf benchmarks).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'vggnet16_2022'
CONFIG_YAML = 'vggnet16_2022.yaml'

CASES = [
    dict(
        desc='vgg spec0 suit (SAT — nominal misclassification, ORT-validated)',
        net='onnx/vgg16-7.onnx',
        vnnlib='vnnlib/spec0_suit.vnnlib',
        expected='sat', timeout=200, max_wall_s=120.0,
    ),
    dict(
        desc='vgg spec1 Scottish_deerhound (UNSAT, sparse-perturbation)',
        net='onnx/vgg16-7.onnx',
        vnnlib='vnnlib/spec1_Scottish_deerhound.vnnlib',
        expected='verified', timeout=200, max_wall_s=120.0,
    ),
    dict(
        desc='vgg spec14 mink (UNSAT, hardest sparse case)',
        net='onnx/vgg16-7.onnx',
        vnnlib='vnnlib/spec14_mink.vnnlib',
        expected='verified', timeout=300, max_wall_s=260.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_vggnet16_2022(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
