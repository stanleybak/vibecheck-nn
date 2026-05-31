"""Integration tests for soundnessbench.

VNNCOMP regular track, SOUNDNESS-STRESS benchmark. 50 instances share ONE conv
net (`128 -> Gemm(12288) -> ReLU -> Reshape(3,64,64) -> Conv x6 -> Gemm(384)`,
~240K ReLUs); each `model_i.vnnlib` is a different input box (width 1.0/dim,
128-D) hiding a different adversarial counterexample. EVERY instance is SAT —
the benchmark exists to catch UNSOUND verifiers: a `verified`/`unsat` here means
the tool certified a property that is actually false.

How vibecheck solves it (see configs/soundnessbench.yaml): the CEXs are found by
DEEP PGD, mirroring AB-CROWN's attack-only config (pgd_steps=1000, alpha=0.005,
lr_decay=0.997). vibecheck routes to verify_graph so Phase-0 PGD runs BEFORE any
bound propagation (the dense forward zonotope would OOM at the wide 3->24 conv
@ 64x64 ~ 98k neurons — but it is never reached: PGD finds the witness first).

SOUNDNESS is enforced at the verdict itself: `_finalize` runs the PGD witness
through onnxruntime (`_validate_sat_witness`) and downgrades any spurious witness
to `unknown`. So every `sat` below is a genuine, onnxruntime-confirmed CEX — and
a future change that made us emit `verified`/`unsat` on these SAT cases would
fail these tests (expected='sat'). That IS the soundness guard for this
benchmark: we never false-verify.

All 50 instances are SAT (the lone `unsat` in AB-CROWN's CSV is the unrelated
`test/test_nano` sanity row). vibecheck cracks 49/50 sat; the one miss, model_26,
is a hard instance AB-CROWN also spends 106 s on. ~30 s/case via Phase-0 PGD
(FASTER than AB-CROWN's ~50-125 s).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'soundnessbench'
CONFIG_YAML = 'soundnessbench.yaml'

CASES = [
    dict(
        desc='soundnessbench model_0 (SAT, deep-PGD finds hidden CEX)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_0.vnnlib',
        expected='sat', timeout=190, max_wall_s=110.0,
    ),
    dict(
        desc='soundnessbench model_1 (SAT, deep-PGD finds hidden CEX)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_1.vnnlib',
        expected='sat', timeout=190, max_wall_s=110.0,
    ),
    dict(
        desc='soundnessbench model_2 (SAT, deep-PGD finds hidden CEX)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_2.vnnlib',
        expected='sat', timeout=190, max_wall_s=110.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_soundnessbench(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
