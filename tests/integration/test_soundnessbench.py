"""Integration tests for soundnessbench.

VNNCOMP regular track, SOUNDNESS-STRESS benchmark. 50 instances share ONE conv
net (`128 -> Gemm(12288) -> ReLU -> Reshape(3,64,64) -> Conv x6 -> Gemm(384)`,
~240K ReLUs); each `model_i.vnnlib` is a different input box (width 1.0/dim,
128-D) hiding a different adversarial counterexample. EVERY instance is SAT —
the benchmark exists to catch UNSOUND verifiers: a `verified`/`unsat` here means
the tool certified a property that is actually false.

How vibecheck solves it (see configs/soundnessbench.yaml): the CEXs are found by
DEEP PGD, mirroring AB-CROWN's attack-only config (pgd_steps=1000, lr_decay=
0.997). vibecheck routes to verify_graph so Phase-0 PGD runs BEFORE any bound
propagation (the dense forward zonotope would OOM at the wide 3->24 conv @ 64x64
~ 98k neurons — but it is never reached: PGD finds the witness first). 49/50
crack on alpha=0.01; model_26's basin needs alpha=0.05, so the config uses a
two-way multi-alpha [0.01, 0.05] at 500 restarts (250 each — keeps the slow
case's full 0.01 density while adding 0.05 for model_26). -> 50/50.

SOUNDNESS is enforced at the verdict itself: `_finalize` runs the PGD witness
through onnxruntime (`_validate_sat_witness`) and downgrades any spurious witness
to `unknown`. So every `sat` below is a genuine, onnxruntime-confirmed CEX — and
a future change that made us emit `verified`/`unsat` on these SAT cases would
fail these tests (expected='sat'). That IS the soundness guard for this
benchmark: we never false-verify.

All 50 instances are SAT (the lone `unsat` in AB-CROWN's CSV is the unrelated
`test/test_nano` sanity row). vibecheck cracks 50/50 sat — full parity with
AB-CROWN. The two-way multi-alpha makes each case ~85 s (still under the 150 s
competition budget).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'soundnessbench'
CONFIG_YAML = 'soundnessbench.yaml'

CASES = [
    dict(
        desc='soundnessbench model_0 (SAT, deep-PGD finds hidden CEX)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_0.vnnlib',
        expected='sat', timeout=150, max_wall_s=130.0,
    ),
    dict(
        desc='soundnessbench model_1 (SAT, deep-PGD finds hidden CEX)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_1.vnnlib',
        expected='sat', timeout=150, max_wall_s=130.0,
    ),
    dict(
        # The straggler: alpha=0.01 misses its basin; only the second alpha=0.05
        # in the two-way multi-alpha cracks it. Pins the multi-alpha fix.
        desc='soundnessbench model_26 (SAT, needs the alpha=0.05 multi-alpha leg)',
        net='onnx/model.onnx', vnnlib='vnnlib/model_26.vnnlib',
        expected='sat', timeout=150, max_wall_s=130.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_soundnessbench(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
