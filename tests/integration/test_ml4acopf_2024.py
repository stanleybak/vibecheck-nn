"""Integration tests for ml4acopf_2024 (AC optimal-power-flow physics nets).

These nets mix Gemm/ReLU MLP blocks with the AC power-flow physics: bilinear
products (V·V, V·V·cos, V·V·sin), self-product squares (V²), Sin/Cos, and Floor.
The verdicts here pin the 2026-06 ml4acopf work on the 14_ieee linear-residual
net (the trig is baked as a ReLU-PWL lookup, so the graph is bilinear+ReLU with
no live Sin/Cos op — the backward-CROWN path applies):

  - prop1 (SAT): the spec is violated at the operating point itself (the whole
    input box is unsafe, ORT margin ~-0.006). The nominal-point CE probe
    (`_acopf_nominal_cex_probe`) finds it on the reference ONNX in ~2 s and
    emits the VNNCOMP counterexample. (α,β-CROWN's clean-input attack does the
    same; vibecheck previously returned `unknown` here.)
  - prop2 (UNSAT): closed at the ROOT by backward-CROWN with topo-order
    intermediate-bound refinement (`_acopf_backward_crown_root`); the spec
    margin is +3.977, BEATING α,β-CROWN's init-CROWN +3.956. This is the case
    that exercises the gather fan-out adjoint soundness fix (index_add_) — the
    refinement is hugely negative (and unsound) without it.
  - prop3 (UNSAT): the binding constraint is a ReLU-only path (lb(Y_5)); plain
    CROWN gives -0.0096, and ReLU-lower-slope α-optimization in
    `_acopf_alpha_opt` lifts it to >0 (matching α,β-CROWN, which also closes it
    purely via ReLU-slope α). Pins the ReLU-α wiring.

The trig path overruns its budget on the big 118/300 nets but the verdict is
still correct (results file pre-seeded `timeout`); those are not pinned here
(both vibecheck and α,β-CROWN time out on the hard 118/300 props).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'ml4acopf_2024'
CONFIG_YAML = 'ml4acopf_2024.yaml'   # absent -> default settings (acopf path
                                     # auto-triggers on the graph, no profile)
NET = 'onnx/14_ieee_ml4acopf-linear-residual.onnx'
FULL_NET = 'onnx/14_ieee_ml4acopf.onnx'   # the real physics net (live Sin/Cos/Sigmoid)

CASES = [
    dict(
        desc='14_ieee prop1 (SAT, nominal-point CE probe ~2s)',
        net=NET, vnnlib='vnnlib/14_ieee_prop1.vnnlib',
        expected='sat', timeout=120, max_wall_s=20.0,
    ),
    dict(
        desc='14_ieee prop2 (UNSAT, backward-CROWN root +3.977, beats ABC)',
        net=NET, vnnlib='vnnlib/14_ieee_prop2.vnnlib',
        expected='verified', timeout=120, max_wall_s=30.0,
    ),
    dict(
        desc='14_ieee prop3 (UNSAT, ReLU-slope α-CROWN closes lb(Y_5))',
        net=NET, vnnlib='vnnlib/14_ieee_prop3.vnnlib',
        expected='verified', timeout=120, max_wall_s=35.0,
    ),
    dict(
        # FULL physics net (live Sin/Cos/Sigmoid). Was within-tol `sat` (forward
        # α-zono plateaued at -1.56e-7) until the dtype-aware band pad: the
        # float32-sized rel_pad (~1e-6) applied in float64 was a constant
        # inflation no α could remove. With it scaled to float64, the existing
        # α-CROWN closes it at +9.88e-7 — matching α,β-CROWN's +8.64e-7
        # (binding atom Y_6>=0.300001, a saturated sigmoid neuron). Pins the
        # rel_pad fix end-to-end through the production path.
        desc='14_ieee FULL-net prop3 (UNSAT, dtype-aware band pad → α-CROWN closes, matches ABC)',
        net=FULL_NET, vnnlib='vnnlib/14_ieee_prop3.vnnlib',
        expected='verified', timeout=120, max_wall_s=45.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_ml4acopf_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
