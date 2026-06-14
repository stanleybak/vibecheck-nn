"""Integration tests for challenging_certified_training_2026.

IBP-trained CNN7s (5 Conv + 2 Gemm, pure ReLU). A structural perturbation-
width gate routes high-uncertainty (eps8) instances to milp_verify's IBP +
α-CROWN + no-reforward β-CROWN BaB, and low-uncertainty (eps2) ones to the
zono+CROWN layers path. The cases below pin the BaB + SAT-finding engine:
  - 1 SAT (eps8, found by PGD in seconds)
  - 2 hard UNSAT (eps8) closed only by the no-reforward β-CROWN BaB — the
    cases that motivated the wire-in (9566_s3 / 8231_s3). α,β-CROWN solves
    these in 16 s / 24 s; we close them in ~91 s / ~39 s.
  - 1 SAT (tinyimagenet, targeted PGD): idx5613_s3 routes to the milp graph
    path via the input-dim gate. Phase 1 leaves only 4 of 199 disjuncts
    open; the default joint-loss PGD dilutes them across the 195 verified
    and misses the CE (we returned unknown@117 s), but the targeted PGD —
    restrict to the open disjuncts with per-restart targeting — finds it in
    ~3.5 s (sat@~7 s). Pins the `milp_graph_targeted_pgd_*` wiring.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'challenging_certified_training_2026/1.0'
CONFIG_YAML = 'challenging_certified_training_2026.yaml'

CASES = [
    dict(
        desc='cifar10 eps8_cnn7 idx3736_s1 (SAT, PGD)',
        net='onnx/cifar10_eps8_cnn7.onnx',
        vnnlib='vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx3736_sample1.vnnlib',
        expected='sat', timeout=30, max_wall_s=15.0,
    ),
    dict(
        # The flagship hard UNSAT: joint α leaves q3/q6 open at ~-0.07;
        # the no-reforward β-CROWN BaB closes them (q3 ~65, q6 ~97 domains).
        # Needs IBP Phase 1 + the self-consistent per-query base + batched α.
        desc='cifar10 eps8_cnn7 idx9566_s3 (hard UNSAT, BaB)',
        net='onnx/cifar10_eps8_cnn7.onnx',
        vnnlib='vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx9566_sample3.vnnlib',
        expected='verified', timeout=120, max_wall_s=115.0,
    ),
    dict(
        desc='cifar10 eps8_wide_cnn7 idx8231_s3 (hard UNSAT, BaB)',
        net='onnx/cifar10_eps8_wide_cnn7.onnx',
        vnnlib='vnnlib/cifar10_eps8_wide_cnn7/cifar10_eps8_wide_cnn7_idx8231_sample3.vnnlib',
        expected='verified', timeout=120, max_wall_s=75.0,
    ),
    dict(
        # tinyimagenet SAT closed by the targeted PGD (4 of 199 disjuncts
        # open → restrict + per-restart targeting). Internal time ~7.4 s,
        # very stable across seeds (8/8 sat with distinct witnesses).
        desc='tinyimagenet eps1_wide_cnn7 idx5613_s3 (SAT, targeted PGD)',
        net='onnx/tinyimagenet_eps1_wide_cnn7.onnx',
        vnnlib='vnnlib/tinyimagenet_eps1_wide_cnn7/'
               'tinyimagenet_eps1_wide_cnn7_idx5613_sample3.vnnlib',
        expected='sat', timeout=120, max_wall_s=20.0,
    ),
    dict(
        # The ONLY real ABC win in the benchmark's remaining-miss set
        # (ABC sat 8 s) — a narrow-CE-basin eps8 case. 5 of 9 disjuncts
        # open; the targeted PGD needs restarts=256 (128 was ~4/5 flaky,
        # OSI init didn't help) to hit the basin → 8/8 sat, CE in ~1 s.
        # Pins both the targeted-PGD wiring AND the 256 restart count.
        desc='cifar10 eps8_cnn7 idx1247_s9 (SAT, targeted PGD, narrow basin)',
        net='onnx/cifar10_eps8_cnn7.onnx',
        vnnlib='vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx1247_sample9.vnnlib',
        expected='sat', timeout=120, max_wall_s=20.0,
    ),
    dict(
        # The hardest UNSAT we close — a case ABC also solves and we used to
        # miss. ABC ablation proved our root bounds MATCH ABC's (-0.175 vs
        # -0.172) and interm-refresh is NOT the lever; the gap was pure BaB
        # throughput. The route-based high-throughput params for large-input
        # nets (n_in 12288 >= 8000: batch=192, α-iters=8, cand=2) reach the
        # ~3549 domains it needs (bound climbs monotone, frontier collapses
        # 1048->8). Pins the route-based BaB throughput config.
        desc='tinyimagenet eps1_cnn7 idx7018_s4 (hard UNSAT, route-based fast BaB)',
        net='onnx/tinyimagenet_eps1_cnn7.onnx',
        vnnlib='vnnlib/tinyimagenet_eps1_cnn7/'
               'tinyimagenet_eps1_cnn7_idx7018_sample4.vnnlib',
        expected='verified', timeout=550, max_wall_s=460.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_challenging_certified_training_2026(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
