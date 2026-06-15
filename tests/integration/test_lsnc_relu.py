"""Integration tests for lsnc_relu (Lyapunov stable neural control).

One ReLU net (6-in/8-out) with a convex quadratic Lyapunov value V(x)=u^T P u
(P PSD) and decrease dV. Spec = (OR of 13 box-violation disjuncts) AND a
per-instance level-set band Y_1=V(x) in [a,b], the band ANDed into every
disjunct. vibecheck = 80/80 (full parity with alpha,beta-CROWN).

Verified by the batched input-split BaB on DOMAIN THROUGHPUT (the McCormick
bound on the indefinite dV is loose; each UNSAT needs ~6.7M domains, closed in
~6-7 s). The cases below pin:
  - 1 SAT (q1): a level-set counterexample, found by the per-disjunct root
    PGD (pgd_per_restart_disjunct) in ~2 s.
  - 2 UNSAT (q0, q62): closed by the throughput fixes (K=1 vectorized split,
    GPU-native SB scoring, chunk worklist, band-query dedup). The dominant one
    is K=1: reverting it (the gate's old K=4 default) routes the split through
    a per-leaf Python loop and blows q0 from ~6.6 s to ~33 s -- the max_wall_s
    guard catches exactly that regression.

NOTE: run on a GPU with >=12 GB (AWS A10G), not the shared desktop GPU. On the
A10G these are ~4-5 s; the wall bounds (~2.5x the RTX-3080 time) absorb GPU
variance while still firing on a real split/worklist/dedup regression.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'lsnc_relu'
CONFIG_YAML = 'lsnc_relu.yaml'

CASES = [
    dict(
        desc='quadrotor2d_state_1 (SAT, level-set CE, per-disjunct root PGD ~2s)',
        net='onnx/relu_quadrotor2d_state.onnx',
        vnnlib='vnnlib/quadrotor2d_state_1.vnnlib',
        expected='sat', timeout=25, max_wall_s=12.0,
    ),
    dict(
        desc='quadrotor2d_state_0 (UNSAT, K=1 throughput BaB ~6.6s)',
        net='onnx/relu_quadrotor2d_state.onnx',
        vnnlib='vnnlib/quadrotor2d_state_0.vnnlib',
        expected='verified', timeout=25, max_wall_s=18.0,
    ),
    dict(
        desc='quadrotor2d_state_62 (UNSAT, K=1 throughput BaB ~6.7s)',
        net='onnx/relu_quadrotor2d_state.onnx',
        vnnlib='vnnlib/quadrotor2d_state_62.vnnlib',
        expected='verified', timeout=25, max_wall_s=18.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_lsnc_relu(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
