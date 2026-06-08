"""Integration tests for cgan_2023.

VNNCOMP regular track. Conditional-GAN benchmark: ConvTranspose+ReLU
generators, plus a Tanh+Sigmoid model, an Upsample-9 model, a
ConvTranspose-with-output_padding model, and a small attention
transformer. 21 instances total.

Probed locally with the cgan_2023 config (input_split_batched + clipping,
batch=16, 120 s/case). 21/21 cases pass; SAT cases run sub-second to ~5 s,
UNSAT cases ~5-15 s.

Cases:
  - small_transformer prop_0 + prop_1 (SAT, ~11 s) — SAGAN-style
    self-attention generator (spectral-norm Conv/Gemm, bilinear-MatMul
    attention, Softmax, max-pool). Loading depends on **onnxsim**
    (`onnx_loader.load_onnx` auto-simplify trigger): it constant-folds
    the spectral-norm weight divisions and the dynamic Shape→Gather→
    Concat reshape subgraphs into static form. Without onnxsim the load
    crashes on the non-constant Gemm weight — this is exactly why the
    full AWS sweep missed both cases (its venv lacked onnxsim, now a
    hard dependency). After load, gpu_graph still can't represent
    attention, so the SAT witness is found via the raw-ONNX PGD
    fallback (`onnx_torch_runner.pgd_via_onnx`) and validated against
    the original ONNX.
  - imgSz32_nCh_3 prop_0 (UNSAT, ~14 s) — baseline ConvTranspose+ReLU
    cGAN. Catches regressions in `propagate_conv_transpose` (zono and
    CROWN backward).
  - imgSz32_nCh_3_nonlinear_activations prop_0 (UNSAT, ~14 s) — Sigmoid
    + Tanh layers. Catches regressions in `_sigmoid_tanh_linear_bounds`
    (closed-form / binary-search Newton tangent points; soundness is
    spotchecked in `tests/test_sigmoid_tanh_bounds.py`).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'cgan_2023'
CONFIG_YAML = 'cgan_2023.yaml'

CASES = [
    dict(
        desc='cgan small_transformer prop_0 (SAT via onnx-PGD fallback)',
        net='onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx',
        vnnlib=('vnnlib/cGAN_imgSz32_nCh_3_small_transformer_prop_0_'
                'input_eps_0.005_output_eps_0.010.vnnlib'),
        expected='sat', timeout=60, max_wall_s=20.0,
        # 272 MB onnx — absent from some thin benchmark mirrors. Skip
        # when missing rather than fail. Requires onnxsim at load time.
        skip_if_missing=True,
    ),
    dict(
        desc='cgan small_transformer prop_1 (SAT via onnx-PGD fallback)',
        net='onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx',
        vnnlib=('vnnlib/cGAN_imgSz32_nCh_3_small_transformer_prop_1_'
                'input_eps_0.010_output_eps_0.015.vnnlib'),
        expected='sat', timeout=60, max_wall_s=20.0,
        skip_if_missing=True,
    ),
    dict(
        desc='cgan imgSz32_nCh_3 prop_0 (UNSAT, ConvTranspose+ReLU)',
        net='onnx/cGAN_imgSz32_nCh_3.onnx',
        vnnlib=('vnnlib/cGAN_imgSz32_nCh_3_prop_0_input_eps_0.015_'
                'output_eps_0.020.vnnlib'),
        expected='verified', timeout=60, max_wall_s=25.0,
    ),
    dict(
        desc='cgan nonlinear_activations prop_0 (UNSAT, Sigmoid+Tanh)',
        net='onnx/cGAN_imgSz32_nCh_3_nonlinear_activations.onnx',
        vnnlib=('vnnlib/cGAN_imgSz32_nCh_3_nonlinear_activations_'
                'prop_0_input_eps_0.015_output_eps_0.020.vnnlib'),
        expected='verified', timeout=60, max_wall_s=25.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cgan_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
