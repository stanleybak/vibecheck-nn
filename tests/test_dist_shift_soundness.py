"""Soundness regression for dist_shift's sigmoid spec-MILP.

dist_shift `mnist_concat` index1285 is a CONFIRMED SAT case (AB-CROWN: sat;
vibecheck's own PGD finds a counterexample with onnxruntime Y_9 - Y_5 = +0.79,
so the unsafe region Y_9 >= Y_5 IS reachable). With sat-finding disabled, the
only way a verdict can come back is the bounds/MILP path — and on a SAT case
that path must NEVER return 'verified'.

Before the fix it did, via TWO independent gen-LP bugs (both measured with the
real builder + the validated CEX, see sound_audit/DIST_SHIFT_ROOTCAUSE.md):

  1. `_build_phase1_lp` / `_build_alpha_zono_lp` pinned the "gap"/trailing
     generator columns (a sigmoid's γ-slack noise symbols) to lb=ub=0, dropping
     their swing from the objective -> under-approximated the zonotope.
  2. `state_from_alpha_zono` seeded its positional column counter with
     `len(x_lo)` (the input DIMENSION, 792) instead of the input-GENERATOR count
     (8 perturbed dims), so cur_n_gens ran 784 too high (1596 vs n_gens=812) and
     every post-input ReLU e_new_col landed on the wrong (zero) column.

Either bug over-restricted the LP into a false INFEASIBLE -> false 'verified',
certifying UNSAT on an input the verifier can itself exhibit as SAT. Both are
fixed; index1285 now reaches the MILP path and returns a non-'verified' verdict.

`bab_refine_adapt_enabled=False` is set only to route past the unrelated
ReLU-adaptive assertion crash so the spec-MILP path is reached.
"""
import gc
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_dist_shift_mnist_concat_index1285_milp_soundness(vnncomp_benchmarks):
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.settings import default_settings
    from vibecheck.config_loader import load_config
    from vibecheck.verify_graph import verify_graph

    root = Path(vnncomp_benchmarks) / 'dist_shift_2023'
    net = root / 'onnx' / 'mnist_concat.onnx'
    vnn = root / 'vnnlib' / 'index1285_delta0.13.vnnlib'
    for p in (net, vnn):
        gz = Path(str(p) + '.gz')
        if not p.exists() and gz.exists():
            subprocess.run(['gunzip', '-kf', str(gz)], check=True)
    if not net.exists() or not vnn.exists():
        pytest.skip(f'dist_shift benchmark files not available: {net}')

    cfg_path = Path(__file__).resolve().parents[1] / 'configs' / 'dist_shift_2023.yaml'
    overrides = load_config(str(cfg_path)) if cfg_path.exists() else {}

    graph = ComputeGraph.from_onnx(str(net), dtype=np.float32)
    spec = load_vnnlib(str(vnn))
    settings = default_settings(
        device='gpu', bits=32, total_timeout=60,
        disable_sat_finding=True,        # isolate the bounds/MILP path
        bab_refine_adapt_enabled=False,  # route past the unrelated relu-adapt crash
        **overrides)
    settings.print_progress = False
    graph.optimize(settings)

    # This test runs last in the full integration suite; on a small (8 GB) GPU
    # the cumulative allocations of ~45 prior GPU tests can exhaust VRAM. Reclaim
    # first, and treat a genuine CUDA OOM as a resource limit (skip), NOT a
    # soundness failure — there is no 'verified' verdict to worry about if the
    # case never finishes. Soundness is still enforced standalone, early in the
    # suite, and on larger GPUs.
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Must reach a verdict (the gen-LP column model is now correctly aligned with
    # the sigmoid γ columns; a NotImplementedError here would be a regression in
    # that alignment, so we deliberately do NOT swallow it).
    try:
        result, _ = verify_graph(graph, spec, settings)
    except torch.cuda.OutOfMemoryError as e:
        pytest.skip(f'CUDA OOM under cumulative suite memory pressure on a small '
                    f'GPU (resource limit, not a soundness signal): {e}')
    assert result != 'verified', (
        f"UNSOUND: spec MILP returned 'verified' on the SAT case index1285 "
        f"(got {result!r}); a SAT case with sat-finding off must be "
        f"'unknown' (or 'sat' from a real MILP witness), never 'verified'")
