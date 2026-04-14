"""BnB verification settings."""

from dotmap import DotMap
import torch


def default_settings(**overrides):
    """Create settings with defaults for BnB verification."""
    s = DotMap(
        device='gpu',
        bits=64,
        pgd_restarts=100,
        pgd_iter=10,
        bnb_order='bfs',
        bnb_timeout=30,
        print_progress=True,
        fuse_gemm_conv=True,
        optimize_relu_relation=True,
        bnb_max_depth=128,
        total_timeout=120.0,
        milp_sample_timeout=5.0,
        milp_scoring='ew_frac',  # 'crown', 'crown_lp_fractional', or 'ew_frac'
        milp_lp_per_worker=True,
        # Tightening options
        milp_tighten_method='lp',       # 'lp' or 'milp' for per-layer tightening
        milp_tighten_sparse=True,       # sparse per-neuron models for conv layers
        milp_tighten_parallel=True,     # parallel workers vs sequential
        milp_tighten_rebuild=False,     # rebuild model per worker vs copy shared
        milp_lp_encoding='compact',     # 'compact' (1 var, 2 constrs) or 'zas' (3 vars, 5 constrs)
        graph_impl='optimized',         # 'reference' or 'optimized' for verify_graph builder (Phase 1)
        spec_impl='gen_lp',             # 'gen_lp' (generator-based GPU) or 'monolithic' for Phase 7/8
        gen_lp_formulation='sparse',    # 'dense' or 'sparse' (applies when spec_impl='gen_lp'); sparse cuts at the last hidden ReLU to avoid numeric trouble
        max_tighten_layer=None,         # if set, only Phase 1 tightens layers <= this idx
        # Callback: called at key points with (event, info) -> bool (False = stop)
        milp_callback=None,
        # Lagrangian-decomposition solver (off by default; gated path in _run_pipeline)
        ld_enabled=False,
        ld_inner_mode='milp',              # 'milp' (big-M binary) or 'lp' (triangle)
        ld_num_iterations=200,
        ld_step_schedule='linear_decay',  # 'linear_decay' | 'adam' | 'constant'
        ld_initial_step=1e-2,
        ld_final_step=1e-4,
        ld_adam_lr=1e-2,
        ld_adam_beta1=0.9,
        ld_adam_beta2=0.999,
        ld_subproblem_timeout=5.0,
        ld_early_stop=True,
        ld_log_interval=10,
        ld_parallel=True,
    )
    s.update(overrides)
    return s


def resolve_torch(settings):
    """Return (torch.device, torch.dtype) from settings."""
    if settings.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype_map = {16: torch.float16, 32: torch.float32, 64: torch.float64}
    dtype = dtype_map[settings.bits]
    return device, dtype
