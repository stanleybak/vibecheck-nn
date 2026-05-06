"""BnB verification settings."""

from dotmap import DotMap
import torch


# Phase-1 per-layer tightening is two orthogonal axes:
#   tighten_formulation: which dependency model / LP matrix is built
#   tighten_solver:      which Gurobi flavor is used
# Legacy `tighten_mode` (single string) is translated into the pair
# via _TIGHTEN_MODE_ALIAS below so existing scripts keep working.
_TIGHTEN_FORMULATIONS = ('weight_walk', 'gen_cone', 'skip')
_TIGHTEN_SOLVERS = ('lp', 'milp', 'probe')
_TIGHTEN_MODE_ALIAS = {
    'probe':         ('weight_walk', 'probe'),  # MILP→LP auto-fallback
    'lp':            ('weight_walk', 'lp'),
    'milp':          ('weight_walk', 'milp'),   # MILP-only (no LP fallback)
    'skip':          ('skip', 'probe'),
    'gen_cone':      ('gen_cone', 'lp'),
    'gen_cone_milp': ('gen_cone', 'milp'),
}


def default_settings(**overrides):
    """Create settings with defaults for BnB verification."""
    # Translate legacy tighten_mode=... into the two-axis API.
    if 'tighten_mode' in overrides:
        legacy = overrides.pop('tighten_mode')
        assert legacy in _TIGHTEN_MODE_ALIAS, (
            f'unknown tighten_mode {legacy!r}; valid: '
            f'{sorted(_TIGHTEN_MODE_ALIAS)}')
        f, sv = _TIGHTEN_MODE_ALIAS[legacy]
        overrides.setdefault('tighten_formulation', f)
        overrides.setdefault('tighten_solver', sv)

    s = DotMap(
        device='gpu',
        bits=64,
        # PGD — matched to α,β-CROWN vnncomp25 defaults. Deep-ResNet loss
        # surfaces have saturated-ReLU plateaus that stall short PGD; 10
        # restarts × 100 Adam-direction+signed-ε-step iters with 0.99 LR
        # decay + hinge loss catch counterexamples missed by the old 100×10.
        pgd_restarts=30,
        pgd_iter=100,
        # Phase 0 PGD: run an attack BEFORE the bab_refine cascade.
        # Mirrors α,β-CROWN's pgd_order='before' default. If SAT is
        # found here, skip the entire cascade. Without this, SAT cases
        # waste 30-60s on cascade work before PGD (Phase 3.5) gets a
        # chance, while AB-CROWN finds them in 3-7s.
        pgd_phase0_enabled=True,
        pgd_time_budget_phase0=10.0,
        pgd_lr_decay=0.99,               # step-size × this every iter
        pgd_hinge_threshold=-1e-5,       # clamp margins at this from below
        pgd_alpha_frac=0.25,             # step_size = eps_input * this
        # PGD optimizer choice. Three modes:
        #   'adam_sign'    — bias-corrected Adam moment, sign-clipped step
        #                    (current vibecheck behavior, kept as default
        #                    for back-compat).
        #   'adam_clipping' — α,β-CROWN AdamClipping (attack_utils.py):
        #                     un-bias-corrected first moment in the sign
        #                     extraction AND step magnitude divided by
        #                     bias_correction1, which makes early-iter
        #                     steps ~10× the asymptotic step (helps escape
        #                     saturated-ReLU plateaus on hard cases).
        #   'sign_sgd'     — pure sign of raw gradient (no Adam state).
        #                    Matches α,β-CROWN's `use_adam=False` branch.
        # On the resnet_large 3585 sidx 3469 case (known SAT) α,β-CROWN
        # finds the counter-example in ~70 ms with adam_clipping; the
        # current 'adam_sign' default mostly misses it.
        pgd_optim='adam_clipping',
        # PGD initialization mode:
        #   'uniform' — random uniform in input box (legacy)
        #   'osi'     — Output Sampling Initialization, mirrors
        #               α,β-CROWN's diversed_PGD. 50 sign-grad steps
        #               maximizing dot(w_d, model(x)) with random
        #               output-space projection w_d per restart, then
        #               normal PGD from those points. Default since
        #               it consistently finds SAT cases uniform misses
        #               (mnist_fc prop_7/prop_2 0.05, cifar_biasfield
        #               prop_40 — the latter found in 5s with osi
        #               vs 390s timeout with uniform).
        pgd_init_mode='osi',
        # α-CROWN hopeless-bound early-exit: when worst best_lb is
        # below `alpha_crown_hopeless_lb` AND the average 3-iter
        # improvement is below `alpha_crown_hopeless_delta` after at
        # least 5 iters, abort α-CROWN and proceed (BaB / spec MILP).
        # AB-CROWN doesn't iterate α-CROWN at all on cifar_biasfield-
        # type cases where init bound is in the −100s; this matches
        # that behavior. Set to None to disable. Default −50 means we
        # only short-circuit the obviously-hopeless cases.
        alpha_crown_hopeless_lb=-50.0,
        alpha_crown_hopeless_delta=0.5,
        # Phase 1 per-neuron adaptive bounds:
        #   phase1_adapt_enabled (default True) — when False, skip the
        #     per-neuron CROWN-backward step entirely (Phase 2.5's
        #     spec-aware α-CROWN handles intermediate bounds). On
        #     cifar_biasfield the all-neurons adapt burned 24 s and
        #     fixed only ~12% of unstables — Phase 2.5 with proper
        #     spec-aware α makes that work redundant on small-input
        #     networks.
        #   phase1_adapt_topk (default None = all) — when set, only
        #     tighten the top-K most impactful unstable neurons per
        #     layer (score = |center|×width). For wide layers this
        #     reduces wall time linearly without losing the high-impact
        #     tightenings.
        phase1_adapt_enabled=True,
        phase1_adapt_topk=None,
        # OSI init iterations (only used when pgd_init_mode='osi').
        pgd_osi_iters=50,
        pgd_middle_enabled=True,         # α,β-CROWN `pgd_order=middle` trick:
                                          # re-run PGD on disjuncts that
                                          # survived Phase 2 CROWN
        # α,β-CROWN's cifar100 yaml sets `pgd_order='middle'` — it runs
        # PGD ONCE, after CROWN. We previously ran it twice (before + middle);
        # disabling Phase 3 matches α,β-CROWN exactly. The "middle" call with
        # restrict_disj still catches counterexamples on the hard disjuncts;
        # pre-CROWN PGD adds ~1.5 s of redundant work on the 97/99 easy ones.
        pgd_before_enabled=False,
        pgd_time_budget_before=5.0,      # max wall for Phase 3 initial PGD (only if pgd_before_enabled)
        pgd_time_budget_middle=5.0,      # max wall for Phase 3.5 PGD
        # CPU fallback on GPU OOM: off by default because CPU Phase 1 is
        # 10-100× slower than GPU and silently masks memory regressions.
        # Opt-in requires BOTH `allow_cpu_fallback=True` AND
        # `raise_on_oom=False` (see below) — two knobs so fallback is
        # never accidental.
        allow_cpu_fallback=False,
        # Phase 1 `_per_neuron_adaptive_bounds` chunk size. None = no
        # chunking (fastest when it fits). Default 256 keeps peak under
        # ~600 MB on resnet_large L1 (1687 unstable), vs 4 GB unchunked.
        # α,β-CROWN's `crown_batch_size` equivalent.
        adapt_chunk_size=256,
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
        # Phase-1 tightening axes
        tighten_formulation='gen_cone',     # 'weight_walk' | 'gen_cone' | 'skip'
        tighten_solver='milp',              # 'lp' | 'milp' | 'probe' (MILP→LP auto)
        # gen_cone-MILP: when True (default) reuse Phase-1 forward rows
        # (rec_zono) for both the LP probe and the MILP solve; the
        # worker dispatches to `_build_gen_cone_lp_phase1` so the
        # zono-form rows are interpreted in their own coordinate
        # system (the historical 'phase1' invariant — see
        # `state_from_phase1`'s soundness note). Setting this False
        # makes the MILP path run a fresh per-layer
        # `_gen_cone_state` rebuild in alpha form, which can produce
        # tighter MILP optima at the cost of one extra forward per
        # layer (helpful on relusplitter mnist_fc 256x6 prop_5; net
        # regression on cifar100 idx=13 verified→timeout in the
        # smoke sample).
        tighten_use_piggyback_milp=True,
        # Phase 1 method:
        #   'legacy' — interleaved forward + per-layer tighten in one
        #              forward pass (the historical default).
        #   'bab_refine' — α,β-CROWN-style cascade: forward zono → for
        #              each layer L do MILP-tighten z_L with sliding
        #              window of K layers + batched α-CROWN refresh
        #              that updates intermediate bounds globally
        #              between layers. On mnist_fc 256x6 prop_5 the
        #              cascade verifies in 190 s vs 380 s with legacy
        #              (no spec MILP needed — Phase 7 LP closes all
        #              disjuncts on the tighter intermediates).
        phase1_method='legacy',
        # Sliding window size for bab_refine — only neurons within K
        # layers of the target are binarized in the per-neuron MILP
        # (older layers stay as the LP triangle relaxation). K=1
        # matches α,β-CROWN's bab-refine; larger K is tighter per-layer
        # but slower. Ignored when phase1_method='legacy'. Can also be
        # overridden by VC_TIGHTEN_WINDOW env var.
        bab_refine_window=1,
        # Per-layer time cap for bab_refine. New per-neuron MILPs stop
        # being submitted to the worker pool once this layer has run
        # this many seconds (already-running tasks continue to their
        # `milp_sample_timeout`). Mirrors α,β-CROWN's
        # `refine_neuron_time_percentage * total_timeout` cap. Caveat:
        # the cap only governs the rate of NEW task submission; on a
        # layer with n_unstable >> n_cores * (budget/sample_to) the
        # actual wall time is `n_unstable * sample_to / n_cores`
        # regardless of this knob. Empirically on prop_4 (256
        # unstables at L=3) layer_budget=30 saved no wall time vs the
        # in-flight tail. Useful mainly to keep deeper layers
        # progressing on benchmarks that DO have low-impact tail
        # neurons (most cases).
        bab_refine_layer_budget=30.0,
        # Topk filter — only the K most-impactful unstable neurons per
        # layer get an MILP-tightening pass; the rest stay at the
        # forward-zono / α-CROWN bound. 0 = no filter (default). With
        # 0, the time cap is set by `bab_refine_layer_budget_frac` and
        # tasks are processed in score order (highest-impact first), so
        # the pool runs the most-useful neurons first and stops when
        # budget runs out. Use a positive K only when you specifically
        # want to *force* a count cap regardless of remaining time.
        bab_refine_topk=0,
        # Per-layer time budget as a FRACTION of total_timeout. When > 0,
        # this overrides `bab_refine_layer_budget` (the absolute-seconds
        # cap). e.g. 0.15 with total_timeout=60s gives 9s per layer. The
        # fractional form scales naturally with shorter/longer runs.
        # When `bab_refine_topk=0` (default), this only acts as a hint;
        # the parallel pool runs until all tasks complete (terminating
        # the pool mid-flight drops queued-but-not-yet-pulled results,
        # so we don't). The actual cost is bounded by per-neuron
        # `milp_sample_timeout` and the global pipeline timeout. Tasks
        # are still dispatched in ew_frac score order so partial
        # progress on a hard layer is dominated by high-impact neurons.
        bab_refine_layer_budget_frac=0.0,
        # Score function for `bab_refine_topk` ranking:
        #   'center_width' — |c_in| × (hi - lo) (legacy default; cheap)
        #   'ew_frac'      — max_over_open_specs |ew[L][j]| × frac[L][j]
        #                    where ew is the spec-direction CROWN-backward
        #                    weight at layer L's pre-relu and
        #                    frac = -lo/(hi-lo). Better signal because it
        #                    measures the spec's actual sensitivity to that
        #                    neuron, weighted by its current relaxation
        #                    looseness. Costs one CROWN backward per still-
        #                    open spec at the start of bab_refine
        #                    (fp64 GPU: ~10ms per spec on mnist_fc 256x4).
        bab_refine_score='ew_frac',
        # Cascade short-circuit: after each layer's MILP+α-refresh, run a
        # quick CROWN spec check on the still-open specs. If ALL closed
        # (spec_lb > 0), exit Phase 1 immediately and skip the remaining
        # layers. Saves work on cases that close at L1/L2 and don't need
        # L3+ tightening.
        bab_refine_short_circuit=True,
        # AB-CROWN's `mip_refine_remove_unstable_neurons` filter
        # (lp_mip_solver.py:1858). For each unstable neuron j at layer L,
        # examine the spec-direction CROWN backward weight ew[i, j]
        # (= lA[i, j], `∂(spec_LB)/∂(z_L,j)`). If ALL spec rows have
        # ew[i, j] > 0, then the spec_LB never uses the upper-bound ReLU
        # relaxation of y_L,j; refining its pre-ReLU bound cannot help.
        # Skip its per-neuron MILP. Mirrors AB-CROWN's "Start on the third
        # linear layer" guard — applied at L>=2 since L=1 has no upstream
        # MIP-refined bounds yet (matches AB's `unstable_neuron_filter`
        # being empty at the first FC layer).
        # Default OFF. Saves 13–26% time on shallow mnist 256x4
        # cases (e.g. prop_0_0.03: 36.3s → 31.6s) but regresses
        # mnist 256x6 prop_5_0.05 from 74.4s verified → 302s
        # timeout regardless of α flavor (trivial or post-Phase-0.5
        # optimized). The regression is robust: AB-CROWN's filter
        # reasoning ("ew > 0 ⇒ refining can't help spec_LB") holds
        # for the LAST FC layer pre-spec (immediate ReLU before the
        # spec output), but on mid-network layers in deeper nets,
        # tightening a "positive-ew" neuron's bounds can still help
        # spec via multi-layer downstream paths that the local sign
        # analysis misses. Re-enable for mnist 256x4 (or shallower)
        # where the filter saves time without soundness issues; keep
        # OFF on deeper architectures.
        bab_refine_remove_unstable=False,
        # Phase 0.5 α-CROWN open-spec detection. Runs BEFORE the per-layer
        # MIP cascade and BEFORE the `min_ew_per_layer` filter sweep:
        #   1. α-CROWN joint intermediate-bound tightening (Adam optim
        #      over per-layer α slopes, computes intermediate bounds at
        #      every unstable ReLU). Iters: bab_refine_phase05_alpha_iters.
        #   2. α-CROWN spec direction with the just-tightened intermediate
        #      bounds frozen (only spec α optimized). Iters:
        #      bab_refine_phase05_spec_iters. The spec lower bounds returned
        #      determine which queries are "still open" (lb <= 0) and feed
        #      `remove_unstable` filter so it considers only un-proved specs.
        # Restricting `min_ew_per_layer` to open queries is materially more
        # selective: a spec already proved by Phase 0.5 cannot improve via
        # per-neuron MIP refinement, so its ew sign is irrelevant.
        bab_refine_phase05_alpha_iters=10,
        bab_refine_phase05_spec_iters=20,
        # Per-spec α in Phase 0.5's spec α-CROWN call (vs shared α
        # across queries). With per-spec α, each open spec gets its own
        # tensor of slopes per ReLU layer — a slope choice that closes
        # one spec but hurts another is no longer a compromise. Mirrors
        # α,β-CROWN's per-(spec, layer) α convention. Default OFF: on
        # the hard mnist 256x4 RSPLITTER cases where shared α leaves
        # all 9 specs in [-10, -0.1], per-spec α gives marginally
        # better margins (~0.01-0.1 lift) but doesn't push any across
        # zero. The 9× extra Adam state plus subtle cascade-state
        # differences caused a prop_7_0.03 regression (79.5s → TIMEOUT).
        # Re-enable when α-CROWN's shared-α LBs are already near 0
        # (0.0 < lb < 0.05 and you want to push them positive without
        # paying the joint α cost).
        bab_refine_phase05_per_spec_alpha=False,
        # Run MIP-tighten + α-CROWN refresh on L=1 INSIDE Phase 0.5
        # (between joint α and spec α). Spec α-CROWN then sees
        # post-MIP bounds, potentially closing more specs and
        # triggering the all-closed short-circuit. The cascade then
        # starts at L=2 (L=1 already done). Default OFF — adds ~5s
        # to Phase 0.5 unconditionally; only profitable if it tips
        # borderline specs across zero. Worth measuring per benchmark.
        bab_refine_phase05_milp_l1=False,
        # Auto-route conv-heavy networks (oval21 cifar_base/deep/wide_kw,
        # cifar_biasfield) to the historical milp_verify pipeline at the
        # top of verify_graph(). bab_refine + alpha-zono Phase 8 was
        # tuned for FC nets like mnist_fc and underperforms on conv
        # ResNets — the alpha-zono LP relaxation is too loose at the
        # spec layer, even with hundreds of binary variables. milp_verify's
        # per-neuron layer-wise MILP encoding + Phase 5 racing escalation
        # empirically closes oval21 medium-eps cases that bab_refine
        # times out on (img1204 eps=0.025: milp_verify 36s vs
        # bab_refine 181s timeout). Detection: graph has any Conv node
        # AND no fork points. Disable to force every case through
        # bab_refine (e.g. for benchmarking).
        auto_route_milp_for_conv=True,
        # Refresh `min_ew_per_layer` after each layer's MIP+α-CROWN
        # round in the bab_refine cascade (mirrors AB-CROWN's per-layer
        # `lA` rebuild between MIP layers). Costs one CROWN backward per
        # still-open spec per layer (~10ms each on mnist_fc). Default
        # OFF: empirically gives mixed results (wins on prop_5/6/7,
        # regresses prop_0/10 — the AB-CROWN anchor — by 4-7s) because
        # we capture ew with trivial α, while AB uses the optimized
        # spec-α from the just-run α-CROWN. Re-enable for experimental
        # comparison, but expect uneven gains until ew capture uses the
        # post-α optimized slopes.
        bab_refine_refresh_filter_per_layer=False,
        # Multi-pass cascade: each pass loops L=0..max_layer applying
        # MILP+α-CROWN refresh. Pass N starts from bounds tightened by
        # pass N-1, so MILPs get a closer starting point and α-CROWN
        # has better intermediate bounds to work with. Default 1 matches
        # the historical single-pass behavior. 2-3 passes can close hard
        # FC cases (mnist_fc 256x4 RSPLITTER prop_6/0/10) that single
        # pass times out on, at the cost of roughly Nx Phase 1 time.
        bab_refine_passes=1,
        # Per-neuron MILP/LP tightening: pick min/max ordering by forwarding
        # concrete witnesses (random + corner inputs) through the actual
        # ReLU network and looking at z_j sign. With all witnesses ≥ 0,
        # only MIN can prove active → run MIN first with BestBdStop. With
        # all ≤ 0, only MAX can prove dead → MAX first. With straddle, fall
        # back to the bound-asymmetry heuristic |cur_lo|<|cur_hi|. Affects
        # `_solve_neuron_both` (verify_milp.py) and `_tighten_neuron_graph`
        # (verify_graph.py). Both still run BOTH directions for genuinely
        # unstable neurons (so bound tightening is identical) — the witness
        # only changes the ORDER, which gives BestBdStop a better chance to
        # early-exit. Empirically saves ~25-45% MILP wallclock on
        # mnist_fc 256x4 prop_5 with no change to bound quality.
        tighten_witness_ordering=True,
        tighten_witness_n_random=8,
        # Phase 2.5 batched α-CROWN at startup tightens intermediate
        # bounds per-query. With this True we merge those bounds back
        # into the GLOBAL bounds_by_relu so the downstream Phase 7/8
        # spec MILP sees them too — the bab-refine pattern from
        # α,β-CROWN. On mnist_fc 256x6 prop_5 (256 unstable at L4/L5
        # after Phase 1 MILP-tightening of L0..L4) the merge drops
        # L4u from 256 to 0 and L5u from 256 to ~12, taking us from
        # worst_lb≈-15 (timeout, unverified) to +0.5 (verified) within
        # the 300s budget. CIFAR100 sample (20 cases): no further
        # regressions beyond the form-fix's idx=13 case.
        merge_alpha_bounds_globally=True,
        # Tightening options
        milp_tighten_method='lp',       # 'lp' or 'milp' for per-layer tightening
        milp_tighten_sparse=True,       # sparse per-neuron models for conv layers
        milp_tighten_parallel=True,     # parallel workers vs sequential
        milp_tighten_rebuild=False,     # rebuild model per worker vs copy shared
        milp_lp_encoding='compact',     # 'compact' (1 var, 2 constrs) or 'zas' (3 vars, 5 constrs)
        graph_impl='optimized',         # 'reference' or 'optimized' for verify_graph builder (Phase 1)
        spec_impl='gen_lp',             # 'gen_lp' (generator-based GPU) or 'monolithic' for Phase 7/8
        gen_lp_formulation='sparse',    # 'dense' or 'sparse' (applies when spec_impl='gen_lp'); sparse cuts at the last hidden ReLU to avoid numeric trouble
        # When True, Phase 7 reuses Phase 1's z_final + rec_zono (populated
        # when tighten_formulation='gen_cone') to build the spec LP without
        # re-forwarding through the network. Skips the dense
        # (n_layer × n_gens) G allocation in `precompute_gen_state` that
        # OOMs on resnet_large CIFAR100 (~30k × 8k × 8 B ≈ 2 GB per conv).
        # Per-query rebuilds with phase8_per_query_tightened_bounds=True
        # still call `precompute_gen_state(merged_bbr)` (Phase-1 form
        # doesn't soundly support changing (λ, μ) post-hoc).
        phase7_reuse_phase1_zono=True,
        gen_lp_parallel_racing=True,    # parallel bin racing across open queries and bin levels (imap_unordered pool) vs sequential per-query bin escalation
        gen_lp_gurobi_threads=1,        # Gurobi 'Threads' per MILP solve in Phase 8
        gen_lp_min_bin=4,               # starting bin count for racing escalation ('legacy' mode only)
        gen_lp_bin_mult=4,              # bin count growth factor per racing level ('legacy' mode only)
        # Phase 8 bin scheduling mode.
        #   'legacy'  = geometric: [gen_lp_min_bin, *bin_mult, *bin_mult, ..., n_scored]
        #   'octaves' = additive:  [8, 16, 24, 32, ..., 8k] with 8k ≤ (n_cores - phase8_leave_cores_open)
        # 'octaves' launches all bin-levels concurrently as CPU workers (with
        # BestBdStop=0 early-exit). First worker whose MILP proves ObjBound>0
        # closes the query. Small-bin workers finish fast but may not prove
        # tightness; large-bin workers are stronger but slower. The pool returns
        # on first UNSAT — so the racing is strictly across bin counts, not
        # SAT-vs-optimization as in the legacy Gurobi feasibility-race.
        phase8_bin_mode='octaves',
        # How many CPU cores to leave unoccupied by Phase 8 MILP workers (so
        # the main thread / GPU driver / OS keep one). n_workers = n_cores -
        # phase8_leave_cores_open.
        phase8_leave_cores_open=1,
        # When True, Phase 8 scoring uses α-CROWN's captured `ew_at_relu`
        # (from Phase 2.5's `capture_ew_per_relu` backward pass) instead of
        # re-running a plain CROWN backward via `_spec_backward_graph`. Saves
        # a CROWN backward per still-open query AND produces tighter scores
        # because α-CROWN's slopes are direction-optimized.
        phase8_use_alpha_ew=True,
        # When True, Phase 8 rebuilds gen_lp_state per-query using the
        # Phase-2.5-tightened intermediate bounds (α-CROWN best_bounds ∧
        # halfspace-LP override). Each rebuild costs one `precompute_gen_state`
        # (~1-2 s on CIFAR100_resnet_medium) but gives the MILP tighter
        # ReLU triangles AND new λ-slope topology (neurons flipped to stable
        # or dead become equality/zero instead of unstable triangles). Sound
        # for per-query spec direction only — tightened bounds are not merged
        # into the global `bounds_by_relu`.
        phase8_per_query_tightened_bounds=True,
        # Phase 8 MILP mode. Two orthogonal axes:
        #   relaxation : 'triangle_lp' (Phase 1 precompute_gen_state)
        #              | 'alpha_zono'  (per-query forward_zono_dir_adaptive)
        #   proof type : 'feasibility'   (UNSAT = ObjBound > 0)
        #              | 'infeasibility' (UNSAT = Gurobi.INFEASIBLE under
        #                                 an inequality halfspace)
        # Encoded as 4 enum values:
        #   'find_sat'                  — triangle_lp + feasibility. Halfspace
        #                                 forced off; BestBdStop=0.
        #   'infeasibility'             — triangle_lp + infeasibility. Adds
        #                                 `qw·y + qb ≤ 0` to the MILP; UNSAT
        #                                 signal is Gurobi.INFEASIBLE.
        #                                 BestBdStop=None.
        #   'alpha_zono_bnb' (default)  — alpha_zono + feasibility. Per-query
        #                                 α-CROWN zonotope (c_α, G_α);
        #                                 parallelogram-only relaxation for
        #                                 non-binarized neurons, full ReLU
        #                                 big-M for binarized.
        #   'alpha_zono_infeasibility'  — alpha_zono + inequality halfspace
        #                                 + Gurobi.INFEASIBLE check. Useful
        #                                 for symmetry; usually worse than
        #                                 alpha_zono_bnb on hard cases.
        phase8_milp_mode='alpha_zono_bnb',
        # Partial triangle relaxation on top of the α-zono parallelogram.
        # When >0, the top-K most-important unstable neurons (by Phase 7
        # / α-CROWN score) get triangle floor constraints (y ≥ 0 and
        # y ≥ z) added inside the alpha_zono LP. Neurons in the MILP
        # binary subset already have those constraints + big-M; this knob
        # tightens the relaxation for the next K most-important
        # non-binarized neurons WITHOUT adding any binary variables.
        # 0 (default) = pure parallelogram-only for non-binarized neurons
        # (current alpha_zono_bnb behavior). Set this when racing fails
        # to push ObjBound > 0 on hard UNSAT cases — triangulating the
        # top neurons may close the gap without binarising them.
        phase8_alpha_zono_triangle_top_k=0,
        # ----- MILP-seeded PGD refinement -----
        # After every Phase 8 MILP worker that returns a feasible (but not
        # spec-violating) solution, take its e_in (the input-generator
        # values) and seed a short PGD attack from there + small random
        # perturbations. The MILP's e_in is often near the spec boundary
        # in the LP-relaxation sense; PGD walks the actual network forward
        # to discover a real counterexample if one is nearby. Runs in the
        # main thread (one PGD per worker result, dispatched via the
        # imap_unordered loop) so it doesn't compete with the CPU MILP
        # pool. Cheap (~30-50 ms per refinement on resnet_medium).
        phase8_pgd_seed_from_milp=True,
        phase8_pgd_seed_iters=20,
        phase8_pgd_seed_perts=8,
        phase8_pgd_seed_noise=0.01,    # fraction of input-box width
        gen_lp_skip_phase7_lp=True,     # skip per-query LP scoring; use α-CROWN/CROWN ew*frac fallback (saves Phase 7 LP wall — was ~4s/query on hard CIFAR100)
        gen_lp_score_method='lp_ew_frac',  # 'lp_ew_frac', 'lp_fractional', 'lp_dual'. lp_dual ranks by |tri_lo|+|tri_up| duals — identifies the actual LP-binding triangles (on CIFAR100_resnet_medium_prop_idx_2477 the duals concentrate in L5 where kfsb/ew_frac promotes L9). lp_dual adds ~1-2s/query Phase-8 overhead to re-solve gen-LP with dual extraction; beneficial on hard queries where the wrong layer is being branched on, neutral-to-slow otherwise. Opt-in via settings.
        skip_phase8_milp=False,         # if True, Phase 8 is skipped and queries Phase 7 LP can't prove UNSAT are returned as 'unknown'
        max_tighten_layer=1,            # only Phase 1 tightens layers <= this idx (None = no cap)
        # When set, Phase 1 extends tightening to layers in the range
        # (max_tighten_layer, max_tighten_layer_lp] using LP only (not MILP).
        # Rationale: L1 benefits from MILP (tight big-M exact triangle), but
        # deeper layers would be too slow for MILP. LP-only per-neuron
        # tightening at L2 is cheap (~1 s avg on CIFAR100_resnet_medium)
        # and shrinks the downstream unstable pool seen by α-CROWN and
        # Phase 8 MILP. Default 2: 200-case CIFAR100 sweep showed +4 new
        # verifications, 0 regressions, 0 unsound, ~+0.36 s avg total time.
        max_tighten_layer_lp=2,
        # Deferred L1-ish MILP tightening — when enabled, Phase 1 skips
        # MILP/LP tightening on `deferred_milp_layers` (adaptive still runs),
        # letting α-CROWN Phase 2.5 close easy cases first. If α-CROWN
        # leaves queries open, we come back and do the expensive parallel
        # per-neuron MILP tightening, then re-enter Phase 2.5 cascade on
        # the tightened bounds. Includes a probe+budget guard so hopeless
        # MILP tightens are skipped rather than burning the timeout.
        deferred_milp_tighten=False,              # opt-in
        deferred_milp_layers=(1,),                # layers to defer
        deferred_milp_probe_timeout=5.0,          # per-neuron timeout during probe
        deferred_milp_probe_neurons=None,         # None = use n_cores
        deferred_milp_budget_frac=0.2,            # skip if est > frac × remaining timeout
        # bab-refine cascade: when True, re-run Phase 2.5 (α-CROWN +
        # halfspace LP) BETWEEN each layer in `deferred_milp_layers` so
        # the new layer's tighter bounds propagate downstream via α-CROWN.
        # Matches α,β-CROWN's `bab-refine` flow which cascades L2 → L3 →
        # L4 with α-CROWN re-runs in between (drove spec_lb from −1924
        # to +0.086 on mnist_fc 256x6 prop_5_0.05). When False, falls back
        # to the legacy single end-of-loop Phase 2.5 re-run.
        bab_refine_cascade=True,
        # Callback: called at key points with (event, info) -> bool (False = stop)
        milp_callback=None,
        # Phase 2.5: iterative zono-lift tightening via closed-form
        # box + 1-halfspace LP (vibecheck.box_halfspace). Runs for each
        # still-open disjunct after Phase 2 CROWN, before Phase 7.
        zono_lift_enabled=True,
        zono_lift_max_passes=10,           # iterations per query (cascade)
        zono_lift_tolerance=1e-4,          # stop when CROWN LB delta < this
        zono_lift_layers=None,             # None = all layers with unstable
        zono_lift_plateau_patience=2,      # stop after K no-bound-change passes
        # Input-space BaB. When the input dim ≤ `input_split_max_dims`,
        # `verify_graph` wraps the pipeline in recursive widest-axis
        # bisection: each leaf runs Phase 1+2 with a tight per-node
        # timeout (no Phase 8 MILP). Closes spec gaps on small-input
        # benchmarks like cifar_biasfield (16 dims) where a single-shot
        # zono enclosure is too loose. No effect when input dim is large.
        input_split_enabled=True,
        input_split_max_dims=20,
        input_split_max_depth=8,
        input_split_node_timeout=8.0,
        # α-CROWN-driven variant D: run α-CROWN per query, reconstruct
        # direction-adaptive forward zonotope with the optimal α's, then
        # apply the halfspace LP on that tighter G. Much stronger than the
        # min-area forward for spec-aligned queries.
        zono_lift_alpha_crown=True,        # use α-CROWN optimization
        zono_lift_alpha_iters=10,          # Adam iters (step68: 10 optimal)
        zono_lift_alpha_lr=0.25,           # Adam lr
        # α-CROWN early-stop on positive spec LB — matches α,β-CROWN's
        # `stop_criterion_final` in `auto_LiRPA::_get_optimized_bounds`:
        # once the spec is provably safe (lb > 0), further Adam steps only
        # waste wall time. On easy α,β-CROWN-provable cases this cuts
        # run_alpha_crown wall from ~3.5 s (10 iters) to ~0.7 s (2 iters).
        alpha_crown_early_stop_on_positive=True,
        # α-CROWN implementation.
        #
        # 'legacy' (default) = joint α-CROWN over all (start_node, L) pairs.
        # Re-optimises intermediate α slopes each Adam iter; effectively re-
        # tightens intermediate pre-ReLU bounds in the spec direction. On
        # mnist_fc 256x6 prop_5 this drops the Phase 2.5 worst spec lb from
        # −995 (fixed_intermediate) to −265 — a 4× improvement. ABC's
        # bab-refine equivalent gets from there to verification by adding
        # per-neuron MIP refinement; we don't, but the joint α-CROWN
        # half is a strict win over fixed_intermediate on the cases tested
        # (relusplitter mnist_fc, oval21, CIFAR100_resnet_medium verified
        # cases — see /tmp/decompose_spec_lb.py).
        #
        # 'v2_fixed_intermediate' = α,β-CROWN's `fix_intermediate_bounds=True`:
        # freeze intermediate bounds to Phase-2 CROWN output, optimise only
        # spec-path α with ExponentialLR. Faster per Adam iter (1 backward vs
        # n_start_nodes×n_relu) but cannot tighten intermediate bounds in the
        # spec direction. On CIFAR100_resnet_large 2993 it closes the spec lb
        # in ≤10 iters where joint plateaued at -0.012 — kept available for
        # cases where joint stalls.
        alpha_crown_impl='legacy',
        # Auto-switch to v2_fixed_intermediate when total unstable
        # neurons (across all hidden ReLUs) exceeds this threshold.
        # Joint legacy α-CROWN does ~138 conv-transpose backwards per
        # Adam iter on cifar_biasfield (1500+ unstable per layer × 6
        # layers), giving 416 ms/iter on RTX 3080. AB-CROWN's
        # equivalent (always fix_intermediate_bounds in their
        # incomplete_verifier) is 4.2 ms/iter on the same hardware.
        # When unstable neurons are below this cap, legacy still wins
        # on bound tightness (mnist_fc 256x6 prop_5: −265 vs −995).
        # 5000 = ~empirical line above which the per-iter cost makes
        # legacy untenable. Set to None to disable auto-switching.
        alpha_crown_impl_auto_switch_threshold=5000,
        # ExponentialLR decay applied to the Adam optimizer in the v2 path.
        # α,β-CROWN uses 0.98 (config value). 1.0 = off (legacy behavior).
        alpha_crown_lr_decay=0.98,
        # Sparse-α (matches α,β-CROWN's `sparse_features_alpha=True` default,
        # see `auto_LiRPA/operators/relu.py:64` and `bound_general.py:84`):
        # allocate Adam parameters only for unstable neurons per ReLU layer.
        # On cifar_biasfield_28 this shrinks the optimiser state from 45056
        # to 10010 floats (4.5×). Per-iter wall is unchanged on the spec-only
        # / fixed-intermediate path (the bottleneck is the conv backward),
        # but Adam state size matters for the joint-α path where multiple
        # (start_node, layer) tensors compound. Default off; set True to
        # match α,β-CROWN. NaN-safe with fp32; fp16 overflows on biasfield.
        alpha_crown_sparse_alpha=False,
        # Skip the α-CROWN cascade and direction-adaptive zonotope rebuild
        # when pass-0 α-CROWN already closed the query. Saves one
        # `capture_ew_per_relu` + `build_dir_adaptive_alpha` per closed
        # query (minor, but eliminates a redundant CROWN backward pass).
        zono_lift_cascade_skip_on_close=True,
        # Batch the pass-0 α-CROWN spec direction across open queries per
        # disjunct (one shared α-Adam graph, spec backward batched (n_q,
        # n_out)). Matches α,β-CROWN's per-spec batching; saves (n_q - 1) ×
        # intermediate-bound-backward cost. Falls back to per-query cascade
        # for any query the batched pass didn't close.
        zono_lift_batch_queries=True,
        # CASCADE mode: re-run α-CROWN on the tightened bounds at each pass
        # of the zono-lift loop. For hard queries this closes the spec in a
        # few passes where single-α zono-lift stalls. Fast queries close on
        # pass 0 before cascade engages.
        zono_lift_cascade_alpha=True,      # re-run α-CROWN per pass on new bbr
        # ----- Phase 2.5 BaB-style split iterations -----
        # After the halfspace-LP cascade fails to close a query, iteratively
        # split on the next-best ranked unstable neuron and apply box+halfspace
        # LP for each side. Three outcomes per split:
        #   (1) both sides verify spec → CLOSE the query
        #   (2) exactly one side verifies → throw it away; commit the other
        #       side's halfspace via tightened bounds (the split neuron flips
        #       stable in those bounds automatically); continue iterating.
        #   (3) neither side verifies → take the union (sound) of both
        #       children's bounds and intersect with the running bounds.
        # Set to 0 to disable (default). Closer to abcrown's BaB; per-split
        # bound is strictly tighter than β-CROWN's β-multiplier relaxation
        # because the box+halfspace LP solves the polytope exactly. EMPIRICAL
        # NOTE: on a 5-case CIFAR100_resnet_medium sweep of α,β-CROWN-provable
        # but vibecheck-unknown instances (2477, 6553, 8523, 1761, 230), this
        # never produces a single commit (always unions, never closes via
        # split) AND adds 9-13 s of pure overhead. Phase 2.5's existing
        # halfspace-LP cascade already exhausts the easy single-split wins
        # before BaB runs (e.g. on 1761, the L9 k=45 neuron — abcrown's first
        # BaB pick — is flipped stable by the cascade before BaB sees it).
        # Kept opt-in for cases where the cascade plateaus early; not enabled
        # by default.
        phase2p5_bab_iters=0,
        # When True, rebuild the spec-adaptive zonotope at each BaB iteration
        # using the freshly tightened bounds (re-runs forward_zono_dir_adaptive).
        # Tighter c, G per neuron at the cost of one forward pass per iter
        # (~30-50 ms on CIFAR100_resnet_medium). False = keep the original
        # zonotope; only update per-neuron (lo, hi). Use the latter when fast
        # iterations matter more than per-iter tightness.
        phase2p5_bab_rebuild_zono=False,
        # OOM-handling policy. True (default) = re-raise any CUDA/CPU OOM
        # so the user sees the real failure. False = callers that have an
        # explicit fallback path (e.g. benchmarking loops recording "OOM"
        # as an outcome, or "retry with patches→dense") may catch it;
        # new code should never catch OOM silently regardless of this
        # flag. This is a global kill-switch — most of the pipeline keys
        # off it to decide "raise vs fall back". Surfacing OOMs early is
        # how we catch regressions (cache lines that grew, forward paths
        # that outgrew their budget, etc.).
        raise_on_oom=True,
        # Forward zonotope implementation. 'dense' = TorchZonotope;
        # 'patches' = PatchesZonotope (per-gen kernel-shaped patches with
        # offsets, exploits conv spatial sparsity to fit large stride-1
        # nets like CIFAR100_resnet_large in ~3× less GPU memory). The
        # patches form auto-falls-back to dense at FC layers, stride>1
        # convs, and once patches reach feature-map size — so dense-only
        # workloads (sequential FC nets, stride-2-from-input ResNets like
        # the 'medium' variant) see no measurable change.
        zono_impl='patches',
    )
    s.update(overrides)
    assert s.tighten_formulation in _TIGHTEN_FORMULATIONS, (
        f'tighten_formulation={s.tighten_formulation!r} not in '
        f'{_TIGHTEN_FORMULATIONS}')
    assert s.tighten_solver in _TIGHTEN_SOLVERS, (
        f'tighten_solver={s.tighten_solver!r} not in {_TIGHTEN_SOLVERS}')
    return s


def resolve_torch(settings):
    """Return (torch.device, torch.dtype) from settings.

    `settings.device` must be 'gpu' or 'cpu'. The historical convention
    uses 'gpu' (not 'cuda') as the GPU identifier — passing any other
    string used to silently fall back to CPU, which masked real perf
    issues (e.g., a CIFAR100 ResNet "GPU" run secretly running on CPU).
    Now an assert fails loudly on unknown values.
    """
    assert settings.device in ('gpu', 'cpu'), (
        f'settings.device must be \'gpu\' or \'cpu\', got '
        f'{settings.device!r}. Use \'gpu\' (not \'cuda\') for GPU.')
    if settings.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype_map = {16: torch.float16, 32: torch.float32, 64: torch.float64}
    dtype = dtype_map[settings.bits]
    return device, dtype
