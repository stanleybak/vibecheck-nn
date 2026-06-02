"""BnB verification settings.

Uses a `Settings` dict-with-attr-access class that RAISES AttributeError on
missing keys (instead of DotMap's silent empty-DotMap return). DotMap's
falsy-default footgun has bitten this project at least 4 times — every
`getattr(settings, 'flag', True)` silently returned False when the flag
wasn't set, because DotMap() is falsy but exists. See CLAUDE.md.
"""

import torch


class Settings:
    """Strict dict-with-attr-access for verification settings.

    Differences from DotMap:
      * Missing attribute raises AttributeError (not returns empty DotMap).
        This means `getattr(s, 'flag', True)` correctly falls back to True.
      * `'flag' in s` works.
      * `s.get('flag', default)` works.
      * Assignment with any name allowed (no schema enforcement) so
        downstream code can stash run-state on the object.
      * No nested DotMap auto-creation on missing — `s.foo.bar` raises
        AttributeError on `.foo` rather than silently building `s.foo`.
    """

    __slots__ = ('_d',)

    def __init__(self, **kwargs):
        object.__setattr__(self, '_d', dict(kwargs))

    def __getattr__(self, name):
        # Note: __getattr__ is only called when normal lookup fails (i.e.
        # name is not in __slots__ or any class attribute), so this only
        # hits user-data attrs.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        d = object.__getattribute__(self, '_d')
        if name in d:
            return d[name]
        raise AttributeError(
            f'Settings has no attribute {name!r}. '
            f'Available: {sorted(d)[:10]}...')

    def __setattr__(self, name, value):
        if name == '_d':
            object.__setattr__(self, name, value)
        else:
            self._d[name] = value

    def __contains__(self, key):
        return key in self._d

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __repr__(self):
        return f'Settings({self._d!r})'

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def update(self, *args, **kwargs):
        self._d.update(*args, **kwargs)

    def to_dict(self):
        return dict(self._d)


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

    s = Settings(
        device='gpu',
        bits=64,
        # PGD — matched to α,β-CROWN vnncomp25 defaults. Deep-ResNet loss
        # surfaces have saturated-ReLU plateaus that stall short PGD; 10
        # restarts × 100 Adam-direction+signed-ε-step iters with 0.99 LR
        # decay + hinge loss catch counterexamples missed by the old 100×10.
        pgd_restarts=30,
        # PGD plateau-based give-up: after `pgd_plateau_iters` iters of
        # no margin improvement AND all restarts above hinge, abandon
        # the attack (no SAT likely). Saves ~80% of PGD time on UNSAT
        # cases. Defaults to 100 iters of plateau before giving up.
        pgd_plateau_iters=100,
        pgd_iter=100,
        # Phase 0 PGD: run an attack BEFORE the bab_refine cascade.
        # Mirrors α,β-CROWN's pgd_order='before' default. If SAT is
        # found here, skip the entire cascade. Without this, SAT cases
        # waste 30-60s on cascade work before PGD (Phase 3.5) gets a
        # chance, while AB-CROWN finds them in 3-7s.
        # Master disable for all SAT-finding paths (PGD phases 0, 3, 3.5,
        # 9, MILP-seeded PGD, and integer-witness check inside Phase 8).
        # Used for soundness testing: when True, the only legitimate
        # outcomes are 'verified' (UNSAT proved) or 'unknown'. If a case
        # AB-CROWN reports SAT comes back as 'verified' here, that's a
        # soundness bug — the LP relaxation incorrectly claims UNSAT.
        # Default False (production behavior).
        disable_sat_finding=False,
        pgd_phase0_enabled=True,
        pgd_time_budget_phase0=10.0,
        pgd_lr_decay=0.99,               # step-size × this every iter
        pgd_hinge_threshold=-1e-5,       # clamp margins at this from below
        pgd_alpha_frac=0.25,             # step_size = eps_input * this
        # Multi-α restart pool: when True, partition the n_restarts pool
        # across the log-spaced alphas in `pgd_alpha_multi_fractions` so
        # each restart gets a different step size. Helps when the right
        # α varies across cases (cersyve: lane_keep wants ~0.01,
        # pendulum ~0.005). Equivalent compute, much broader coverage.
        pgd_alpha_multi=False,
        pgd_alpha_multi_fractions=(0.25, 0.05, 0.01, 0.002),
        # Per-leaf PGD inside input_split BaB: when an `unknown` leaf is
        # encountered, run a short PGD attack on that localized sub-box
        # before splitting. Helps when root-PGD missed a narrow SAT
        # region that's easier to find in a smaller sub-box. Off by
        # default; opt-in per benchmark (cersyve uses this).
        input_split_leaf_pgd_enabled=False,
        input_split_leaf_pgd_time=0.1,
        # Per-leaf JOINT-query CROWN check for AND conjuncts. Multi-
        # query disjuncts (cersyve: 2 queries per disjunct) can fail
        # the per-query "any q has lb > 0" check on UNSAT leaves
        # because closing requires JOINT reasoning. A linear-combo
        # λ·q0 + (1-λ)·q1 with lb > 0 proves the disjunct's unsafe
        # AND-region is empty (any unsafe point satisfies both
        # constraints, contradicting positive combo lb). Pass a list
        # of λ ∈ [0,1] to try. `[0.5]` is the cheapest useful setting.
        # `None` (default) disables. Opt-in per benchmark.
        input_split_leaf_joint_lambdas=None,
        # Per-leaf joint-AND INPUT-space LP check. Uses CROWN's
        # per-query linear lower-bound coefficients in input space to
        # check whether the joint unsafe-AND halfspaces have a
        # non-empty intersection with the leaf input box. Catches
        # CURVED separations between safe regions that λ-combo CROWN
        # misses (the λ-combo collapses to a single hyperplane).
        # Costs one extra spec backward + tiny LP per leaf. Disabled
        # by default; opt-in per benchmark (cersyve).
        input_split_leaf_joint_input_lp=False,
        # Stronger fallback: when the CROWN-input-space joint LP
        # doesn't close a disjunct, also try the LP on the OUTPUT
        # ZONOTOPE. The zonotope captures dependence between outputs
        # via shared e-generators (cifar pretrain models share the
        # ReLU error structure across both Y_0 and Y_1) — strictly
        # tighter than the single-hyperplane CROWN input bound. Only
        # used when `input_split_leaf_joint_input_lp=True`; on by
        # default because the extra cost is ~1 ms per leaf.
        input_split_leaf_joint_zono_lp=True,
        # Final fallback: full per-leaf TRIANGLE-LP that builds the
        # whole network's LP relaxation with BOTH unsafe constraints
        # added jointly. Captures correlations across the network that
        # neither CROWN linearization nor the output zonotope can.
        # Costs ~5-20 ms per leaf on tiny networks; opt-in per
        # benchmark (cersyve 4D-input UNSAT cases — 2/12 boundary
        # leaves not closable by zono LP).
        input_split_leaf_joint_triangle_lp=False,
        # Batched input-split BaB. When True, switches the input-split
        # dispatch from sequential per-leaf processing to a worklist-
        # based driver that stacks up to `input_split_batch_size` boxes
        # into one tensor and runs a single batched forward zono + spec
        # backward. Skips joint LP / α-CROWN at the leaf — pure per-
        # query CROWN. The throughput jump (1000s of leaves/sec on GPU
        # vs ~30/sec sequential) compensates by going deeper. Used by
        # cersyve 4D-input UNSAT cases that the sequential path
        # couldn't crack within budget.
        input_split_batched_enabled=False,
        input_split_batch_size=4096,
        # Memory cap: bail out of batched BaB if worklist grows past
        # this many open boxes (each is a (n_in,) tensor pair — 32 B
        # for 4-D input, so 200K boxes ≈ 6 MB; safety net for
        # divergent splits).
        input_split_batched_max_worklist=200_000,
        # Domain clipping inside batched BaB. After CROWN backward gives
        # per-query input-space linearization (A_q, b_q), clip each
        # leaf's box to the bounding box of the intersection of unsafe
        # halfspaces `A_q·x + b_q ≤ 0`. For AND-conjuncts, leaves
        # where the polytope is empty are verified directly; leaves
        # where it's just smaller go into the next iteration with
        # tighter bounds. Mirrors AB-CROWN's `clip_input_domain:
        # complete`. On by default whenever `input_split_batched_enabled`.
        input_split_batched_clip_enabled=True,
        # Iterate the per-halfspace clip until fixed point — box
        # shrinks → L_other tightens → halfspace projection on x_i
        # tightens further. Empirically converges in 1-2 passes for
        # cersyve / acasxu (no extra shrinkage past iter 1). Knob kept
        # for benchmarks where the fixed point isn't reached fast.
        input_split_batched_clip_iters=1,
        # Second-stage full-LP clipping after per-halfspace. For each
        # leaf that's still feasible after per-halfspace, run a full
        # Gurobi LP: 1 feasibility check + 2×n_in projection LPs.
        # Strictly tighter than per-halfspace (captures joint
        # constraints). Parallelized across CPU cores with persistent
        # Gurobi envs per worker. Mirrors AB-CROWN's
        # `clip_type: complete`. Costs ~5-30 ms per leaf per iter
        # depending on n_in; only enable when per-halfspace isn't
        # converging (acasxu prop_1/5/6/9 boundary cases).
        input_split_batched_clip_full_lp=False,
        input_split_batched_clip_lp_workers=None,  # default: cpu-1
        # SB (smart branching) axis selection — pick split dim by
        # `width × sum_q |A_q|` from CROWN's input linearization. Falls
        # back to widest-axis if A_lin not available. Empirically on
        # acasxu the simpler widest-axis converged about as fast (the
        # sensitivity score barely re-orders the picks because the
        # input box is approximately isotropic after splits). Default
        # off; AB-CROWN's `naive` (= widest) is the empirical winner.
        input_split_batched_branch_sb=False,
        # Route input-split-eligible nets to the freeze-replay α-CROWN verifier
        # (verify_hybrid_acasxu) with TIGHTENED intermediate bounds. The batched
        # input-split BaB's forward-zono intermediate bounds are ~1000x too
        # loose for ACAS Xu's amplifying weights → it diverges. Default off;
        # acasxu turns it on.
        use_hybrid_acasxu=False,
        # verify_hybrid tuning (only read when use_hybrid_acasxu). The
        # between-rounds PGD is wasted on the 139 UNSAT cases — dialing it down
        # speeds them up; freeze_iters trades freeze tightness for speed.
        hybrid_pgd_between_every=1,
        hybrid_pgd_between_restarts=1000,
        hybrid_freeze_iters=100,
        # Backward-CROWN intermediate bounds in the batched input-split BaB
        # (AB-CROWN's bound_prop_method: crown). ~2x tighter than forward zono
        # -> far fewer leaves on ACAS Xu. Default off.
        input_split_crown_intermediate=False,
        # Leaf-level SAT search in the batched input-split BaB. Phase-0 PGD on
        # the WIDE root box misses a narrow SAT witness (acasxu 1_5/1_9
        # prop_2/prop_7); the witness-containing leaf survives + keeps splitting
        # until narrow, where batched-PGD inside that leaf's box finds it. Every
        # `_every` iters, PGD the `_max_leaves` narrowest surviving leaves.
        # Default off (0 = disabled); acasxu turns it on.
        input_split_leaf_pgd_every=0,
        input_split_leaf_pgd_max_leaves=64,
        input_split_leaf_pgd_restarts=128,
        input_split_leaf_pgd_iters=50,
        # Clip → re-CROWN inner cycles. After clipping a leaf, the OLD
        # CROWN bounds are still sound on the smaller box but loose.
        # Re-running CROWN on the clipped box gives tighter spec lbs
        # that may close the leaf without splitting. Costs +1 forward
        # zono + spec backward per cycle (~5-20 ms per batch). Stops
        # early if clipping no longer shrinks. Default 0 (off);
        # opt-in per benchmark.
        input_split_batched_clip_recrown_cycles=0,
        # MILP escalation on stuck boundary leaves. After CROWN +
        # α-CROWN + clip, if a leaf still won't close AND its unstable
        # count ≤ `milp_max_unstable`, try the full triangle MILP
        # (exact ReLU encoding). At deep enough splits (~60 unstable
        # remaining), MILP closes in <50 ms. Per-leaf serial Gurobi;
        # capped at `milp_max_leaves` per iter to bound cost.
        input_split_batched_milp_escalate=False,
        input_split_batched_milp_max_unstable=80,
        input_split_batched_milp_max_leaves=20,
        input_split_batched_milp_tl=2.0,
        # Selective α-CROWN on boundary leaves of batched BaB. Per iter,
        # for leaves whose worst per-disjunct best-query lb is within
        # `boundary_eps` of 0, run per-leaf α-CROWN with up to
        # `alpha_iters` Adam steps and `early_stop_on_positive=True`.
        # Closes leaves on the convergence plateau where per-query
        # CROWN tightens but doesn't quite cross 0. Capped at
        # `alpha_max_leaves` per iter to bound serial cost. Default
        # disabled (eps=0); opt-in per benchmark.
        input_split_batched_alpha_boundary_eps=0.0,
        input_split_batched_alpha_iters=10,
        input_split_batched_alpha_max_leaves=200,
        # Per-disjunct-input (subbox) verification: give each subbox the
        # FULL remaining budget serially instead of an upfront rem/n_left
        # fraction. Lets acasxu prop_6's 2 subboxes (each needing >half the
        # cap on a slow GPU) both close within `total`. Default off — the
        # fractional split is better when there are many subboxes (mscn).
        input_split_serial_disjuncts=False,
        # When True, run the batched α-CROWN on EVERY unclosed leaf in the
        # batch (not just the eps-boundary band). Needed when per-leaf CROWN
        # lb sits far below 0 (tllverifybench: -30..-180) so the boundary band
        # never fires and the input-split explodes. Default OFF; opt-in.
        input_split_batched_alpha_all_leaves=False,
        # Exponent on the (1+n_unstable_in_dominant_shallow_layer) split-
        # selection boost. >1 sharpens the preference for bound-critical
        # unstable-branch dims. The non-LP forcing block raises this to
        # 2.0 for the pensieve signature (n_var>8), where it cut leaves
        # 10–198× (see docs/benchmarks/nn4sys.md). 1.0 = linear (no-op
        # exponent), the default elsewhere. Env BRANCH_BOOST_EXP overrides.
        input_split_batched_branch_boost_exp=1.0,
        # Per-phase profiler for the batched input-split BaB: prints a
        # [vc-phase] line with bound/clip/split time split + leaf/closure
        # counts. Off by default and fully short-circuited when off (no
        # perf_counter / cuda sync); env VC_PHASE_TIMING also forces it on.
        input_split_batched_phase_timing=False,
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
        # Floating-point-soundness inflation for spec-MILP pre-ReLU bounds.
        # The pre-ReLU interval bounds (lo, hi) are imposed as *hard* variable
        # bounds in the spec MILP/LP. They are computed in float32 (zono/CROWN),
        # but the MILP recomputes the affine in float64 (Gurobi + sparse W).
        # When a neuron's bound is near-degenerate (lo≈hi, e.g. a tiny
        # perturbation box where most neurons are nearly constant — collins_rul),
        # the float32↔float64 gap exceeds the bound width, so a genuinely
        # reachable point lands just outside [lo,hi] → the relaxation excludes
        # reachable points → falsely-infeasible spec LP → **false verified**.
        # Outward inflation (lo-=tol, hi+=tol; tol = atol + rtol·max|bound|)
        # restores the over-approximation: it can only make the feasibility LP
        # *more* feasible, never create a false-verify. Cost is completeness for
        # true-UNSAT margins below tol (negligible vs typical margins).
        milp_bound_inflation_atol=1e-5,
        milp_bound_inflation_rtol=1e-5,
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
        #              forward pass (the historical default; kept for
        #              the input-split sub-pipeline forced at L6360).
        #   'bab_refine' — α,β-CROWN-style cascade: forward zono → for
        #              each layer L do MILP-tighten z_L with sliding
        #              window of K layers + batched α-CROWN refresh
        #              that updates intermediate bounds globally
        #              between layers. PRODUCTION DEFAULT. Recovered
        #              +6 mnist_fc cases over legacy on the relusplitter
        #              benchmark with no oval21/cifar regressions
        #              (input-split fast-leaf and auto_route_milp_for_conv
        #              skip Phase 1 on those families anyway).
        phase1_method='bab_refine',
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
        # Auto-route small FC nets (≤2 ReLU layers, no conv, no bilinear,
        # input_dim > input-split cap) to milp_verify's exact per-neuron
        # MILP — the graph relaxation is too loose and input-split explodes
        # on them, but the exact MILP is sub-second (e.g. safenlp_2024).
        auto_route_milp_for_small_fc=True,
        # α-CROWN intermediate-bound tightening as Phase 1.5 in
        # milp_verify, BEFORE the per-layer LP/MILP loop. On conv
        # ResNets (oval21 deep_kw img3039: 1.4s α-CROWN closes 3/9
        # specs vs 29s LP+MILP closing 1/9 with WORSE worst-LB). The
        # joint α optimization tightens deep layers (L3-L5) that
        # per-neuron LP times out on, and preserves spec-direction
        # consistency. Phase 2 still runs after to catch remaining
        # unstables. Default ON.
        milp_alpha_tighten=True,
        milp_alpha_tighten_iters=10,
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
        # Timeout guard for the per-query α-zono state build loop: stop building
        # new per-query states (and skip the box-halfspace scoring) once fewer
        # than this many seconds remain, leaving room for the actual Phase 8
        # solve. Skipped queries fall back to the shared state. Prevents the
        # per-query loop blowing the budget on big conv nets (tinyimagenet).
        phase8_per_query_state_reserve_s=8.0,
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
        # High-bin infeasibility fallback. After standard Phase 8 racing
        # exhausts its octave/legacy schedule (capped at 8*n_workers bins
        # for `octaves` mode), each still-open query gets ONE more shot:
        # build the alpha_zono LP with the top-K most-impactful neurons as
        # binaries AND `qw·y + qb ≤ 0` halfspace constraint, look for
        # Gurobi `INFEASIBLE` (proves UNSAT). Empirically (oval21 deep_kw
        # img3039 q8): standard racing maxes at bins=32 → lb≈−0.144;
        # the fallback at bins=200 returns INFEASIBLE in ~30s → q8 closed,
        # full instance verified in 73s (matches AB-CROWN's 70s solve).
        # Sound: relaxation∩{qw·y+qb≤0}=∅ ⇔ relaxation min > 0, proves
        # spec. The cross-check via `_resolve_standard_lb` rejects any
        # numerical false-INFEASIBLE; if it can't confirm, we keep the
        # query open. Default ON.
        phase8_high_bin_fallback=True,
        # Max neurons the fallback binarizes. Sentinel 'all' (also None / inf /
        # <=0) means a full MILP over every unstable neuron — use it instead of
        # a magic 'bigger than any net' integer.
        phase8_high_bin_count=200,
        phase8_high_bin_time_limit=60.0,
        # High-bin fallback proof method. Default-on: minimize the spec margin
        # and stop via BestBdStop once its proven lower bound >= tol > 0 (an
        # explicit, auditable certificate). Disable to use the legacy
        # halfspace+INFEASIBLE proof (relies on Gurobi infeasibility detection).
        phase8_high_bin_bestbdstop=True,
        phase8_high_bin_bestbdstop_tol=1e-6,
        # Default-on: queue an all-neurons "complete" task (cuts ON + BestBdStop)
        # at the FRONT of the parallel racing pool, racing it concurrently with
        # the small-bin levels. Closes cases where every neuron matters (the
        # small bins plateau) without a per-benchmark flag. Disable if it slows
        # many-spec benchmarks (one extra full-MILP worker per open spec).
        phase8_race_all_bins=True,
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
        # ----- Phase 8 split scoring -----
        # Use box+halfspace per-neuron delta_LB scoring to rank BaB splits
        # (replaces the older `lp_ew_frac = |ew_k| * triangle_area`).
        # Bug 2026-05-14: DotMap's missing-attr returns DotMap() not True,
        # so `bool(getattr(settings, '...', True))` was always False and
        # this rerank was silently disabled. Now an explicit setting.
        phase8_score_box_halfspace=True,
        # Also a separate fix in score_box_halfspace_delta_lb: derive
        # ew_k = d[e_new_col]/μ from state instead of using the α-CROWN
        # capture_ew_per_relu output. They differ by ~1e-8 (FP32 noise),
        # but for neurons with lam=0 (degenerate parallelogram), the
        # off-side substitution's only adjustment is `d[e_new_col] -= ew*μ`.
        # Using state-derived ew exactly zeros this; α-CROWN-derived ew
        # leaves a tiny residual that gets amplified into wildly different
        # rankings across runs (with the same input).
        #
        # ----- GPU dual-ascent BaB (drop-in for Phase 8 MILP racing) -----
        # When True, replace `parallel_query_racing` with the substitution-
        # form GPU BaB in `dual_ascent_bab.verify_query_dual_ascent_bab`.
        # Each query is solved in ~0.3-1s on RTX 3080 (vs ~70s for the
        # MILP racing default on hard tinyimagenet cases). Sound: any λ ≥ 0
        # yields g(λ) ≤ LP_min by weak duality; only certifies when
        # computed best_g > 0. Sanity-checked vs Gurobi on prop_4260
        # (250 nodes, 100% decision match, 0 unsoundness). FP32 by default.
        phase8_use_dual_ascent_gpu=False,
        phase8_dual_ascent_max_iter=1,         # K — hard iter cap per node
        # Phase 8 minimum-budget floor as fraction of total_timeout. The
        # pipeline rebudgets so Phase 8 always gets at least this fraction
        # of the wall, trimming earlier phases if they overran. 0.0 = off.
        phase8_min_budget_frac=0.0,
        # Phase 2 (CROWN spec direction). When False, reuses the spec_lbs
        # already produced by Phase 0.5's batched α-CROWN — strictly tighter
        # than basic CROWN, so skipping is sound and saves ~1-3s/case on
        # cifar100/tinyimagenet ResNets.
        phase2_crown_enabled=True,
        phase8_dual_ascent_repair_steps=5,
        gen_lp_skip_phase7_lp=True,     # skip per-query LP scoring; use α-CROWN/CROWN ew*frac fallback (saves Phase 7 LP wall — was ~4s/query on hard CIFAR100)
        gen_lp_score_method='lp_ew_frac',  # 'lp_ew_frac', 'lp_fractional', 'lp_dual'. lp_dual ranks by |tri_lo|+|tri_up| duals — identifies the actual LP-binding triangles (on CIFAR100_resnet_medium_prop_idx_2477 the duals concentrate in L5 where kfsb/ew_frac promotes L9). lp_dual adds ~1-2s/query Phase-8 overhead to re-solve gen-LP with dual extraction; beneficial on hard queries where the wrong layer is being branched on, neutral-to-slow otherwise. Opt-in via settings.
        skip_phase8_milp=False,         # if True, Phase 8 is skipped and queries Phase 7 LP can't prove UNSAT are returned as 'unknown'
        # Phase 1 MILP-tightens layers ≤ this idx; LP-only at deeper
        # layers up to `max_tighten_layer_lp`. Default raised from 1 to
        # 2 on 2026-05-10 (+5 mnist_fc verifications), then to 3 later
        # the same day after profiling mnist_256x6 prop_4_0.03 with a
        # 30 s budget: ML=2 timed out at 30.5 s but ML=3 verified in
        # 24.7 s — the deeper L=2 MILP gives Phase 8 racing tighter
        # bounds and closes all queries within budget. Mini-sweep on
        # 4 mnist cases (256x4 prop_3/5/8, 256x6 prop_5) showed walls
        # within 0.5 s of ML=2 — no regressions.
        max_tighten_layer=3,
        # When set, Phase 1 extends tightening to layers in the range
        # (max_tighten_layer, max_tighten_layer_lp] using LP only (not MILP).
        # Rationale: L1 benefits from MILP (tight big-M exact triangle), but
        # deeper layers would be too slow for MILP. LP-only per-neuron
        # tightening at L2 is cheap (~1 s avg on CIFAR100_resnet_medium)
        # and shrinks the downstream unstable pool seen by α-CROWN and
        # Phase 8 MILP. Default 2: 200-case CIFAR100 sweep showed +4 new
        # verifications, 0 regressions, 0 unsound, ~+0.36 s avg total time.
        max_tighten_layer_lp=2,
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
        # In `_verify_per_disjunct_subboxes` (batched per-disjunct subbox
        # path for many-disjunct specs like mscn cardinality), use forward
        # LiRPA instead of forward zonotope. LiRPA gives tighter
        # intermediate bounds for sigmoid/tanh/mul_bilinear (mscn uses
        # all three). Same caller signature via
        # `forward_lirpa_compat_zono_batched` adapter.
        use_forward_lirpa_subboxes=True,
        # Mini-group size for `_multi_sub_input_split_bab`. 60 is the safe
        # default; `default_settings_for` overrides to 120/200 for instances
        # with many disjuncts (see `_adapt_per_disjunct`). Env MINI_GROUP_SIZE
        # wins over both.
        mini_group_size=60,
        # Multi-dim simultaneous split in input-split BaB. K=1 = single
        # widest-dim split (2 children, matches ABC's runtime behavior
        # when queue ≥ min_batch_size=25.6 — see ABC's
        # `input_split/split.py:get_split_depth`). Tried K>1 on mscn_240
        # and it REGRESSED because most leaves have 1 varying dim of 8
        # total → K>1 picks zero-width dims and produces duplicate
        # children that compound exponentially. The right fix is
        # ABC-style ADAPTIVE depth (small queue → high K, large queue →
        # K=1); not implementing that yet.
        bab_split_depth=1,
        # Time is the budget; no depth cap by default (was 8 — too
        # aggressive; on cersyve UNSAT cases the BaB needs ~1k nodes
        # per leaf-verification path). Set to a positive int to opt
        # back into a depth cap.
        input_split_max_depth=None,
        input_split_node_timeout=8.0,
        # Fast-leaf path for input-split BaB. When True, each leaf runs
        # only forward zono + CROWN backward + α-CROWN(N iters) + spec
        # check, bypassing the full _run_pipeline's PGD/multiprocess
        # overhead (~60s per leaf). On cifar_biasfield_0 a leaf takes
        # ~3.4s here vs ~65s via _run_pipeline. Default ON.
        input_split_fast_leaf=True,
        # Truncate joint α-CROWN's `intermediate_start_nodes` per leaf
        # to at most this many (deepest first); 0 = use all unstable
        # layers (legacy). Joint α cost is ~linear in start_node count
        # but bound improvement saturates after the deepest 2-3 layers
        # — the split itself tightens shallow layers anyway. See
        # `_input_split_fast_leaf` for the exact dispatch.
        input_split_alpha_max_start_nodes=0,
        # α-CROWN iters per leaf. Reduced from 3 to 1 on 2026-05-10:
        # measured -13s on cifar_biasfield_8 (60.7s → 47s), -5s on
        # _29 (21s → 16s), -19s on _28 (57s → 38s) — the win is that
        # leaves fail fast, the BaB tree explores more splits within
        # the budget, and tighter sub-leaves close in 1 iter via the
        # `early_stop_on_positive` exit anyway. Higher iters help
        # individual leaves converge but the wall budget is dominated
        # by the leaf count, not per-leaf depth.
        input_split_alpha_iters=1,
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
        # Hard wall-time budget for the entire Phase 2.5 cascade across all
        # disjuncts. None = unlimited (legacy). When set, the per-disjunct
        # outer loop checks elapsed before starting each disjunct and the
        # per-query inner loop checks before each query; once exceeded,
        # remaining queries are left open and Phase 2.5 returns. Bound is
        # soundness-preserving: every query starts from `bounds_by_relu`
        # and only tightens, so an aborted query is no worse than skipped.
        zono_lift_time_budget=None,
        # Disjunct-level early abort. After the batched pass-0 α-CROWN, if
        # the worst (most-negative) `lb_alpha` over all queries in a
        # disjunct is below this threshold, skip the per-query cascade for
        # the entire disjunct — it cannot close in 2.5 (a disjunct only
        # closes when ALL its queries close). None = disabled (legacy).
        # Negative value typical (e.g. -0.3); the smaller (more negative)
        # the threshold, the more aggressive the skip.
        zono_lift_disjunct_hopeless_lb=None,
        # Promising-first ordering across disjuncts and within a disjunct.
        # When True (default once enabled), disjuncts are processed in
        # descending order of their min-query lb_alpha (most likely to
        # fully close first), and queries within a disjunct in ASCENDING
        # order of lb_alpha (bottleneck first — if it cannot close, the
        # disjunct cannot either, so we detect hopeless cases earliest).
        # Requires batched pass-0 results (`zono_lift_batch_queries=True`).
        zono_lift_promising_first=True,
        # Phase 2.6: per-spec targeted PGD after Phase 2.5 closes easy
        # disjuncts. For each still-open disjunct, run a small focused
        # PGD with `restrict_disj={di}` so the loss landscape isolates
        # ONE spec's margins (vs Phase 3.5's union-of-margins PGD which
        # can dilute the gradient when many disjuncts are open). Total
        # wall is hard-capped by `phase26_pgd_per_spec_time_budget`;
        # per-disjunct time is `total / n_open`, floored at 0.2s. Stops
        # on first SAT witness.
        phase26_pgd_per_spec_enabled=True,
        phase26_pgd_per_spec_time_budget=3.0,
        phase26_pgd_per_spec_min_per_spec=0.2,
        # `strict_min=True`: each open spec gets at least min_per_spec
        # regardless of total budget. False = stop when total exhausted.
        phase26_pgd_per_spec_strict_min=True,
        # Pre-α-CROWN PGD hook (zono-sorted). Fires after forward zono +
        # adaptive CROWN, before α-CROWN. If SAT found, skip α-CROWN +
        # cascade entirely. Budget per open spec is
        # max(n × pgd_per_spec_min, time_left × total_frac) / n, capped
        # at per_spec_cap. See verify_graph._pre_cascade_pgd_hook.
        phase26_pre_cascade_enabled=True,
        phase26_pre_cascade_total_frac=0.10,
        phase26_pre_cascade_per_spec_cap=5.0,
        # Parallel PGD (background THREAD) that runs PGD attacks during
        # Phase 1's pure-CPU MILP windows. A GPU lock is held by main
        # during all GPU work (α-CROWN, gen_cone_state setup, forward
        # zono) and released only around `pool.imap_unordered` in the
        # MILP cascade — the genuinely GPU-idle window (~13s on
        # tinyimagenet medium). The thread acquires the lock per attack,
        # so PGD's CUDA kernels only run during MILP windows and α-CROWN
        # is not slowed (validated: α-CROWN time unchanged at 2.5s
        # locked vs 2.5s solo). Cap on attacks via `parallel_pgd_max_attacks`
        # prevents thread from bleeding into Phase 7/8.
        parallel_pgd_enabled=False,
        parallel_pgd_max_attacks=20,
        # ONNXRuntime SAT-witness validation (defense-in-depth).
        # Before returning 'sat' from any path, run the witness through
        # the ORIGINAL ONNX model + check it actually violates the spec
        # within `sat_validate_atol` (mirrors VNNCOMP scoring's
        # COUNTEREXAMPLE_ATOL=1e-4). Spurious witnesses are downgraded
        # to 'unknown' with `details['spurious_witness']` populated.
        # `skip_sat_validation=True` opts out (e.g. for ORT-free envs).
        sat_validate_atol=1e-4,
        skip_sat_validation=False,
        # ONNXRuntime VERIFIED-witness validation (defense-in-depth).
        # When a verdict comes back 'verified', sample N points from the
        # input box, forward them through the ORIGINAL ONNX model, and
        # check that NONE counterexamples the spec. If any does, the
        # verified verdict was unsound — downgrade to 'unknown' with
        # `details['spurious_verified']` set. Finite sampling is NOT a
        # soundness proof of UNSAT (we may miss adversarial inputs), but
        # ANY counterexample found is a true counterexample. Catches
        # Class-1 unsoundness (verifier silently certified a SAT spec)
        # at near-zero cost. `skip_verified_validation=True` opts out
        # (e.g. for ORT-free envs or speed-critical sweeps).
        verified_validation_samples=32,
        skip_verified_validation=False,
        # Fallback policy for nonlinear bilinear ops (mul_bilinear with
        # both sides varying; div_bilinear with non-point denominator).
        # 'raise' = strict (NotImplementedError if no exact handling).
        # 'box'   = sound decorrelated box-enclosure: new error generator
        #            per output element, no x-y correlation preserved.
        #            Looses tightness but unblocks pensieve_*_parallel
        #            (Pow→ReduceSum→Div softmax-style normalization) and
        #            mscn cases with non-point masks. Defaults to 'box'
        #            since a sound looser bound is strictly better than
        #            an unhandled-op error verdict.
        nonlin_div_fallback='box',
        nonlin_mul_fallback='box',
        # Pow relaxation form. 'chord' = chord-tangent parallelogram
        # per element (tighter; preserves input-output correlation via
        # the chord slope; sound on uniform-curvature intervals).
        # 'box' = box-decorrelated (sound, simpler, loose). Defaults to
        # chord since it's tighter on the common case (post-ReLU input
        # is non-negative so chord is uniformly valid).
        pow_relaxation='chord',
        # Phase 1 gen-LP conv chunking. Default 256 = chunked with safe
        # block size + OOM-halve-retry fallback. The chunk loop itself
        # is ~0.3% overhead vs un-chunked; OOM halving costs ~0.2% per
        # event. Set to None to force the un-chunked legacy path.
        # Read via env var VC_GEN_LP_CONV_CHUNK inside _gen_cone_state.
        gen_lp_conv_chunk=256,
        # Storage form for gen-LP `G_out` allocations in precompute_gen_state.
        # 'dense' = legacy (n × n_gens) torch.zeros.
        # 'sparse' = `_StructuredSparseG` (dense rows for stable passthrough +
        #   identity-entry lists for unstable/stable-new). Identical results
        #   (materialize-equivalent), ~0% overhead on cases that fit, ~4×
        #   memory savings on big nets (cifar100_resnet_large L1: 1.55 GB →
        #   ~400 MB).  Default 'sparse' since it's strictly equal or better.
        # Read via env var VC_GEN_LP_G_STORAGE inside _gen_cone_state.
        gen_lp_g_storage='sparse',
        # α-CROWN backward direction-batching mode for run_alpha_crown_batched
        # in `_alpha_refresh_best_bounds`.
        #   'joint' — compute LB and UB backwards per chunk together; both
        #             autograd graphs live until loss.backward(). Tightest
        #             per-iter, highest memory.
        #   'split' — separate spec-loss passes for LB-only and UB-only;
        #             each pass's autograd graph dies at its loss.backward().
        #             ~half memory peak, ~1.5-2× wall per iter.
        #   'auto'  — start in 'joint', sticky-downgrade to 'split' on OOM.
        # Default 'split' (not 'auto'): the joint-mode OOM-then-fallback
        # path leaves ~2 GB of joint-attempt autograd state behind that
        # gc.collect can't reclaim, causing the split fallback to fail
        # on big nets that split alone would handle (cifar100_resnet_large
        # prop_2461 etc.). Split is ~5% slower than joint on cases that
        # fit in either; we accept that for the robustness.
        alpha_crown_dir_mode='split',
        # When >1, split sorted(intermediate_start_nodes) into N groups in
        # `run_alpha_crown_batched._do_pass`. Each group runs its own
        # spec backward + loss.backward(), freeing autograd between groups.
        # Cuts peak autograd retention to ~1/N at cost of N× spec backwards
        # (and N-times-looser per-group gradient signal since spec backward
        # sees partial bbr updates). Default 2 since the s_split=1→2 OOM
        # fallback ladder is unreliable (partial OOM state leaks across
        # retry); ~7% wall overhead on cases that fit either, recovers
        # cifar100_resnet_large cases that need s_split≥2.
        alpha_crown_s_split_n=2,
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
