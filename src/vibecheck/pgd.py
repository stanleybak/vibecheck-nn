"""PGD counterexample search — α,β-CROWN-style implementation.

Design choices (matched to α,β-CROWN vnncomp25 cifar100 defaults):
  - Batched: all restarts run concurrently on GPU as [R, *input_shape]
  - AdamClipping optimiser: Adam momentum for direction, signed-ε step magnitude
  - Hinge loss: clamp per-constraint margin at `hinge_threshold=-1e-5` so
    satisfied constraints stop contributing gradient; optimiser focuses on
    still-positive margins
  - Step-size decay `lr_decay=0.99` per iter
  - Budget 10 restarts × 100 steps (deep-ResNet optima need long runs;
    short 10-step PGD stalls in saturated-ReLU plateaus)
  - GPU-side witness detection: per-sample per-disjunct max margin on GPU;
    only fall to CPU `spec.check` for candidates flagged by GPU. Keeps the
    per-iter hot path GPU-only — no 1000× CPU `.check()` calls.
  - Wall-time budget: caller can cap the total attack time with
    `pgd_time_budget` so PGD cannot burn the whole timeout on a hopeless case.
  - Optional `restrict_disj` — re-attack specifically the disjuncts that
    CROWN couldn't verify (α,β-CROWN `pgd_order='middle'` trick)
"""
import time
import numpy as np
import torch

# α,β-CROWN does this at attack-module import: avoids PyTorch's
# ProfilingExecutor re-profiling each call, which costs 20-50 ms/iter on
# deep resnets (see /tmp/abcrown_runs/.../attack/attack_pgd.py:26-27).
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


def _freeze_weight_grads(gg):
    """Disable gradient tracking on all graph weights. PGD only needs grad
    wrt the adversarial input — keeping grads on weights forces PyTorch to
    retain the full autograd graph through every conv/fc per iter.
    `gg_pgd` is the attack-time float32 graph, so this is an irreversible
    flag flip only on its tensors.
    """
    for op in gg['ops']:
        for k, v in list(op.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.requires_grad_(False)


def _get_traced_forward(gg, sample_x):
    """Return a TorchScript-traced forward of the graph, cached on `gg`.

    α,β-CROWN runs PGD on the raw PyTorch ONNX-imported nn.Sequential
    (`abcrown.py:56-64`: `model = model_ori`). For our graph-dict
    representation, `_forward_batch_graph` iterates ops in Python. For
    ~40-op resnets × 100 PGD iters that's ~100× more interpreter calls
    than a raw nn.Module would pay. torch.jit.trace records the tensor
    ops once and replays them as a fused graph — matches α,β-CROWN's
    per-iter cost (~5 ms on RTX 3080 at batch 10 for resnet_large).

    The cached trace is invalidated if batch-size or input shape changes;
    caller is responsible for using the same shape across iters.
    """
    from .verify_zono_bnb import _forward_batch_graph
    sig = (sample_x.shape, sample_x.dtype, sample_x.device)
    cache = gg.get('_pgd_traced')
    if cache is not None and cache['sig'] == sig:
        return cache['fn']

    def _fn(x):
        return _forward_batch_graph(x, gg)

    with torch.no_grad():
        # Trace with the concrete sample; forward batches of this exact
        # shape afterward. PGD uses fixed (n_restarts, *input_shape).
        traced = torch.jit.trace(_fn, sample_x.detach(), check_trace=False)
    gg['_pgd_traced'] = {'sig': sig, 'fn': traced}
    return traced


def _build_constraint_matrices(per_disj_constraints, n_out, dtype, device):
    """Pre-compute (W, b) matrices so margins = out @ W.T + b (one matmul).

    Sign convention MUST match ``spec.py``'s ``Constraint.margin`` /
    ``PairwiseConstraint.margin``: margin > 0 = SAFE for that constraint,
    margin ≤ 0 = unsafe-side (PGD wants to push margins ≤ 0). For a
    point input, margins are evaluated as:

      pairwise (Y_comp >= Y_pred unsafe):  margin = y[pred] - y[comp]
      threshold Y[i] >= val (unsafe):      margin = val - y[i]
      threshold Y[i] <= val (unsafe):      margin = y[i] - val

    A prior version sign-flipped both threshold cases (used `y[i] - val`
    for '>=' and `val - y[i]` for '<='). On pairwise-only benchmarks
    (cifar100, tinyimagenet) the bug was silent. On threshold-spec
    benchmarks (cersyve: 6 SAT cases) the PGD loss = sum of clamped
    margins was effectively MAXIMIZED for SAT — gradient descent ran
    AWAY from witnesses that brute-force sampling finds in <100k
    samples. Documented here because the bug is easy to reintroduce
    when refactoring; see ``tests/test_pgd_margin_signs.py``.

    Returns list of (W, b) per disjunct.
    """
    mats = []
    for constraints in per_disj_constraints:
        n_c = len(constraints)
        W = torch.zeros(n_c, n_out, dtype=dtype, device=device)
        b = torch.zeros(n_c, dtype=dtype, device=device)
        for i, c in enumerate(constraints):
            if hasattr(c, 'pred'):
                W[i, c.pred] = 1.0
                W[i, c.comp] = -1.0
            elif c.op == '>=':
                # margin = val - y[i] → W=-e_i, b=+val
                W[i, c.index] = -1.0
                b[i] = float(c.value)
            elif c.op == '<=':
                # margin = y[i] - val → W=+e_i, b=-val
                W[i, c.index] = 1.0
                b[i] = -float(c.value)
            else:
                raise ValueError(f'unknown constraint op {c.op!r}')
        mats.append((W, b))
    return mats


def _compute_margins_per_disj_batched(out, mats):
    """Returns list of [n_batch, n_constraints_d] margin tensors (one matmul
    per disjunct). `out @ W.T + b` replaces the 99-op Python loop."""
    return [out @ W.T + b for (W, b) in mats]


class _PGDOptim:
    """Pluggable PGD optimizer with three modes:

      'sign_sgd'      — sign of raw gradient (no Adam state).
                        Matches α,β-CROWN's `use_adam=False` branch in
                        `attack_pgd.py`: `delta + alpha * sign(delta.grad)`.
      'adam_sign'     — sign of bias-corrected Adam direction
                        (m_hat / sqrt(v_hat) + eps). vibecheck's
                        historical default.
      'adam_clipping' — α,β-CROWN's `AdamClipping` (attack_utils.py):
                        update = exp_avg / denom * step_size where
                        step_size = lr / bias_correction1, then
                        scaled_update = sign(update) * step_size.
                        Effective step magnitude is `lr / bias_correction1`
                        which is ~10× lr at iter 1 and decays to lr.
                        (`exp_avg` is unbias-corrected — sign matches
                        bias-corrected sign though, so the only
                        operationally meaningful difference vs adam_sign
                        is the bias-corrected step magnitude.)

    `compute_delta(grad, base_step)` returns the per-element delta to be
    SUBTRACTED from x for gradient descent on the loss (caller convention:
    minimize loss, so x_new = x - delta).
    """

    _MODES = ('sign_sgd', 'adam_sign', 'adam_clipping')

    def __init__(self, mode, shape, dtype, device,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        assert mode in self._MODES, (
            f'unknown pgd_optim {mode!r}; valid: {self._MODES}')
        self.mode = mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        if mode != 'sign_sgd':
            self.m = torch.zeros(shape, dtype=dtype, device=device)
            self.v = torch.zeros(shape, dtype=dtype, device=device)

    def compute_delta(self, grad, base_step):
        """base_step: tensor (broadcastable to grad) or scalar.
        Returns the descent delta — caller does `x_new = x - delta`.
        """
        self.t += 1
        if self.mode == 'sign_sgd':
            return base_step * grad.sign()
        b1, b2, eps_a = self.beta1, self.beta2, self.eps
        self.m.mul_(b1).add_(grad, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        if self.mode == 'adam_sign':
            m_hat = self.m / bc1
            v_hat = self.v / bc2
            direction = m_hat / (v_hat.sqrt() + eps_a)
            return base_step * direction.sign()
        # 'adam_clipping': α,β-CROWN's per-iter step.
        denom = (self.v.sqrt() / (bc2 ** 0.5)) + eps_a
        direction = (self.m / denom).sign()
        return (base_step / bc1) * direction


def _gpu_witness_candidates(margins_per_disj):
    """GPU-only witness screen mirroring ``VNNSpec.check`` semantics.

    A spec output is 'unknown' (= candidate counterexample) iff the
    worst-case margin = ``min over (d, c) of margin_dc`` is ≤ 0.

    Returns a 1-D bool mask of candidate-indices. Caller should run
    ``spec.check`` only on these to confirm.
    """
    if not margins_per_disj:
        return None
    # Concatenate every (d, c) constraint margin into one (n_batch, total_c)
    # tensor and find the per-sample minimum. Anywhere min ≤ 0 there exists
    # some constraint ≤ 0 — exactly what spec.check looks for.
    all_margins = torch.cat(margins_per_disj, dim=1)
    overall_min = all_margins.min(dim=1).values  # [n_batch]
    return overall_min <= 0.0


def pgd_attack_general(xl, xh, spec, gg, settings,
                        restrict_disj=None, time_budget=None):
    """PGD counterexample search for any VNNSpec (DNF).

    Args:
        xl, xh: input-box tensors on the compute device.
        spec: `VNNSpec` with `.disjuncts` (DNF of conjunctions).
        gg: forward-friendly GPU graph (fp32 recommended).
        settings: DotMap with pgd_restarts, pgd_iter, pgd_lr_decay,
            pgd_hinge_threshold, pgd_alpha_frac.
        restrict_disj: optional set of disjunct indices to attack (others
            assumed already verified). None = attack every disjunct.
        time_budget: optional wall-time cap (seconds). If exceeded, abort
            cleanly and return (False, None). Overrides `pgd_time_budget`
            setting.

    Returns:
        (is_sat: bool, witness: np.ndarray or None).
    """
    _freeze_weight_grads(gg)  # idempotent, cheap flag flip

    dev = xl.device
    dt = xl.dtype
    n_restarts = int(getattr(settings, 'pgd_restarts', 10))
    n_iter = int(getattr(settings, 'pgd_iter', 100))
    lr_decay = float(getattr(settings, 'pgd_lr_decay', 0.99))
    hinge_thr = float(getattr(settings, 'pgd_hinge_threshold', -1e-5))
    alpha_frac = float(getattr(settings, 'pgd_alpha_frac', 0.25))
    if time_budget is None:
        _tb = getattr(settings, 'pgd_time_budget', None)
        # DotMap's getattr returns an empty DotMap for missing keys, not None
        if isinstance(_tb, (int, float)):
            time_budget = float(_tb)
    t_start = time.perf_counter()

    eps_input = (xh - xl) / 2.0
    # Multi-α restart pool: a single α is brittle on benchmarks with
    # narrow-SAT regions where the right step size varies across cases
    # (cersyve: lane_keep wants α≈0.01, pendulum wants α≈0.005, and
    # different alphas crack different SAT cases). When
    # `pgd_alpha_multi=True` (default False), partition the restart
    # pool across the log-spaced alphas in `pgd_alpha_multi_fractions`
    # so each restart gets its own α. Equivalent throughput, much
    # broader coverage.
    _alpha_multi = bool(getattr(settings, 'pgd_alpha_multi', False))
    if _alpha_multi:
        _alpha_fracs = list(getattr(
            settings, 'pgd_alpha_multi_fractions', [0.25, 0.05, 0.01, 0.002]))
        # Per-restart α: cycle through the alpha list
        # step_size_tensor shape will be (n_restarts, *eps_input.shape)
        per_restart_alpha = torch.tensor(
            [_alpha_fracs[i % len(_alpha_fracs)] for i in range(n_restarts)],
            dtype=dt, device=dev)
        # Broadcast (n_restarts,) × eps_input shape via outer product on
        # dimension 0
        while per_restart_alpha.dim() < eps_input.dim() + 1:
            per_restart_alpha = per_restart_alpha.unsqueeze(-1)
        step_size_tensor = eps_input.unsqueeze(0) * per_restart_alpha
    else:
        step_size_tensor = eps_input * alpha_frac
    pgd_optim = str(getattr(settings, 'pgd_optim', 'adam_sign'))

    # Select disjuncts
    if restrict_disj is None:
        active = list(spec.disjuncts)
    else:
        active = [spec.disjuncts[di] for di in sorted(restrict_disj)]
    if not active:
        return False, None

    per_disj_constraints = [list(dj.constraints) for dj in active
                             if dj.constraints]
    if not per_disj_constraints:
        return False, None

    # Batched initial point. Two init modes:
    #   'uniform' — random uniform in input box (historical default)
    #   'osi'     — Output Sampling Initialization, mirrors α,β-CROWN's
    #               diversed_PGD: 50 sign-grad steps maximizing
    #               `dot(w_d, model(x))` with random `w_d` per restart.
    #               Diversifies starting points across the network's
    #               output regions, helping PGD find counterexamples on
    #               non-trivial loss landscapes (mnist_fc, cifar_biasfield
    #               SAT cases AB finds via this init that uniform PGD
    #               misses).
    init_mode = str(getattr(settings, 'pgd_init_mode', 'uniform'))
    x_adv = xl + (xh - xl) * torch.rand(
        n_restarts, *xl.shape, dtype=dt, device=dev)
    x_adv = x_adv.detach().requires_grad_(True)

    # Compile the forward once (cached on gg).
    traced_forward = _get_traced_forward(gg, x_adv)
    if init_mode == 'osi':
        with torch.no_grad():
            _probe_for_osi = traced_forward(x_adv[:1].detach())
            n_out_for_osi = _probe_for_osi.shape[-1]
        # Random projection per restart in [-1, 1]^n_out.
        w_d = (torch.rand(n_restarts, n_out_for_osi,
                          dtype=dt, device=dev) - 0.5) * 2.0
        osi_iters = int(getattr(settings, 'pgd_osi_iters', 50))
        # OSI step size matches α,β-CROWN: same as PGD's eps step.
        osi_step = step_size_tensor
        x_osi = x_adv.detach().clone().requires_grad_(True)
        for _ in range(osi_iters):
            out = traced_forward(x_osi)
            # Sum-over-batch of dot(w_d_i, out_i) → scalar; one backward
            # gives per-restart gradients (each restart's gradient is
            # only influenced by its own w_d row because batch reductions
            # are linear).
            loss = (w_d * out).sum()
            grad = torch.autograd.grad(loss, x_osi, create_graph=False)[0]
            with torch.no_grad():
                x_osi = x_osi + osi_step * grad.sign()
                x_osi = torch.clamp(x_osi, xl, xh)
            x_osi = x_osi.detach().requires_grad_(True)
        x_adv = x_osi.detach().requires_grad_(True)

    # Pre-build the (W, b) constraint matrices so margin computation is one
    # matmul per disjunct instead of N_constraints Python-side ops.
    with torch.no_grad():
        _probe = traced_forward(x_adv[:1].detach())
        n_out = _probe.shape[-1]
    spec_mats = _build_constraint_matrices(
        per_disj_constraints, n_out, dt, dev)

    optim = _PGDOptim(pgd_optim, x_adv.shape, dt, dev)
    cur_step = step_size_tensor.clone()

    def _confirm_witness(x_batch, out_batch, cand_mask):
        """For each candidate index, run the canonical spec.check. Returns
        (witness_np or None)."""
        out_cpu = out_batch.detach().cpu().numpy()
        idxs = cand_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        for b in idxs:
            result, _ = spec.check(out_cpu[b], out_cpu[b])
            if result == 'unknown':
                return x_batch[b].detach().cpu().numpy()
        return None

    # Plateau-based give-up: track the best worst-margin (≤0 means
    # some restart found a constraint in unsafe direction). If it
    # doesn't improve toward zero for `plateau_iters` consecutive
    # iters AND no restart is below zero, give up — no SAT likely.
    # Saves ~80% of PGD time on UNSAT cases. AB-CROWN's
    # `pgd_restart_when_stuck` does similar.
    plateau_iters = int(getattr(settings, 'pgd_plateau_iters', 100))
    best_min_margin = float('inf')
    iters_without_improvement = 0

    for t in range(1, n_iter + 1):
        if time_budget is not None and (time.perf_counter() - t_start) > time_budget:
            break

        out = traced_forward(x_adv)
        margins_pd = _compute_margins_per_disj_batched(out, spec_mats)

        # GPU-side witness screen
        cand = _gpu_witness_candidates(margins_pd)
        if cand is not None and cand.any():
            witness = _confirm_witness(x_adv, out, cand)
            if witness is not None:
                return True, witness

        # Plateau check: take the per-sample min margin across all
        # constraints; if no sample is anywhere near unsafe AND it's
        # not getting closer over many iters, abandon.
        with torch.no_grad():
            if margins_pd:
                all_m = torch.cat(margins_pd, dim=1)
                curr_min = float(all_m.min().item())
                # Only count "no improvement" when ALL restarts are
                # above the hinge (no candidate to refine).
                if curr_min > hinge_thr:
                    if curr_min < best_min_margin - 1e-6:
                        best_min_margin = curr_min
                        iters_without_improvement = 0
                    else:
                        iters_without_improvement += 1
                    if iters_without_improvement >= plateau_iters:
                        break
                else:
                    iters_without_improvement = 0  # active region, keep going

        # Flatten margins into one tensor for hinge loss.
        # Loss formulation matches α,β-CROWN's `default_pgd_loss`:
        # sum (over surviving spec rows) of clamp(margin, min=-1e-5).
        # Tried per-disjunct max-over-constraints loss (focuses gradient
        # on the AND group's least-violated constraint), but on cersyve
        # it oscillates between constraints and lands on 1/12 SAT vs the
        # sum loss's 2/12. Sum gives every still-positive constraint
        # gradient simultaneously; once one drops below the hinge it
        # freezes and the optimizer redistributes effort. Required to
        # match α,β-CROWN's PGD success rate on near-boundary CEXes.
        flat_margins = torch.cat(margins_pd, dim=1) if margins_pd else None
        if flat_margins is None or flat_margins.numel() == 0:
            return False, None
        clamped = torch.clamp(flat_margins, min=hinge_thr)
        per_sample = clamped.sum(dim=1)
        loss = per_sample.sum()
        x_adv.grad = None
        loss.backward()

        with torch.no_grad():
            delta = optim.compute_delta(x_adv.grad, cur_step)
            x_new = torch.clamp(x_adv - delta, xl, xh)
            x_adv = x_new.detach().requires_grad_(True)
            cur_step = cur_step * lr_decay

    # Final check on last iterate.
    with torch.no_grad():
        out = traced_forward(x_adv)
        margins_pd = _compute_margins_per_disj_batched(out, spec_mats)
        cand = _gpu_witness_candidates(margins_pd)
        if cand is not None and cand.any():
            witness = _confirm_witness(x_adv, out, cand)
            if witness is not None:
                return True, witness
    return False, None


def pgd_attack_from_init(x_init_batch, xl, xh, spec, gg, settings, *,
                          restrict_disj=None, n_iter=20,
                          time_budget=None):
    """PGD from a caller-supplied batch of initial points (vs random).

    Used to refine MILP near-boundary witnesses: the MILP's relaxation
    gave us a point at the spec boundary that may or may not correspond
    to a real counterexample; a short PGD from there + small nearby
    perturbations has a good chance of walking to a real violation if
    one is nearby. Returns (is_sat, witness_np) — witness is the real
    network input (not the MILP's relaxation), verified via spec.check.

    Args:
      x_init_batch: (B, *input_shape) tensor of starting points, on gg's
        device/dtype. Caller ensures entries are inside [xl, xh].
      xl, xh: input box tensors.
      spec: `VNNSpec`.
      gg: forward-friendly GPU graph.
      settings: DotMap — `pgd_lr_decay`, `pgd_hinge_threshold`,
        `pgd_alpha_frac` read; `pgd_restarts`/`pgd_iter` IGNORED (caller
        controls the batch size and iteration count).
      restrict_disj: optional set of disjunct indices to attack.
      n_iter: PGD iters (default 20 — short, since starting close).
      time_budget: optional wall-time cap.

    Returns (is_sat: bool, witness: np.ndarray or None).
    """
    _freeze_weight_grads(gg)
    dev = xl.device; dt = xl.dtype
    lr_decay = float(getattr(settings, 'pgd_lr_decay', 0.99))
    hinge_thr = float(getattr(settings, 'pgd_hinge_threshold', -1e-5))
    alpha_frac = float(getattr(settings, 'pgd_alpha_frac', 0.25))
    pgd_optim = str(getattr(settings, 'pgd_optim', 'adam_sign'))
    t_start = time.perf_counter()

    # Clamp init to box (defensive — caller may have tiny float slack).
    x_adv = torch.clamp(x_init_batch.detach(), xl, xh).to(
        device=dev, dtype=dt).requires_grad_(True)

    eps_input = (xh - xl) / 2.0
    step_size_tensor = eps_input * alpha_frac
    cur_step = step_size_tensor.clone()

    if restrict_disj is None:
        active = list(spec.disjuncts)
    else:
        active = [spec.disjuncts[di] for di in sorted(restrict_disj)]
    if not active:
        return False, None
    per_disj_constraints = [list(dj.constraints) for dj in active
                             if dj.constraints]
    if not per_disj_constraints:
        return False, None

    traced_forward = _get_traced_forward(gg, x_adv)
    with torch.no_grad():
        _probe = traced_forward(x_adv[:1].detach())
        n_out = _probe.shape[-1]
    spec_mats = _build_constraint_matrices(
        per_disj_constraints, n_out, dt, dev)

    def _confirm(xb, ob, cm):
        out_cpu = ob.detach().cpu().numpy()
        for b in cm.nonzero(as_tuple=False).view(-1).tolist():
            res, _ = spec.check(out_cpu[b], out_cpu[b])
            if res == 'unknown':
                return xb[b].detach().cpu().numpy()
        return None

    # Initial check (the MILP witness may already be a real counterexample).
    with torch.no_grad():
        out = traced_forward(x_adv)
        margins_pd = _compute_margins_per_disj_batched(out, spec_mats)
        cand = _gpu_witness_candidates(margins_pd)
        if cand is not None and cand.any():
            w = _confirm(x_adv, out, cand)
            if w is not None:
                return True, w

    optim = _PGDOptim(pgd_optim, x_adv.shape, dt, dev)
    for t in range(1, n_iter + 1):
        if time_budget is not None and (time.perf_counter() - t_start) > time_budget:
            break
        out = traced_forward(x_adv)
        margins_pd = _compute_margins_per_disj_batched(out, spec_mats)
        cand = _gpu_witness_candidates(margins_pd)
        if cand is not None and cand.any():
            w = _confirm(x_adv, out, cand)
            if w is not None:
                return True, w
        flat = torch.cat(margins_pd, dim=1) if margins_pd else None
        if flat is None or flat.numel() == 0:
            return False, None
        # SUM-of-hinged-margins (α,β-CROWN match); see comment in
        # `pgd_attack_general` for why MAX is wrong.
        loss = torch.clamp(flat, min=hinge_thr).sum(dim=1).sum()
        x_adv.grad = None; loss.backward()
        with torch.no_grad():
            delta = optim.compute_delta(x_adv.grad, cur_step)
            x_new = torch.clamp(x_adv - delta, xl, xh)
            x_adv = x_new.detach().requires_grad_(True)
            cur_step = cur_step * lr_decay

    with torch.no_grad():
        out = traced_forward(x_adv)
        margins_pd = _compute_margins_per_disj_batched(out, spec_mats)
        cand = _gpu_witness_candidates(margins_pd)
        if cand is not None and cand.any():
            w = _confirm(x_adv, out, cand)
            if w is not None:
                return True, w
    return False, None
