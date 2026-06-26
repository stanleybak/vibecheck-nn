"""Sign-BNN attack mode — for binarized networks vibecheck (and ABC's onnx2pytorch) can't
bound soundly because the `Sign` activation is a hard step. Incomplete / attack-only (never
returns unsat): find an adversarial counterexample by PGD on a straight-through-estimator
(STE) surrogate of `Sign`, then VALIDATE the witness on the ORIGINAL (true-`Sign`) model via
CPU onnxruntime (the scoring engine) — so a mismatched surrogate can never yield a false sat.

The recipe (validated against α,β-CROWN on traffic_signs_recognition_2023, a GTSRB BNN):
  - forward = true `Sign`; backward = CLIPPED STE `grad[|x|>=eps]=0; grad/=eps`.
  - the benchmark's `Sign -> Add(+c) -> Sign` merge: STE the FIRST sign (clips on the real
    pre-activation x); the SECOND sign is functionally identity on {-1,+1} so it gets a
    PASS-THROUGH backward — STE-ing it would zero the gradient (its input is always ~±1 >= eps).
  - optimize the PRE-softmax logits (a trailing softmax saturates to ~one-hot -> 0 gradient).
  - loss = -worst_margin over the spec disjuncts (Adam + ExponentialLR), eps loose/tight
    alternation, a gentle "push Sign pre-acts off zero" penalty to escape the plateau.
"""
import os
import time

import numpy as np

from .surrogate_pgd import _load_onnx_model, _model_input_shapes, _ort_eval


def has_sign_ops(onnx_path):
    """True if the ONNX uses `Sign` (a binarized / BNN activation vibecheck can't bound)."""
    return any(n.op_type == 'Sign' for n in _load_onnx_model(onnx_path).graph.node)


def _disjunct_loss(y, disjuncts, torch):
    """Differentiable `-worst_margin` from a (pre-softmax) output vector `y`. worst_margin =
    min over disjuncts of (max over constraints of the SAFE margin); pushing it < 0 makes some
    disjunct's unsafe region reachable (a counterexample). Mirrors spec.py margin conventions:
    PairwiseConstraint(pred,comp) safe = y[pred]-y[comp]; Constraint(idx,'>=',v) safe = v-y[idx];
    Constraint(idx,'<=',v) safe = y[idx]-v."""
    conj_safes = []
    for conj in disjuncts:
        cmar = []
        for c in conj.constraints:
            if hasattr(c, 'pred'):                       # PairwiseConstraint
                cmar.append(y[c.pred] - y[c.comp])
            elif c.op == '>=':                           # Constraint  Y[idx] >= value
                cmar.append(c.value - y[c.index])
            else:                                        # Constraint  Y[idx] <= value
                cmar.append(y[c.index] - c.value)
        conj_safes.append(torch.stack(cmar).max())       # conjunct safe iff ANY constraint safe
    worst = torch.stack(conj_safes).min()
    return -worst                                         # maximize -> drive worst < 0 (sat)


def _worst_margin_np(y, disjuncts):
    """Numpy `worst_margin` for the ORT validation: <0 clear CE, in [0,atol] within-tol."""
    conj = []
    for c in disjuncts:
        cm = []
        for k in c.constraints:
            if hasattr(k, 'pred'):
                cm.append(float(y[k.pred] - y[k.comp]))
            elif k.op == '>=':
                cm.append(float(k.value - y[k.index]))
            else:
                cm.append(float(y[k.index] - k.value))
        conj.append(max(cm))
    return min(conj)


def sign_attack(onnx_path, vnnlib_path, settings, timeout, log=print):
    """Run the Sign-BNN STE-PGD attack. Returns (verdict, witness) where verdict in
    {'sat','timeout','unknown'} and witness is a list with the single input np.ndarray
    (None unless sat). The verdict is decided ONLY by the original model on ORT-CPU."""
    import torch
    from onnx2torch import convert
    from .io_util import ensure_decompressed
    from .vnnlib_loader import load_vnnlib

    t0 = time.time()
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    mshapes = _model_input_shapes(onnx_path)
    if len(mshapes) != 1:
        raise NotImplementedError(f'sign_attack expects a single-input model, got {len(mshapes)}')
    in_shape = mshapes[0]
    atol = float(getattr(settings, 'sat_validate_atol', 1e-4))   # INPUT-box tolerance only
    restarts = int(getattr(settings, 'sign_attack_restarts', 50))
    steps = int(getattr(settings, 'sign_attack_steps', 200))
    pen_coef = float(getattr(settings, 'sign_preact_penalty', 1.0))
    fracs = [float(f) for f in getattr(settings, 'sign_attack_clip_fracs', [0.05, 0.2, 0.1, 0.02])]
    per_disjunct = bool(getattr(settings, 'sign_per_disjunct', False))
    _bs = getattr(settings, 'pgd_seed', None)
    base_seed = int(_bs) if isinstance(_bs, (int, float)) else 0
    _want_gpu = (getattr(settings, 'device', 'gpu') == 'gpu')
    dev = 'cuda' if (_want_gpu and torch.cuda.is_available()) else 'cpu'

    class _SignSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, frac):
            # Per-layer ADAPTIVE clip: eps tracks THIS layer's pre-activation scale, so the
            # gradient survives whatever the magnitude is. Binarized-conv pre-acts span orders
            # of magnitude between layers; a fixed eps zeros a whole layer's gradient -> stall.
            eps = frac * float(x.detach().abs().median()) + 1e-12
            ctx.save_for_backward(x)
            ctx.eps = eps
            return torch.sign(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return (g / ctx.eps).masked_fill(x.abs() >= ctx.eps, 0.0), None

    class _SignPass(torch.autograd.Function):    # 2nd sign of a Sign->Add->Sign merge:
        @staticmethod                            # identity on {-1,+1}; pass the gradient
        def forward(ctx, x):
            return torch.sign(x)

        @staticmethod
        def backward(ctx, g):
            return g

    frac_box = [fracs[0]]                         # mutable; cycled over clip_fracs per restart
    model = convert(onnx_path).eval().to(dev)

    def _leaf(name):                             # last path component of an onnx2torch module name
        return name.rsplit('/', 1)[-1]

    n_first = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != 'OnnxFunction':
            continue
        if _leaf(name) == 'Sign':                # FIRST sign of a merge -> clipped STE on x
            mod.function = (lambda x, _f=frac_box: _SignSTE.apply(x, _f[0]))
            n_first += 1
        elif _leaf(name) == 'Sign_1':            # SECOND sign -> identity-grad pass-through
            mod.function = (lambda x: _SignPass.apply(x))
    # capture the LAST softmax's input (pre-softmax logits) + every Sign pre-activation
    presm, preacts = {}, []
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'Softmax':
            mod.register_forward_pre_hook(lambda m, inp: presm.__setitem__('z', inp[0]))
        elif type(mod).__name__ == 'OnnxFunction' and _leaf(name) in ('Sign', 'Sign_1'):
            mod.register_forward_pre_hook(lambda m, inp: preacts.append(inp[0]))
    log(f'[sign] loaded on {dev} in {time.time()-t0:.1f}s; {n_first} sign-merges, '
        f'restarts={restarts} steps={steps} disjuncts={len(spec.disjuncts)} '
        f'{"per-disjunct" if per_disjunct else "general"}')

    lo = torch.tensor(np.asarray(spec.x_lo, np.float32).reshape(in_shape), device=dev)
    hi = torch.tensor(np.asarray(spec.x_hi, np.float32).reshape(in_shape), device=dev)
    cen = (lo + hi) / 2
    half = (hi - lo) / 2
    n_val = [0]

    def logits(pts):
        out = model(pts)
        _ = out[0] if isinstance(out, (list, tuple)) else out
        return presm['z'].reshape(-1) if 'z' in presm else _.reshape(-1)

    def ort_consider(pts, tag):
        """Validate the witness on the ORIGINAL (true-Sign) model via ORT-CPU. CLEAR CE
        (worst_margin <= 0, boundary inclusive) -> return ('sat', witness). Under the
        2026 output-strict rule an output-near-boundary point (worst_margin > 0) is
        INCORRECT, not a usable CE; the within-tol stash is reachable only when
        out_atol>0 is configured (not scorer-accepted). Else None."""
        feed = [pts.detach().cpu().numpy().reshape(in_shape).astype(np.float32)]
        ff = feed[0].ravel()
        assert (ff >= spec.x_lo - atol).all() and (ff <= spec.x_hi + atol).all(), \
            'sign_attack produced an out-of-box witness'
        y = _ort_eval(onnx_path, feed)
        n_val[0] += 1
        m = _worst_margin_np(y, spec.disjuncts)
        if m <= 0.0:
            log(f'[sign] CLEAR SAT at {tag} (worst_margin={m:.3e})')
            return ('sat', feed)
        return None        # m > 0: no violation, no within-output-tol fallback (2026 rule)

    rng = torch.Generator(device='cpu')
    n_targets = len(spec.disjuncts) if per_disjunct else 1
    for r in range(restarts):
        if time.time() - t0 > timeout:
            break
        frac_box[0] = fracs[r % len(fracs)]
        alpha = float(half.max()) / 4.0
        if r == 0:
            delta = torch.zeros_like(cen)            # restart 0: the original (box center)
        else:
            rng.manual_seed(base_seed + r)
            # random box VERTEX: L-inf adversaries land on the boundary, and the CEs on this
            # benchmark are vertex-like (most pixels pinned to lo/hi); start there.
            delta = half * torch.sign(2 * torch.rand(cen.shape, generator=rng).to(dev) - 1)
        delta = delta.detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=alpha)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)
        tgt = spec.disjuncts[r % n_targets] if per_disjunct else None
        for it in range(steps):
            if time.time() - t0 > timeout:
                break
            preacts.clear()
            opt.zero_grad()
            x = torch.minimum(torch.maximum(cen + delta, lo), hi)
            y = logits(x)
            loss = (_disjunct_loss(y, [tgt], torch) if per_disjunct
                    else _disjunct_loss(y, spec.disjuncts, torch))
            pen = sum((1e-4 - p.abs()).clamp(min=0).sum() for p in preacts)
            (-(loss + pen_coef * pen)).backward()
            opt.step()
            sched.step()
            if it % 10 == 0 or it == steps - 1:
                res = ort_consider(x, f'restart{r}(frac={frac_box[0]}) step{it}')
                if res is not None:
                    return 'sat', res[1]
    # Incomplete (attack-only) mode: no strict CE found -> timeout (cannot prove unsat).
    log(f'[sign] no CE (t={time.time()-t0:.1f}s, restarts={restarts}, val={n_val[0]})')
    return 'timeout', None
