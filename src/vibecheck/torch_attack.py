"""Generic onnx2torch PGD attack mode — for nets vibecheck can't bound soundly/cheaply but
that ARE differentiable through onnx2torch (e.g. the collins_aerospace YOLOv5-nano: 60 Conv +
LeakyRelu + Sigmoid, a 640x640 robustness spec with a tiny perturbed patch). Incomplete /
attack-only (never returns unsat): PGD on the real ops finds a counterexample, then the
witness is VALIDATED on the ORIGINAL model via CPU onnxruntime (the scoring engine), so a
torch/ORT mismatch can never yield a false sat.

Shares the spec-margin + ORT helpers with the Sign-BNN attack; the only difference is there is
no `Sign` surrogate to patch — autograd flows through the genuine ops. PGD runs only on the
PERTURBED input dims (where hi>lo); the rest stay pinned at their fixed value.
"""
import time

import numpy as np

from .surrogate_pgd import _model_input_shapes, _ort_eval
from .sign_attack import _disjunct_loss, _worst_margin_np


def torch_attack(onnx_path, vnnlib_path, settings, timeout, log=print):
    """Run the generic onnx2torch STE-free PGD attack. Returns (verdict, witness) with verdict
    in {'sat','timeout','unknown'} and witness a list holding the single input np.ndarray (None
    unless sat). The verdict is decided ONLY by the original model on ORT-CPU."""
    import torch
    from onnx2torch import convert
    from .io_util import ensure_decompressed
    from .vnnlib_loader import load_vnnlib

    t0 = time.time()
    spec = load_vnnlib(ensure_decompressed(vnnlib_path))
    mshapes = _model_input_shapes(onnx_path)
    if len(mshapes) != 1:
        raise NotImplementedError(f'torch_attack expects a single-input model, got {len(mshapes)}')
    in_shape = mshapes[0]
    atol = float(getattr(settings, 'sat_validate_atol', 1e-4))
    restarts = int(getattr(settings, 'torch_attack_restarts', 20))
    steps = int(getattr(settings, 'torch_attack_steps', 200))
    keep_searching = bool(getattr(settings, 'keep_searching_within_tol', True))
    _bs = getattr(settings, 'pgd_seed', None)
    base_seed = int(_bs) if isinstance(_bs, (int, float)) else 0
    _want_gpu = (getattr(settings, 'device', 'gpu') == 'gpu')
    dev = 'cuda' if (_want_gpu and torch.cuda.is_available()) else 'cpu'

    model = convert(onnx_path).eval().to(dev)
    lo = torch.tensor(np.asarray(spec.x_lo, np.float32).reshape(in_shape), device=dev)
    hi = torch.tensor(np.asarray(spec.x_hi, np.float32).reshape(in_shape), device=dev)
    cen = (lo + hi) / 2
    half = (hi - lo) / 2
    free = int((half > 0).sum())
    log(f'[torch] loaded on {dev} in {time.time()-t0:.1f}s; in={list(in_shape)} free={free} '
        f'disjuncts={len(spec.disjuncts)} restarts={restarts} steps={steps}')

    within_tol = [None]
    n_val = [0]

    def flat_out(x):
        out = model(x)
        o = out[0] if isinstance(out, (list, tuple)) else out
        return o.reshape(-1)

    def ort_consider(pts, tag):
        feed = [pts.detach().cpu().numpy().reshape(in_shape).astype(np.float32)]
        ff = feed[0].ravel()
        assert (ff >= spec.x_lo - atol).all() and (ff <= spec.x_hi + atol).all(), \
            'torch_attack produced an out-of-box witness'
        y = _ort_eval(onnx_path, feed)
        n_val[0] += 1
        m = _worst_margin_np(y, spec.disjuncts)
        if m < 0.0:
            log(f'[torch] CLEAR SAT at {tag} (worst_margin={m:.3e})')
            return ('sat', feed)
        if m <= atol and within_tol[0] is None:
            within_tol[0] = feed
            log(f'[torch] within-tol CE at {tag} (worst_margin={m:.3e}, atol={atol:g})'
                + ('' if keep_searching else ' — accepting (keep_searching off)'))
            if not keep_searching:
                return ('sat', feed)
        return None

    rng = torch.Generator(device='cpu')
    amax = float(half.max())
    for r in range(restarts):
        if time.time() - t0 > timeout:
            break
        if r == 0:
            delta = torch.zeros_like(cen)            # restart 0: the original (box center)
        else:
            rng.manual_seed(base_seed + r)
            # random box VERTEX on the perturbed dims (L-inf adversaries land on the boundary)
            delta = half * torch.sign(2 * torch.rand(cen.shape, generator=rng).to(dev) - 1)
        delta = delta.detach().requires_grad_(True)
        alpha = amax / 4 if amax > 0 else 1e-3
        opt = torch.optim.Adam([delta], lr=alpha)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)
        for it in range(steps):
            if time.time() - t0 > timeout:
                break
            opt.zero_grad()
            x = torch.minimum(torch.maximum(cen + delta, lo), hi)
            loss = _disjunct_loss(flat_out(x), spec.disjuncts, torch)
            (-loss).backward()
            opt.step()
            sched.step()
            if it % 10 == 0 or it == steps - 1:
                res = ort_consider(x, f'restart{r} step{it}')
                if res is not None:
                    return 'sat', res[1]
    timed_out = (time.time() - t0) > timeout
    if within_tol[0] is not None:
        log(f'[torch] no clear CE — emitting within-tol CE (t={time.time()-t0:.1f}s, val={n_val[0]})')
        return 'sat', within_tol[0]
    log(f'[torch] no CE (t={time.time()-t0:.1f}s, restarts={restarts}, val={n_val[0]})')
    return ('timeout' if timed_out else 'unknown'), None
