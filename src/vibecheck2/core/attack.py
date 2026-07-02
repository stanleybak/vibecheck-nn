"""The attack engine: one PGD implementation for every falsification need
(design 2.2 / survey F1-F10). Plug points, not copies:

  backend:  the net's own traced forward (default; onnx2torch/STE surrogates
            arrive with their handlers)
  init:     'uniform' | 'center' | 'vertex' | caller-provided seed batch
  accept:   every candidate must pass the v1 ORT-CPU chokepoint before it is
            a counterexample (strict output violation, in-box within 1e-4);
            the search margin is only a ranking signal
  project:  clamp to the global input box; per-disjunct subboxes are handled
            by validation (a candidate outside its disjunct's box fails
            check_witness) until the mega-disjunct handler lands

Recipe follows v1 pgd.pgd_attack_general: all restarts as one batch, Adam
on the inputs, exponential step decay, hinged per-disjunct margin loss,
plateau give-up, wall budget.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import forward


def spec_margins(spec, n_out, device, dtype):
    """Per-disjunct margin closure: margins(y) (B, D) where a value < 0
    means y satisfies EVERY constraint of that disjunct (a counterexample,
    up to input-box membership). Built once from as_linear_queries."""
    rows = spec.as_linear_queries(n_out)
    W = torch.tensor(np.stack([w for _, w, _ in rows]), device=device,
                     dtype=dtype)
    b = torch.tensor([bias for _, _, bias in rows], device=device,
                     dtype=dtype)
    di = torch.tensor([d for d, _, _ in rows], device=device)
    D = len(spec.disjuncts)

    def margins(y):
        m = y @ W.T + b                     # (B, q); row > 0 = row violated
        out = torch.full((y.shape[0], D), -torch.inf, device=y.device,
                         dtype=y.dtype)
        out.scatter_reduce_(1, di.unsqueeze(0).expand(y.shape[0], -1), m,
                            reduce='amax', include_self=True)
        return out

    return margins


def _restart_boxes(lo, hi, restarts, device, dtype):
    """Per-restart boxes (R, n): a single box tiles; a (K, n) batch of boxes
    (BaB subdomains) is assigned round-robin so every box gets restarts."""
    lo = lo.to(device, dtype)
    hi = hi.to(device, dtype)
    if lo.dim() == 1:
        return (lo.unsqueeze(0).expand(restarts, -1),
                hi.unsqueeze(0).expand(restarts, -1))
    idx = torch.arange(restarts, device=device) % lo.shape[0]
    return lo[idx], hi[idx]


def _init_points(lo_r, hi_r, mode, seeds, generator):
    """Start points inside per-restart boxes (R, n)."""
    R, n = lo_r.shape
    u = torch.rand(R, n, device=lo_r.device, dtype=lo_r.dtype,
                   generator=generator)
    if mode == 'vertex':
        u = (u > 0.5).to(lo_r.dtype)
    x = lo_r + u * (hi_r - lo_r)
    x[0] = (lo_r[0] + hi_r[0]) / 2                     # one center start
    if seeds is not None and len(seeds):
        s = torch.as_tensor(np.asarray(seeds), device=lo_r.device,
                            dtype=lo_r.dtype).reshape(-1, n)[:R]
        x[-s.shape[0]:] = torch.maximum(torch.minimum(s, hi_r[-s.shape[0]:]),
                                        lo_r[-s.shape[0]:])
    return x


def _osi_diversify(net, x, lo, hi, generator, steps=30):
    """Output-space diversification (abcrown's diversed PGD / v1 OSI init):
    push each restart toward a random output direction with sign-gradient
    steps, spreading the starting points across the reachable output set
    before the real attack. Sound: purely an init heuristic."""
    y0 = forward.point(net, x[:1])
    d = torch.randn(x.shape[0], y0.shape[1], device=x.device, dtype=x.dtype,
                    generator=generator)
    step = 0.1 * (hi - lo)
    x = x.detach().clone().requires_grad_(True)
    for _ in range(steps):
        obj = (forward.point(net, x) * d).sum()
        g, = torch.autograd.grad(obj, x)
        with torch.no_grad():
            x += step * g.sign()
            x.clamp_(lo, hi)
    return x.detach()


def pgd(net, spec, lo=None, hi=None, restarts=64, iters=100, seed=0,
        device='cpu', time_budget=10.0, init='uniform', seeds=None,
        lr_frac=0.25, lr_decay=0.99, plateau=40, log=lambda m: None):
    """Search for a spec counterexample. Returns (witness_np | None, info).

    The returned witness is ONLY a candidate: callers must gate it through
    validate() (the ORT chokepoint) before any 'sat' verdict.
    """
    dev = torch.device(device)
    dt = torch.float32
    if lo is None:
        lo = torch.tensor(spec.x_lo, dtype=dt)
    if hi is None:
        hi = torch.tensor(spec.x_hi, dtype=dt)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    margins = spec_margins(spec, net.n_out, dev, dt)

    lo, hi = _restart_boxes(lo, hi, restarts, dev, dt)   # (R, n) each
    x = _init_points(lo, hi, init, seeds, gen)
    if init == 'osi':
        x = _osi_diversify(net, x, lo, hi, gen)
    x = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=float(lr_frac * (hi - lo).max()))
    # NB: lr is global; per-restart boxes of very different width could
    # warrant per-restart lr (not needed by current categories)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    t0 = time.time()
    best_m = torch.full((x.shape[0],), torch.inf, device=dev)
    best_x = x.detach().clone()
    # per-restart disjunct targeting (v1 pgd_per_restart_disjunct): restart r
    # descends disjunct r mod D, so every disjunct gets dedicated restarts
    # and the gradient signal is dense instead of min-of-max sparse
    D = len(spec.disjuncts)
    target = torch.arange(x.shape[0], device=dev) % max(D, 1)
    since_improve = 0
    it = 0
    for it in range(iters):
        y = forward.point(net, x)
        m = margins(y)                                # (R, D)
        overall = m.min(dim=1).values                 # (R,)
        improved = overall < best_m
        best_x[improved] = x.detach()[improved]
        best_m = torch.minimum(best_m, overall)
        if (best_m <= 0).any():        # VNNLIB constraints are NON-strict:
            break                      # equality satisfies (sat_relu Y_1<=0)
        loss = m.gather(1, target.unsqueeze(1)).clamp(min=-0.05).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(lo, hi)
        since_improve = 0 if bool(improved.any()) else since_improve + 1
        if since_improve > plateau or time.time() - t0 > time_budget:
            break
    order = torch.argsort(best_m)
    info = {'best_margin': float(best_m.min()), 'iters': it + 1,
            'time': time.time() - t0}
    log(f'[vc2/pgd] best_margin={info["best_margin"]:+.3e} '
        f'iters={info["iters"]} t={info["time"]:.2f}s')
    if best_m[order[0]] <= 0:
        return best_x[order[0]].detach().cpu().numpy().astype(np.float64), info
    return None, info


def validate(onnx_path, spec, witness):
    """The one acceptance gate: v1's ORT-CPU chokepoint (input box within
    1e-4, output STRICTLY violating). Returns (ok, info) where info may
    carry a float32-safe clamped witness ('witness_inbox') and the ORT
    output ('out')."""
    from vibecheck.verify_graph import _validate_sat_witness
    return _validate_sat_witness(onnx_path, spec, witness,
                                 atol=1e-4, out_atol=0.0)
