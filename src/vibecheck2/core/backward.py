"""The backward propagator: CROWN adjoint walk over the DAG (design 2.2).

One implementation. Computes sound LOWER bounds on q linear output rows
W (q, n_out) over B input boxes, walking the flat DAG in reverse topological
order. Per edge the state is an adjoint tensor A (B, q, n_edge) plus an
accumulated offset d (B, q); a fork's consumers sum their adjoints (exact),
a LinMap applies lin_t, a nonlinearity applies its RelaxLib planes split by
adjoint sign, and the input concretizes A over the box.

alpha: optimizable lower-plane slopes for unstable relus, one tensor per
relu op of shape (B, q, n) in [0,1]; when absent the sound adaptive default
from RelaxLib is used. The whole walk is differentiable w.r.t. alpha, so
`alpha_crown` just Adam-ascends the bound. beta (split constraints) and
gamma (output constraints) extend this same walk in later milestones.

Intermediate (pre-activation) bounds come from the forward propagator and
are treated as constants (fixed-intermediate mode).
"""
from __future__ import annotations

import torch

from . import forward as fwd
from .relax import REL


def _neg_part(A):
    return A.clamp(max=0)


def _pos_part(A):
    return A.clamp(min=0)


def _zono_cost_bytes(net, B):
    """Projected peak cost of a dense forward zonotope: max over edges of
    B * n_edge * (n_in + #relu elements so far) * 4 bytes. Cheap shape-only
    estimate for the generator-lifecycle decision (design 3.3); the full
    reduce/drop-and-continue lifecycle replaces this in M5."""
    g = net.n_in
    worst = 0
    for name in net.order:
        op = net.ops[name]
        worst = max(worst, op.n * g)
        if op.kind == 'nonlin' and op.fn == 'relu':
            g += op.n                          # worst case: all unstable
    return worst * B * 4


def intermediates(net, lo, hi):
    """Pre-activation bounds for every nonlinear edge: forward zonotope when
    the projected dense cost fits the memory budget, else interval (the
    CROWN-IBP regime for big conv nets until patches/lifecycle land in M5).
    Also falls back to interval when zono lacks an op's relaxation."""
    from . import memory
    B = lo.shape[0]
    if _zono_cost_bytes(net, B) < memory.free_bytes(lo.device) * memory.SAFETY:
        try:
            _lo, _hi, state = fwd.zono(net, lo, hi, return_state=True)
            return {name: state[net.ops[name].inputs[0]].bounds()
                    for name in net.order if net.ops[name].kind == 'nonlin'}
        except NotImplementedError:
            pass                # an op without a zono relaxation yet
    state = fwd.interval(net, lo, hi, return_state=True)
    return {name: state[net.ops[name].inputs[0]]
            for name in net.order if net.ops[name].kind == 'nonlin'}


def crown(net, lo, hi, W, inter=None, alpha=None, start=None,
          return_input_adjoint=False):
    """Lower bounds on W @ y_edge for x in [lo, hi], where y_edge is the
    value of edge `start` (default: the network output). Bounding an
    INTERMEDIATE edge is the same walk seeded there; ops after it never
    accumulate an adjoint and are skipped naturally.

    lo, hi: (B, n_in); W: (q, n_edge) or (B, q, n_edge).
    inter: {nonlin_op_name: (lo, hi)} pre-activation bounds ((B, n) each).
    alpha: {relu_op_name: (B, q, n) in [0, 1]} optimizable lower slopes.
    Returns lb (B, q).
    """
    B = lo.shape[0]
    dev, dt = lo.device, lo.dtype
    if inter is None:
        inter = intermediates(net, lo, hi)
    if W.dim() == 2:
        W = W.unsqueeze(0).expand(B, -1, -1)
    q = W.shape[1]

    A = {start or net.output_name: W.to(device=dev, dtype=dt)}
    d = torch.zeros(B, q, device=dev, dtype=dt)

    def take(name):
        """Pop the accumulated adjoint for edge `name` (zeros if unused)."""
        return A.pop(name)

    def put(name, val):
        A[name] = A[name] + val if name in A else val

    for name in reversed(net.order):
        if name not in A:
            continue                     # edge does not influence the queries
        op = net.ops[name]
        Ao = take(name)
        if op.kind == 'linmap':
            d = d + Ao @ op.lm.bias_vec(Ao)
            Ain = op.lm.lin_t(Ao.reshape(B * q, -1)).reshape(B, q, -1)
            put(op.inputs[0], Ain)
        elif op.kind == 'add':
            put(op.inputs[0], Ao)
            put(op.inputs[1], Ao)
        elif op.kind == 'concat':
            base = torch.as_tensor(op.params['base'], device=dev, dtype=dt)
            d = d + Ao @ base
            for src, pos in zip(op.inputs, op.params['positions']):
                put(src, Ao[:, :, torch.as_tensor(pos, device=dev)])
        elif op.kind == 'nonlin':
            l, h = inter[name]
            rel = REL[op.fn]
            if not hasattr(rel, 'planes'):
                raise NotImplementedError(
                    f'crown: no planes for nonlinearity {op.fn!r} yet')
            al, bl, au, bu = rel.planes(l, h)
            if alpha and name in alpha:
                # optimizable lower slope on unstable neurons only
                unstable = ((l < 0) & (h > 0)).unsqueeze(1)
                al = torch.where(unstable, alpha[name].clamp(0.0, 1.0),
                                 al.unsqueeze(1))
            if al.dim() == 2:
                al = al.unsqueeze(1)
            Ap, An = _pos_part(Ao), _neg_part(Ao)
            # lower bound: positive adjoint takes the lower plane,
            # negative adjoint the upper plane
            Ain = Ap * al + An * au.unsqueeze(1)
            d = d + (Ap * bl.unsqueeze(1) + An * bu.unsqueeze(1)).sum(dim=2)
            put(op.inputs[0], Ain)
        elif op.kind in ('mul', 'maxpool'):
            raise NotImplementedError(
                f'crown: {op.kind} relaxation arrives with its category')
        else:
            raise NotImplementedError(f'crown: op kind {op.kind!r}')

    Ain = A.pop(net.input_name)
    assert not A, f'unconsumed adjoints: {list(A)}'
    c = (hi + lo) / 2
    r = (hi - lo) / 2
    lb = d + torch.einsum('bqn,bn->bq', Ain, c) \
           - torch.einsum('bqn,bn->bq', Ain.abs(), r)
    if return_input_adjoint:
        return lb, Ain
    return lb


def intermediates_crown(net, lo, hi, base_inter=None):
    """Pre-activation bounds per nonlin edge via per-edge backward CROWN
    (chunked identity queries, both signs in one pass). Strictly tighter
    than interval; the regime for conv nets whose dense zonotope does not
    fit (until patches land in M5). Earlier edges' CROWN bounds feed later
    edges' relaxations (topo order)."""
    from . import memory
    B = lo.shape[0]
    dev, dt = lo.device, lo.dtype
    # interval bounds seed every edge; CROWN refines ONLY the neurons whose
    # interval sign is ambiguous (planes are already exact on stable ones),
    # which is a small fraction on certified/robust nets
    if base_inter is None:
        state = fwd.interval(net, lo, hi, return_state=True)
        base_inter = {name: state[net.ops[name].inputs[0]]
                      for name in net.order
                      if net.ops[name].kind == 'nonlin'}
    inter = dict(base_inter)
    widest = max(net.ops[o].n for o in net.order)
    for name in net.order:
        op = net.ops[name]
        if op.kind != 'nonlin':
            continue
        e = op.inputs[0]
        n = net.ops[e].n
        l0, h0 = inter[name]
        idx = torch.nonzero(((l0 < 0) & (h0 > 0)).any(dim=0),
                            as_tuple=False).flatten()
        if not idx.numel():
            continue
        # identity blocks per chunk (never a full n x n eye); both signs in
        # one walk so lo and hi share it. A deep walk holds several live
        # adjoints plus conv temporaries, hence the generous per-row factor;
        # chunked_indices halves on an OOM anyway (the one sanctioned catch).
        per_row = B * widest * 4 * 12
        lb = l0.clone()
        ub = h0.clone()

        def refine(sel, _e=e, _n=n, _lb=lb, _ub=ub):
            m = sel.numel()
            Wc = torch.zeros(2 * m, _n, device=dev, dtype=dt)
            ar = torch.arange(m, device=dev)
            Wc[ar, sel] = 1.0
            Wc[m + ar, sel] = -1.0
            out = crown(net, lo, hi, Wc.unsqueeze(0).expand(B, -1, -1),
                        inter, start=_e)
            _lb[:, sel] = torch.maximum(_lb[:, sel], out[:, :m])
            _ub[:, sel] = torch.minimum(_ub[:, sel], -out[:, m:])

        memory.chunked_indices(refine, idx, per_row)
        inter[name] = (lb, ub)
    return inter


def alpha_crown(net, lo, hi, W, inter=None, iters=20, lr=0.25,
                thresholds=None):
    """Adam-optimized alpha-CROWN lower bounds (fixed intermediates).

    Maximizes each query's lb independently (sum of hinged bounds: a query
    already past its threshold contributes nothing, focusing the optimizer
    on the still-open ones). Returns the elementwise-best lb seen (sound:
    every iterate is a valid bound).
    """
    B = lo.shape[0]
    if inter is None:
        inter = intermediates(net, lo, hi)
    q = W.shape[-2]
    alpha = {}
    for name in net.order:
        op = net.ops[name]
        if op.kind == 'nonlin' and op.fn == 'relu':
            l, h = inter[name]
            al0 = REL['relu'].planes(l, h)[0]           # adaptive default
            alpha[name] = al0.detach().clone().unsqueeze(1) \
                .expand(B, q, l.shape[1]).contiguous().requires_grad_(True)
    if not alpha:
        return crown(net, lo, hi, W, inter)
    opt = torch.optim.Adam(list(alpha.values()), lr=lr)
    best = None
    thr = (torch.zeros(q, device=lo.device, dtype=lo.dtype)
           if thresholds is None else thresholds)
    for _ in range(max(1, iters)):
        lb = crown(net, lo, hi, W, inter, alpha)
        best = lb.detach() if best is None \
            else torch.maximum(best, lb.detach())
        loss = -(torch.minimum(lb, thr.unsqueeze(0) + 1.0)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for t in alpha.values():
                t.clamp_(0.0, 1.0)
    lb = crown(net, lo, hi, W, inter, alpha)
    return torch.maximum(best, lb.detach())
