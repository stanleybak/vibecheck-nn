"""Batched-over-directions reverse-mode state build: process all D spec directions
in ONE backward (shared weights, per-direction lam/mu), with GPU-vectorized
nonzero extraction exploiting the static sparsity pattern (same cols across D).
Returns a list of D state dicts identical to build_state_reverse per direction."""
import numpy as np, torch
import torch.nn.functional as F
import scipy.sparse as sp


def _relax_np(lo, hi, alpha):
    lo = np.asarray(lo); hi = np.asarray(hi); alpha = np.asarray(alpha)
    active = lo >= 0; dead = hi <= 0; ust = (~active) & (~dead)
    lam = np.zeros_like(lo); lam[active] = 1.0; lam[ust] = alpha[ust]
    mu = np.zeros_like(lo)
    mu[ust] = np.maximum((1 - alpha[ust]) * hi[ust] / 2.0, -alpha[ust] * lo[ust] / 2.0)
    return lam, mu, ust


def build_states_reverse_batched(gg, xl, xh, bbr, alphas, dev, dt):
    """alphas: list of D alpha dicts {L: tensor/array}. Returns list of D states."""
    def _np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    def _t(x): return x.detach() if torch.is_tensor(x) else torch.as_tensor(x, device=dev, dtype=dt)
    D = len(alphas)
    xl_n = _np(xl); xh_n = _np(xh)
    ops = gg['ops']; in_name = gg['input_name']
    relu_ops = {op['layer_idx']: op for op in ops if op['type'] == 'relu' and 'layer_idx' in op}
    Ls = sorted(relu_ops.keys())
    radii = torch.as_tensor((xh_n - xl_n) / 2.0, device=dev, dtype=dt)
    n_input = int((xh_n > xl_n).sum())
    # per-direction lam/mu stacked to (D, width); unstable set + e_new_col shared (bbr-fixed)
    lamD = {}; muD = {}; ust_L = {}; e_new_col = {}; col = n_input
    for L in Ls:
        lo, hi = bbr[L]
        lams = []; mus = []
        for d in range(D):
            a = _np(alphas[d][L]) if (L in alphas[d]) else np.zeros_like(np.asarray(lo))
            lm, m, ust = _relax_np(lo, hi, a)
            lams.append(lm); mus.append(m)
        lamD[L] = torch.as_tensor(np.stack(lams), device=dev, dtype=dt)   # (D, width)
        muD[L] = torch.as_tensor(np.stack(mus), device=dev, dtype=dt)
        _, _, ust0 = _relax_np(lo, hi, _np(alphas[0][L]) if L in alphas[0] else np.zeros_like(np.asarray(lo)))
        ust_L[L] = np.where(ust0)[0]   # unstable set: bbr-determined (alpha-independent)
        for j in ust_L[L]:
            e_new_col[(L, int(j))] = col; col += 1
    n_gens = col
    # center point-forward per direction (cheap) -> pre-relu centers (D, width)
    centers = []  # centers[d] = {tensor_name: vec}
    pre_center = {L: None for L in Ls}
    for d in range(D):
        center = {in_name: (xl_n + xh_n) / 2.0}; shapes = {}
        for op in ops:
            t = op['type']; nm = op['name']
            if t == 'conv':
                x = center[op['inputs'][0]].reshape(op['in_shape'])
                k = _np(op['kernel']); b = op['bias']; b = _np(b) if b is not None else None
                y = F.conv2d(torch.tensor(x).unsqueeze(0), torch.tensor(k),
                             bias=torch.tensor(b) if b is not None else None,
                             stride=op['stride'], padding=op['padding'])[0].numpy()
                shapes[nm] = y.shape; center[nm] = y.flatten()
            elif t == 'fc':
                W = _np(op['W']); b = op['bias']; b = _np(b) if b is not None else 0.0
                center[nm] = W @ center[op['inputs'][0]] + b
            elif t == 'relu':
                x = center[op['inputs'][0]]
                if 'layer_idx' in op:
                    L = op['layer_idx']
                    if pre_center[L] is None: pre_center[L] = np.zeros((D, x.size))
                    pre_center[L][d] = x
                    center[nm] = _np(lamD[L][d]) * x + _np(muD[L][d])
                else:
                    center[nm] = np.maximum(x, 0)
            elif t == 'reshape':
                center[nm] = center[op['inputs'][0]]
            elif t == 'add':
                if op.get('is_merge'):
                    center[nm] = center[op['inputs'][0]] + center[op['inputs'][1]]
                else:
                    b = op.get('bias'); center[nm] = center[op['inputs'][0]] + (np.asarray(b).flatten() if b is not None else 0.0)
            elif t == 'sub':
                b = op.get('bias'); center[nm] = center[op['inputs'][0]] - (np.asarray(b).flatten() if b is not None else 0.0)
            else:
                raise NotImplementedError(f'center fwd: op {t!r}')
        centers.append((center, shapes))
    shapes = centers[0][1]
    out_name = gg['ops'][-1]['name']; n_out = centers[0][0][out_name].size

    def _backward(seed_tensor, seed_idx, L_self):
        ns = len(seed_idx)
        rowG = torch.zeros(D, ns, n_gens, device=dev, dtype=dt)
        seed = torch.zeros(D, ns, centers[0][0][seed_tensor].size, device=dev, dtype=dt)
        seed[:, torch.arange(ns, device=dev), torch.as_tensor(seed_idx, device=dev, dtype=torch.long)] = 1.0
        sens = {seed_tensor: seed}
        for op in reversed(ops):
            nm = op['name']
            if nm not in sens: continue
            s = sens[nm]; t = op['type']
            if t == 'conv':
                out_sh = shapes[nm]; in_sh = tuple(op['in_shape'])
                ker = _t(op['kernel']); kH, kW = ker.shape[-2], ker.shape[-1]
                st = op['stride']; pd = op['padding']
                sH, sW = (st, st) if isinstance(st, int) else (st[0], st[1])
                pH, pW = (pd, pd) if isinstance(pd, int) else (pd[0], pd[1])
                opH = in_sh[1] - ((out_sh[1]-1)*sH - 2*pH + kH); opW = in_sh[2] - ((out_sh[2]-1)*sW - 2*pW + kW)
                dx = F.conv_transpose2d(s.reshape(D*ns, *out_sh), ker, stride=st, padding=pd, output_padding=(opH, opW))
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + dx.reshape(D, ns, -1)
            elif t == 'fc':
                W = _t(op['W']); inp = op['inputs'][0]
                sens[inp] = sens.get(inp, 0) + (s.reshape(D*ns, -1) @ W).reshape(D, ns, -1)
            elif t == 'relu':
                if 'layer_idx' in op:
                    Lp = op['layer_idx']
                    if Lp != L_self:
                        uj = torch.as_tensor(ust_L[Lp], device=dev, dtype=torch.long)
                        cols = torch.as_tensor([e_new_col[(Lp, int(j))] for j in ust_L[Lp]], device=dev, dtype=torch.long)
                        rowG[:, :, cols] += s[:, :, uj] * muD[Lp][:, uj][:, None, :]
                        s = s * lamD[Lp][:, None, :]
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + s
                else:
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + s
            elif t == 'reshape':
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + s
            elif t == 'add':
                if op.get('is_merge'):
                    for inp in op['inputs'][:2]: sens[inp] = sens.get(inp, 0) + s
                else:
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + s
            elif t == 'sub':
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + s
            else:
                raise NotImplementedError(f'reverse: op {t!r}')
        s_in = sens.get(in_name)
        if s_in is not None:
            rowG[:, :, :n_input] += s_in[:, :, :n_input] * radii[None, None, :n_input]
        return rowG   # (D, ns, n_gens)

    def _extract(rowG, seed_idx):
        # static pattern across D: cols nonzero in ANY direction
        pat = (rowG.abs() > 0).any(0)              # (ns, n_gens)
        nz = pat.nonzero(as_tuple=False)           # (K,2): (neuron, col) sorted by neuron
        if nz.numel() == 0:
            return [[(np.empty(0,np.int64), np.empty(0)) for _ in seed_idx] for _ in range(D)]
        valsD = rowG[:, nz[:, 0], nz[:, 1]].cpu().numpy()   # (D, K)
        nrow = nz[:, 0].cpu().numpy(); ncol = nz[:, 1].cpu().numpy()
        # segment boundaries per neuron
        counts = np.bincount(nrow, minlength=len(seed_idx))
        bnd = np.concatenate([[0], np.cumsum(counts)])
        out = [[None]*len(seed_idx) for _ in range(D)]
        for i in range(len(seed_idx)):
            sl = slice(bnd[i], bnd[i+1]); ci = ncol[sl]
            for d in range(D):
                out[d][i] = (ci.astype(np.int64), valsD[d, sl].astype(np.float64))
        return out

    # formulation MUST be 'alpha_zono' (same contract as build_state_reverse) —
    # the per-neuron data below is the α-zono parametrization, so the gen-LP/MILP
    # builder must dispatch to _build_alpha_zono_lp, NOT the generic 'sparse'
    # direct-ReLU builder. 'sparse' here is the same unsound-MILP-fallback bug
    # fixed in reverse_g.py; see tests/test_reverse_g.py.
    states = [dict(n_gens=n_gens, n_input=n_input, unstable_list=[],
                   input_name=in_name, output_op_name=out_name,
                   formulation='alpha_zono',
                   stable_list=[], x_lo=xl_n, x_hi=xh_n, sigmoid_tanh_layer_ids=set())
              for _ in range(D)]
    for L in Ls:
        u = ust_L[L]
        if len(u) == 0: continue
        rowG = _backward(relu_ops[L]['inputs'][0], u, L)
        ext = _extract(rowG, u)
        lamn = _np(lamD[L]); mun = _np(muD[L])
        for d in range(D):
            for i, j in enumerate(u):
                ri, rv = ext[d][i]
                states[d]['unstable_list'].append(dict(
                    layer_idx=L, neuron_idx=int(j), c_in=float(pre_center[L][d, j]),
                    lo=float(np.asarray(bbr[L][0])[j]), hi=float(np.asarray(bbr[L][1])[j]),
                    lam=float(lamn[d, j]), mu=float(mun[d, j]),
                    e_new_col=e_new_col[(L, int(j))], form='alpha_zono',
                    row_indices=ri, row_values=rv))
        del rowG
    # output objective per direction
    rowG = _backward(out_name, np.arange(n_out), None)
    ext = _extract(rowG, np.arange(n_out))
    for d in range(D):
        og = np.zeros((n_out, n_gens))
        for i in range(n_out):
            ri, rv = ext[d][i]; og[i, ri] = rv
        states[d]['obj_c_out'] = centers[d][0][out_name].astype(np.float64)
        states[d]['obj_G_out_csr'] = sp.csr_matrix(og.astype(np.float64))
    return states


def build_states_reverse_batched_safe(gg, xl, xh, bbr, alphas, dev, dt,
                                      chunk=None, _bench=None):
    """OOM-safe wrapper: process the D directions in chunks; on CUDA OOM halve the
    chunk and retry (down to 1 = sequential). Remembers the smaller safe chunk for
    subsequent chunks. Returns the D states in order. `_bench` (optional dict) is
    filled with {'final_chunk': int, 'n_chunks': int} for tests/measurement."""
    D = len(alphas)
    if chunk is None:
        chunk = D
    out = [None] * D
    i = 0; n_chunks = 0; final_chunk = chunk
    while i < D:
        step = min(chunk, D - i)
        while True:
            try:
                res = build_states_reverse_batched(
                    gg, xl, xh, bbr, alphas[i:i + step], dev, dt)
                for j, s in enumerate(res):
                    out[i + j] = s
                n_chunks += 1; final_chunk = step
                break
            except torch.cuda.OutOfMemoryError:
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                if step == 1:
                    raise            # can't shrink further — genuinely OOM
                step = max(1, step // 2)
                chunk = step         # remember the smaller safe size
        i += step
    if _bench is not None:
        _bench['final_chunk'] = final_chunk; _bench['n_chunks'] = n_chunks
    return out
