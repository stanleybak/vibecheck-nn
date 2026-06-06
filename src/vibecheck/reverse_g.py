"""Reverse-mode alpha-zono state build over the gg DAG, GPU-native hot path.
Reproduces forward_zono_dir_adaptive+state_from_alpha_zono backward from the
unstable neurons (cheap) instead of forward over all generators (expensive).
Validated bit-for-bit (machine eps) vs the forward on conv/relu/add(skip)/fc."""
import numpy as np, torch
import torch.nn.functional as F


def _relax_np(lo, hi, alpha):
    lo = np.asarray(lo); hi = np.asarray(hi); alpha = np.asarray(alpha)
    active = lo >= 0; dead = hi <= 0; ust = (~active) & (~dead)
    lam = np.zeros_like(lo); lam[active] = 1.0; lam[ust] = alpha[ust]
    mu = np.zeros_like(lo)
    mu[ust] = np.maximum((1 - alpha[ust]) * hi[ust] / 2.0, -alpha[ust] * lo[ust] / 2.0)
    return lam, mu, ust


def build_state_reverse(gg, xl, xh, bbr, alpha, dev, dt):
    def _np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    def _t(x): return x.detach() if torch.is_tensor(x) else torch.as_tensor(x, device=dev, dtype=dt)
    xl_n = _np(xl); xh_n = _np(xh)
    ops = gg['ops']; in_name = gg['input_name']
    relu_ops = {op['layer_idx']: op for op in ops if op['type'] == 'relu' and 'layer_idx' in op}
    Ls = sorted(relu_ops.keys())
    radii_n = (xh_n - xl_n) / 2.0
    radii = torch.as_tensor(radii_n, device=dev, dtype=dt)
    n_input = int((xh_n > xl_n).sum())  # all input dims are generators here (dense box)
    # --- lam/mu/unstable + e_new_col (forward order) ---
    lam_L = {}; mu_L = {}; ust_L = {}; e_new_col = {}; col = n_input
    for L in Ls:
        lo, hi = bbr[L]
        a = _np(alpha[L]) if (L in alpha) else np.zeros_like(np.asarray(lo))
        lam, mu, ust = _relax_np(lo, hi, a)
        lam_L[L] = torch.as_tensor(lam, device=dev, dtype=dt)
        mu_L[L] = torch.as_tensor(mu, device=dev, dtype=dt)
        ust_L[L] = np.where(ust)[0]
        for j in ust_L[L]:
            e_new_col[(L, int(j))] = col; col += 1
    n_gens = col
    # --- center point-forward (numpy, cheap) -> pre-relu centers ---
    center = {in_name: (xl_n + xh_n) / 2.0}; shapes = {}; pre_center = {}
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
                L = op['layer_idx']; pre_center[L] = x.copy()
                center[nm] = _np(lam_L[L]) * x + _np(mu_L[L])
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
    # --- reverse backward helper: rows (n_seed x n_gens) for a seed set ---
    import scipy.sparse as _sp
    def _backward(seed_tensor, seed_idx, L_self):
        ns = len(seed_idx)
        rowG = torch.zeros(ns, n_gens, device=dev, dtype=dt)
        sens = {seed_tensor: torch.zeros(ns, center[seed_tensor].size, device=dev, dtype=dt)}
        sens[seed_tensor][torch.arange(ns, device=dev),
                          torch.as_tensor(seed_idx, device=dev, dtype=torch.long)] = 1.0
        for op in reversed(ops):
            nm = op['name']
            if nm not in sens:
                continue
            sx = sens[nm]; t = op['type']
            if t == 'conv':
                out_sh = shapes[nm]; in_sh = tuple(op['in_shape'])
                ker = _t(op['kernel']); kH, kW = ker.shape[-2], ker.shape[-1]
                st = op['stride']; pd = op['padding']
                sH, sW = (st, st) if isinstance(st, int) else (st[0], st[1])
                pH, pW = (pd, pd) if isinstance(pd, int) else (pd[0], pd[1])
                opH = in_sh[1] - ((out_sh[1] - 1) * sH - 2 * pH + kH)
                opW = in_sh[2] - ((out_sh[2] - 1) * sW - 2 * pW + kW)
                dx = F.conv_transpose2d(sx.reshape(ns, *out_sh), ker, stride=st,
                                        padding=pd, output_padding=(opH, opW))
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + dx.reshape(ns, -1)
            elif t == 'fc':
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx @ _t(op['W'])
            elif t == 'relu':
                if 'layer_idx' in op:
                    Lp = op['layer_idx']
                    if Lp != L_self:
                        uj = torch.as_tensor(ust_L[Lp], device=dev, dtype=torch.long)
                        cols = torch.as_tensor([e_new_col[(Lp, int(j))] for j in ust_L[Lp]],
                                               device=dev, dtype=torch.long)
                        rowG[:, cols] += sx[:, uj] * mu_L[Lp][uj][None, :]
                        sx = sx * lam_L[Lp][None, :]
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx
                else:
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx
            elif t == 'reshape':
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx
            elif t == 'add':
                if op.get('is_merge'):
                    for inp in op['inputs'][:2]:
                        sens[inp] = sens.get(inp, 0) + sx
                else:
                    inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx
            elif t == 'sub':
                inp = op['inputs'][0]; sens[inp] = sens.get(inp, 0) + sx
            else:
                raise NotImplementedError(f'reverse: op {t!r}')
        s_in = sens.get(in_name)
        if s_in is not None:
            rowG[:, :n_input] += s_in[:, :n_input] * radii[None, :n_input]
        return rowG
    # --- per unstable layer ---
    unstable_list = []
    for L in Ls:
        u = ust_L[L]
        if len(u) == 0:
            continue
        rowG = _backward(relu_ops[L]['inputs'][0], u, L).cpu().numpy()
        for i, j in enumerate(u):
            nz = np.nonzero(rowG[i])[0]
            unstable_list.append(dict(
                layer_idx=L, neuron_idx=int(j), c_in=float(pre_center[L][j]),
                lo=float(np.asarray(bbr[L][0])[j]), hi=float(np.asarray(bbr[L][1])[j]),
                lam=float(_np(lam_L[L])[j]), mu=float(_np(mu_L[L])[j]),
                e_new_col=e_new_col[(L, int(j))], form='alpha_zono',
                row_indices=nz.astype(np.int64),
                row_values=rowG[i, nz].astype(np.float64)))
    # --- output objective (seed from the output neurons; L_self=None) ---
    out_name = gg['ops'][-1]['name']
    n_out = center[out_name].size
    obj_G = _backward(out_name, np.arange(n_out), None).cpu().numpy()
    obj_c_out = center[out_name].astype(np.float64)
    obj_G_out_csr = _sp.csr_matrix(obj_G.astype(np.float64))
    return dict(n_gens=n_gens, n_input=n_input, unstable_list=unstable_list,
                obj_c_out=obj_c_out, obj_G_out_csr=obj_G_out_csr,
                input_name=in_name, output_op_name=out_name, formulation='sparse',
                stable_list=[], x_lo=xl_n, x_hi=xh_n, sigmoid_tanh_layer_ids=set())
