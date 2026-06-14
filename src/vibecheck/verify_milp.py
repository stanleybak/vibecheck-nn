"""MILP verification pipeline: zonotope + per-layer tightening + spec MILP.

Strategy:
  1. GPU zonotope forward + adaptive CROWN backward (fast initial bounds)
  2. Per-layer tightening: sample MILP/LP timing, then solve all neurons
     with the fastest method that doesn't timeout. Degrades MILP → LP → zono.
  3. Spec verification: CROWN backward, then LP, then MILP with doubling
     binary budget scored by per-neuron gap contribution.
"""

import time
import os
import numpy as np
import multiprocessing
import torch
import torch.nn.functional as F

from .settings import default_settings, resolve_torch
from .gurobi_util import optimize_checked


class VerifyStats:
    """Collects timing, neuron stats, and model info during verification."""

    def __init__(self):
        self.timing = {}
        self.neuron_stats = {'per_layer': {}, 'total_unstable': 0,
                             'total_neurons': 0, 'neurons_fixed_by_tightening': 0}
        self.model_size = None

    def record_timing(self, phase, elapsed):
        self.timing[phase] = elapsed

    def record_layer_stats(self, layer_idx, total, unstable, avg_width):
        self.neuron_stats['per_layer'][layer_idx] = {
            'total': total, 'unstable': unstable, 'avg_width': avg_width}
        self.neuron_stats['total_neurons'] += total
        self.neuron_stats['total_unstable'] += unstable

    def record_bounds(self, bounds_dict, label=''):
        """Compute per-layer stats from a bounds dict (layer_idx -> (lo, hi))."""
        self.neuron_stats['per_layer'] = {}
        self.neuron_stats['total_neurons'] = 0
        self.neuron_stats['total_unstable'] = 0
        for li in sorted(bounds_dict.keys()):
            lo, hi = bounds_dict[li]
            if hasattr(lo, 'numpy'):
                lo, hi = lo.cpu().numpy(), hi.cpu().numpy()
            n = len(lo)
            ust = int(((lo < 0) & (hi > 0)).sum())
            widths = hi - lo
            ust_mask = (lo < 0) & (hi > 0)
            avg_w = float(widths[ust_mask].mean()) if ust > 0 else 0.0
            self.record_layer_stats(li, n, ust, avg_w)

    def record_model(self, n_vars, n_constrs, n_binaries):
        self.model_size = {'n_vars': n_vars, 'n_constrs': n_constrs,
                           'n_binaries': n_binaries}

    def to_dict(self):
        d = {'timing': self.timing, 'neuron_stats': self.neuron_stats}
        if self.model_size:
            d['model_size'] = self.model_size
        # Compute overhead
        total = sum(self.timing.values())
        d['timing']['total_phases'] = total
        return d


def _fire_callback(settings, event, info):
    """Fire callback if set. Returns False if callback requests termination."""
    cb = getattr(settings, 'milp_callback', None)
    if cb is None:
        return True
    return cb(event, info)


def _make_result(result, details, stats=None):
    """Merge VerifyStats into result details dict."""
    if stats:
        details.update(stats.to_dict())
    return result, details
from .zonotope import TorchZonotope
from .verify_zono_bnb import (
    _make_slopes, _build_spec_ew, _spec_backward, _evaluate_region,
    _pgd_attack, _forward_batch,
    _forward_zonotope_graph, _forward_batch_graph, _ibp_forward_graph,
    _ibp_forward_graph_batched, _spec_backward_graph_batched,
    _spec_backward_graph, _build_spec_ew_graph,
    _pgd_attack_graph,
)


# Loose Gurobi feasibility tolerance for the spec MILP. Default
# (FeasibilityTol=1e-6) is too tight against float32-zono / float64-LP
# round-trip; isolated test with bound width 1 ulp + constraint
# residual ~1.15e-6 → INFEAS at default, OPT at 1e-5. See block comment
# in `_solve_spec_worker` for the metaroom unsoundness this fixes.
_GUROBI_FEAS_TOL = 1e-5

# Below this many unstable neurons, the spec-MILP racing skips its gradual
# bin-escalation and solves the EXACT MILP (all unstable binarized) in one
# shot — the exact solve is sub-second at this scale, and escalating wastes
# a Pool+Gurobi-env spawn per intermediate level (which all return loose
# "feasibility SAT" until the exact level anyway). Big conv nets stay on the
# gradual schedule (the exact MILP would be intractable). Tuned for
# safenlp_2024 (≤128 unstable): exact <1s vs escalation timing out at 20s.
_DIRECT_EXACT_MAX_UNSTABLE = 256


def _pgd_attack_general(xl, xh, spec, gg, settings,
                         restrict_disj=None, time_budget=None,
                         per_restart_disj=None):
    """Thin wrapper over `vibecheck.pgd.pgd_attack_general`.

    Kept here so existing imports still work; all algorithmic logic lives
    in `pgd.py` (α,β-CROWN-style AdamClipping + hinge + 10×100 schedule).
    """
    from . import pgd as _pgd
    return _pgd.pgd_attack_general(xl, xh, spec, gg, settings,
                                     restrict_disj=restrict_disj,
                                     time_budget=time_budget,
                                     per_restart_disj=per_restart_disj)

# ---------------------------------------------------------------------------
# Shared state for multiprocessing workers (COW via fork)
# ---------------------------------------------------------------------------
_shared_model = None
_shared_layer_np = None
_shared_prev_layer_idx = None
_shared_sparse_args = None


# ---------------------------------------------------------------------------
# Conv sparse connections
# ---------------------------------------------------------------------------

def _conv_connections(j, kernel, in_shape, stride, padding):
    """Sparse input connections for conv output neuron j.

    Returns list of (flat_input_idx, weight_value).
    """
    C_in, H_in, W_in = in_shape
    C_out, _, kH, kW = kernel.shape
    sH, sW = stride
    pH, pW = padding
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1

    c_out = j // (H_out * W_out)
    rem = j % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    conns = []
    for c_in in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                h_in = h_out * sH - pH + kh
                w_in = w_out * sW - pW + kw
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    w = float(kernel[c_out, c_in, kh, kw])
                    if w != 0:
                        flat = c_in * H_in * W_in + h_in * W_in + w_in
                        conns.append((flat, w))
    return conns


def _conv_sparse_matrix(kernel, in_shape, stride, padding):
    """Build sparse weight matrix for a conv layer (vectorized, fast).

    Returns scipy.sparse.csr_matrix of shape (n_out, n_in) and bias_vec.
    """
    import scipy.sparse as sp
    C_in, H_in, W_in = in_shape
    C_out, _, kH, kW = kernel.shape
    sH, sW = stride
    pH, pW = padding
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    n_out = C_out * H_out * W_out
    n_in = C_in * H_in * W_in
    spatial_out = H_out * W_out

    ho = np.arange(H_out)
    wo = np.arange(W_out)
    ci = np.arange(C_in)
    kh_r = np.arange(kH)
    kw_r = np.arange(kW)

    HO, WO, CI, KH, KW = np.meshgrid(ho, wo, ci, kh_r, kw_r, indexing='ij')
    H_IN = HO * sH - pH + KH
    W_IN = WO * sW - pW + KW
    valid = (H_IN >= 0) & (H_IN < H_in) & (W_IN >= 0) & (W_IN < W_in)
    FLAT_IN = CI * H_in * W_in + H_IN * W_in + W_IN

    ROW_BASE = (np.arange(H_out)[:, None] * W_out
                + np.arange(W_out)[None, :])[:, :, None, None, None]
    ROW_BASE = np.broadcast_to(ROW_BASE, valid.shape)

    rows_list, cols_list, vals_list = [], [], []
    for c in range(C_out):
        w_vals = np.broadcast_to(kernel[c], (H_out, W_out, C_in, kH, kW))
        mask = valid & (w_vals != 0)
        rows_list.append((c * spatial_out + ROW_BASE[mask]).ravel())
        cols_list.append(FLAT_IN[mask].ravel())
        vals_list.append(w_vals[mask].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_out, n_in))


def _conv_bias_idx(j, kernel, in_shape, stride, padding):
    """Get bias index (output channel) for flat conv neuron j."""
    C_out = kernel.shape[0]
    H_out = (in_shape[1] + 2 * padding[0] - kernel.shape[2]) // stride[0] + 1
    W_out = (in_shape[2] + 2 * padding[1] - kernel.shape[3]) // stride[1] + 1
    spatial = H_out * W_out
    return j // spatial


def _n_out(layer_np):
    """Number of output neurons for a numpy layer dict."""
    if layer_np['type'] == 'fc':
        return layer_np['W'].shape[0]
    k = layer_np['kernel']
    ins = layer_np['in_shape']
    s, p = layer_np['stride'], layer_np['padding']
    H_out = (ins[1] + 2 * p[0] - k.shape[2]) // s[0] + 1
    W_out = (ins[2] + 2 * p[1] - k.shape[3]) // s[1] + 1
    return k.shape[0] * H_out * W_out


# ---------------------------------------------------------------------------
# Neuron scoring
# ---------------------------------------------------------------------------

def score_neurons_by_relaxation(bounds, layers_np, nh):
    """Score unstable neurons by triangle relaxation area.

    Returns dict: (layer_idx, neuron_idx) -> score.
    Higher score = more relaxation error = more important to tighten.
    """
    scores = {}
    for l in range(nh):
        lo, hi = bounds[l]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        for i in unstable:
            area = float(hi[i]) * abs(float(lo[i])) / 2.0
            scores[(l, int(i))] = area
    return scores


def score_neurons_by_crown(bounds, ew_at_layer, nh):
    """Score unstable neurons by CROWN effective weight × relaxation error.

    score(l, i) = |ew_l[i]| × mu_i
    where mu_i = -hi*lo / (2*(hi-lo)) is the relaxation half-width.

    Args:
        bounds: dict l -> (lo, hi) numpy arrays
        ew_at_layer: dict l -> numpy array of per-neuron effective weights
                     from CROWN backward for the spec
        nh: number of hidden layers

    Returns dict: (layer_idx, neuron_idx) -> score.
    """
    scores = {}
    for l in range(nh):
        if l not in ew_at_layer:
            continue
        lo, hi = bounds[l]
        ew = ew_at_layer[l]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        for i in unstable:
            mu = -float(hi[i]) * float(lo[i]) / (2.0 * (float(hi[i]) - float(lo[i])))
            scores[(l, int(i))] = abs(float(ew[i])) * mu
    return scores


def score_neurons_crown_lp_fractional(bounds_np, ew_at_layer, nh,
                                       lp_model):
    """Combined scoring: CROWN × LP-fractional.

    score(l, i) = crown_score(l, i) × frac_score(l, i)

    crown_score = |ew[i]| × mu[i]  (spec-relevance × relaxation error)
    frac_score = |a_LP - max(0, z_LP)|  (how far LP exploits the triangle)

    The product picks neurons that are both exploited by the LP AND
    relevant to the spec — these are where making the relaxation exact
    will most tighten the spec bound.

    Args:
        bounds_np: dict l -> (lo, hi) numpy arrays
        ew_at_layer: dict l -> numpy array of CROWN effective weights
        nh: number of hidden layers
        lp_model: solved Gurobi LP model (status=2) with z_l_j and a_l_j vars

    Returns dict: (layer_idx, neuron_idx) -> score.
    """
    scores = {}
    for l in range(nh):
        if l not in ew_at_layer:
            continue
        lo, hi = bounds_np[l]
        ew = ew_at_layer[l]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        for i in unstable:
            i = int(i)
            mu = -float(hi[i]) * float(lo[i]) / (
                2.0 * (float(hi[i]) - float(lo[i])))
            crown = abs(float(ew[i])) * mu

            # Read LP solution values
            z_var = lp_model.getVarByName(f'z_{l}_{i}')
            a_var = lp_model.getVarByName(f'a_{l}_{i}')
            if z_var is not None and a_var is not None:
                z_lp = z_var.X
                a_lp = a_var.X
                frac = abs(a_lp - max(0.0, z_lp))
            else:
                frac = 0.0

            scores[(l, i)] = crown * frac
    return scores


def score_neurons_ew_frac(bounds_np, ew_at_layer, nh, lp_model):
    """Score: |ew[i]| × frac (no mu term).

    frac = |a_LP - max(0, z_LP)| measures how much the LP exploits the
    triangle relaxation.  Combined with |ew| this picks neurons that are
    both LP-exploited and spec-relevant.

    Args:
        bounds_np: dict l -> (lo, hi) numpy arrays
        ew_at_layer: dict l -> numpy array of CROWN effective weights
        nh: number of hidden layers
        lp_model: solved Gurobi LP model (status=2)

    Returns dict: (layer_idx, neuron_idx) -> score.
    """
    scores = {}
    for l in range(nh):
        if l not in ew_at_layer:
            continue
        lo, hi = bounds_np[l]
        ew = ew_at_layer[l]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        for i in unstable:
            i = int(i)
            z_var = lp_model.getVarByName(f'z_{l}_{i}')
            a_var = lp_model.getVarByName(f'a_{l}_{i}')
            if z_var is not None and a_var is not None:
                frac = abs(a_var.X - max(0.0, z_var.X))
            else:
                frac = 0.0
            scores[(l, i)] = abs(float(ew[i])) * frac
    return scores


def _compute_crown_layer_weights(bounds_np, layers_np, spec_ew, pred, comp, nh):
    """Run CROWN backward and record effective weight at each layer.

    Returns dict: layer_idx -> numpy array of effective weights.
    """
    ew_at_layer = {}

    # Start from spec weight at output. Two output-layer shapes:
    #   FC : `final['W'][pred] - final['W'][comp]` is the spec direction
    #        already in (n_in,) shape, ready to walk backward.
    #   Conv (e.g. cifar_biasfield's last layer is Conv(in=128, out=10)
    #        with output spatial 1×1, then Flatten → (10,)): treat the
    #        spec direction as a one-hot vector in output shape
    #        (out_channels, H_out, W_out), then conv-transpose it back
    #        to (in_channels, H_in, W_in) flattened — same as how the
    #        hidden Conv branch below propagates ew through earlier
    #        Convs. This recovers 15 cifar_biasfield cases that
    #        otherwise hit `NotImplementedError` here and report
    #        `error` in the harness.
    final = layers_np[nh]
    if final['type'] == 'fc':
        ew = final['W'][pred] - final['W'][comp]
        acc = float(final['bias'][pred]) - float(final['bias'][comp])
    else:
        kernel = final['kernel']
        in_shape = final['in_shape']
        stride = final['stride']
        padding = final['padding']
        C_out = kernel.shape[0]
        H_out = (in_shape[1] + 2*padding[0] - kernel.shape[2]) // stride[0] + 1
        W_out = (in_shape[2] + 2*padding[1] - kernel.shape[3]) // stride[1] + 1
        # Spec direction in flattened output space = pred-row minus comp-row
        # of the one-hot identity matrix. Reshape to (out_c, H_out, W_out)
        # — for the cifar_biasfield case H_out=W_out=1 so this is a
        # one-hot at (pred, 0, 0) minus one-hot at (comp, 0, 0).
        spec_dir = np.zeros(C_out * H_out * W_out, dtype=np.float64)
        spec_dir[pred * H_out * W_out:(pred + 1) * H_out * W_out] = 1.0
        spec_dir[comp * H_out * W_out:(comp + 1) * H_out * W_out] -= 1.0
        bias_arr = final['bias']
        acc = float(bias_arr[pred]) - float(bias_arr[comp])
        ew_t = torch.tensor(spec_dir, dtype=torch.float64).reshape(
            1, C_out, H_out, W_out)
        k_t = torch.tensor(kernel, dtype=torch.float64)
        oph = in_shape[1] - ((H_out - 1)*stride[0] - 2*padding[0]
                              + kernel.shape[2])
        opw = in_shape[2] - ((W_out - 1)*stride[1] - 2*padding[1]
                              + kernel.shape[3])
        ew = F.conv_transpose2d(
            ew_t, k_t, stride=stride, padding=padding,
            output_padding=(oph, opw)).flatten().numpy()

    for k in range(nh - 1, -1, -1):
        lo, hi = bounds_np[k]
        # Adaptive slopes
        lb_r = np.clip(lo, a_min=None, a_max=0)
        ub_r = np.clip(hi, a_min=0, a_max=None)
        ub_r = np.maximum(ub_r, lb_r + 1e-8)
        up_s = ub_r / (ub_r - lb_r)
        up_t = -lb_r * up_s
        active = lo >= 0
        dead = hi <= 0
        lm = active.astype(np.float64)
        um = dead.astype(np.float64)
        lo_s = (up_s > 0.5).astype(np.float64) * (1 - lm) * (1 - um) + lm

        # Record effective weight at this layer (before ReLU slope application)
        ew_at_layer[k] = ew.copy()

        # Apply slopes
        ep = np.clip(ew, a_min=0, a_max=None)
        en = np.clip(ew, a_min=None, a_max=0)
        acc += float((en * up_t).sum())
        ew = ep * lo_s + en * up_s

        # Through linear layer
        layer = layers_np[k]
        if layer['type'] == 'fc':
            acc += float(ew @ layer['bias'])
            ew = ew @ layer['W']
        else:
            # Conv: need conv_transpose equivalent in numpy
            # For now, just use dense matmul via constructing the conv matrix
            # This is only for scoring, doesn't need to be ultra-fast
            kernel = layer['kernel']
            in_shape = layer['in_shape']
            stride = layer['stride']
            padding = layer['padding']
            n_out_l = len(lo)
            n_in_l = in_shape[0] * in_shape[1] * in_shape[2]

            # Compute bias contribution
            C_out = kernel.shape[0]
            H_out = (in_shape[1] + 2*padding[0] - kernel.shape[2]) // stride[0] + 1
            W_out = (in_shape[2] + 2*padding[1] - kernel.shape[3]) // stride[1] + 1
            spatial = H_out * W_out
            for c in range(C_out):
                acc += float(ew[c*spatial:(c+1)*spatial].sum()) * float(layer['bias'][c])

            # Conv transpose via torch (quick)
            ew_t = torch.tensor(ew, dtype=torch.float64).reshape(1, C_out, H_out, W_out)
            k_t = torch.tensor(kernel, dtype=torch.float64)
            oph = in_shape[1] - ((H_out-1)*stride[0] - 2*padding[0] + kernel.shape[2])
            opw = in_shape[2] - ((W_out-1)*stride[1] - 2*padding[1] + kernel.shape[3])
            ew = F.conv_transpose2d(ew_t, k_t, stride=stride, padding=padding,
                                     output_padding=(oph, opw)).flatten().numpy()

    return ew_at_layer


# ---------------------------------------------------------------------------
# Gurobi model building
# ---------------------------------------------------------------------------

def _build_sparse_neuron_model(layers_np, x_lo, x_hi, bounds,
                                target_layer, target_neuron):
    """Build a sparse Gurobi model for a single neuron's bounds.

    Only includes neurons in the receptive field chain from the target
    neuron back to the input. For conv layers this is much smaller than
    the full model.

    Returns (model, env) with a '_target' variable for the neuron's
    pre-ReLU value.
    """
    import gurobipy as grb

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    # Trace receptive field backward from target neuron
    # needed[l] = set of neuron indices needed at layer l's output (post-ReLU)
    needed = {}

    # Target neuron connects to previous layer neurons
    tl = layers_np[target_layer]
    if tl['type'] == 'conv':
        conns = _conv_connections(
            target_neuron, tl['kernel'], tl['in_shape'],
            tl['stride'], tl['padding'])
        needed_prev = set(fi for fi, _ in conns)
    else:
        W = tl['W']
        needed_prev = set(int(k) for k in range(W.shape[1]) if W[target_neuron, k] != 0)

    # Walk backward through layers to find which neurons are needed
    for l in range(target_layer - 1, -1, -1):
        needed[l] = needed_prev
        layer = layers_np[l]
        lo, hi = bounds[l]
        # For each needed neuron at layer l, trace its inputs
        new_needed = set()
        for j in needed_prev:
            if hi[j] <= 0:
                continue  # dead, contributes 0
            if layer['type'] == 'conv':
                conns = _conv_connections(
                    j, layer['kernel'], layer['in_shape'],
                    layer['stride'], layer['padding'])
                for fi, _ in conns:
                    new_needed.add(fi)
            else:
                W = layer['W']
                for k in range(W.shape[1]):
                    if W[j, k] != 0:
                        new_needed.add(int(k))
        needed_prev = new_needed

    # needed_prev now has the input dimensions needed
    needed_inputs = needed_prev

    # Build model: only create variables for needed neurons

    # Input variables (only needed ones)
    inp_vars = {}
    for i in sorted(needed_inputs):
        v = m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]), name=f'inp_{i}')
        inp_vars[i] = v
    m.update()

    # Layer variables
    layer_vars = {}  # (l, j) -> var for a_{l}_{j}
    for l in range(target_layer):
        if l not in needed:
            continue
        layer = layers_np[l]
        lo, hi = bounds[l]
        prev_prefix = 'inp' if l == 0 else None

        for j in sorted(needed[l]):
            if hi[j] <= 0:
                # Dead: fixed to 0
                v = m.addVar(lb=0.0, ub=0.0, name=f'a_{l}_{j}')
                layer_vars[(l, j)] = v
            elif lo[j] >= 0:
                # Active: a = z (no binary needed)
                v = m.addVar(lb=0.0, ub=grb.GRB.INFINITY,
                             name=f'a_{l}_{j}')
                layer_vars[(l, j)] = v
            else:
                # Unstable: z, a, s
                zv = m.addVar(lb=float(lo[j]), ub=float(hi[j]),
                              name=f'z_{l}_{j}')
                av = m.addVar(lb=0.0, ub=float(hi[j]),
                              name=f'a_{l}_{j}')
                sv = m.addVar(vtype=grb.GRB.BINARY, name=f's_{l}_{j}')
                layer_vars[(l, j)] = av
        m.update()

        # Constraints
        for j in sorted(needed[l]):
            if hi[j] <= 0:
                continue
            expr = grb.LinExpr()
            if layer['type'] == 'conv':
                conns = _conv_connections(
                    j, layer['kernel'], layer['in_shape'],
                    layer['stride'], layer['padding'])
                for fi, w in conns:
                    if l == 0:
                        v = inp_vars.get(fi)
                    else:
                        v = layer_vars.get((l-1, fi))
                    if v is not None:
                        expr.add(v, w)
                b_j = float(layer['bias'][
                    _conv_bias_idx(j, layer['kernel'],
                                   layer['in_shape'],
                                   layer['stride'],
                                   layer['padding'])])
            else:
                W = layer['W']
                for k in range(W.shape[1]):
                    if W[j, k] != 0:
                        if l == 0:
                            v = inp_vars.get(k)
                        else:
                            v = layer_vars.get((l-1, k))
                        if v is not None:
                            expr.add(v, float(W[j, k]))
                b_j = float(layer['bias'][j])

            if lo[j] >= 0:
                m.addConstr(layer_vars[(l, j)] == expr + b_j)
            else:
                z = m.getVarByName(f'z_{l}_{j}')
                a = m.getVarByName(f'a_{l}_{j}')
                s = m.getVarByName(f's_{l}_{j}')
                lo_j, hi_j = float(lo[j]), float(hi[j])
                m.addConstr(z == expr + b_j)
                m.addConstr(a >= 0)
                m.addConstr(a >= z)
                m.addConstr(a <= hi_j * s)
                m.addConstr(a <= z - lo_j * (1 - s))
        m.update()

    # Add target variable
    target = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                       name='_target')
    m.update()
    expr = grb.LinExpr()
    if tl['type'] == 'conv':
        conns = _conv_connections(
            target_neuron, tl['kernel'], tl['in_shape'],
            tl['stride'], tl['padding'])
        for fi, w in conns:
            if target_layer == 0:
                v = inp_vars.get(fi)
            else:
                v = layer_vars.get((target_layer-1, fi))
            if v is not None:
                expr.add(v, w)
        b_j = float(tl['bias'][
            _conv_bias_idx(target_neuron, tl['kernel'],
                           tl['in_shape'],
                           tl['stride'],
                           tl['padding'])])
    else:
        W = tl['W']
        for k in range(W.shape[1]):
            if W[target_neuron, k] != 0:
                if target_layer == 0:
                    v = inp_vars.get(k)
                else:
                    v = layer_vars.get((target_layer-1, k))
                if v is not None:
                    expr.add(v, float(W[target_neuron, k]))
        b_j = float(tl['bias'][target_neuron])

    m.addConstr(target == expr + b_j)
    m.update()

    return m, env


def _build_base_model(layers_np, x_lo, x_hi, bounds, up_to_layer,
                      milp_set=None, lp_encoding='zas'):
    """Build Gurobi LP/MILP encoding layers 0..up_to_layer-1.

    Dead neurons get a fixed-zero variable. Active neurons get a variable
    with a=z equality constraint. Unstable neurons use either:
    - 'zas': z, a, s variables (3 vars, 5 constraints per unstable LP neuron)
    - 'compact': a variable only (1 var, 2 constraints per unstable LP neuron)

    For MILP neurons (in milp_set), always uses z, a, s with binary s.

    Args:
        layers_np: list of layer dicts with numpy weights
        x_lo, x_hi: input bounds (numpy float64)
        bounds: dict l -> (lo, hi) numpy arrays (pre-ReLU bounds)
        up_to_layer: encode layers 0..up_to_layer-1
        milp_set: set of (layer, neuron) pairs to encode with binaries.
                  None = all unstable get binaries.
                  Empty set = pure LP (triangle relaxation for all).
        lp_encoding: 'zas' or 'compact' for LP neurons.

    Returns: (model, env)
    """
    import gurobipy as grb

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    n_input = len(x_lo)
    for i in range(n_input):
        m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]), name=f'inp_{i}')
    m.update()

    for l in range(up_to_layer):
        layer = layers_np[l]
        lo, hi = bounds[l]
        n = _n_out(layer)
        prev = 'inp' if l == 0 else f'a_{l-1}'

        # Add variables
        for j in range(n):
            if hi[j] <= 0:
                m.addVar(lb=0.0, ub=0.0, name=f'a_{l}_{j}')
            elif lo[j] >= 0:
                m.addVar(lb=0.0, ub=grb.GRB.INFINITY,
                         name=f'a_{l}_{j}')
            else:
                use_milp = (milp_set is None or (l, j) in milp_set)
                if use_milp or lp_encoding == 'zas':
                    m.addVar(lb=float(lo[j]), ub=float(hi[j]),
                             name=f'z_{l}_{j}')
                    m.addVar(lb=0.0, ub=float(hi[j]), name=f'a_{l}_{j}')
                    if use_milp:
                        m.addVar(vtype=grb.GRB.BINARY, name=f's_{l}_{j}')
                    else:
                        m.addVar(lb=0.0, ub=1.0, name=f's_{l}_{j}')
                else:
                    # Compact: only a variable
                    m.addVar(lb=0.0, ub=float(hi[j]), name=f'a_{l}_{j}')
        m.update()

        # Add constraints
        for j in range(n):
            if hi[j] <= 0:
                continue

            # Build affine expression
            expr = grb.LinExpr()
            if layer['type'] == 'conv':
                conns = _conv_connections(
                    j, layer['kernel'], layer['in_shape'],
                    layer['stride'], layer['padding'])
                for flat_in, w in conns:
                    v = m.getVarByName(f'{prev}_{flat_in}')
                    if v is not None:
                        expr.add(v, w)
                b_j = float(layer['bias'][
                    _conv_bias_idx(j, layer['kernel'],
                                   layer['in_shape'],
                                   layer['stride'],
                                   layer['padding'])])
            else:
                W = layer['W']
                for k in range(W.shape[1]):
                    if W[j, k] != 0:
                        v = m.getVarByName(f'{prev}_{k}')
                        if v is not None:
                            expr.add(v, float(W[j, k]))
                b_j = float(layer['bias'][j])

            if lo[j] >= 0:
                # Active: a = z
                m.addConstr(m.getVarByName(f'a_{l}_{j}') == expr + b_j)
            else:
                use_milp = (milp_set is None or (l, j) in milp_set)
                if use_milp or lp_encoding == 'zas':
                    z = m.getVarByName(f'z_{l}_{j}')
                    a = m.getVarByName(f'a_{l}_{j}')
                    s = m.getVarByName(f's_{l}_{j}')
                    lo_j, hi_j = float(lo[j]), float(hi[j])
                    m.addConstr(z == expr + b_j)
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= hi_j * s)
                    m.addConstr(a <= z - lo_j * (1 - s))
                else:
                    # Compact: a >= z, a <= slope*z - slope*lo
                    a = m.getVarByName(f'a_{l}_{j}')
                    lo_j, hi_j = float(lo[j]), float(hi[j])
                    m.addConstr(a >= expr + b_j)
                    slope = hi_j / (hi_j - lo_j)
                    m.addConstr(a <= slope * (expr + b_j) - slope * lo_j)
        m.update()

    return m, env


# ---------------------------------------------------------------------------
# Parallel neuron solving
# ---------------------------------------------------------------------------

def _solve_neuron(args):
    """Worker: solve one neuron's bound (min or max of pre-ReLU value).

    Uses sparse model if available (_shared_sparse_args set), otherwise
    copies the full shared model and adds the neuron objective.
    """
    import gurobipy as grb

    idx, direction, timeout = args

    if _shared_sparse_args is not None:
        layers_np, x_lo, x_hi, bounds, target_layer = _shared_sparse_args
        model, env = _build_sparse_neuron_model(
            layers_np, x_lo, x_hi, bounds, target_layer, idx)
        target_var = model.getVarByName('_target')
    else:
        model = _shared_model.copy()
        env = None
        layer = _shared_layer_np
        l_prev = _shared_prev_layer_idx
        prev = 'inp' if l_prev < 0 else f'a_{l_prev}'

        target_var = model.addVar(lb=-grb.GRB.INFINITY,
                                   ub=grb.GRB.INFINITY, name='_target')
        model.update()
        expr = grb.LinExpr()
        if layer['type'] == 'conv':
            conns = _conv_connections(
                idx, layer['kernel'], layer['in_shape'],
                layer['stride'], layer['padding'])
            for flat_in, w in conns:
                v = model.getVarByName(f'{prev}_{flat_in}')
                if v is not None:
                    expr.add(v, w)
            b_val = float(layer['bias'][
                _conv_bias_idx(idx, layer['kernel'],
                               layer['in_shape'],
                               layer['stride'],
                               layer['padding'])])
        else:
            W = layer['W']
            for k in range(W.shape[1]):
                if W[idx, k] != 0:
                    v = model.getVarByName(f'{prev}_{k}')
                    if v is not None:
                        expr.add(v, float(W[idx, k]))
            b_val = float(layer['bias'][idx])
        model.addConstr(target_var == expr + b_val)
        model.update()

    model.setParam('TimeLimit', timeout)

    if direction == 'min':
        model.setObjective(target_var, grb.GRB.MINIMIZE)
    else:
        model.setObjective(target_var, grb.GRB.MAXIMIZE)

    t0 = time.perf_counter()
    optimize_checked(model)
    dt = time.perf_counter() - t0

    timed_out = model.status == 9
    bound = None
    try:
        bound = model.ObjBound
    except (grb.GurobiError, AttributeError):
        # ObjBound is unavailable when no LP root relaxation has produced
        # a bound (e.g. infeasible or pre-solve abort); fall back to incumbent.
        if model.SolCount > 0:
            bound = model.ObjVal

    model.dispose()
    if env is not None:
        env.dispose()

    return idx, direction, bound, dt, timed_out


def _forward_witnesses_layered(layers_np, witnesses, target_layer):
    """Forward witness inputs (n_w, n_in_flat) through the actual ReLU
    network up to `target_layer` and return pre-activation z at that layer
    of shape (n_w, n_neur_at_target). Used to pick MIN/MAX ordering: if all
    witnesses give z ≥ 0 at neuron j, MAX cannot prove dead → MIN first.

    Conv layers are evaluated via torch.nn.functional.conv2d at fp64 to
    avoid precision drift (the LP/MILP runs at fp64 by default).
    """
    y = witnesses.astype(np.float64, copy=False)
    for L in range(target_layer + 1):
        layer = layers_np[L]
        if layer['type'] == 'fc':
            z = y @ layer['W'].T.astype(np.float64) + \
                layer['bias'].astype(np.float64)
        else:  # conv
            kernel = layer['kernel'].astype(np.float64)
            bias = layer['bias'].astype(np.float64)
            stride = layer['stride']; padding = layer['padding']
            in_shape = layer['in_shape']
            n_w = y.shape[0]
            y_4d = y.reshape(n_w, in_shape[0], in_shape[1], in_shape[2])
            z_t = F.conv2d(
                torch.from_numpy(y_4d), torch.from_numpy(kernel),
                torch.from_numpy(bias), stride=stride, padding=padding)
            z = z_t.numpy().reshape(n_w, -1)
        if L == target_layer:
            return z
        y = np.maximum(z, 0.0)
    return z   # unreachable


def _pick_milp_direction(cur_lo, cur_hi, z_w_min, z_w_max):
    """Return (first_obj, second_obj) — Gurobi sense for first/second MILP.
    Witness rule: all witnesses ≥ 0 → MIN first (proving active is the only
    chance). All ≤ 0 → MAX first (proving dead). Straddle / no witness →
    fall back to |cur_lo|<|cur_hi| asymmetry heuristic.
    """
    import gurobipy as grb
    if z_w_min is not None and z_w_max is not None:
        if z_w_min >= -1e-9:
            return grb.GRB.MINIMIZE, grb.GRB.MAXIMIZE
        if z_w_max <= 1e-9:
            return grb.GRB.MAXIMIZE, grb.GRB.MINIMIZE
    if abs(cur_lo) < abs(cur_hi):
        return grb.GRB.MINIMIZE, grb.GRB.MAXIMIZE
    return grb.GRB.MAXIMIZE, grb.GRB.MINIMIZE


def _solve_neuron_both(args):
    """Worker: solve BOTH min and max for a neuron on the same model.

    Returns (idx, lb, ub, total_time, any_timeout).

    args is (idx, timeout, cur_lo, cur_hi) for the legacy bound-asymmetry
    ordering, or (idx, timeout, cur_lo, cur_hi, z_w_min, z_w_max) for
    witness-guided ordering (per `tighten_witness_ordering=True`).
    """
    import gurobipy as grb

    if len(args) == 6:
        idx, timeout, cur_lo, cur_hi, z_w_min, z_w_max = args
    else:
        idx, timeout, cur_lo, cur_hi = args
        z_w_min = z_w_max = None

    if _shared_sparse_args is not None:
        sparse_args = _shared_sparse_args
        if len(sparse_args) == 6 and sparse_args[5] == 'fc_lp':
            # Per-worker LP: build fresh model from scratch
            layers_np, x_lo, x_hi, bounds, target_layer, _ = sparse_args
            model, env = _build_base_model(
                layers_np, x_lo, x_hi, bounds, target_layer,
                milp_set=set())
            layer = layers_np[target_layer]
            l_prev = target_layer - 1
            prev = 'inp' if l_prev < 0 else f'a_{l_prev}'
            target_var = model.addVar(lb=-grb.GRB.INFINITY,
                                       ub=grb.GRB.INFINITY, name='_target')
            model.update()
            expr = grb.LinExpr()
            W = layer['W']
            for k in range(W.shape[1]):
                if W[idx, k] != 0:
                    v = model.getVarByName(f'{prev}_{k}')
                    if v is not None:
                        expr.add(v, float(W[idx, k]))
            b_val = float(layer['bias'][idx])
            model.addConstr(target_var == expr + b_val)
            model.update()
        else:
            layers_np, x_lo, x_hi, bounds, target_layer = sparse_args
            model, env = _build_sparse_neuron_model(
                layers_np, x_lo, x_hi, bounds, target_layer, idx)
            target_var = model.getVarByName('_target')
    else:
        model = _shared_model.copy()
        env = None
        layer = _shared_layer_np
        l_prev = _shared_prev_layer_idx
        prev = 'inp' if l_prev < 0 else f'a_{l_prev}'

        target_var = model.addVar(lb=-grb.GRB.INFINITY,
                                   ub=grb.GRB.INFINITY, name='_target')
        model.update()
        expr = grb.LinExpr()
        if layer['type'] == 'conv':
            conns = _conv_connections(
                idx, layer['kernel'], layer['in_shape'],
                layer['stride'], layer['padding'])
            for flat_in, w in conns:
                v = model.getVarByName(f'{prev}_{flat_in}')
                if v is not None:
                    expr.add(v, w)
            b_val = float(layer['bias'][
                _conv_bias_idx(idx, layer['kernel'],
                               layer['in_shape'],
                               layer['stride'],
                               layer['padding'])])
        else:
            W = layer['W']
            for k in range(W.shape[1]):
                if W[idx, k] != 0:
                    v = model.getVarByName(f'{prev}_{k}')
                    if v is not None:
                        expr.add(v, float(W[idx, k]))
            b_val = float(layer['bias'][idx])
        model.addConstr(target_var == expr + b_val)
        model.update()

    model.setParam('TimeLimit', timeout)
    any_timeout = False
    lb, ub = cur_lo, cur_hi

    # Solve "proving stable" direction first (witness-guided when available,
    # else the |cur_lo|<|cur_hi| asymmetry heuristic).
    first, second = _pick_milp_direction(cur_lo, cur_hi, z_w_min, z_w_max)

    # First direction with BestBdStop for early exit
    model.setObjective(target_var, first)
    if first == grb.GRB.MINIMIZE:
        model.setParam('BestBdStop', 1e-6)  # stop if lb > 0 (proven active)
    else:
        model.setParam('BestBdStop', -1e-6)  # stop if ub < 0 (proven dead)
    optimize_checked(model)
    if model.status == 9:
        any_timeout = True
    try:
        b = model.ObjBound
    except (grb.GurobiError, AttributeError):
        b = model.ObjVal if model.SolCount > 0 else None
    if b is not None:
        if first == grb.GRB.MINIMIZE:
            lb = max(lb, b)
        else:
            ub = min(ub, b)

    # Skip second direction if already proven stable
    if lb >= 0 or ub <= 0:
        model.dispose()
        if env is not None:
            env.dispose()
        return idx, lb, ub, 0, any_timeout

    # Second direction
    model.setObjective(target_var, second)
    if second == grb.GRB.MINIMIZE:
        model.setParam('BestBdStop', 1e-6)
    else:
        model.setParam('BestBdStop', -1e-6)
    optimize_checked(model)
    if model.status == 9:
        any_timeout = True
    try:
        b = model.ObjBound
    except (grb.GurobiError, AttributeError):
        b = model.ObjVal if model.SolCount > 0 else None
    if b is not None:
        if second == grb.GRB.MINIMIZE:
            lb = max(lb, b)
        else:
            ub = min(ub, b)

    model.dispose()
    if env is not None:
        env.dispose()
    return idx, lb, ub, 0, any_timeout


def _tighten_layer_parallel(layers_np, x_lo, x_hi, bounds, l,
                             use_milp, timeout, n_cores,
                             neuron_subset=None, lp_per_worker=False,
                             witness_n_random=8, deadline=None):
    """Tighten unstable neurons at layer l using parallel LP or MILP.

    For conv layers, uses sparse per-neuron models (much faster).
    For FC layers with lp_per_worker=True, each worker builds its own LP.

    Two-pass: first solve the tighter side (min if |lo|<|hi|, else max),
    then solve the other direction for still-unstable neurons.

    Returns (new_lo, new_hi, any_timeout).
    """
    global _shared_model, _shared_layer_np, _shared_prev_layer_idx
    global _shared_sparse_args

    lo, hi = bounds[l]
    if neuron_subset is not None:
        unstable = neuron_subset
    else:
        unstable = np.where((lo < 0) & (hi > 0))[0]
    if len(unstable) == 0:
        return lo.copy(), hi.copy(), False

    is_conv = layers_np[l]['type'] == 'conv'
    model = None
    env = None

    if is_conv and use_milp:
        # Sparse per-neuron models for conv MILP — each worker builds its
        # own small model with only receptive field neurons. Gives identical
        # bounds to the full model (verified empirically) but solves ~8x faster.
        _shared_sparse_args = (layers_np, x_lo, x_hi, bounds, l)
        _shared_model = None
    elif not is_conv and not use_milp and lp_per_worker:
        # Per-worker LP for FC: each worker builds its own model from
        # scratch, avoiding model.copy() overhead and enabling true
        # parallelism.
        _shared_sparse_args = (layers_np, x_lo, x_hi, bounds, l, 'fc_lp')
        _shared_model = None
    else:
        # Full shared model for FC MILP or conv LP
        _shared_sparse_args = None
        milp_set = None if use_milp else set()
        model, env = _build_base_model(layers_np, x_lo, x_hi, bounds, l,
                                       milp_set=milp_set)
        _shared_model = model
        _shared_layer_np = layers_np[l]
        _shared_prev_layer_idx = l - 1

    new_lo, new_hi = lo.copy(), hi.copy()
    any_timeout = False

    # Witness-guided ordering: forward `witness_n_random` random + 3 corner
    # witnesses (x_lo, x_hi, midpoint) through the actual ReLU network up
    # to layer l. For each unstable neuron j, pass z_w_min[j], z_w_max[j]
    # so the worker can put the "proving stable" MILP direction first
    # (the only direction that has a chance to fire BestBdStop early).
    z_w_min = z_w_max = None
    if witness_n_random > 0:
        rng = np.random.default_rng(7)
        rand = (x_lo[None, :] +
                rng.random((witness_n_random, x_lo.size)) * (x_hi - x_lo))
        witnesses = np.vstack([
            rand,
            x_lo[None, :],
            x_hi[None, :],
            ((x_lo + x_hi) / 2)[None, :],
        ]).astype(np.float64)
        z_w = _forward_witnesses_layered(layers_np, witnesses, l)
        z_w_min = z_w.min(axis=0); z_w_max = z_w.max(axis=0)

    if z_w_min is not None:
        tasks = [(int(idx), timeout, float(lo[idx]), float(hi[idx]),
                  float(z_w_min[idx]), float(z_w_max[idx]))
                 for idx in unstable]
    else:
        tasks = [(int(idx), timeout, float(lo[idx]), float(hi[idx]))
                 for idx in unstable]
    chunksize = max(1, len(tasks) // (n_cores * 4))

    with multiprocessing.Pool(n_cores) as pool:
        if deadline is None:
            results = pool.map(_solve_neuron_both, tasks,
                               chunksize=chunksize)
        else:
            # Hard wall-clock cap: the per-neuron `timeout` bounds ONE
            # solve, but a 250-neuron layer is 500 solves - without
            # this the phase blows through total_timeout 2x+ (observed
            # 30s budget -> 70s walls on cora). Each partial result is
            # a valid bound, so stopping early is sound.
            results = []
            # chunksize MUST be 1: CPython's imap_unordered returns the
            # timeout-capable IMapUnorderedIterator only for chunksize=1
            # (chunked mode wraps it in a plain generator with no
            # .next(timeout)). Per-task IPC overhead is negligible next
            # to an LP/MILP solve.
            it = pool.imap_unordered(_solve_neuron_both, tasks,
                                     chunksize=1)
            while True:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    any_timeout = True
                    pool.terminate()
                    break
                try:
                    results.append(it.next(timeout=max(0.1, rem)))
                except StopIteration:
                    break
                except multiprocessing.TimeoutError:
                    any_timeout = True
                    pool.terminate()
                    break

    for idx, lb, ub, _, timed_out in results:
        if timed_out:
            any_timeout = True
        new_lo[idx] = max(new_lo[idx], lb)
        new_hi[idx] = min(new_hi[idx], ub)

    _shared_model = None
    _shared_sparse_args = None
    if model is not None:
        model.dispose()
    if env is not None:
        env.dispose()
    return new_lo, new_hi, any_timeout


# ---------------------------------------------------------------------------
# Spec MILP/LP
# ---------------------------------------------------------------------------

def _build_spec_model(layers_np, x_lo, x_hi, bounds, pred, comp,
                      milp_neurons=None, n_threads=0, lp_encoding='zas'):
    """Build spec MILP: minimize y_pred - y_comp.

    Args:
        milp_neurons: set of (layer, neuron) to encode with binaries.
                      None = all, set() = pure LP.
        n_threads: Gurobi threads (0 = auto).
        lp_encoding: 'zas' or 'compact' for LP neurons.

    Returns: (model, env)
    """
    import gurobipy as grb

    nh = len(layers_np) - 1
    milp_set = milp_neurons if milp_neurons is not None else None

    model, env = _build_base_model(
        layers_np, x_lo, x_hi, bounds, nh, milp_set=milp_set,
        lp_encoding=lp_encoding)
    model.setParam('Threads', n_threads if n_threads > 0
                    else multiprocessing.cpu_count())

    final = layers_np[nh]
    prev = f'a_{nh-1}'

    expr = grb.LinExpr()
    if final['type'] == 'fc':
        W = final['W']
        bias = final['bias']
        w_diff = W[pred] - W[comp]
        b_diff = float(bias[pred]) - float(bias[comp])
        for k in range(W.shape[1]):
            if w_diff[k] != 0:
                v = model.getVarByName(f'{prev}_{k}')
                if v is not None:
                    expr.add(v, float(w_diff[k]))
    else:
        kernel = final['kernel']
        in_shape = final['in_shape']
        stride = final['stride']
        padding = final['padding']
        conns_p = _conv_connections(pred, kernel, in_shape, stride, padding)
        conns_c = _conv_connections(comp, kernel, in_shape, stride, padding)
        diff = {}
        for fi, w in conns_p:
            diff[fi] = diff.get(fi, 0) + w
        for fi, w in conns_c:
            diff[fi] = diff.get(fi, 0) - w
        bp = float(final['bias'][
            _conv_bias_idx(pred, kernel, in_shape, stride, padding)])
        bc = float(final['bias'][
            _conv_bias_idx(comp, kernel, in_shape, stride, padding)])
        b_diff = bp - bc
        for fi, w in diff.items():
            if w != 0:
                v = model.getVarByName(f'{prev}_{fi}')
                if v is not None:
                    expr.add(v, w)

    model.setObjective(expr + b_diff, grb.GRB.MINIMIZE)
    model.setParam('BestBdStop', 1e-6)

    return model, env


# ---------------------------------------------------------------------------
# Compact spec model — dead neurons inlined as 0
# ---------------------------------------------------------------------------

def _build_spec_model_compact(layers_np, x_lo, x_hi, bounds, pred, comp,
                               milp_neurons=None, n_threads=0,
                               lp_encoding='zas'):
    """Build spec MILP with dead neurons inlined (no variable, no constraint).

    Same as _build_spec_model but skips creating variables and constraints
    for dead neurons (hi <= 0), reducing model size.

    Args:
        lp_encoding: 'zas' (z,a,s with continuous s) or 'compact' (a only, 2 constraints)

    Returns (model, env)
    """
    import gurobipy as grb

    nh = len(layers_np) - 1
    milp_set = milp_neurons if milp_neurons is not None else None

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    n_input = len(x_lo)
    for i in range(n_input):
        m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]), name=f'inp_{i}')
    m.update()

    for l in range(nh):
        layer = layers_np[l]
        lo, hi = bounds[l]
        n = _n_out(layer)
        prev = 'inp' if l == 0 else f'a_{l-1}'

        # Variables: skip dead neurons entirely
        for j in range(n):
            if hi[j] <= 0:
                pass  # dead: no variable
            elif lo[j] >= 0:
                m.addVar(lb=0.0, ub=grb.GRB.INFINITY,
                         name=f'a_{l}_{j}')
            else:
                use_milp = (milp_set is None or (l, j) in milp_set)
                if use_milp or lp_encoding == 'zas':
                    m.addVar(lb=float(lo[j]), ub=float(hi[j]),
                             name=f'z_{l}_{j}')
                    m.addVar(lb=0.0, ub=float(hi[j]), name=f'a_{l}_{j}')
                    if use_milp:
                        m.addVar(vtype=grb.GRB.BINARY, name=f's_{l}_{j}')
                    else:
                        m.addVar(lb=0.0, ub=1.0, name=f's_{l}_{j}')
                else:
                    m.addVar(lb=0.0, ub=float(hi[j]), name=f'a_{l}_{j}')
        m.update()

        # Constraints: skip dead neurons, dead predecessors contribute 0
        for j in range(n):
            if hi[j] <= 0:
                continue

            expr = grb.LinExpr()
            if layer['type'] == 'conv':
                conns = _conv_connections(
                    j, layer['kernel'], layer['in_shape'],
                    layer['stride'], layer['padding'])
                for flat_in, w in conns:
                    v = m.getVarByName(f'{prev}_{flat_in}')
                    if v is not None:
                        expr.add(v, w)
                    # else: dead predecessor, contributes 0
                b_j = float(layer['bias'][
                    _conv_bias_idx(j, layer['kernel'],
                                   layer['in_shape'],
                                   layer['stride'],
                                   layer['padding'])])
            else:
                W = layer['W']
                for k in range(W.shape[1]):
                    if W[j, k] != 0:
                        v = m.getVarByName(f'{prev}_{k}')
                        if v is not None:
                            expr.add(v, float(W[j, k]))
                b_j = float(layer['bias'][j])

            if lo[j] >= 0:
                m.addConstr(m.getVarByName(f'a_{l}_{j}') == expr + b_j)
            else:
                use_milp = (milp_set is None or (l, j) in milp_set)
                if use_milp or lp_encoding == 'zas':
                    z = m.getVarByName(f'z_{l}_{j}')
                    a = m.getVarByName(f'a_{l}_{j}')
                    s = m.getVarByName(f's_{l}_{j}')
                    lo_j, hi_j = float(lo[j]), float(hi[j])
                    m.addConstr(z == expr + b_j)
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= hi_j * s)
                    m.addConstr(a <= z - lo_j * (1 - s))
                else:
                    a = m.getVarByName(f'a_{l}_{j}')
                    lo_j, hi_j = float(lo[j]), float(hi[j])
                    m.addConstr(a >= expr + b_j)
                    slope = hi_j / (hi_j - lo_j)
                    m.addConstr(a <= slope * (expr + b_j) - slope * lo_j)
        m.update()

    # Spec objective
    final = layers_np[nh]
    prev = f'a_{nh-1}'

    expr = grb.LinExpr()
    if final['type'] == 'fc':
        W = final['W']
        bias = final['bias']
        w_diff = W[pred] - W[comp]
        b_diff = float(bias[pred]) - float(bias[comp])
        for k in range(W.shape[1]):
            if w_diff[k] != 0:
                v = m.getVarByName(f'{prev}_{k}')
                if v is not None:
                    expr.add(v, float(w_diff[k]))
    else:
        kernel = final['kernel']
        in_shape = final['in_shape']
        stride = final['stride']
        padding = final['padding']
        conns_p = _conv_connections(pred, kernel, in_shape, stride, padding)
        conns_c = _conv_connections(comp, kernel, in_shape, stride, padding)
        diff = {}
        for fi, w in conns_p:
            diff[fi] = diff.get(fi, 0) + w
        for fi, w in conns_c:
            diff[fi] = diff.get(fi, 0) - w
        bp = float(final['bias'][
            _conv_bias_idx(pred, kernel, in_shape, stride, padding)])
        bc = float(final['bias'][
            _conv_bias_idx(comp, kernel, in_shape, stride, padding)])
        b_diff = bp - bc
        for fi, w in diff.items():
            if w != 0:
                v = m.getVarByName(f'{prev}_{fi}')
                if v is not None:
                    expr.add(v, w)

    m.setObjective(expr + b_diff, grb.GRB.MINIMIZE)
    m.setParam('BestBdStop', 1e-6)
    m.setParam('Threads', n_threads if n_threads > 0
                else multiprocessing.cpu_count())

    return m, env




# ---------------------------------------------------------------------------
# Racing escalation workers
# ---------------------------------------------------------------------------

def _solve_spec_worker(args):
    """Worker: build cascading MILP from scratch and solve.

    Builds a compact model (dead neurons skipped, LP neurons use 2-constraint
    triangle with no z/s vars) and solves in either feasibility or
    optimization mode.

    Args tuple: (mode, layers_np, x_lo, x_hi, bounds, pred, comp,
                  scored_keys, n_bins, n_threads, timeout)
        mode: 'feasibility' or 'optimize'

    Returns: (result_str, time, bound_or_None)
        result_str: 'SAT', 'UNSAT', or 'UNKNOWN'
    """
    mode, layers_np, x_lo, x_hi, bounds, pred, comp, \
        scored_keys, n_bins, n_threads, timeout = args
    import gurobipy as grb

    milp_set = set(scored_keys[:n_bins]) if n_bins > 0 else set()
    milp_by_layer = {}
    for (l, i) in milp_set:
        milp_by_layer.setdefault(l, set()).add(i)

    nh = len(layers_np) - 1
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)
    # Loosen Gurobi's default FeasibilityTol (1e-6) — zono forward in
    # float32 produces bounds tight to ~1 ulp, and the float64 LP
    # arithmetic computes `expr + b_j` 1-3 ulp outside those bounds,
    # which is enough to declare the model spuriously INFEASIBLE.
    # Caught on metaroom_2023 (4cnn_ry_99_16 / spec_43): 1/10 runs
    # returned 'verified' on a real SAT case via racing
    # `feasibility UNSAT @ bins=0`. 1e-5 absorbs the float32→float64
    # round-trip without loosening the relaxation in any
    # algorithmically-meaningful way (the gap is at the precision floor).
    m.setParam('FeasibilityTol', _GUROBI_FEAS_TOL)

    # Input variables
    n_input = len(x_lo)
    xv = [m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
          for i in range(n_input)]
    m.update()
    prev_vars = list(xv)

    # Hidden layers
    for l in range(nh):
        layer = layers_np[l]
        lo, hi = bounds[l]
        ms = milp_by_layer.get(l, set())
        n = _n_out(layer)
        new_vars = []

        for j in range(n):
            if hi[j] <= 0:
                # Dead neuron: no variable, None placeholder
                new_vars.append(None)
                continue

            # Build affine expression
            expr = grb.LinExpr()
            if layer['type'] == 'conv':
                conns = _conv_connections(
                    j, layer['kernel'], layer['in_shape'],
                    layer['stride'], layer['padding'])
                for fi, w in conns:
                    if prev_vars[fi] is not None:
                        expr.add(prev_vars[fi], w)
                b_j = float(layer['bias'][
                    _conv_bias_idx(j, layer['kernel'],
                                   layer['in_shape'],
                                   layer['stride'],
                                   layer['padding'])])
            else:
                W = layer['W']
                for k in range(W.shape[1]):
                    if W[j, k] != 0 and prev_vars[k] is not None:
                        expr.add(prev_vars[k], float(W[j, k]))
                b_j = float(layer['bias'][j])

            if lo[j] >= 0:
                # Active: a = z.
                a = m.addVar(lb=float(lo[j]), ub=float(hi[j]))
                m.update()
                m.addConstr(a == expr + b_j)
                new_vars.append(a)
            elif j in ms:
                # Binary MILP encoding
                z = m.addVar(lb=float(lo[j]), ub=float(hi[j]))
                a = m.addVar(lb=0.0, ub=float(hi[j]))
                s = m.addVar(vtype=grb.GRB.BINARY)
                m.update()
                m.addConstr(z == expr + b_j)
                m.addConstr(a >= 0)
                m.addConstr(a >= z)
                m.addConstr(a <= float(hi[j]) * s)
                m.addConstr(a <= z - float(lo[j]) * (1 - s))
                new_vars.append(a)
            else:
                # LP triangle: 1 var, 2 constraints (no z/s)
                a = m.addVar(lb=0.0, ub=float(hi[j]))
                m.update()
                m.addConstr(a >= expr + b_j)
                slope = float(hi[j]) / (float(hi[j]) - float(lo[j]))
                m.addConstr(a <= slope * (expr + b_j)
                            - slope * float(lo[j]))
                new_vars.append(a)

        prev_vars = new_vars
    m.update()

    # Spec expression: y_pred - y_comp
    final = layers_np[nh]
    spec_expr = grb.LinExpr()
    if final['type'] == 'fc':
        W = final['W']
        bias = final['bias']
        w_diff = W[pred] - W[comp]
        b_diff = float(bias[pred]) - float(bias[comp])
        for k in range(W.shape[1]):
            if w_diff[k] != 0 and prev_vars[k] is not None:
                spec_expr.add(prev_vars[k], float(w_diff[k]))
    else:
        kernel = final['kernel']
        in_shape = final['in_shape']
        stride = final['stride']
        padding = final['padding']
        conns_p = _conv_connections(pred, kernel, in_shape, stride, padding)
        conns_c = _conv_connections(comp, kernel, in_shape, stride, padding)
        diff = {}
        for fi, w in conns_p:
            diff[fi] = diff.get(fi, 0) + w
        for fi, w in conns_c:
            diff[fi] = diff.get(fi, 0) - w
        bp = float(final['bias'][
            _conv_bias_idx(pred, kernel, in_shape, stride, padding)])
        bc = float(final['bias'][
            _conv_bias_idx(comp, kernel, in_shape, stride, padding)])
        b_diff = bp - bc
        for fi, w in diff.items():
            if w != 0 and prev_vars[fi] is not None:
                spec_expr.add(prev_vars[fi], w)

    m.setParam('TimeLimit', timeout)
    t0 = time.perf_counter()

    if mode == 'feasibility':
        m.addConstr(spec_expr + b_diff <= 0, 'spec_leq0')
        m.update()
        optimize_checked(m)
        dt = time.perf_counter() - t0
        if m.Status == 2:
            result = 'SAT'
        elif m.Status == 3:
            result = 'UNSAT'
        else:
            result = 'UNKNOWN'
        m.dispose()
        env.dispose()
        return result, dt, None
    else:
        m.setParam('BestBdStop', 0.0)
        m.setObjective(spec_expr + b_diff, grb.GRB.MINIMIZE)
        m.update()
        optimize_checked(m)
        dt = time.perf_counter() - t0
        lb = None
        if m.Status in (2, 15):
            lb = m.ObjBound
            result = 'UNSAT' if lb > 0 else 'SAT'
        elif m.Status == 9 and m.SolCount > 0:
            lb = m.ObjBound
            result = 'SAT'
        else:
            result = 'UNKNOWN'
        m.dispose()
        env.dispose()
        return result, dt, lb


def _racing_escalation(layers_np, x_lo, x_hi, bounds_np, pred, comp,
                        sorted_neurons, n_cores, time_left_fn,
                        print_progress=False):
    """Race feasibility (1 thread) vs optimization (n-1 threads) per bin level.

    At each level, spawn both workers. First to finish wins:
    - Feasibility SAT → bin count insufficient, escalate immediately
    - Feasibility UNSAT → verified (spec > 0 always)
    - Optimization UNSAT (lb > 0) → verified
    - Optimization SAT → escalate

    Returns (verified: bool, n_binaries: int).
    """
    # Bin schedule: 0, 2, 4, 8, 16, 32, 64, 128, 256, ...
    # For small nets the EXACT MILP (all unstable binarized) solves in well
    # under a second, so the gradual escalation just wastes a full
    # Pool(2)+Gurobi-env spawn (~0.4s local, ~3.7s on slower boxes) on each
    # intermediate level — all of which return loose-relaxation
    # "feasibility SAT" until the exact level anyway. Below
    # `_DIRECT_EXACT_MAX_UNSTABLE` neurons, go straight to the exact level
    # (one solve). This is what made safenlp_2024 time out on server1
    # (reached only bins=16 in 21s) while the exact MILP is <1s.
    n_uns = len(sorted_neurons)
    # Only short-circuit for SHALLOW pure-FC nets. Number of ReLU layers =
    # affine layers minus the (relu-free) output layer.
    _is_fc_only = not any(l['type'] == 'conv' for l in layers_np)
    _n_relu_layers = sum(
        1 for l in layers_np if l['type'] in ('fc', 'conv')) - 1
    # For a 1-2 relu-layer FC net the full-binary MILP is a near-flat big-M
    # system Gurobi cracks in <1s, so the gradual escalation just wastes a
    # Pool(2)+Gurobi-env spawn per intermediate level. For DEEPER nets (e.g.
    # acasxu's 6 relu layers) the full-binary MILP is combinatorially hard and
    # the gradual schedule instead verifies at an early intermediate bin count
    # — forcing direct-exact there makes it time out to 'unknown' (regressed
    # the acasxu sequential-vs-graph equivalence test). The n_unstable cap is a
    # secondary guard (one very wide layer can still be slow).
    if (_is_fc_only and _n_relu_layers <= 2
            and 0 < n_uns <= _DIRECT_EXACT_MAX_UNSTABLE):
        bin_schedule = [n_uns]
    else:
        bin_schedule = [0]
        b = 2
        while b <= n_uns:
            bin_schedule.append(b)
            b *= 2
        if bin_schedule[-1] < n_uns:
            bin_schedule.append(n_uns)

    opt_threads = max(1, n_cores - 1)

    for n_bins in bin_schedule:
        tl = time_left_fn()
        if tl <= 0:
            break

        feas_args = ('feasibility', layers_np, x_lo, x_hi, bounds_np,
                     pred, comp, sorted_neurons, n_bins, 1, tl)
        opt_args = ('optimize', layers_np, x_lo, x_hi, bounds_np,
                    pred, comp, sorted_neurons, n_bins, opt_threads, tl)

        pool = multiprocessing.Pool(2)
        async_feas = pool.apply_async(_solve_spec_worker, (feas_args,))
        async_opt = pool.apply_async(_solve_spec_worker, (opt_args,))

        winner = None
        while True:
            if async_feas.ready():
                feas_result, feas_dt, _ = async_feas.get()
                if feas_result == 'SAT':
                    if print_progress:
                        print(f'  Racing bins={n_bins}: '
                              f'feasibility SAT ({feas_dt:.1f}s) → escalate')
                    pool.terminate()
                    pool.join()
                    winner = 'escalate'
                    break
                elif feas_result == 'UNSAT':
                    if print_progress:
                        print(f'  Racing bins={n_bins}: '
                              f'feasibility UNSAT ({feas_dt:.1f}s) → verified')
                    pool.terminate()
                    pool.join()
                    return True, n_bins
                else:
                    # Same fix as `_racing_escalation_graph` — UNKNOWN /
                    # TIMEOUT is NOT a verification signal.
                    if print_progress:
                        print(f'  Racing bins={n_bins}: '
                              f'feasibility {feas_result} ({feas_dt:.1f}s) → escalate')
                    pool.terminate()
                    pool.join()
                    winner = 'escalate'
                    break
            if async_opt.ready():
                opt_result, opt_dt, opt_lb = async_opt.get()
                if opt_result == 'UNSAT':
                    if print_progress:
                        lb_s = f'{opt_lb:.4f}' if opt_lb is not None else '?'
                        print(f'  Racing bins={n_bins}: '
                              f'optimization lb={lb_s} ({opt_dt:.1f}s) '
                              f'→ verified')
                    pool.terminate()
                    pool.join()
                    return True, n_bins
                else:
                    lb_s = f'{opt_lb:.4f}' if opt_lb is not None else '?'
                    if print_progress:
                        print(f'  Racing bins={n_bins}: '
                              f'optimization lb={lb_s} ({opt_dt:.1f}s) '
                              f'→ escalate')
                    pool.terminate()
                    pool.join()
                    winner = 'escalate'
                    break
            # Deadline guard: a worker can run past the budget while BUILDING
            # the spec MILP (Gurobi's TimeLimit bounds optimize(), not model
            # construction — a CNN7's unfolded-conv MILP takes tens of
            # seconds to build). Without this the racing loop spins until the
            # outer shell `timeout` SIGKILLs the process mid-build, before it
            # can write the results file (→ a NOFILE rather than a clean
            # 'unknown'). Terminate the workers and return not-verified once
            # the deadline passes.
            if time_left_fn() <= 1.0:
                pool.terminate()
                pool.join()
                return False, n_bins
            time.sleep(0.05)

    return False, bin_schedule[-1] if bin_schedule else 0


# ---------------------------------------------------------------------------
# Graph per-neuron tightening worker
# ---------------------------------------------------------------------------

_graph_tighten_args = None
_graph_shared_model = None
_graph_shared_target_indices = None


def _inflate_milp_bounds(bounds_by_relu, atol, rtol):
    """Outward-inflate pre-ReLU bounds for floating-point soundness.

    The spec MILP/LP imposes `(lo, hi)` as *hard* variable bounds, but it
    recomputes the affine in float64 while the bounds came from float32
    (zono/CROWN). For near-degenerate neurons (lo≈hi, e.g. a tiny perturbation
    box) that float32↔float64 gap can exceed the bound width, so a genuinely
    reachable point lands just outside [lo,hi] and the relaxation falsely
    excludes it → false `verified`. Widening by `tol = atol + rtol·max|bound|`
    keeps the over-approximation sound (a larger feasible set can only make the
    feasibility LP *less* infeasible — never a false-verify). See
    `settings.milp_bound_inflation_{atol,rtol}`.

    The inflation MUST NOT flip a neuron's active/dead classification. Flipping
    is sound but turns a trivially-stable neuron into a binarised one, and on
    integer-weighted nets (sat_relu) MANY always-active neurons have lo == 0
    exactly — lo - tol < 0 would reclassify every one as unstable, exploding the
    spec MILP from ~250 to ~320 degenerate binaries (60 s timeout vs 0.2 s). So
    clamp: an active neuron (lo >= 0) keeps lo_new >= 0, a dead neuron (hi <= 0)
    keeps hi_new <= 0. The clamped bound still contains the true reachable range
    (max(lo-tol,0) <= lo), so soundness is preserved; only the FP-noise slack
    below 0 is dropped for an already-active neuron, where it is not needed.
    """
    if atol <= 0 and rtol <= 0:
        return bounds_by_relu
    out = {}
    for li, (lo, hi) in bounds_by_relu.items():
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        tol = atol + rtol * np.maximum(np.abs(lo), np.abs(hi))
        lo_new = lo - tol
        hi_new = hi + tol
        # Preserve active (lo>=0) and dead (hi<=0) classification.
        lo_new = np.where(lo >= 0, np.maximum(lo_new, 0.0), lo_new)
        hi_new = np.where(hi <= 0, np.minimum(hi_new, 0.0), hi_new)
        out[li] = (lo_new, hi_new)
    return out


def _compute_dead_at(gg_ops, bounds_by_relu):
    """Compute dead neuron mask for each op in the graph.

    Propagates backward: relu dead → through reshape/sub/add → mark inputs.
    If ALL outputs of a conv/fc are dead, mark its input as fully dead too.
    """
    dead_at = {}
    for op in gg_ops:
        if op['type'] == 'relu' and 'layer_idx' in op:
            li = op['layer_idx']
            if li not in bounds_by_relu:
                # Layer bounds not yet computed (e.g., interleaved
                # forward builder LP-probing at an earlier layer); skip
                # this relu's dead mask — it only affects downstream.
                continue
            _, hi = bounds_by_relu[li]
            dead_at[op['name']] = (hi <= 0)

    consumer_count = {}
    for op in gg_ops:
        for inp in op.get('inputs', []):
            consumer_count[inp] = consumer_count.get(inp, 0) + 1

    for op in reversed(gg_ops):
        name = op['name']
        if name not in dead_at:
            continue
        my_dead = dead_at[name]
        if op['type'] in ('relu', 'reshape', 'sub'):
            inp = op['inputs'][0]
            if consumer_count.get(inp, 1) == 1:
                dead_at[inp] = my_dead
            else:
                if inp in dead_at:
                    dead_at[inp] = dead_at[inp] & my_dead
                else:
                    dead_at[inp] = my_dead.copy()
        elif op['type'] in ('conv', 'fc'):
            # If ALL outputs are dead, predecessor is fully dead
            if my_dead.all():
                inp = op['inputs'][0]
                if consumer_count.get(inp, 1) == 1:
                    dead_at.setdefault(inp, np.ones_like(my_dead, dtype=bool))
        elif op['type'] == 'add' and op.get('is_merge'):
            for inp in op['inputs']:
                if consumer_count.get(inp, 1) == 1:
                    dead_at[inp] = my_dead
                else:
                    if inp in dead_at:
                        dead_at[inp] = dead_at[inp] & my_dead
                    else:
                        dead_at[inp] = my_dead.copy()
    return dead_at


def _build_graph_model_to_relu(gg_ops, x_lo, x_hi, bounds_by_relu,
                                 target_layer_idx, input_name, fork_pts,
                                 use_milp=False):
    """Build a graph LP/MILP model up to (and including the linear op before) a target relu.

    When use_milp=True, prior relu layers get binary encoding (exact MILP).
    When use_milp=False, all relus get LP triangle encoding.

    Returns (model, env, target_vars_list) where target_vars_list[j] is the
    Gurobi variable for neuron j's pre-relu value (or None if dead).
    """
    import gurobipy as grb

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    inp_vars = [m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
                for i in range(len(x_lo))]
    m.update()

    dead_at = _compute_dead_at(gg_ops, bounds_by_relu)

    # Find target relu's input op name
    target_input_name = None
    for op in gg_ops:
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            target_input_name = op['inputs'][0]
            break

    op_var_refs = {input_name: inp_vars}
    target_vars = None

    for op in gg_ops:
        nm = op['name']
        t = op['type']

        if t in ('conv', 'fc'):
            prev = op_var_refs[op['inputs'][0]]
            n_prev = len(prev)
            if t == 'conv':
                n_out = op['n_out']
                spatial = op['out_shape'][1] * op['out_shape'][2]
            else:
                n_out = op['W_np'].shape[0]

            if t == 'conv':
                W_sp = op.get('W_sp')
                if W_sp is None:
                    W_sp = _conv_sparse_matrix(
                        op['kernel_np'], op['in_shape'], op['stride'], op['padding'])

            my_dead = dead_at.get(nm)
            if my_dead is not None and len(my_dead) != n_out:
                my_dead = None
            all_prev_dead = all(p is None for p in prev)

            out = []
            for j in range(n_out):
                if my_dead is not None and my_dead[j]:
                    out.append(None)
                    continue
                if t == 'conv':
                    b_j = float(op['bias_np'][j // spatial])
                else:
                    b_j = float(op['bias_np'][j])
                if all_prev_dead:
                    # Bug #1 fix: dead inputs → output is bias[j].
                    out.append(m.addVar(lb=b_j, ub=b_j))
                    continue
                expr = grb.LinExpr()
                if t == 'conv':
                    row = W_sp.getrow(j)
                    for fi, w in zip(row.indices, row.data):
                        if fi < n_prev and prev[fi] is not None:
                            expr.add(prev[fi], float(w))
                else:
                    W = op['W_np']
                    for k in range(n_prev):
                        if W[j, k] != 0 and prev[k] is not None:
                            expr.add(prev[k], float(W[j, k]))
                if expr.size() == 0:
                    # Bug #1 fix: every live input has zero weight.
                    out.append(m.addVar(lb=b_j, ub=b_j))
                    continue
                v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                out.append(v)
                m.addConstr(v == expr + b_j)
            m.update()
            op_var_refs[nm] = out
            if nm == target_input_name:
                target_vars = out
                break

        elif t == 'relu':
            if 'layer_idx' not in op:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]
                continue
            li = op['layer_idx']
            lo_r, hi_r = bounds_by_relu[li]
            prev = op_var_refs[op['inputs'][0]]
            out = []
            for j in range(len(prev)):
                if hi_r[j] <= 0:
                    out.append(None)
                elif lo_r[j] >= 0:
                    a = m.addVar(lb=float(lo_r[j]), ub=float(hi_r[j]))
                    m.addConstr(a == prev[j])
                    out.append(a)
                elif use_milp:
                    # MILP: exact binary encoding
                    a = m.addVar(lb=0.0, ub=float(hi_r[j]))
                    s = m.addVar(vtype=grb.GRB.BINARY)
                    z = prev[j]
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= float(hi_r[j]) * s)
                    m.addConstr(a <= z - float(lo_r[j]) * (1 - s))
                    out.append(a)
                else:
                    # LP: triangle relaxation
                    a = m.addVar(lb=0.0, ub=float(hi_r[j]))
                    m.addConstr(a >= prev[j])
                    slope = float(hi_r[j]) / (float(hi_r[j]) - float(lo_r[j]))
                    m.addConstr(a <= slope * prev[j] - slope * float(lo_r[j]))
                    out.append(a)
            m.update()
            op_var_refs[nm] = out

        elif t == 'add':
            if op.get('is_merge'):
                va = op_var_refs[op['inputs'][0]]
                vb = op_var_refs[op['inputs'][1]]
                out = []
                for j in range(len(va)):
                    if va[j] is None and vb[j] is None:
                        out.append(None)
                    else:
                        expr = grb.LinExpr()
                        if va[j] is not None:
                            expr.add(va[j], 1.0)
                        if vb[j] is not None:
                            expr.add(vb[j], 1.0)
                        v = m.addVar(lb=-grb.GRB.INFINITY,
                                     ub=grb.GRB.INFINITY)
                        m.addConstr(v == expr)
                        out.append(v)
                m.update()
                op_var_refs[nm] = out
                if nm == target_input_name:
                    target_vars = out
                    break
            else:
                # Same fix as `_solve_spec_graph_worker` — apply the
                # bias of a non-merge add op. Silent bias-drop here
                # encoded the whole network bias-less for ACASXU.
                prev = op_var_refs[op['inputs'][0]]
                bias = op.get('bias')
                if bias is None:
                    op_var_refs[nm] = prev
                else:
                    bias_flat = bias.flatten().astype(np.float64)
                    out = []
                    for j in range(len(prev)):
                        if prev[j] is None:
                            v = m.addVar(lb=float(bias_flat[j]),
                                          ub=float(bias_flat[j]))
                            out.append(v)
                        else:
                            v = m.addVar(lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY)
                            m.addConstr(v == prev[j] + float(bias_flat[j]))
                            out.append(v)
                    m.update()
                    op_var_refs[nm] = out
                # An ACASXU-style network's target_input_name is the
                # bias-add op feeding the next ReLU. Have to capture
                # target_vars + break here too (the merge-add branch
                # above did this for ResNets; the non-merge add path
                # was silently skipping it).
                if nm == target_input_name:
                    target_vars = op_var_refs[nm]
                    break

        elif t == 'sub':
            # Sub with constant: create shifted variables
            prev = op_var_refs[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                import gurobipy as grb
                bias_flat = bias.flatten().astype(np.float64)
                out = []
                for j in range(len(prev)):
                    if prev[j] is None:
                        out.append(None)
                    else:
                        v = m.addVar(lb=-grb.GRB.INFINITY,
                                     ub=grb.GRB.INFINITY)
                        m.addConstr(v == prev[j] - float(bias_flat[j]))
                        out.append(v)
                m.update()
                op_var_refs[nm] = out
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'reshape':
            op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        else:
            raise NotImplementedError(
                f'milp graph builder: unsupported op {t!r} at {nm!r} — '
                f'skipping it would encode a different network')

    m.update()
    return m, env, target_vars


def _tighten_neuron_graph(args):
    """Worker: copy shared model, solve one neuron's min/max bounds."""
    j, timeout, cur_lo, cur_hi = args[:4]
    import gurobipy as grb

    # Use shared model (COW via fork) — copy is ~1ms
    m = _graph_shared_model.copy()
    var_idx = _graph_shared_target_indices[j]

    lb, ub = cur_lo, cur_hi

    if var_idx >= 0:
        tv = m.getVars()[var_idx]
        m.setParam('TimeLimit', timeout)

        # Minimize (tighter side first for early exit)
        if abs(cur_lo) < abs(cur_hi):
            m.setObjective(tv, grb.GRB.MINIMIZE)
            m.setParam('BestBdStop', 1e-6)
        else:
            m.setObjective(tv, grb.GRB.MAXIMIZE)
            m.setParam('BestBdStop', -1e-6)
        optimize_checked(m)
        try:
            b = m.ObjBound
        except (grb.GurobiError, AttributeError):
            b = None
        if b is not None:
            if m.ModelSense == 1:
                lb = max(lb, b)
            else:
                ub = min(ub, b)

        if lb < 0 and ub > 0:
            m.reset()
            if abs(cur_lo) < abs(cur_hi):
                m.setObjective(tv, grb.GRB.MAXIMIZE)
                m.setParam('BestBdStop', -1e-6)
            else:
                m.setObjective(tv, grb.GRB.MINIMIZE)
                m.setParam('BestBdStop', 1e-6)
            optimize_checked(m)
            try:
                b = m.ObjBound
            except (grb.GurobiError, AttributeError):
                b = None
            if b is not None:
                if m.ModelSense == 1:
                    lb = max(lb, b)
                else:
                    ub = min(ub, b)

    m.dispose()
    return j, lb, ub


# ---------------------------------------------------------------------------
# Graph MILP model building
# ---------------------------------------------------------------------------

def _solve_spec_graph_worker(args):
    """Worker: build graph-aware MILP from scratch and solve.

    Uses direct variable references (not getVarByName) for speed.
    Dead neurons are skipped entirely (compact encoding).

    Returns: (result_str, time, bound_or_None)
    """
    (mode, gg_ops, x_lo, x_hi, bounds_by_relu, query_w, query_bias,
     scored_keys, n_bins, n_threads, timeout, input_name, fork_pts) = args
    import gurobipy as grb

    milp_set = set(scored_keys[:n_bins]) if n_bins > 0 else set()
    milp_by_layer = {}
    for (li, ni) in milp_set:
        milp_by_layer.setdefault(li, set()).add(ni)

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    # Input variables — direct list, no names needed
    n_input = len(x_lo)
    inp_vars = [m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
                for i in range(n_input)]
    m.update()

    # Track output variable refs: name -> list of (grb.Var or None)
    op_var_refs = {input_name: inp_vars}
    last_op_name = None

    # Build successor map and precompute dead masks
    succ_map = {}
    for op2 in gg_ops:
        for inp in op2.get('inputs', []):
            succ_map.setdefault(inp, []).append(op2)

    dead_at = _compute_dead_at(gg_ops, bounds_by_relu)

    for op in gg_ops:
        name = op['name']
        t = op['type']
        last_op_name = name

        if t in ('conv', 'fc'):
            prev = op_var_refs[op['inputs'][0]]
            n_prev = len(prev)

            if t == 'conv':
                kernel = op['kernel_np']
                bias_np = op['bias_np']
                in_shape = op['in_shape']
                stride = op['stride']
                padding = op['padding']
                n_out = op['n_out']
            else:
                W = op['W_np']
                bias_np = op['bias_np']
                n_out = W.shape[0]

            my_dead = dead_at.get(name)
            if my_dead is not None and len(my_dead) != n_out:
                my_dead = None

            if t == 'conv':
                W_sp = op.get('W_sp')
                if W_sp is None:
                    W_sp = _conv_sparse_matrix(kernel, in_shape, stride, padding)
                spatial = op['out_shape'][1] * op['out_shape'][2]

            all_prev_dead = all(p is None for p in prev)

            out = []
            for j in range(n_out):
                if my_dead is not None and my_dead[j]:
                    out.append(None)
                    continue

                if t == 'conv':
                    b_j = float(bias_np[j // spatial])
                else:
                    b_j = float(bias_np[j])

                if all_prev_dead:
                    # Bug #1 fix: all inputs dead → output is `bias[j]`,
                    # not zero. Emit a fixed-bound variable so downstream
                    # adds, subs, and the spec expression still see the
                    # constant contribution.
                    out.append(m.addVar(lb=b_j, ub=b_j))
                    continue

                expr = grb.LinExpr()
                if t == 'conv':
                    row = W_sp.getrow(j)
                    for fi, w in zip(row.indices, row.data):
                        if fi < n_prev and prev[fi] is not None:
                            expr.add(prev[fi], float(w))
                else:
                    for k in range(n_prev):
                        if W[j, k] != 0 and prev[k] is not None:
                            expr.add(prev[k], float(W[j, k]))

                if expr.size() == 0:
                    # Bug #1 fix: every live input has zero weight → the
                    # output collapses to the bias constant.
                    out.append(m.addVar(lb=b_j, ub=b_j))
                    continue

                v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                out.append(v)
                m.addConstr(v == expr + b_j)
            m.update()
            op_var_refs[name] = out

        elif t == 'relu':
            if 'layer_idx' not in op:
                op_var_refs[name] = op_var_refs[op['inputs'][0]]
                continue

            layer_idx = op['layer_idx']
            lo, hi = bounds_by_relu[layer_idx]
            prev = op_var_refs[op['inputs'][0]]
            n = len(prev)
            ms = milp_by_layer.get(layer_idx, set())

            out = []
            for j in range(n):
                if hi[j] <= 0:
                    out.append(None)  # dead
                elif lo[j] >= 0:
                    # Active: pass through
                    a = m.addVar(lb=float(lo[j]), ub=float(hi[j]))
                    m.addConstr(a == prev[j])
                    out.append(a)
                elif j in ms:
                    # Binary
                    a = m.addVar(lb=0.0, ub=float(hi[j]))
                    s = m.addVar(vtype=grb.GRB.BINARY)
                    z = prev[j]
                    m.addConstr(a >= 0); m.addConstr(a >= z)
                    m.addConstr(a <= float(hi[j]) * s)
                    m.addConstr(a <= z - float(lo[j]) * (1 - s))
                    out.append(a)
                else:
                    # LP triangle
                    a = m.addVar(lb=0.0, ub=float(hi[j]))
                    z = prev[j]
                    m.addConstr(a >= z)
                    slope = float(hi[j]) / (float(hi[j]) - float(lo[j]))
                    m.addConstr(a <= slope * z - slope * float(lo[j]))
                    out.append(a)
            m.update()
            op_var_refs[name] = out

        elif t == 'add':
            if op.get('is_merge'):
                va_list = op_var_refs[op['inputs'][0]]
                vb_list = op_var_refs[op['inputs'][1]]
                n = len(va_list)
                # Fast path: if both inputs are all-dead, output is all-dead
                if all(v is None for v in va_list) and all(v is None for v in vb_list):
                    op_var_refs[name] = [None] * n
                    continue
                out = []
                for j in range(n):
                    if va_list[j] is None and vb_list[j] is None:
                        out.append(None)
                    else:
                        expr = grb.LinExpr()
                        if va_list[j] is not None:
                            expr.add(va_list[j], 1.0)
                        if vb_list[j] is not None:
                            expr.add(vb_list[j], 1.0)
                        v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                        m.addConstr(v == expr)
                        out.append(v)
                m.update()
                op_var_refs[name] = out
            else:
                # Constant bias add (non-merge add with a `bias` field —
                # common ACASXU/safenlp pattern where ONNX exports the
                # bias as a separate Add op after Gemm). Forwarding the
                # input vars unchanged would silently drop the bias →
                # entire network encoded bias-less → wildly wrong y
                # bounds → falsely-infeasible spec LPs (observed: prop_2
                # `4_4` got "feas UNSAT" in 0.0s even though brute-force
                # found 500 real SAT witnesses).
                prev = op_var_refs[op['inputs'][0]]
                bias = op.get('bias')
                if bias is None:
                    op_var_refs[name] = prev
                else:
                    bias_flat = bias.flatten().astype(np.float64)
                    out = []
                    for j in range(len(prev)):
                        if prev[j] is None:
                            # all-dead upstream: emit a constant var.
                            v = m.addVar(lb=float(bias_flat[j]),
                                          ub=float(bias_flat[j]))
                            out.append(v)
                        else:
                            v = m.addVar(lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY)
                            m.addConstr(v == prev[j] + float(bias_flat[j]))
                            out.append(v)
                    m.update()
                    op_var_refs[name] = out

        elif t == 'sub':
            prev = op_var_refs[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bias_flat = bias.flatten().astype(np.float64)
                out = []
                for j in range(len(prev)):
                    if prev[j] is None:
                        out.append(None)
                    else:
                        v = m.addVar(lb=-grb.GRB.INFINITY,
                                     ub=grb.GRB.INFINITY)
                        m.addConstr(v == prev[j] - float(bias_flat[j]))
                        out.append(v)
                m.update()
                op_var_refs[name] = out
            else:
                op_var_refs[name] = op_var_refs[op['inputs'][0]]

        elif t == 'reshape':
            op_var_refs[name] = op_var_refs[op['inputs'][0]]

        else:
            raise NotImplementedError(
                f'milp spec worker: unsupported op {t!r} at {name!r} — '
                f'skipping it would encode a different network')

    m.update()

    # Spec. `query_w`/`query_bias` are EITHER a single halfspace (1-D w,
    # scalar b — the common case) OR a CONJUNCTION of halfspaces (2-D w with
    # one row per conjunct, 1-D b). A conjunctive disjunct's unsafe region is
    # the INTERSECTION of its halfspaces (e.g. sat_relu's "Y_0>=1 AND Y_1<=0"),
    # which is often empty even when each halfspace alone is reachable — so
    # feasibility must enforce ALL of them jointly. optimize/score act on the
    # first halfspace only (proving it empty still proves the intersection
    # empty — a sound shortcut; the joint feasibility solve is the general one).
    last_vars = op_var_refs[last_op_name]
    qw_arr = np.atleast_2d(np.asarray(query_w, dtype=np.float64))
    qb_arr = np.atleast_1d(np.asarray(query_bias, dtype=np.float64))

    def _halfspace_expr(h):
        e = grb.LinExpr()
        row = qw_arr[h]
        for j in range(row.shape[0]):
            if row[j] != 0 and j < len(last_vars) and last_vars[j] is not None:
                e.add(last_vars[j], float(row[j]))
        return e

    spec_expr = _halfspace_expr(0)

    m.setParam('TimeLimit', timeout)
    t0 = time.perf_counter()

    if mode == 'feasibility':
        for h in range(qw_arr.shape[0]):
            m.addConstr(_halfspace_expr(h) + float(qb_arr[h]) <= 0)
        m.update(); optimize_checked(m)
        dt = time.perf_counter() - t0
        result = ('SAT' if m.Status == 2 else
                  'UNSAT' if m.Status == 3 else 'UNKNOWN')
        m.dispose(); env.dispose()
        return result, dt, None
    elif mode == 'score':
        # Solve LP, extract fractional scores for scoring neurons
        m.setObjective(spec_expr + float(qb_arr[0]), grb.GRB.MINIMIZE)
        m.update(); optimize_checked(m)
        dt = time.perf_counter() - t0
        scores = {}
        if m.Status == 2:
            # Read fractional values at each relu layer
            for op in gg_ops:
                if op['type'] != 'relu' or 'layer_idx' not in op:
                    continue
                li = op['layer_idx']
                lo_r, hi_r = bounds_by_relu[li]
                inp_name = op['inputs'][0]
                prev = op_var_refs.get(inp_name, [])
                relu_out = op_var_refs.get(op['name'], [])
                for j in range(len(relu_out)):
                    if relu_out[j] is None or lo_r[j] >= 0 or hi_r[j] <= 0:
                        continue
                    try:
                        a_val = relu_out[j].X
                        z_val = prev[j].X if prev[j] is not None else 0
                        frac = abs(a_val - max(0.0, z_val))
                        scores[(li, j)] = frac
                    except (grb.GurobiError, AttributeError):
                        # .X unavailable when no MIP feasible solution exists;
                        # skip scoring this unstable ReLU.
                        pass
        m.dispose(); env.dispose()
        return 'SCORED', dt, scores
    else:
        m.setParam('BestBdStop', 0.0)
        m.setObjective(spec_expr + float(qb_arr[0]), grb.GRB.MINIMIZE)
        m.update(); optimize_checked(m)
        dt = time.perf_counter() - t0
        lb = None
        if m.Status in (2, 15):
            lb = m.ObjBound
            result = 'UNSAT' if lb > 0 else 'SAT'
        elif m.Status == 9 and m.SolCount > 0:
            lb = m.ObjBound; result = 'SAT'
        else:
            result = 'UNKNOWN'
        m.dispose(); env.dispose()
        return result, dt, lb


def _racing_escalation_graph(gg_ops, x_lo, x_hi, bounds_by_relu,
                              query_w, query_bias, scored_keys, n_cores,
                              time_left_fn, input_name, fork_pts,
                              print_progress=False, feas_only=False):
    """Racing escalation for graph networks.

    `feas_only=True` runs ONLY the feasibility worker, with all cores. Use it
    for a CONJUNCTIVE disjunct (multiple halfspaces): the optimize worker
    minimises only the FIRST halfspace's margin, which for a conjunction is
    reachable on its own (never verifies) — so it is pure waste, and the
    single-thread feasibility worker that DOES prove the joint MILP infeasible
    is starved of cores (sat_relu unsat_v65: 96.6 s single-threaded → times
    out). feas_only hands it every core.
    """
    bin_schedule = [0]
    b = 2
    while b <= len(scored_keys):
        bin_schedule.append(b)
        b *= 2
    if scored_keys and bin_schedule[-1] < len(scored_keys):
        bin_schedule.append(len(scored_keys))

    opt_threads = max(1, n_cores - 1)

    for n_bins in bin_schedule:
        tl = time_left_fn()
        if tl <= 0:
            break

        common = (gg_ops, x_lo, x_hi, bounds_by_relu, query_w, query_bias,
                  scored_keys, n_bins)

        if feas_only:
            feas_args = ('feasibility',) + common + (
                max(1, n_cores), tl, input_name, fork_pts)
            pool = multiprocessing.Pool(1)
            async_feas = pool.apply_async(_solve_spec_graph_worker,
                                          (feas_args,))
            while not async_feas.ready():
                time.sleep(0.05)
            feas_result, feas_dt, _ = async_feas.get()
            pool.terminate(); pool.join()
            if feas_result == 'UNSAT':
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'feas UNSAT ({feas_dt:.1f}s) → verified')
                return True, n_bins
            if print_progress:
                print(f'    Racing bins={n_bins}: '
                      f'feas {feas_result} ({feas_dt:.1f}s) → escalate')
            continue

        feas_args = ('feasibility',) + common + (1, tl, input_name, fork_pts)
        opt_args = ('optimize',) + common + (opt_threads, tl, input_name, fork_pts)

        pool = multiprocessing.Pool(2)
        async_feas = pool.apply_async(_solve_spec_graph_worker, (feas_args,))
        async_opt = pool.apply_async(_solve_spec_graph_worker, (opt_args,))

        # feas (spec LP/MILP infeasible → safe) and opt (ObjBound > 0 → safe)
        # are two INDEPENDENT verification signals at this bin count — EITHER
        # proving safe is enough. Crucially they can DISAGREE: for a conjunctive
        # disjunct the opt worker minimises only the FIRST halfspace's margin
        # (reachable → "escalate") while the feas worker enforces ALL halfspaces
        # jointly (infeasible → verified). So we must NOT terminate the pool the
        # instant one worker says "escalate" — that used to kill a still-running
        # feas worker about to return UNSAT (sat_relu's unsat cases). Escalate
        # only once BOTH workers have finished without verifying.
        feas_done = opt_done = False
        while True:
            if not feas_done and async_feas.ready():
                feas_result, feas_dt, _ = async_feas.get()
                feas_done = True
                if feas_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'feas UNSAT ({feas_dt:.1f}s) → verified')
                    pool.terminate(); pool.join()
                    return True, n_bins
                # SAT (relaxation feasible) or UNKNOWN/TIMEOUT (not a sound
                # signal) — feas cannot verify at this bin count; wait for opt.
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'feas {feas_result} ({feas_dt:.1f}s)')
            if not opt_done and async_opt.ready():
                opt_result, opt_dt, opt_lb = async_opt.get()
                opt_done = True
                lb_s = f'{opt_lb:.4f}' if opt_lb is not None else '?'
                if opt_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'opt lb={lb_s} ({opt_dt:.1f}s) → verified')
                    pool.terminate(); pool.join()
                    return True, n_bins
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'opt lb={lb_s} ({opt_dt:.1f}s)')
            if feas_done and opt_done:
                # Neither feas (infeasible) nor opt (lb>0) verified at this bin
                # count → escalate to more binaries.
                if print_progress:
                    print(f'    Racing bins={n_bins}: neither → escalate')
                pool.terminate(); pool.join()
                break
            # Deadline guard: terminate if the budget runs out mid-build (the
            # spec-MILP construction is not bounded by Gurobi's TimeLimit) so
            # the caller returns a clean verdict instead of being SIGKILLed by
            # the outer shell timeout before it can write the results file.
            if time_left_fn() <= 1.0:
                pool.terminate(); pool.join()
                return False, n_bins
            time.sleep(0.05)

    return False, bin_schedule[-1] if bin_schedule else 0


# ---------------------------------------------------------------------------
# Graph-aware verification pipeline (ResNets / skip connections)
# ---------------------------------------------------------------------------

def _ibp_crown_bab(gg, xl, xh, sb_root, w_q, b_q, device, dtype, *,
                   time_left, ew_w=None, batch=64, alpha_iters=8,
                   lr=0.25, max_domains=500000, print_progress=False,
                   alpha_init=None, kfsb_k=4, split_deepest=False,
                   no_reforward=False):
    """ReLU-split BaB with per-domain IBP intermediate refresh for ONE query.

    The IBP-route counterpart of `attn_crown.attn_beta_bab`. That kernel
    is no-reforward (root intermediates + split clamps only) because the
    vit zonotope forward is expensive; here an IBP forward costs ~ms, so
    every domain REFRESHES all intermediate bounds under its split clamps
    (`_ibp_forward_graph_batched`) — each split tightens every downstream
    layer, which is what makes splits productive on deep conv nets
    (measured on cct tinyimagenet idx7018 q86: no-reforward β-BaB crept
    -0.36 → -0.15 in 13 343 domains/468 s; ABC with per-domain bound
    refresh closed it in 1 108 domains).

    Per batched step: pop the `batch` worst domains, batched IBP refresh,
    `alpha_iters` Adam steps of per-domain α-CROWN
    (`_spec_backward_graph_batched(alpha_at_layer=...)`), prune lb > 0,
    split the rest on the highest ew-weighted-width unstable neuron
    (clamped neurons read stable in the refreshed bounds, so they are
    never re-picked).

    Soundness: split clamps hold on each child's subdomain; intersecting
    the root enclosure, the parent IBP enclosure and the clamp is still an
    enclosure; the two children cover the parent; a domain is pruned only
    when its best α-CROWN lb > 0. fp arithmetic follows the caller's
    dtype — same convention as the rest of the milp graph path.

    Returns (closed, n_domains, reason): closed=True iff every domain was
    pruned (query verified on the whole box).
    """
    import heapq
    w_t = (w_q if torch.is_tensor(w_q)
           else torch.as_tensor(np.asarray(w_q), device=device, dtype=dtype))
    spec_ew_q = {0: (w_t, float(b_q))}
    sb_root_d = {L: (lo.detach(), hi.detach())
                 for L, (lo, hi) in sb_root.items()}

    def _sb_for_domains(relu_clamps, Bn):
        """Per-domain pre-ReLU bounds. `no_reforward`: broadcast the ROOT
        α-tight bounds and intersect the split clamps (no IBP forward) —
        the IBP refresh LOOSENS deeper layers on tight-root nets
        (measured 9566: split -0.065 -> -0.8); keeping root bounds + β
        propagating the split lets the bound climb like ABC (-0.065 ->
        +0.002, 23 domains). Else the IBP refresh (helps loose-root
        nets where re-forward tightens)."""
        if no_reforward:
            sb_b = {}
            for L, (lo_r, hi_r) in sb_root_d.items():
                lo = lo_r.unsqueeze(0).expand(Bn, -1).clone()
                hi = hi_r.unsqueeze(0).expand(Bn, -1).clone()
                if relu_clamps and L in relu_clamps:
                    cl, ch = relu_clamps[L]
                    lo = torch.maximum(lo, cl)
                    hi = torch.minimum(hi, ch)
                    hi = torch.maximum(hi, lo)   # crossing repair
                sb_b[L] = (lo, hi)
            return sb_b
        xl_b = xl.unsqueeze(0).expand(Bn, -1).contiguous()
        xh_b = xh.unsqueeze(0).expand(Bn, -1).contiguous()
        return _ibp_forward_graph_batched(
            xl_b, xh_b, gg, device, dtype,
            relu_clamps=relu_clamps or None, root_sb=sb_root_d)

    counter = 0
    frontier = [(-float('inf'), counter, {})]   # (lb, tiebreak, clamps)
    n_domains = 0
    n_report = 0
    while frontier:
        if time_left() <= 3.0:
            return False, n_domains, 'time'
        if n_domains >= max_domains:
            return False, n_domains, 'max_domains'
        batch_doms = [heapq.heappop(frontier)
                      for _ in range(min(batch, len(frontier)))]
        B = len(batch_doms)
        n_domains += B
        xl_b = xl.unsqueeze(0).expand(B, -1).contiguous()
        xh_b = xh.unsqueeze(0).expand(B, -1).contiguous()
        relu_clamps = {}
        for bi, (_, _, clamps) in enumerate(batch_doms):
            for (L, j), side in clamps.items():
                if L not in relu_clamps:
                    n_L = sb_root_d[L][0].numel()
                    relu_clamps[L] = (
                        torch.full((B, n_L), -np.inf, device=device,
                                   dtype=dtype),
                        torch.full((B, n_L), np.inf, device=device,
                                   dtype=dtype))
                if side == 0:
                    relu_clamps[L][1][bi, j] = 0.0   # OFF: z_j <= 0
                else:
                    relu_clamps[L][0][bi, j] = 0.0   # ON:  z_j >= 0
        with torch.no_grad():
            sb_b = _sb_for_domains(relu_clamps, B)
        # Per-domain α-CROWN, warm-started from the converged ROOT α
        # when provided (`alpha_init`: {L: (n,)}). Without the warm
        # start the min-area init needs many more Adam iters than the
        # per-domain budget allows and domains evaluate LOOSER than the
        # root bound — measured on cct idx7018 q86: 46 847 domains, 0
        # pruned, frontier == domains (every child stuck near the
        # min-area bound -0.65 while the root α gave -0.356).
        alpha_at_layer = {}
        for L, (lo_b, hi_b) in sb_b.items():
            if alpha_init is not None and L in alpha_init:
                init_a = alpha_init[L].detach().to(dtype).unsqueeze(0) \
                    .expand(lo_b.shape[0], -1)
            else:
                _, up_s, _, _, _, unstable = _make_slopes(lo_b, hi_b)
                init_a = (up_s > 0.5).to(dtype) * unstable.to(dtype)
            alpha_at_layer[L] = init_a.detach().clone().requires_grad_(True)
        # β for the split halfspaces (β-CROWN): sign +1 for OFF
        # (constraint z <= 0), -1 for ON (-z <= 0); β >= 0 learnable.
        # The IBP clamp only captures the BOX consequence of a split;
        # β couples the halfspace linearly through earlier layers —
        # without it deep domains plateau (24k domains at lb ~-0.05
        # with zero prunes on cct idx7018 q86).
        beta_sign = {}
        beta_params = {}
        for L, (cl, ch) in relu_clamps.items():
            sgn = torch.zeros_like(cl)
            sgn[ch == 0] = 1.0
            sgn[cl == 0] = -1.0
            beta_sign[L] = sgn
            beta_params[L] = torch.zeros_like(cl).requires_grad_(True)
        opt = torch.optim.Adam(
            list(alpha_at_layer.values()) + list(beta_params.values()),
            lr=lr)
        best_lbs = None
        for _it in range(max(1, alpha_iters)):
            opt.zero_grad()
            rsb = ({L: beta_params[L] * beta_sign[L] for L in beta_params}
                   or None)
            lbs = _spec_backward_graph_batched(
                sb_b, xl_b, xh_b, gg, spec_ew_q, device, dtype,
                alpha_at_layer=alpha_at_layer, relu_split_beta=rsb)[:, 0]
            with torch.no_grad():
                best_lbs = (lbs.detach().clone() if best_lbs is None
                            else torch.maximum(best_lbs, lbs.detach()))
            if (best_lbs > 0).all() or time_left() <= 3.0:
                break
            (-lbs.sum()).backward()
            opt.step()
            with torch.no_grad():
                for L in alpha_at_layer:
                    alpha_at_layer[L].clamp_(0.0, 1.0)
                for L in beta_params:
                    beta_params[L].clamp_(min=0.0)
        with torch.no_grad():
            # kfsb-lite: per unpruned domain, take the top-k unstable
            # candidates by width score, evaluate ALL their children
            # bounds in one batched pass (init α, no Adam — a valid
            # though looser bound), commit to the candidate whose worse
            # child is best (max-min), and push children with their
            # EVALUATED lbs. Children that already evaluate > 0 are
            # pruned immediately. Plain width-argmax splits gave a
            # logarithmic trajectory (vc9: -0.177 -> -0.027 in 39k
            # domains without closing); candidate evaluation is ABC's
            # kfsb mechanism.
            cand_specs = []   # (slot, bi, (L, j))
            dom_cands = {}    # bi -> [(L, j), ...]
            # Per-domain BaBSR ew: recompute the backward coefficients with
            # THIS domain's (split-tightened) bounds when no_reforward is
            # on. Using the stale ROOT ew_w for deep domains picks the
            # wrong neurons and plateaus (measured 9566: stale-root ew ->
            # -0.004 plateau; per-domain ew -> CLOSES in 45 domains). When
            # reforward is on (loose-root nets) the passed ew_w is fine.
            ew_per_dom = None
            if no_reforward and ew_w is not None:
                ew_per_dom = []
                for bi in range(B):
                    sb_bi = {L: (sb_b[L][0][bi], sb_b[L][1][bi])
                             for L in sb_b}
                    try:
                        _, _, _ewd = _spec_backward_graph(
                            sb_bi, xl, xh, gg, spec_ew_q, {0}, len(sb_bi),
                            device, dtype, return_ew=True)
                        ew_per_dom.append({
                            L: torch.as_tensor(
                                np.abs(np.asarray(e, np.float64)),
                                device=device, dtype=dtype)
                            for L, e in _ewd.get(0, {}).items()})
                    except NotImplementedError:
                        ew_per_dom.append(ew_w)
            for bi, (_, _, clamps) in enumerate(batch_doms):
                if float(best_lbs[bi]) > 0:
                    continue
                ew_bi = ew_per_dom[bi] if ew_per_dom is not None else ew_w
                # Optionally restrict candidates to the DEEPEST layer
                # with unstable neurons: splits near the output have a
                # short backward path where β couples strongest — ABC
                # closes cct s3 cases with ~10 splits, ALL at the last
                # ReLU (their log: every decision at /input-40).
                if split_deepest:
                    deep_layers = [L for L in sorted(sb_b)
                                   if bool(((sb_b[L][0][bi] < 0)
                                            & (sb_b[L][1][bi] > 0)).any())]
                    layer_pool = deep_layers[-1:] if deep_layers else []
                else:
                    layer_pool = list(sb_b)
                scored = []
                for L in layer_pool:
                    lo_b, hi_b = sb_b[L]
                    lo_d, hi_d = lo_b[bi], hi_b[bi]
                    unstable = (lo_d < 0) & (hi_d > 0)
                    if not bool(unstable.any()):
                        continue
                    # BaBSR score: |lA| × the ReLU triangle intercept
                    # (-lo·hi/(hi-lo), the max relaxation gap), NOT width.
                    # This is ABC's babsr_score intercept term — it ranks
                    # by how much a neuron's upper relaxation loosens the
                    # bound, and is far better than width (measured 9566:
                    # width plateaus / explodes; intercept climbs to
                    # -0.004). ew_w carries |lA| (backward coeffs).
                    denom = (hi_d - lo_d).clamp(min=1e-12)
                    gap = (-lo_d * hi_d) / denom
                    score = gap * unstable.to(dtype)
                    if ew_bi is not None and L in ew_bi:
                        score = score * ew_bi[L]
                    kk = min(kfsb_k, int(unstable.sum()))
                    vals, idxs = torch.topk(score, kk)
                    for v, j in zip(vals.tolist(), idxs.tolist()):
                        if v > 0:
                            scored.append((v, (L, int(j))))
                if not scored:
                    # Fully split ReLU pattern, bound still <= 0: the
                    # backward bound over a fixed pattern is exact for
                    # the affine composition — cannot improve by
                    # splitting further.
                    return False, n_domains, 'exhausted_pattern'
                scored.sort(reverse=True)
                picks = [key for _, key in scored[:kfsb_k]]
                dom_cands[bi] = picks
                for key in picks:
                    cand_specs.append((len(cand_specs), bi, key))
            if not cand_specs:
                continue
            # Evaluate all children (2 per candidate) in chunks.
            child_lbs = torch.empty(2 * len(cand_specs), dtype=dtype)
            CH = max(8, batch)
            for c0 in range(0, 2 * len(cand_specs), CH):
                rows = range(c0, min(c0 + CH, 2 * len(cand_specs)))
                Bc = len(rows)
                xl_c = xl.unsqueeze(0).expand(Bc, -1).contiguous()
                xh_c = xh.unsqueeze(0).expand(Bc, -1).contiguous()
                rc_c = {}
                al_c = {}
                for r_i, r in enumerate(rows):
                    slot, side = r // 2, r % 2
                    _, bi, key = cand_specs[slot]
                    clamps = dict(batch_doms[bi][2])
                    clamps[key] = side
                    for (L, j), sd in clamps.items():
                        if L not in rc_c:
                            n_L = sb_root_d[L][0].numel()
                            rc_c[L] = (
                                torch.full((Bc, n_L), -np.inf,
                                           device=device, dtype=dtype),
                                torch.full((Bc, n_L), np.inf,
                                           device=device, dtype=dtype))
                        if sd == 0:
                            rc_c[L][1][r_i, j] = 0.0
                        else:
                            rc_c[L][0][r_i, j] = 0.0
                sb_c = _sb_for_domains(rc_c, Bc)
                for L, (lo_c, hi_c) in sb_c.items():
                    if alpha_init is not None and L in alpha_init:
                        al_c[L] = alpha_init[L].detach().to(dtype) \
                            .unsqueeze(0).expand(Bc, -1)
                lbs_c = _spec_backward_graph_batched(
                    sb_c, xl_c, xh_c, gg, spec_ew_q, device, dtype,
                    alpha_at_layer=al_c or None)[:, 0]
                child_lbs[c0:c0 + Bc] = lbs_c.detach().cpu()
            # Pick per-domain best candidate (max over candidates of
            # min(child lbs)); push its children with evaluated lbs.
            slot_of = {}
            for slot, bi, key in cand_specs:
                slot_of.setdefault(bi, []).append((slot, key))
            for bi, slot_keys in slot_of.items():
                best_val, best_slot, best_key = -float('inf'), None, None
                for slot, key in slot_keys:
                    v = float(torch.minimum(child_lbs[2 * slot],
                                            child_lbs[2 * slot + 1]))
                    if v > best_val:
                        best_val, best_slot, best_key = v, slot, key
                clamps = batch_doms[bi][2]
                for side in (0, 1):
                    lb_child = float(child_lbs[2 * best_slot + side])
                    if lb_child > 0:
                        continue   # pruned at candidate-eval time
                    counter += 1
                    child = dict(clamps)
                    child[best_key] = side
                    heapq.heappush(frontier, (lb_child, counter, child))
        if print_progress and n_domains >= n_report + 2048:
            n_report = n_domains
            wl = min((d[0] for d in frontier), default=0.0)
            print(f'    [ibp-bab] {n_domains} domains, frontier '
                  f'{len(frontier)}, worst lb {wl:.4f}', flush=True)
    return True, n_domains, 'closed'


def _crown_bab_noreforward(gg, xl, xh, sb, alpha_spec, w_q, b_q, device,
                           dtype, *, time_left, batch=16, main_iters=20,
                           cand_iters=8, prefilter=12, multilevel=2,
                           max_domains=500000, print_progress=False):
    """No-reforward β-CROWN BaB with per-domain BaBSR branching, ONE query.

    Validated standalone recipe (closes cct2026 cifar10_eps8 9566_s3 q3 in
    41 domains / 37 s, q6 in 45 / 40 s; ABC: 9 domains / 11 s). Four
    load-bearing pieces, each a measured win over its alternative
    (`scratch/cct2026/NOTES.md`, reference `own_kfsb_pd.py`):

      1. NO-REFORWARD — keep the root α-tight intermediate bounds; per
         domain intersect ONLY the split neuron's own pre-ReLU bound with
         the clamp (`min(hi,0)` OFF, `max(lo,0)` ON). Do NOT IBP-re-forward:
         on tight-root nets that LOOSENS deeper layers (9566 split
         -0.065 → -0.8). The split constraint is instead propagated by β.
      2. BaBSR triangle-intercept split score `|clamp(ew,max=0)| ·
         (-lo·hi/(hi-lo))` — ABC's `babsr_score` intercept term (the ReLU
         triangle's max relaxation gap weighted by the backward coeff).
         NOT width (width plateaus).
      3. PER-DOMAIN ew — recompute the backward coeffs (`return_ew`) with
         THIS domain's split-tightened bounds. Scoring deep domains with
         the stale root ew picks wrong neurons → -0.004 plateau.
      4. kfsb candidate evaluation + multilevel — per open domain take the
         top-`prefilter` candidates per layer by the score, evaluate each
         candidate's 2 children with a cheap `cand_iters`-step α+β bound,
         pick the top-`multilevel` by best-worst-child, and branch all
         picks simultaneously (2^multilevel children).

    The caller has already run root α-CROWN and merged `root_bounds` into
    `sb` (the root α-tight pre-ReLU bounds, `{layer: (lo,hi)}` tensors);
    `alpha_spec` is the per-query spec α (`{layer: tensor}` in [0,1]).

    Soundness is identical to `_ibp_crown_bab`: a valid CROWN lower bound
    under fixed split clamps + a β-Lagrangian relaxation of the split
    halfspaces (β ≥ 0, sign +1 OFF / -1 ON); the two children of a split
    partition the unstable neuron, so they cover the parent; a domain is
    pruned only when its best lb > 0. fp arithmetic follows `dtype`.

    Returns (closed, n_domains, reason): closed=True iff the frontier
    emptied (query verified on the whole box).
    """
    import heapq
    import itertools
    w_t = (w_q if torch.is_tensor(w_q)
           else torch.as_tensor(np.asarray(w_q), device=device, dtype=dtype))
    spec_ew = {0: (w_t, float(b_q))}
    sb = {L: (lo.detach(), hi.detach()) for L, (lo, hi) in sb.items()}
    ai = {L: a.detach().to(dtype) for L, a in alpha_spec.items()}
    nL = {L: sb[L][0].numel() for L in sb}
    LS = sorted(sb)
    zero = torch.zeros((), device=device, dtype=dtype)

    def make_sb(clamp_list):
        """Per-domain pre-ReLU bounds + β sign. Broadcast the ROOT α-tight
        bounds and intersect ONLY the split neuron's own bound with the
        clamp (no IBP forward). sgn carries +1 for an OFF split (s = +z ≤ 0)
        and -1 for an ON split (s = -z ≤ 0)."""
        Bn = len(clamp_list)
        sb_b = {L: (sb[L][0].unsqueeze(0).expand(Bn, -1).clone(),
                    sb[L][1].unsqueeze(0).expand(Bn, -1).clone()) for L in sb}
        sgn = {L: torch.zeros(Bn, nL[L], device=device, dtype=dtype)
               for L in sb}
        for bi, clamps in enumerate(clamp_list):
            for (L, j), side in clamps.items():
                if side == 0:   # OFF: z_j <= 0
                    sb_b[L][1][bi, j] = torch.minimum(sb_b[L][1][bi, j], zero)
                    sgn[L][bi, j] = 1.0
                else:           # ON: z_j >= 0
                    sb_b[L][0][bi, j] = torch.maximum(sb_b[L][0][bi, j], zero)
                    sgn[L][bi, j] = -1.0
        return sb_b, sgn

    def bound(clamp_list, opt_iters):
        """Best CROWN lb over `opt_iters` Adam steps on (α, β), per domain.
        α warm-starts from `alpha_spec`, β from 0; both clamp to their
        feasible ranges each step. Returns (Bn,) best lb."""
        Bn = len(clamp_list)
        sb_b, sgn = make_sb(clamp_list)
        alpha_at = {L: ai[L].unsqueeze(0).expand(Bn, -1).clone()
                    .requires_grad_(True) for L in ai}
        beta = {L: torch.zeros(Bn, nL[L], device=device, dtype=dtype,
                               requires_grad=True) for L in sb}
        opt = torch.optim.Adam(
            list(alpha_at.values()) + list(beta.values()), lr=0.2)
        best = None
        xlb = xl.unsqueeze(0).expand(Bn, -1)
        xhb = xh.unsqueeze(0).expand(Bn, -1)
        for _ in range(max(1, opt_iters)):
            opt.zero_grad()
            rsb = {L: beta[L].clamp(min=0) * sgn[L] for L in beta}
            lb = _spec_backward_graph_batched(
                sb_b, xlb, xhb, gg, spec_ew, device, dtype,
                alpha_at_layer=alpha_at, relu_split_beta=rsb)[:, 0]
            best = (lb.detach().clone() if best is None
                    else torch.maximum(best, lb.detach()))
            if bool((best > 0).all()) or time_left() <= 0:
                break
            (-lb.sum()).backward()
            opt.step()
            with torch.no_grad():
                for L in alpha_at:
                    alpha_at[L].clamp_(0.0, 1.0)
                for L in beta:
                    beta[L].clamp_(min=0.0)
        return best

    counter = 0
    n_domains = 0
    n_report = 0
    t_start = time.perf_counter()
    frontier = [(-float('inf'), counter, {})]   # (lb, tiebreak, clamps)
    while frontier:
        if time_left() <= 0:
            return False, n_domains, 'time'
        if n_domains >= max_domains:
            return False, n_domains, 'max_domains'
        doms = [heapq.heappop(frontier)
                for _ in range(min(batch, len(frontier)))]
        clamp_list = [d[2] for d in doms]
        n_domains += len(doms)
        best = bound(clamp_list, main_iters)
        sb_b, _ = make_sb(clamp_list)
        for bi, clamps in enumerate(clamp_list):
            if float(best[bi]) > 0:
                continue
            # PER-DOMAIN ew: recompute backward coeffs with THIS domain's
            # split-tightened bounds (stale root ew plateaus at -0.004).
            sb_bi = {L: (sb_b[L][0][bi], sb_b[L][1][bi]) for L in sb_b}
            _, _, ewb = _spec_backward_graph(
                sb_bi, xl, xh, gg, spec_ew, {0}, len(sb_bi),
                device, dtype, return_ew=True)
            # Candidates: top-`prefilter` unstable per layer by the BaBSR
            # triangle-intercept score, excluding already-split neurons.
            cand = []
            for L in LS:
                lo_d, hi_d = sb_b[L][0][bi], sb_b[L][1][bi]
                umask = (lo_d < 0) & (hi_d > 0)
                e = ewb.get(0, {}).get(L)
                ew = (torch.as_tensor(np.asarray(e, np.float64),
                                      device=device, dtype=dtype)
                      if e is not None else torch.zeros_like(lo_d))
                gap = (-lo_d * hi_d) / torch.clamp(hi_d - lo_d, min=1e-12)
                score = torch.clamp(ew, max=0).abs() * gap * umask.to(dtype)
                k = min(prefilter, int(umask.sum()))
                if k > 0:
                    vals, idxs = torch.topk(score, k)
                    for vv, j in zip(vals.tolist(), idxs.tolist()):
                        if vv > 0 and (L, int(j)) not in clamps:
                            cand.append((L, int(j)))
            if not cand:
                # Fully split (or all-stable) pattern, bound still <= 0:
                # the backward bound over a fixed pattern is exact for the
                # affine composition — cannot improve by splitting further.
                return False, n_domains, 'exhausted'
            # Evaluate each candidate's 2 children (cheap α+β), pick the
            # top-`multilevel` by worst child, branch all picks at once.
            trial = [{**clamps, key: side}
                     for key in cand for side in (0, 1)]
            tb = bound(trial, cand_iters)
            csc = [(min(float(tb[2 * i]), float(tb[2 * i + 1])), cand[i])
                   for i in range(len(cand))]
            csc.sort(reverse=True)
            picks = [key for _, key in csc[:multilevel]]
            for combo in itertools.product([0, 1], repeat=len(picks)):
                counter += 1
                child = dict(clamps)
                for key, side in zip(picks, combo):
                    child[key] = side
                heapq.heappush(frontier, (float(best[bi]), counter, child))
        if print_progress and n_domains >= n_report + 128:
            n_report = n_domains
            wl = min((d[0] for d in frontier), default=0.0)
            print(f'    [crown-bab-nr] {n_domains} domains, frontier '
                  f'{len(frontier)}, worst lb {wl:.4f}, '
                  f'{time.perf_counter() - t_start:.1f}s', flush=True)
    return True, n_domains, 'closed'


def _alpha_start_cap(settings, max_layer_neurons):
    """Max start-node layer size (neurons) for the per-target α-CROWN
    backward, derived from a MEMORY budget rather than a fixed neuron
    count. The (non-chunked) per-target backward materializes a peak
    tensor of ~n_targets × max_layer_neurons elements (measured exactly:
    cifar10 normal L1 = 65536 targets × 65536 max layer × 4 B = 17.2 GB),
    so a layer is a full-batch α start node iff
    n_targets × max_layer ≤ `milp_graph_alpha_start_mem_elems`; bigger
    layers are routed to the chunked, memory-safe
    `tighten_layer_alpha_crown` (batched over target neurons, so its base
    tightening is still applied). The budget is the PEAK element count
    (× 4 B = bytes) — sized to leave headroom on a 24 GB GPU. This auto-
    includes the layers that fit and auto-excludes the wide/tinyimagenet
    layers whose backward would OOM, with no per-net constant.
    `milp_graph_alpha_start_mem_elems=None/0` → fall back to the fixed
    `milp_graph_alpha_start_cap` neuron count (legacy behavior)."""
    mem = getattr(settings, 'milp_graph_alpha_start_mem_elems', None)
    if mem:
        return max(1, int(mem) // max(1, int(max_layer_neurons)))
    return int(getattr(settings, 'milp_graph_alpha_start_cap', 32768))


def _milp_verify_graph(graph, spec, settings, device, dtype,
                        deadline, total_timeout):
    """Verification pipeline for networks with skip connections.

    Supports any spec type (pairwise, threshold, mixed) via linear queries.
    """
    print_progress = settings.print_progress
    stats = VerifyStats()
    _wnr = (int(getattr(settings, 'tighten_witness_n_random', 8))
            if getattr(settings, 'tighten_witness_ordering', True) else 0)

    def time_left():
        return max(0, deadline - time.perf_counter())

    gg = graph.gpu_graph(device, dtype)
    nh = gg['n_relu']

    # Find output size from last linear op
    n_output = None
    for op in reversed(gg['ops']):
        if op['type'] == 'fc':
            n_output = op['W'].shape[0]
            break
        elif op['type'] == 'conv':
            n_output = op['n_out']
            break
    assert n_output is not None

    # Convert spec to linear queries
    queries = spec.as_linear_queries(n_output)
    # Group by disjunct: all queries in a disjunct must be verified
    disj_queries = {}  # disjunct_idx -> list of (query_idx, w, bias)
    for qi, (di, w, bias) in enumerate(queries):
        disj_queries.setdefault(di, []).append((qi, w, bias))

    # Build spec_ew dict keyed by query_idx
    spec_ew = {}
    for qi, (di, w, bias) in enumerate(queries):
        w_t = torch.tensor(w, dtype=dtype, device=device)
        # spec_ew needs w in terms of the last hidden layer's output
        # For now, pass raw output weights — _spec_backward_graph
        # will propagate through the final linear layer
        spec_ew[qi] = (w_t, float(bias))

    x_lo_f32 = spec.x_lo.astype(np.float32)
    x_hi_f32 = spec.x_hi.astype(np.float32)
    xl_g = torch.tensor(x_lo_f32, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi_f32, dtype=dtype, device=device)

    # --- Phase 1: GPU zonotope forward + CROWN ---
    # (or IBP forward — CROWN-IBP — for large-input nets where the
    # zonotope generator tensors cannot fit GPU memory; see
    # settings.phase1_ibp_input_dim_threshold.)
    _ibp_thresh = int(getattr(settings, 'phase1_ibp_input_dim_threshold', 0))
    # `milp_force_ibp_phase1`: use the IBP forward regardless of input dim —
    # for benchmarks routed here by the perturbation-width gate (large-eps
    # cifar conv) whose α-CROWN root is materially tighter on IBP than on the
    # zonotope forward (cct cifar10_eps8 9566 q3: IBP→-0.0655 vs zono→-0.29).
    _use_ibp = (bool(getattr(settings, 'milp_force_ibp_phase1', False))
                or (_ibp_thresh > 0
                    and int(np.prod(spec.x_lo.shape)) >= _ibp_thresh))
    t0 = time.perf_counter()
    if _use_ibp:
        with torch.no_grad():
            sb = _ibp_forward_graph(xl_g, xh_g, gg, device, dtype)
        if print_progress:
            print('[phase1] IBP forward (CROWN-IBP): input dim '
                  f'{int(np.prod(spec.x_lo.shape))} >= '
                  f'{_ibp_thresh}', flush=True)
        _phase1_skip_zono = True
    else:
        _phase1_skip_zono = False
    try:
        if not _phase1_skip_zono:
            with torch.no_grad():
                sb, z_final = _forward_zonotope_graph(
                    xl_g, xh_g, gg, device, dtype)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # CPU fallback silently masks real regressions (a forward that
        # used to fit suddenly doesn't). Require explicit opt-in via
        # `allow_cpu_fallback` AND `raise_on_oom=False`; otherwise re-raise.
        cpu_ok = (device.type != 'cpu'
                  and getattr(settings, 'allow_cpu_fallback', False)
                  and not getattr(settings, 'raise_on_oom', True))
        if cpu_ok:
            if print_progress:
                print(f'  GPU OOM ({e!s:.60}); falling back to CPU '
                      '(allow_cpu_fallback=True, raise_on_oom=False)')
            device = torch.device('cpu')
            gg = graph.gpu_graph(device, dtype)
            xl_g = xl_g.cpu()
            xh_g = xh_g.cpu()
            spec_ew = {qi: (w.cpu(), b) for qi, (w, b) in spec_ew.items()}
            with torch.no_grad():
                sb, z_final = _forward_zonotope_graph(
                    xl_g, xh_g, gg, device, dtype)
        else:
            raise

    with torch.no_grad():
        all_qids = set(spec_ew.keys())
        spec_lbs, still_open_q = _spec_backward_graph(
            sb, xl_g, xh_g, gg, spec_ew, all_qids, nh, device, dtype)
    t_phase1 = time.perf_counter() - t0

    # Check which disjuncts are fully verified
    verified_disj = set()
    for di, qlist in disj_queries.items():
        if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist):
            verified_disj.add(di)
    still_open_disj = set(disj_queries.keys()) - verified_disj
    n_total = len(disj_queries)

    if print_progress:
        worst = min(spec_lbs.values()) if spec_lbs else 0
        print(f'Phase 1 (zonotope+CROWN graph): {t_phase1:.2f}s  '
              f'verified={len(verified_disj)}/{n_total}  '
              f'worst={worst:.4f}')

    stats.record_timing('crown', t_phase1)
    stats.record_bounds(sb)
    _fire_callback(settings, 'phase_done', {
        'phase': 'crown', 'elapsed': t_phase1,
        'bounds_summary': stats.neuron_stats['per_layer']})

    if not still_open_disj:
        return _make_result('verified', {
            'time': time.perf_counter() - (deadline - total_timeout),
            'phase': 'crown_graph'}, stats)

    # --- PGD attack ---
    # The default attack is one joint-loss PGD over EVERY disjunct. When a
    # handful of disjuncts remain open among many already-verified ones
    # (199 disjuncts, 4 open on cct2026 idx5613), that joint loss dilutes
    # the open disjuncts' gradient among the verified majority and never
    # descends into the open basins — it misses CEs α,β-CROWN's diversified
    # attack finds. `milp_graph_targeted_pgd_max_open` enables a TARGETED
    # attack: restrict to the still-open disjuncts and (per_restart_disj)
    # give each its own restarts. Gated on a small open count — a strong SAT
    # signal — so a hard UNSAT case with many open disjuncts does NOT fire a
    # long focused sweep that would only burn budget finding nothing; and
    # time-capped (`milp_graph_targeted_pgd_budget`) so even a few-open UNSAT
    # case loses at most that many seconds before the BaB.
    _disable_sat = bool(getattr(settings, 'disable_sat_finding', False))
    if not _disable_sat:
        t_pgd = time.perf_counter()
        _n_open = len(still_open_disj)
        _max_open = int(getattr(settings, 'milp_graph_targeted_pgd_max_open', 0))
        _focus = _max_open > 0 and 0 < _n_open <= _max_open
        _tr = int(getattr(settings, 'milp_graph_targeted_pgd_restarts', 0))
        _orig_restarts = int(getattr(settings, 'pgd_restarts', 10))
        if _focus and _tr > 0:
            settings.pgd_restarts = _tr
        try:
            if _focus:
                _tb = float(getattr(
                    settings, 'milp_graph_targeted_pgd_budget', 8.0))
                _tb = min(_tb, max(0.5, time_left() - 2.0))
                pgd_sat, pgd_witness = _pgd_attack_general(
                    xl_g, xh_g, spec, gg, settings,
                    restrict_disj=still_open_disj, per_restart_disj=True,
                    time_budget=_tb)
            else:
                pgd_sat, pgd_witness = _pgd_attack_general(
                    xl_g, xh_g, spec, gg, settings)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        finally:
            settings.pgd_restarts = _orig_restarts
        dt_pgd = time.perf_counter() - t_pgd
        stats.record_timing('pgd', dt_pgd)
        if print_progress:
            print(f'PGD attack (graph): {dt_pgd:.2f}s  sat={pgd_sat}'
                  f'  (focus={_focus} open={_n_open})', flush=True)
        if pgd_sat:
            return _make_result('sat', {
                'time': time.perf_counter() - (deadline - total_timeout),
                'phase': 'pgd_graph', 'witness': pgd_witness}, stats)

    if time_left() <= 0:
        return _make_result('unknown', {'time': total_timeout,
            'phase': 'crown_graph_timeout',
            'remaining': len(still_open_disj)}, stats)

    # Stable per-query BaB base: the Phase-1 bounds plus query-INDEPENDENT
    # tightening (Phase 1.5 `tighten_big_layers`), but NOT the joint
    # α-CROWN bounds nor any other query's per-query root_bounds. The
    # no-reforward BaB (`_crown_bab_noreforward`) is fed `sb_bab_base ∩
    # this-query's-own-root_bounds`. Feeding it the cumulatively/jointly
    # tightened `sb` instead — though those are sound enclosures — makes
    # the per-domain bounds NOISIER and the BaBSR split scoring worse, so
    # the same query swings 41→393 domains run-to-run and times out
    # (measured cct2026 9566 q3: stable base = reliable ~41 domains / 6 s;
    # joint/cumulative base = 45–393 variable). The joint α-CROWN below is
    # kept — it still closes the easy queries cheaply — but its bounds do
    # not contaminate the BaB base.
    sb_bab_base = {L: (lo.detach().clone(), hi.detach().clone())
                   for L, (lo, hi) in sb.items()}

    # --- Phase 1.5: batched joint α-CROWN on open queries (graph path) ---
    # Mirrors the layers-path `milp_alpha_tighten` block: optimize shared
    # α slopes (spec + size-capped intermediate start nodes) on the
    # still-open queries, prune what closes, and merge tightened
    # intermediate bounds into `sb` for the later phases. Default OFF —
    # other milp-graph benchmarks keep their measured behavior; enabled
    # per-benchmark (challenging_certified_training tinyimagenet: CROWN-IBP
    # leaves ~9-12 open queries at worst lb ~-2 that the MILP tightener
    # provably cannot close — 546 s for -1.80 -> -1.79 on idx7018_sample4).
    if (bool(getattr(settings, 'milp_graph_alpha_enabled', False))
            and still_open_disj and time_left() > 5.0):
        t_a15 = time.perf_counter()
        from . import alpha_crown as ac
        # Targeted per-target α-CROWN tightening of layers TOO BIG to be
        # α start nodes (above milp_graph_alpha_start_cap). Without
        # this, those layers keep raw Phase-1 bounds and stay looser
        # than ABC's (cct idx7018: L3 3940 unstable vs ABC 2618; after
        # this pass every layer beats ABC: 7464/2295/2546 vs
        # 7472/2307/2618). Chunked over target neurons so the per-target
        # backward (chunk x sum-of-prior-layer-sizes) stays ~1 GB.
        if bool(getattr(settings, 'milp_graph_tighten_big_layers', False)):
            from .alpha_tighten import tighten_layer_alpha_crown
            _max_layer = max(t[0].numel() for t in sb.values())
            _cap_b = _alpha_start_cap(settings, _max_layer)
            _bbr_np = {li: (sb[li][0].detach().cpu().numpy().astype(np.float64),
                            sb[li][1].detach().cpu().numpy().astype(np.float64))
                       for li in sb}
            _sizes = {li: len(_bbr_np[li][0]) for li in _bbr_np}
            for _Lb in sorted(_bbr_np):
                if _Lb == 0 or _sizes[_Lb] <= _cap_b:
                    continue   # L0 pre-relu bounds are exact (first affine)
                lo_b, hi_b = _bbr_np[_Lb]
                _unb = np.where((lo_b < 0) & (hi_b > 0))[0]
                if len(_unb) == 0 or time_left() <= 30.0:
                    continue
                _prior = sum(_sizes[k] for k in _sizes if k < _Lb) \
                    + int(np.prod(spec.x_lo.shape))
                _chunk = max(256, min(2048, int(3.0e8 // max(1, _prior))))
                t_bl = time.perf_counter()
                new_lo, new_hi = lo_b.copy(), hi_b.copy()
                for _c0 in range(0, len(_unb), _chunk):
                    if time_left() <= 30.0:
                        break
                    _idx = _unb[_c0:_c0 + _chunk]
                    _tl, _th = tighten_layer_alpha_crown(
                        gg, spec.x_lo.flatten(), spec.x_hi.flatten(),
                        _bbr_np, _Lb, device=device, dtype=dtype,
                        n_iters=int(getattr(
                            settings, 'milp_graph_tighten_big_iters', 15)),
                        lr=0.05, target_indices=_idx, per_target=True)
                    new_lo[_idx] = np.maximum(new_lo[_idx], _tl[_idx])
                    new_hi[_idx] = np.minimum(new_hi[_idx], _th[_idx])
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                _bbr_np[_Lb] = (new_lo, new_hi)
                sb[_Lb] = (torch.as_tensor(new_lo, device=device,
                                           dtype=dtype),
                           torch.as_tensor(new_hi, device=device,
                                           dtype=dtype))
                # big-layer tightening is query-independent → also fold it
                # into the stable BaB base (tinyimagenet needs it; the raw
                # IBP bounds are too loose for the no-reforward BaB there).
                sb_bab_base[_Lb] = (sb[_Lb][0].clone(), sb[_Lb][1].clone())
                if print_progress:
                    _un2 = int(((new_lo < 0) & (new_hi > 0)).sum())
                    print(f'Phase 1.5 big-layer tighten L{_Lb}: '
                          f'{len(_unb)} -> {_un2} unstable '
                          f'({time.perf_counter() - t_bl:.1f}s, '
                          f'chunk {_chunk})', flush=True)
        open_q = [(qi, w, b) for di in sorted(still_open_disj)
                  for (qi, w, b) in disj_queries[di]
                  if spec_lbs.get(qi, -1.0) <= 0]
        if open_q:
            w_qs_a = np.stack([w for _, w, _ in open_q])
            b_qs_a = np.array([b for _, _, b in open_q], dtype=np.float64)
            bbr_a = {li: (sb[li][0].detach().cpu().numpy().astype(np.float64),
                          sb[li][1].detach().cpu().numpy().astype(np.float64))
                     for li in sb}
            # Memory guard: intermediate start nodes only for layers small
            # enough that the per-target backward fits (the first conv
            # layers of a 64x64-input CNN7 have 262k neurons — backward
            # to input would be the same 12 GiB tensor the IBP route
            # exists to avoid).
            _cap = _alpha_start_cap(
                settings, max(t[0].numel() for t in sb.values()))
            start_nodes_a = [
                L for L in sorted(bbr_a) if L > 0
                and len(bbr_a[L][0]) <= _cap
                and bool(((bbr_a[L][0] < 0) & (bbr_a[L][1] > 0)).any())]
            un_idx_a = {L: np.where((bbr_a[L][0] < 0)
                                    & (bbr_a[L][1] > 0))[0].tolist()
                        for L in bbr_a}
            n_iters_a = int(getattr(settings, 'milp_graph_alpha_iters', 20))
            best_lbs_a, _, best_bounds_a, _ = ac.run_alpha_crown_batched(
                gg, xl_g, xh_g, bbr_a, w_qs_a, b_qs_a,
                start_nodes_a, un_idx_a, device, dtype,
                n_iters=n_iters_a, lr=0.25, lr_decay=0.98,
                early_stop_on_positive=True, sparse_alpha=True,
                time_left_fn=time_left)
            for k, (qi, _, _) in enumerate(open_q):
                lb_q = float(best_lbs_a[k])
                if lb_q > spec_lbs.get(qi, -float('inf')):
                    spec_lbs[qi] = lb_q
            # Merge tightened intermediates into sb (max lo / min hi).
            for Lk, (lo_t, hi_t) in best_bounds_a.items():
                if Lk in sb:
                    lo_old, hi_old = sb[Lk]
                    sb[Lk] = (torch.maximum(lo_old, lo_t.detach().to(lo_old)),
                              torch.minimum(hi_old, hi_t.detach().to(hi_old)))
            verified_disj_a = {
                di for di in still_open_disj
                if all(spec_lbs.get(qi, -1.0) > 0
                       for qi, _, _ in disj_queries[di])}
            still_open_disj -= verified_disj_a
            dt_a15 = time.perf_counter() - t_a15
            stats.record_timing('alpha_crown_graph', dt_a15)
            if print_progress:
                worst_a = (float(np.min(best_lbs_a))
                           if len(best_lbs_a) else 0.0)
                print(f'Phase 1.5 (joint α-CROWN graph, {n_iters_a} iters, '
                      f'{len(start_nodes_a)} start nodes): {dt_a15:.2f}s  '
                      f'open queries {len(open_q)} -> '
                      f'{sum(1 for qi, _, _ in open_q if spec_lbs.get(qi, -1.0) <= 0)}  '
                      f'worst={worst_a:.4f}', flush=True)
            if not still_open_disj:
                return _make_result('verified', {
                    'time': time.perf_counter() - (deadline - total_timeout),
                    'phase': 'alpha_crown_graph'}, stats)

    # --- Phase 1.6: IBP-refresh ReLU-split BaB on remaining queries ---
    # Per open query: `_ibp_crown_bab` (per-domain IBP intermediate
    # refresh + batched per-domain α-CROWN). Default OFF; enabled by the
    # benchmarks that route Phase 1 through IBP. Runs before Phase 2 —
    # on wide conv nets it is strictly stronger per second than the
    # per-neuron MILP tightener.
    if (bool(getattr(settings, 'milp_graph_ibp_bab_enabled', False))
            and still_open_disj and time_left() > 10.0):
        t_b16 = time.perf_counter()
        open_q16 = [(qi, w, b) for di in sorted(still_open_disj)
                    for (qi, w, b) in disj_queries[di]
                    if spec_lbs.get(qi, -1.0) <= 0]
        _bab_batch = int(getattr(settings, 'milp_graph_ibp_bab_batch', 64))
        _bab_iters = int(getattr(settings, 'milp_graph_ibp_bab_alpha_iters', 8))
        n_closed_16 = 0
        from . import alpha_crown as ac16
        _root_iters = int(getattr(settings,
                                  'milp_graph_ibp_bab_root_alpha_iters', 50))
        _cap16 = _alpha_start_cap(
            settings, max(t[0].numel() for t in sb_bab_base.values()))
        for k16, (qi, w, b) in enumerate(open_q16):
            if time_left() <= 10.0:
                break
            # Converged per-query ROOT α: often closes the query without
            # any splitting, and otherwise warm-starts every BaB
            # domain's α (without it the per-domain Adam budget can't
            # recover the root bound — see `_ibp_crown_bab` docstring).
            # Root α (and the BaB) run on the STABLE base, NOT the
            # cumulatively/jointly tightened `sb` — see sb_bab_base above.
            bbr16 = {li: (sb_bab_base[li][0].detach().cpu().numpy()
                          .astype(np.float64),
                          sb_bab_base[li][1].detach().cpu().numpy()
                          .astype(np.float64))
                     for li in sb_bab_base}
            start16 = [L for L in sorted(bbr16) if L > 0
                       and len(bbr16[L][0]) <= _cap16
                       and bool(((bbr16[L][0] < 0)
                                 & (bbr16[L][1] > 0)).any())]
            un16 = {L: np.where((bbr16[L][0] < 0)
                                & (bbr16[L][1] > 0))[0].tolist()
                    for L in bbr16}
            # Use the BATCHED (single-query) α-CROWN: its sparse/chunked
            # start-node backward peaks at ~12.9 GB on cifar10-normal where
            # the non-batched `run_alpha_crown` peaks at 17.2 GB and OOMs
            # under memory pressure (measured eps2_cnn7 s2/s3). Same root_lb
            # (9566 q3 = -0.0655), and `alpha_params['spec']` is the dense
            # per-layer α the BaB warm-starts from. start16 is already
            # max-layer-capped so the wide cnn7's 131072-neuron layers stay
            # off the dense start-node path.
            _lbs16, root_alpha, root_bounds, _ = ac16.run_alpha_crown_batched(
                gg, xl_g, xh_g, bbr16, np.stack([w]),
                np.array([float(b)], dtype=np.float64), start16, un16,
                device, dtype, n_iters=_root_iters, lr=0.25,
                lr_decay=0.99, early_stop_on_positive=True,
                sparse_alpha=False, time_left_fn=time_left)
            root_lb = float(_lbs16[0])
            # Query-local BaB bounds: the stable base intersected with ONLY
            # this query's own root_bounds. NOT cumulatively folded into the
            # shared base — cross-query / joint tightening makes the
            # no-reforward BaB unreliable (see sb_bab_base).
            sb_q = {L: (lo.clone(), hi.clone())
                    for L, (lo, hi) in sb_bab_base.items()}
            for Lk, (lo_t, hi_t) in root_bounds.items():
                if Lk in sb_q:
                    sb_q[Lk] = (
                        torch.maximum(sb_q[Lk][0],
                                      lo_t.detach().to(sb_q[Lk][0])),
                        torch.minimum(sb_q[Lk][1],
                                      hi_t.detach().to(sb_q[Lk][1])))
            if print_progress:
                print(f'Phase 1.6 root α q{qi}: lb={root_lb:.4f}',
                      flush=True)
            if root_lb > 0:
                spec_lbs[qi] = float(root_lb)
                n_closed_16 += 1
                continue
            alpha_init16 = {
                L: t.detach() for L, t in root_alpha.get('spec', {}).items()}
            # no_reforward routes to the validated `_crown_bab_noreforward`
            # (its own per-domain-ew BaBSR branching + kfsb/multilevel);
            # `_ibp_crown_bab` is kept for the IBP-reforward / loose-root
            # path (per-domain IBP intermediate refresh, width-only split).
            if bool(getattr(settings, 'milp_graph_ibp_bab_no_reforward',
                            False)):
                closed, n_dom, reason = _crown_bab_noreforward(
                    gg, xl_g, xh_g, sb_q, alpha_init16, w, float(b),
                    device, dtype, time_left=lambda: time_left() - 5.0,
                    batch=_bab_batch, main_iters=_bab_iters,
                    cand_iters=int(getattr(
                        settings, 'milp_graph_ibp_bab_cand_iters', 8)),
                    prefilter=int(getattr(
                        settings, 'milp_graph_ibp_bab_prefilter', 12)),
                    multilevel=int(getattr(
                        settings, 'milp_graph_ibp_bab_multilevel', 2)),
                    print_progress=print_progress)
            else:
                closed, n_dom, reason = _ibp_crown_bab(
                    gg, xl_g, xh_g, sb_q, w, float(b), device, dtype,
                    time_left=lambda: time_left() - 5.0, ew_w=None,
                    batch=_bab_batch, alpha_iters=_bab_iters,
                    print_progress=print_progress,
                    alpha_init=alpha_init16 or None,
                    split_deepest=bool(getattr(
                        settings, 'milp_graph_ibp_bab_split_deepest', False)),
                    no_reforward=False)
            if closed:
                spec_lbs[qi] = 1e-9   # marker: closed by BaB
                n_closed_16 += 1
            if print_progress:
                print(f'Phase 1.6 (ibp-bab) q{qi}: '
                      f'{"closed" if closed else "open"} '
                      f'domains={n_dom} ({reason})', flush=True)
            if not closed:
                # Worst-first: if the worst query won't close, siblings
                # in the same disjunct can't rescue the disjunct alone —
                # but other disjuncts might still close; keep going only
                # while time is plentiful.
                continue
        verified_disj_16 = {
            di for di in still_open_disj
            if all(spec_lbs.get(qi, -1.0) > 0
                   for qi, _, _ in disj_queries[di])}
        still_open_disj -= verified_disj_16
        stats.record_timing('ibp_bab', time.perf_counter() - t_b16)
        if print_progress:
            print(f'Phase 1.6 (ibp-bab): '
                  f'{time.perf_counter() - t_b16:.2f}s  closed '
                  f'{n_closed_16}/{len(open_q16)} queries  still-open '
                  f'disjuncts={len(still_open_disj)}', flush=True)
        if not still_open_disj:
            return _make_result('verified', {
                'time': time.perf_counter() - (deadline - total_timeout),
                'phase': 'ibp_bab_graph'}, stats)

    # --- Phase 2: Per-layer tightening ---
    t_phase2_start = time.perf_counter()
    bounds_by_relu = {}
    for li in range(nh):
        lo, hi = sb[li]
        bounds_by_relu[li] = (lo.cpu().numpy().astype(np.float64),
                               hi.cpu().numpy().astype(np.float64))

    # Serialize ops for MILP workers
    gg_ops_ser = []
    for op in gg['ops']:
        d = {'name': op['name'], 'type': op['type'], 'inputs': op['inputs']}
        if op['type'] == 'conv':
            d['kernel_np'] = op['kernel_np']
            d['bias_np'] = op['bias_np']
            d['in_shape'] = op['in_shape']
            d['out_shape'] = op['out_shape']
            d['stride'] = op['stride']
            d['padding'] = op['padding']
            d['n_out'] = op['n_out']
        elif op['type'] == 'fc':
            d['W_np'] = op['W_np']
            d['bias_np'] = op['bias_np']
        elif op['type'] == 'relu':
            if 'layer_idx' in op:
                d['layer_idx'] = op['layer_idx']
        elif op['type'] == 'add':
            d['is_merge'] = op.get('is_merge', False)
            # Carry the bias for non-merge bias-adds (ACASXU / safenlp
            # ONNXes export `Gemm → Add(bias)` separately). Dropping
            # this used to silently encode the network bias-less in
            # MILP workers → falsely-infeasible spec LPs.
            d['bias'] = op.get('bias')
        elif op['type'] == 'sub':
            d['bias'] = op.get('bias')
        elif op['type'] == 'reshape':
            pass    # flat passthrough; consumers alias the input vars
        else:
            raise NotImplementedError(
                f"milp serializer: unsupported op {op['type']!r} at "
                f"{op['name']!r} — serializing without its params would "
                f"make the MILP encode a different network")
        gg_ops_ser.append(d)

    x_lo_64 = spec.x_lo.astype(np.float64)
    x_hi_64 = spec.x_hi.astype(np.float64)
    n_cores = multiprocessing.cpu_count()
    sample_timeout = getattr(settings, 'milp_sample_timeout', 5.0)

    # Tighten each relu layer — but skip if CROWN gap is too large
    # (tightening won't close a large gap, MILP escalation is needed)

    total_build_time = 0.0
    total_solve_time = 0.0

    # Per-layer MILP tightening can be disabled per-benchmark: on nets
    # with very wide conv layers it burns the whole budget for no bound
    # movement (cct tinyimagenet: 546 s, worst lb -1.80 -> -1.79) — the
    # α phase above is the effective tightener there.
    _tighten_on = bool(getattr(settings, 'milp_graph_tighten_enabled', True))

    for li in range(nh if _tighten_on else 0):
        if time_left() <= 1:
            break

        # Skip first relu layer: bounds are exact (no prior relu relaxation)
        # Check if any prior relu exists in the graph path to this relu
        has_prior_relu = li > 0

        lo, hi = bounds_by_relu[li]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        if len(unstable) == 0 or not has_prior_relu:
            continue

        if print_progress:
            print(f'  Tightening relu {li}: {len(unstable)} unstable...', end='')

        t_tight_start = time.perf_counter()

        # Check if subgraph to this relu is purely sequential (no merge-Add)
        has_add_before = False
        for op in gg_ops_ser:
            if op['type'] == 'relu' and op.get('layer_idx') == li:
                break
            if op['type'] == 'add' and op.get('is_merge'):
                has_add_before = True

        if not has_add_before:
            # Sequential subgraph — delegate to fast sequential tightening
            # Only include ops that are ancestors of the target relu
            target_relu_name = None
            for op in gg_ops_ser:
                if op['type'] == 'relu' and op.get('layer_idx') == li:
                    target_relu_name = op['name']
                    break

            # Walk backward from target relu to find ancestor ops
            ancestors = set()
            stack = [target_relu_name]
            op_by_name = {op['name']: op for op in gg_ops_ser}
            while stack:
                n = stack.pop()
                if n in ancestors:
                    continue
                ancestors.add(n)
                op = op_by_name.get(n)
                if op:
                    for inp in op['inputs']:
                        stack.append(inp)

            layers_np_seq = []
            for op in gg_ops_ser:
                if op['name'] not in ancestors:
                    continue
                if op['type'] == 'relu' and op.get('layer_idx') == li:
                    break
                if op['type'] == 'conv':
                    layers_np_seq.append({
                        'type': 'conv', 'kernel': op['kernel_np'],
                        'bias': op['bias_np'].copy(),
                        'in_shape': op['in_shape'],
                        'stride': op['stride'], 'padding': op['padding']})
                elif op['type'] == 'fc':
                    layers_np_seq.append({
                        'type': 'fc', 'W': op['W_np'],
                        'bias': op['bias_np'].copy()})
                elif (op['type'] == 'add' and not op.get('is_merge')
                        and op.get('bias') is not None
                        and layers_np_seq):
                    # Fold this constant bias into the preceding fc/conv's
                    # bias. ONNX exports `Gemm(no bias) → Add(bias) →
                    # ReLU` for ACASXU / safenlp; without folding, the
                    # sequential tightener sees a bias-less network and
                    # produces UNSOUND tight bounds (proved neurons
                    # always positive when reality says they straddle 0).
                    bias_add = np.asarray(op['bias']).flatten().astype(
                        layers_np_seq[-1]['bias'].dtype)
                    layers_np_seq[-1]['bias'] = (
                        layers_np_seq[-1]['bias'] + bias_add)
            seq_li = len(layers_np_seq) - 1
            seq_bounds = {}
            rc = 0
            for op in gg_ops_ser:
                if op['name'] not in ancestors:
                    continue
                if op['type'] == 'relu' and 'layer_idx' in op:
                    if op['layer_idx'] < li:
                        seq_bounds[rc] = bounds_by_relu[op['layer_idx']]
                        rc += 1
                    elif op['layer_idx'] == li:
                        seq_bounds[rc] = bounds_by_relu[li]
                        break

            lp_pw = getattr(settings, 'milp_lp_per_worker', True)
            is_conv = layers_np_seq[seq_li]['type'] == 'conv'

            # Probe MILP vs LP
            n_samp = min(n_cores, len(unstable))
            samp = np.random.RandomState(li).choice(unstable, n_samp, replace=False)
            half = n_samp // 2
            milp_to = True
            if half > 0 and is_conv:
                _, _, milp_to = _tighten_layer_parallel(
                    layers_np_seq, x_lo_64, x_hi_64, seq_bounds, seq_li,
                    use_milp=True, timeout=sample_timeout,
                    n_cores=n_cores, neuron_subset=samp[:half],
                    witness_n_random=_wnr, deadline=deadline)

            if not milp_to and is_conv:
                new_lo, new_hi, _ = _tighten_layer_parallel(
                    layers_np_seq, x_lo_64, x_hi_64, seq_bounds, seq_li,
                    use_milp=True, timeout=sample_timeout, n_cores=n_cores,
                    witness_n_random=_wnr, deadline=deadline)
                method_label = 'MILP-sparse'
            else:
                new_lo, new_hi, _ = _tighten_layer_parallel(
                    layers_np_seq, x_lo_64, x_hi_64, seq_bounds, seq_li,
                    use_milp=False, timeout=sample_timeout,
                    n_cores=n_cores, lp_per_worker=lp_pw,
                    witness_n_random=_wnr, deadline=deadline)
                method_label = 'LP-seq'

            tightened = int(np.sum((new_lo >= 0) | (new_hi <= 0)))
            orig_tight = int(np.sum((lo >= 0) | (hi <= 0)))
            bounds_by_relu[li] = (new_lo, new_hi)
            sb[li] = (torch.tensor(new_lo, dtype=dtype, device=device),
                       torch.tensor(new_hi, dtype=dtype, device=device))
            new_ust = int(np.sum((new_lo < 0) & (new_hi > 0)))
            dt_tight = time.perf_counter() - t_tight_start
            total_solve_time += dt_tight
            if print_progress:
                print(f' → {new_ust} unstable '
                      f'({tightened - orig_tight} fixed, {dt_tight:.1f}s [{method_label}])')
            continue

        # Graph subgraph with Add nodes — use graph model builder
        # But only if the model isn't too large. For deep graph layers,
        # the model can have 40k+ vars which makes LP too slow.
        # Skip if we've already spent significant time on tightening.
        if time_left() < 10:
            if print_progress:
                print(f' skip (time)')
            continue

        new_lo = lo.copy()
        new_hi = hi.copy()
        relu_op = None
        for op in gg_ops_ser:
            if op['type'] == 'relu' and op.get('layer_idx') == li:
                relu_op = op
                break
        target_name = relu_op['inputs'][0]

        t_build = time.perf_counter()
        global _graph_shared_model, _graph_shared_target_indices

        # Build LP model — if build alone exceeds sample_timeout, skip
        m_shared, env_shared, tvars = _build_graph_model_to_relu(
            gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
            li, gg['input_name'], gg['fork_points'], use_milp=False)
        dt_model_build = time.perf_counter() - t_build

        if dt_model_build > sample_timeout:
            m_shared.dispose()
            env_shared.dispose()
            if print_progress:
                print(f' skip (build={dt_model_build:.1f}s)')
            continue

        # Probe: solve 1 neuron, estimate total layer time
        import gurobipy as grb
        probe_j = int(unstable[0])
        skip_layer = False
        if tvars and probe_j < len(tvars) and tvars[probe_j] is not None:
            cm = m_shared.copy()
            cm.setObjective(cm.getVars()[tvars[probe_j].index], grb.GRB.MINIMIZE)
            cm.setParam('TimeLimit', sample_timeout)
            t_probe = time.perf_counter()
            optimize_checked(cm)
            dt_probe = time.perf_counter() - t_probe
            timed_out = cm.Status == 9
            cm.dispose()

            if timed_out:
                skip_layer = True
            else:
                est_time = (len(unstable) / n_cores) * dt_probe * 2
                if est_time > time_left() * 0.5:
                    skip_layer = True

        if skip_layer:
            m_shared.dispose()
            env_shared.dispose()
            if print_progress:
                print(f' skip (probe={dt_probe:.1f}s, est={est_time:.0f}s)')
            continue

        # Map neuron index → Gurobi var index (-1 if dead)
        target_indices = {}
        if tvars:
            for j in range(len(tvars)):
                target_indices[j] = tvars[j].index if tvars[j] is not None else -1
        _graph_shared_model = m_shared
        _graph_shared_target_indices = target_indices
        dt_build = time.perf_counter() - t_build

        t_solve = time.perf_counter()
        tasks = [(int(j), sample_timeout, float(lo[j]), float(hi[j]))
                 for j in unstable]
        chunksize = max(1, len(tasks) // (n_cores * 4))
        with multiprocessing.Pool(n_cores) as pool:
            results = pool.map(_tighten_neuron_graph, tasks,
                               chunksize=chunksize)

        dt_solve = time.perf_counter() - t_solve
        _graph_shared_model = None
        _graph_shared_target_indices = None
        n_vars = m_shared.NumVars
        n_constrs = m_shared.NumConstrs
        m_shared.dispose()
        env_shared.dispose()

        for j, lb_j, ub_j in results:
            new_lo[j] = max(new_lo[j], lb_j)
            new_hi[j] = min(new_hi[j], ub_j)

        tightened = int(np.sum((new_lo >= 0) | (new_hi <= 0)))
        orig_tightened = int(np.sum((lo >= 0) | (hi <= 0)))
        bounds_by_relu[li] = (new_lo, new_hi)
        sb[li] = (torch.tensor(new_lo, dtype=dtype, device=device),
                   torch.tensor(new_hi, dtype=dtype, device=device))
        new_ust = int(np.sum((new_lo < 0) & (new_hi > 0)))
        dt_tight = time.perf_counter() - t_tight_start
        total_build_time += dt_build
        total_solve_time += dt_solve
        method_label = 'LP-graph'
        if print_progress:
            print(f' → {new_ust} unstable '
                  f'({tightened - orig_tightened} fixed, {dt_tight:.1f}s '
                  f'[{method_label} build={dt_build:.1f}s solve={dt_solve:.1f}s '
                  f'model={n_vars}v/{n_constrs}c])')

    t_phase2 = time.perf_counter() - t_phase2_start
    stats.record_timing('tightening', t_phase2)
    stats.record_timing('tighten_build', total_build_time)
    stats.record_timing('tighten_solve', total_solve_time)
    stats.record_bounds(sb)
    if print_progress:
        print(f'Phase 2 (tightening): {t_phase2:.2f}s')

    # Re-run CROWN spec backward with tightened bounds
    if t_phase2 > 0.1:
        t_recheck = time.perf_counter()
        with torch.no_grad():
            spec_lbs, still_open_q = _spec_backward_graph(
                sb, xl_g, xh_g, gg, spec_ew, all_qids, nh, device, dtype)
        stats.record_timing('crown_recheck', time.perf_counter() - t_recheck)
        verified_disj = set()
        for di, qlist in disj_queries.items():
            if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist):
                verified_disj.add(di)
        still_open_disj = set(disj_queries.keys()) - verified_disj
        if print_progress:
            worst = min(spec_lbs.values()) if spec_lbs else 0
            print(f'Phase 3a (CROWN recheck): '
                  f'verified={len(verified_disj)}/{n_total}  worst={worst:.4f}')
        if not still_open_disj:
            return _make_result('verified', {
                'time': time.perf_counter() - (deadline - total_timeout),
                'phase': 'crown_tightened_graph'}, stats)

    # --- Phase 3: MILP escalation for remaining queries ---
    # Skippable per-benchmark: on wide conv nets the spec MILP cannot
    # finish, overruns the CLI deadline (observed NOFILE verdicts at
    # ext-kill: vc6/probe4), and Gurobi numeric trouble aborts the case.
    if not bool(getattr(settings, 'milp_graph_escalation_enabled', True)):
        return _make_result('unknown', {
            'time': time.perf_counter() - (deadline - total_timeout),
            'phase': 'no_escalation',
            'remaining': len(still_open_disj)}, stats)
    # Collect still-open query indices
    remaining_qids = set()
    for di in still_open_disj:
        for qi, _, _ in disj_queries[di]:
            if spec_lbs.get(qi, -1) <= 0:
                remaining_qids.add(qi)

    # Prepare numpy bounds for MILP
    bounds_by_relu = {}
    for li in range(nh):
        lo, hi = sb[li]
        bounds_by_relu[li] = (lo.cpu().numpy().astype(np.float64),
                               hi.cpu().numpy().astype(np.float64))
    # Float-point-soundness inflation: these bounds become *hard* variable
    # bounds in the spec MILP/LP, which recomputes the affine in float64 while
    # the bounds are float32. Without inflation, near-degenerate bounds (tiny
    # perturbation box → nearly-constant neurons) exclude genuinely reachable
    # points → false `verified` (observed: collins_rul). See _inflate_milp_bounds.
    bounds_by_relu = _inflate_milp_bounds(
        bounds_by_relu,
        float(getattr(settings, 'milp_bound_inflation_atol', 1e-5)),
        float(getattr(settings, 'milp_bound_inflation_rtol', 1e-5)))

    # Score neurons: solve LP for each open query, score by fractional gap
    per_query_scored = {}

    x_lo_64 = spec.x_lo.astype(np.float64)
    x_hi_64 = spec.x_hi.astype(np.float64)
    n_cores = multiprocessing.cpu_count()

    # Serialize ops for worker (strip torch tensors, keep numpy)
    gg_ops_ser = []
    for op in gg['ops']:
        d = {'name': op['name'], 'type': op['type'], 'inputs': op['inputs']}
        if op['type'] == 'conv':
            d['kernel_np'] = op['kernel_np']
            d['bias_np'] = op['bias_np']
            d['in_shape'] = op['in_shape']
            d['out_shape'] = op['out_shape']
            d['stride'] = op['stride']
            d['padding'] = op['padding']
            d['n_out'] = op['n_out']
        elif op['type'] == 'fc':
            d['W_np'] = op['W_np']
            d['bias_np'] = op['bias_np']
        elif op['type'] == 'relu':
            if 'layer_idx' in op:
                d['layer_idx'] = op['layer_idx']
        elif op['type'] == 'add':
            d['is_merge'] = op.get('is_merge', False)
            d['bias'] = op.get('bias')
        elif op['type'] == 'sub':
            d['bias'] = op.get('bias')
        elif op['type'] == 'reshape':
            pass    # flat passthrough; consumers alias the input vars
        else:
            raise NotImplementedError(
                f"milp serializer: unsupported op {op['type']!r} at "
                f"{op['name']!r} — serializing without its params would "
                f"make the MILP encode a different network")
        gg_ops_ser.append(d)

    # Precompute sparse matrices (after GPU memory is freed by tightening)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])

    # Get CROWN effective weights at each relu for scoring
    with torch.no_grad():
        _, _, ew_at_relu = _spec_backward_graph(
            sb, xl_g, xh_g, gg, spec_ew,
            remaining_qids, nh, device, dtype, return_ew=True)

    # Score neurons per-query: solve LP per query, extract frac values
    t_score_start = time.perf_counter()
    for qi in sorted(remaining_qids):
        if time_left() <= 2:
            break
        _, q_w, q_bias = queries[qi]

        score_args = ('score', gg_ops_ser, x_lo_64, x_hi_64,
                      bounds_by_relu, q_w, q_bias, [], 0, 1,
                      min(30, time_left()), gg['input_name'],
                      gg['fork_points'])
        _, score_dt, frac_scores = _solve_spec_graph_worker(score_args)

        q_ew = ew_at_relu.get(qi, {})
        q_scores = {}
        if frac_scores:
            for (li, j), frac in frac_scores.items():
                ew_j = abs(float(q_ew.get(li, np.zeros(1))[j])) if li in q_ew and j < len(q_ew[li]) else 1.0
                q_scores[(li, j)] = frac * ew_j
        else:
            for li in range(nh):
                lo, hi = bounds_by_relu[li]
                unstable = np.where((lo < 0) & (hi > 0))[0]
                for i in unstable:
                    q_scores[(li, int(i))] = float(hi[i]) * abs(float(lo[i])) / 2
        per_query_scored[qi] = sorted(
            q_scores.keys(), key=lambda k: q_scores[k], reverse=True)

    dt_score_total = time.perf_counter() - t_score_start
    if print_progress:
        print(f'  Scoring: {len(remaining_qids)} queries ({dt_score_total:.1f}s)')

    n_binaries = 0
    for qi in sorted(remaining_qids):
        if time_left() <= 0:
            break
        _, q_w, q_bias = queries[qi]
        scored_keys = per_query_scored.get(qi, [])

        if print_progress:
            print(f'  MILP query {qi} (disjunct {queries[qi][0]}):')

        verified, n_bins = _racing_escalation_graph(
            gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
            q_w, q_bias, scored_keys, n_cores, time_left,
            gg['input_name'], gg['fork_points'], print_progress)
        if verified:
            spec_lbs[qi] = 1.0  # mark as verified
            n_binaries = max(n_binaries, n_bins)

    # Conjunctive disjuncts: a disjunct with >1 conjunct is unsafe only where
    # ALL its halfspaces hold simultaneously (their INTERSECTION). The per-query
    # loop above verifies it only if some single conjunct's halfspace is empty;
    # when each halfspace alone is reachable but the intersection is empty
    # (sat_relu's "Y_0>=1 AND Y_1<=0"), prove the JOINT feasibility MILP
    # infeasible instead. Sound: relaxation ∩ (all halfspaces) = ∅ ⇒ disjunct
    # safe (feas-mode worker adds every conjunct as a constraint).
    for di in sorted(disj_queries):
        qlist = disj_queries[di]
        if len(qlist) < 2:
            continue  # single-conjunct already handled per-query
        if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist):
            continue  # already verified (some single halfspace was empty)
        if time_left() <= 0:
            break
        qw_2d = np.stack([w for _, w, _ in qlist]).astype(np.float64)
        qb_1d = np.array([b for _, _, b in qlist], dtype=np.float64)
        # Neurons to binarise: reuse a member query's scored keys; else
        # box-area score over all unstable so the bin schedule can escalate to
        # exact binarisation.
        conj_scored = []
        for qi0, _, _ in qlist:
            if per_query_scored.get(qi0):
                conj_scored = per_query_scored[qi0]
                break
        if not conj_scored:
            sc = {}
            for li in range(nh):
                lo, hi = bounds_by_relu[li]
                for i in np.where((lo < 0) & (hi > 0))[0]:
                    sc[(li, int(i))] = float(hi[i]) * abs(float(lo[i])) / 2
            conj_scored = sorted(sc, key=lambda k: sc[k], reverse=True)
        if print_progress:
            print(f'  MILP conjunctive disjunct {di} '
                  f'({len(qlist)} conjuncts):')
        verified, n_bins = _racing_escalation_graph(
            gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
            qw_2d, qb_1d, conj_scored, n_cores, time_left,
            gg['input_name'], gg['fork_points'], print_progress,
            feas_only=True)
        if verified:
            for qi, _, _ in qlist:
                spec_lbs[qi] = 1.0
            n_binaries = max(n_binaries, n_bins)

    # Re-check which disjuncts are now fully verified
    verified_disj = set()
    for di, qlist in disj_queries.items():
        if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist):
            verified_disj.add(di)
    still_open_disj = set(disj_queries.keys()) - verified_disj

    t_escalation = time.perf_counter() - t_phase2_start - t_phase2
    stats.record_timing('milp_escalation', t_escalation)

    t_total = time.perf_counter() - (deadline - total_timeout)
    if not still_open_disj:
        return _make_result('verified', {'time': t_total, 'phase': 'milp_graph',
                            'n_binaries': n_binaries}, stats)
    return _make_result('unknown', {'time': t_total, 'phase': 'milp_graph_timeout',
                        'remaining': len(still_open_disj)}, stats)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def milp_verify(graph, spec, settings=None):
    """MILP verification pipeline.

    Returns ('verified'|'unknown'|'sat', details_dict).
    """
    if settings is None:
        settings = default_settings()
    device, dtype = resolve_torch(settings)
    torch.set_num_threads(1)
    total_timeout = settings.total_timeout
    print_progress = settings.print_progress
    deadline = time.perf_counter() + total_timeout

    # Dispatch to graph-aware path for networks with skip connections,
    # OR for specs that don't fit the "single pred, many comps" shape
    # the sequential path requires (e.g. acasxu prop_2 has 4 different
    # preds with comp=0 — "COC is maximal" form). The graph path uses
    # `spec.as_linear_queries` which handles arbitrary DNF specs.
    # IBP routing: large-input nets whose zonotope forward cannot fit GPU
    # memory go through the graph path, where Phase 1 swaps in
    # `_ibp_forward_graph` (see settings.phase1_ibp_input_dim_threshold).
    _ibp_thresh = int(getattr(settings, 'phase1_ibp_input_dim_threshold', 0))
    _use_ibp = (bool(getattr(settings, 'milp_force_ibp_phase1', False))
                or (_ibp_thresh > 0
                    and int(np.prod(spec.x_lo.shape)) >= _ibp_thresh))
    # `milp_force_graph_path`: route through the graph path with the
    # normal zonotope Phase 1 — for benchmarks whose hard cases need the
    # graph path's α/BaB phases and clean deadline behavior while their
    # input is small enough for the (tighter) zonotope forward
    # (challenging_certified_training cifar10: IBP Phase 1 starts at
    # worst -8.6 where zono is near-verifying).
    _force_graph = bool(getattr(settings, 'milp_force_graph_path', False))

    if (graph.fork_points() or spec.as_pairwise() is None or _use_ibp
            or _force_graph):
        return _milp_verify_graph(graph, spec, settings, device, dtype,
                                   deadline, total_timeout)

    sample_timeout = settings.milp_sample_timeout
    lp_per_worker = getattr(settings, 'milp_lp_per_worker', True)
    _wnr = (int(getattr(settings, 'tighten_witness_n_random', 8))
            if getattr(settings, 'tighten_witness_ordering', True) else 0)
    n_cores = multiprocessing.cpu_count()

    def time_left():
        return max(0, deadline - time.perf_counter())

    pred, comps = spec.as_pairwise()

    gpu_layers_list, fwd_data = graph.gpu_layers(device, dtype)
    nh = len(gpu_layers_list) - 1

    spec_ew = _build_spec_ew(gpu_layers_list, pred, comps, device, dtype)

    x_lo_f32 = spec.x_lo.astype(np.float32)
    x_hi_f32 = spec.x_hi.astype(np.float32)
    xl_g = torch.tensor(x_lo_f32, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi_f32, dtype=dtype, device=device)

    # --- Phase 1: GPU zonotope + CROWN ---
    t0 = time.perf_counter()
    spec_lbs, still_open, _ = _evaluate_region(
        xl_g, xh_g, set(comps), gpu_layers_list, spec_ew,
        pred, nh, device, dtype)
    t_phase1 = time.perf_counter() - t0

    if print_progress:
        verified_count = len(comps) - len(still_open)
        worst = min(spec_lbs.values()) if spec_lbs else 0
        print(f'Phase 1 (zonotope+CROWN): {t_phase1:.2f}s  '
              f'verified={verified_count}/{len(comps)}  '
              f'worst={worst:.4f}')

    if not still_open:
        return 'verified', {'time': time.perf_counter() - (deadline - total_timeout),
                            'phase': 'crown'}

    if time_left() <= 0:
        return 'unknown', {'time': total_timeout, 'phase': 'crown_timeout'}

    # --- PGD attack (GPU, fast) ---
    # Use `pgd_attack_general` (spec-aware) instead of the legacy
    # `_pgd_attack`. The legacy function flattens the spec to (pred,
    # comps_set) via `as_pairwise`, losing the disjunct structure, and
    # checks SAT as `min(Y_pred - Y_comp) < 0` over comps — that's OR
    # semantics. For AND-conjunct specs (acasxu prop_3: single
    # disjunct with 4 pairwise constraints) this falsely claims SAT
    # when ANY constraint is in the unsafe direction, even if siblings
    # are provably safe (witness on prop_3 / `1_1` had Y_3 > Y_0,
    # disproving the AND, but legacy PGD accepted it because Y_1, Y_2,
    # Y_4 < Y_0). `pgd_attack_general` uses `spec.check` in its
    # confirm step, so it correctly handles both AND and OR conjuncts.
    _disable_sat = bool(getattr(settings, 'disable_sat_finding', False))
    pgd_sat, pgd_witness = False, None
    if not _disable_sat:
        t_pgd = time.perf_counter()
        gg_for_pgd = graph.gpu_graph(device, dtype)
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_g, xh_g, spec, gg_for_pgd, settings)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        if print_progress:
            print(f'PGD attack: {time.perf_counter()-t_pgd:.2f}s  '
                  f'sat={pgd_sat}')
        if pgd_sat:
            return 'sat', {'time': time.perf_counter() - (deadline - total_timeout),
                            'phase': 'pgd', 'witness': pgd_witness}

    # --- Extract bounds to CPU numpy ---
    # Re-run zonotope forward to get sb bounds (Phase 1 of _evaluate_region)
    z = TorchZonotope.from_input_bounds(xl_g, xh_g, device, dtype)
    sb = {}
    for l in range(nh):
        gl = gpu_layers_list[l]
        if gl['type'] == 'conv':
            z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                             gl['stride'], gl['padding'])
        else:
            z.propagate_fc(gl['W'], gl['bias'])
        lo, hi = z.apply_relu()
        sb[l] = (lo.clone(), hi.clone())

    # Also get the tightened bounds from CROWN backward
    # (re-run Phase 2 of _evaluate_region to get tight)
    tight_gpu = {0: (sb[0][0].clone(), sb[0][1].clone())} if nh > 0 else {}
    for l in range(1, nh):
        lo_std, hi_std = sb[l]
        ust_idx = torch.where((lo_std < 0) & (hi_std > 0))[0]
        if len(ust_idx) == 0:
            tight_gpu[l] = (lo_std.clone(), hi_std.clone())
            continue
        # Use sb bounds (skip full backward tighten for speed; Phase 1 already
        # ran the full _evaluate_region which does tightening)
        tight_gpu[l] = (lo_std.clone(), hi_std.clone())

    # Transfer to CPU float64 for Gurobi
    bounds_np = {}
    for l in range(nh):
        lo_l = sb[l][0].cpu().numpy().astype(np.float64)
        hi_l = sb[l][1].cpu().numpy().astype(np.float64)
        bounds_np[l] = (lo_l, hi_l)

    # --- Phase 1.5: α-CROWN intermediate-bound tightening ---
    # Joint α optimization (Adam, ~10 iters on GPU) tightens every
    # unstable layer's pre-ReLU bounds in 1-2s. On conv ResNets this
    # closes more specs than the per-layer LP/MILP loop (Phase 2)
    # despite running 10-20× faster, because:
    #   1. It tightens deeper layers that LP per-neuron times out on.
    #   2. It optimizes α slopes jointly across layers, preserving spec
    #      direction sensitivity (per-neuron LP can degrade joint
    #      consistency — observed worst-LB *worsening* on img3039 from
    #      -3.97 to -5.17 after Phase 2).
    # Phase 2 still runs after this on the tightened bbr to catch
    # remaining unstables that α-CROWN didn't close.
    alpha_tighten = bool(getattr(settings, 'milp_alpha_tighten', True))
    if alpha_tighten and time_left() > 5.0:
        t_at = time.perf_counter()
        from . import alpha_crown as ac
        gg = graph.gpu_graph(device, dtype)
        xl_at = torch.as_tensor(spec.x_lo.flatten().astype(np.float64),
                                 device=device, dtype=dtype)
        xh_at = torch.as_tensor(spec.x_hi.flatten().astype(np.float64),
                                 device=device, dtype=dtype)
        # Build linear queries from spec.
        # n_out from the last *linear* op (fc or conv); the network's
        # last op may be an `add` (separate bias-add common in ACASXU
        # / safenlp ONNXes), which has no `out_shape` and would have
        # silently defaulted to 10 — wrong for non-10-output nets, and
        # produced a downstream shape mismatch in α-CROWN backward
        # (ew @ bias_at_add: ew shape (4, 10), bias (5,) → crash).
        n_out = None
        for op in reversed(gg['ops']):
            if op.get('type') == 'fc':
                W = op.get('W_np') if 'W_np' in op else op.get('W')
                if W is not None:
                    n_out = int(W.shape[0]); break
            if op.get('type') == 'conv':
                _no = int(op.get('n_out', 0))
                if _no > 0:
                    n_out = _no; break
        assert n_out is not None, 'no fc/conv op found for n_out'
        queries_at = spec.as_linear_queries(n_out)
        if queries_at:
            w_qs_at = np.stack([q[1] for q in queries_at])
            b_qs_at = np.array([q[2] for q in queries_at])
            # Use the bbr from sb_init (zono forward) for α-CROWN.
            # Note: gg uses ReLU layer_idx as keys; sb uses 0..nh-1.
            # On non-fork networks these align (every linear layer
            # produces a ReLU at the same index).
            bbr_at = {l: bounds_np[l] for l in bounds_np}
            start_nodes = [Lk for Lk in bbr_at if Lk > 0 and
                            ((bbr_at[Lk][0] < 0)
                             & (bbr_at[Lk][1] > 0)).any()]
            un_idx = {Lk: np.where((bbr_at[Lk][0] < 0)
                                     & (bbr_at[Lk][1] > 0))[0].tolist()
                       for Lk in bbr_at}
            if start_nodes:
                n_iters_at = int(getattr(settings,
                                          'milp_alpha_tighten_iters', 10))
                # `best_lbs_joint` is the joint α-CROWN's spec-direction
                # lower bounds — Phase 1.5 always optimised them as part
                # of the joint pass (the 'spec' start_node is appended
                # inside run_alpha_crown_batched), pre-2026-05-11 the
                # caller silently dropped them via `_, _, best_bounds, _`.
                # On oval21 deep_kw img3782 those joint-spec lbs close
                # 7/9 OR specs in ~5 s (matching AB-CROWN's
                # `prune_after_crown` flow), saving the per-layer LP/MILP
                # work that was previously the entire 60 s budget.
                best_lbs_joint, _, best_bounds, _ = ac.run_alpha_crown_batched(
                    gg, xl_at, xh_at, bbr_at, w_qs_at, b_qs_at,
                    start_nodes, un_idx,
                    device, dtype, n_iters=n_iters_at, lr=0.25,
                    lr_decay=0.98, early_stop_on_positive=False)
                # Merge tightened bounds (max lo, min hi)
                n_tightened = 0
                for Lk in best_bounds:
                    lo_t, hi_t = best_bounds[Lk]
                    lo_a = lo_t.detach().cpu().numpy().astype(np.float64)
                    hi_a = hi_t.detach().cpu().numpy().astype(np.float64)
                    if Lk in bounds_np:
                        lo_g, hi_g = bounds_np[Lk]
                        new_lo = np.maximum(lo_g, lo_a)
                        new_hi = np.minimum(hi_g, hi_a)
                        n_tightened += int(((lo_g < 0) & (hi_g > 0)
                                             & ~((new_lo < 0)
                                                 & (new_hi > 0))).sum())
                        bounds_np[Lk] = (new_lo, new_hi)
                if print_progress:
                    print(f'Phase 1.5 (α-CROWN tighten, {n_iters_at} iters): '
                          f'{time.perf_counter()-t_at:.2f}s  '
                          f'closed {n_tightened} unstables')

                # Short-circuit + still_open prune: joint α-CROWN's
                # spec lbs let us mark verified queries early so Phase 2
                # only works on the remainder. The query order from
                # `as_linear_queries` is the same `comps` order milp_verify
                # tracks (`spec_ew` is built from `comps` in order).
                worst_lb_joint = float(best_lbs_joint.min()) if len(
                    best_lbs_joint) > 0 else -1e30
                n_closed_joint = int((best_lbs_joint > 0).sum()) if len(
                    best_lbs_joint) > 0 else 0
                # Only safe to prune if query count matches comp count
                # AND the order is the per-comp scan; on more elaborate
                # multi-disjunct specs the mapping differs and we just
                # skip the prune (still get the short-circuit when ALL
                # close).
                if len(best_lbs_joint) == len(comps):
                    comps_list = sorted(comps)
                    for qi, lb_q in enumerate(best_lbs_joint):
                        if float(lb_q) > 0:
                            still_open.discard(comps_list[qi])
                if print_progress:
                    print(f'  joint-α spec lbs: closed '
                          f'{n_closed_joint}/{len(best_lbs_joint)} queries '
                          f'worst={worst_lb_joint:.4f}  '
                          f'still_open now={len(still_open)}')
                if worst_lb_joint > 0:
                    return 'verified', {
                        'time': time.perf_counter() - (deadline
                                                         - total_timeout),
                        'phase': 'alpha_crown_joint',
                    }

    # Extract numpy layer weights
    layers_np = []
    for gl in gpu_layers_list:
        d = {'type': gl['type']}
        if gl['type'] == 'fc':
            d['W'] = gl['W'].cpu().numpy().astype(np.float64)
            d['bias'] = gl['bias'].cpu().numpy().astype(np.float64)
        else:
            d['kernel'] = gl['kernel'].cpu().numpy().astype(np.float64)
            d['bias'] = gl['bias'].cpu().numpy().astype(np.float64)
            d['in_shape'] = gl['in_shape']
            d['stride'] = gl['stride']
            d['padding'] = gl['padding']
        layers_np.append(d)

    x_lo_64 = spec.x_lo.astype(np.float64)
    x_hi_64 = spec.x_hi.astype(np.float64)

    # --- Phase 2: Per-layer tightening ---
    # Skip Phase 2 entirely when Phase 1.5's joint α-CROWN already
    # closed most specs. Phase 2 tightens unstable neurons across ALL
    # layers regardless of which spec needs them — wasted work when
    # only 1-2 specs remain open. Going straight to Phase 5/8 spec MILP
    # racing on the few remaining queries is much faster.
    # Threshold: skip if ≤ 33 % of comps still open. On oval21 deep_kw
    # img3782 the joint-α leaves 2/9 open (22 %), the per-layer LP/MILP
    # would burn the entire 60 s budget; with the skip Phase 5 racing
    # closes the remainder in seconds.
    _skip_phase2_thr_frac = 0.33
    if (len(still_open) <= int(_skip_phase2_thr_frac * len(comps))
            and len(still_open) > 0):
        if print_progress:
            print(f'Phase 2 SKIPPED: {len(still_open)}/{len(comps)} '
                  f'still open after Phase 1.5 — going straight to '
                  f'Phase 5 spec MILP')
        # Jump to Phase 5 by leaving bounds_np as-is (still tightened by
        # Phase 1.5). The for-loop below is a no-op via the early break.
        nh_iter = 0
    else:
        nh_iter = nh
    t_phase2_start = time.perf_counter()
    tighten_mode = 'sample'  # 'sample', 'lp', 'zono'

    for l in range(nh_iter):
        if time_left() <= 0:
            break

        # Skip first layer: bounds are exact from zonotope (no prior ReLU)
        if l == 0:
            continue

        lo, hi = bounds_np[l]
        unstable = np.where((lo < 0) & (hi > 0))[0]
        if len(unstable) == 0:
            continue

        if tighten_mode == 'sample':
            # Sample n_cores neurons: half MILP, half LP
            n_sample = min(n_cores, len(unstable))
            sample_idx = np.random.RandomState(l).choice(
                unstable, n_sample, replace=False)
            half = n_sample // 2

            # For FC layers after a conv layer used MILP, skip MILP
            # sample (too slow at deeper layers) and go straight to LP
            is_fc = layers_np[l]['type'] == 'fc'
            skip_milp_sample = is_fc and l > 0

            # Sample MILP on first half (unless skipped)
            milp_any_timeout = skip_milp_sample
            if half > 0 and not skip_milp_sample:
                _, _, milp_any_timeout = _tighten_layer_parallel(
                    layers_np, x_lo_64, x_hi_64, bounds_np, l,
                    use_milp=True, timeout=sample_timeout,
                    n_cores=n_cores, neuron_subset=sample_idx[:half],
                    witness_n_random=_wnr, deadline=deadline)

            # Sample LP on second half
            lp_any_timeout = False
            if n_sample - half > 0:
                _, _, lp_any_timeout = _tighten_layer_parallel(
                    layers_np, x_lo_64, x_hi_64, bounds_np, l,
                    use_milp=False, timeout=sample_timeout,
                    n_cores=n_cores, neuron_subset=sample_idx[half:],
                    lp_per_worker=lp_per_worker, witness_n_random=_wnr, deadline=deadline)

            if not milp_any_timeout:
                if print_progress:
                    print(f'  L{l}: {len(unstable)} unstable, MILP OK'
                          f' → solving all with MILP')
                new_lo, new_hi, _ = _tighten_layer_parallel(
                    layers_np, x_lo_64, x_hi_64, bounds_np, l,
                    use_milp=True, timeout=sample_timeout,
                    n_cores=n_cores, witness_n_random=_wnr, deadline=deadline)
                bounds_np[l] = (new_lo, new_hi)
            elif not lp_any_timeout:
                if print_progress:
                    print(f'  L{l}: {len(unstable)} unstable, LP OK'
                          f' → solving all with LP')
                new_lo, new_hi, _ = _tighten_layer_parallel(
                    layers_np, x_lo_64, x_hi_64, bounds_np, l,
                    use_milp=False, timeout=sample_timeout,
                    n_cores=n_cores, lp_per_worker=lp_per_worker,
                    witness_n_random=_wnr, deadline=deadline)
                bounds_np[l] = (new_lo, new_hi)
                tighten_mode = 'lp'
            else:
                if print_progress:
                    print(f'  L{l}: {len(unstable)} unstable, both timeout'
                          f' → zono (skip)')
                tighten_mode = 'zono'

        elif tighten_mode == 'lp':
            if print_progress:
                print(f'  L{l}: {len(unstable)} unstable, LP mode')
            new_lo, new_hi, any_to = _tighten_layer_parallel(
                layers_np, x_lo_64, x_hi_64, bounds_np, l,
                use_milp=False, timeout=sample_timeout,
                n_cores=n_cores, lp_per_worker=lp_per_worker,
                witness_n_random=_wnr, deadline=deadline)
            bounds_np[l] = (new_lo, new_hi)
            if any_to:
                tighten_mode = 'zono'

        # zono mode: skip (bounds_np[l] stays as zonotope bounds)

        if print_progress and tighten_mode != 'zono':
            new_lo, new_hi = bounds_np[l]
            new_ust = np.sum((new_lo < 0) & (new_hi > 0))
            print(f'    → {len(unstable)} → {new_ust} unstable '
                  f'({time.perf_counter() - t_phase2_start:.1f}s elapsed)')

    t_phase2 = time.perf_counter() - t_phase2_start
    if print_progress:
        print(f'Phase 2 (tightening): {t_phase2:.2f}s  mode={tighten_mode}')

    if time_left() <= 0:
        return 'unknown', {'time': total_timeout, 'phase': 'tighten_timeout'}

    # --- Phase 3: Spec verification with tightened bounds ---

    # Transfer tightened bounds back to GPU
    tight_gpu = {}
    for l in range(nh):
        lo_np, hi_np = bounds_np[l]
        tight_gpu[l] = (
            torch.tensor(lo_np, dtype=dtype, device=device),
            torch.tensor(hi_np, dtype=dtype, device=device),
        )

    # Re-run CROWN spec backward with tightened bounds
    t_spec_start = time.perf_counter()
    spec_lbs, still_open = _spec_backward(
        tight_gpu, xl_g, xh_g, gpu_layers_list, spec_ew,
        still_open, nh, device, dtype)

    if print_progress:
        verified_now = len(comps) - len(still_open)
        worst = min(spec_lbs[c] for c in still_open) if still_open else 0
        print(f'Phase 3a (CROWN recheck): {time.perf_counter()-t_spec_start:.2f}s  '
              f'verified={verified_now}/{len(comps)}  worst={worst:.4f}')

    if not still_open:
        return 'verified', {
            'time': time.perf_counter() - (deadline - total_timeout),
            'phase': 'crown_tightened'}

    # --- PGD attack again (seeded with best adversarial from first PGD) ---
    # Same fix as the first PGD call above — use spec-aware general PGD
    # to avoid the OR-flatten-AND unsoundness in legacy `_pgd_attack`.
    if time_left() > 0 and not _disable_sat:
        t_pgd2 = time.perf_counter()
        try:
            pgd_sat2, pgd_witness2 = _pgd_attack_general(
                xl_g, xh_g, spec, gg_for_pgd, settings)
        except RuntimeError:
            pgd_sat2, pgd_witness2 = False, None
        if print_progress:
            print(f'PGD attack (post-tighten): {time.perf_counter()-t_pgd2:.2f}s  '
                  f'sat={pgd_sat2}')
        if pgd_sat2:
            return 'sat', {'time': time.perf_counter() - (deadline - total_timeout),
                            'phase': 'pgd_post_tighten', 'witness': pgd_witness2}

    # --- Score neurons for spec MILP ---
    remaining = set(still_open)

    use_lp = settings.milp_scoring in ('crown_lp_fractional', 'ew_frac')

    # Compute per-comp scoring
    per_comp_sorted = {}
    for comp in remaining:
        ew_at_layer = _compute_crown_layer_weights(
            bounds_np, layers_np, spec_ew, pred, comp, nh)

        if use_lp:
            t_lp0 = time.perf_counter()
            lp_m, lp_e = _build_spec_model_compact(
                layers_np, x_lo_64, x_hi_64, bounds_np, pred, comp,
                milp_neurons=set(), n_threads=1)
            lp_m.setParam('TimeLimit', min(30, time_left()))
            optimize_checked(lp_m)
            dt_lp = time.perf_counter() - t_lp0

            if lp_m.status == 2:
                if settings.milp_scoring == 'ew_frac':
                    comp_scores = score_neurons_ew_frac(
                        bounds_np, ew_at_layer, nh, lp_m)
                else:
                    comp_scores = score_neurons_crown_lp_fractional(
                        bounds_np, ew_at_layer, nh, lp_m)
                if print_progress:
                    print(f'  {settings.milp_scoring} scoring comp={comp}: '
                          f'{dt_lp:.2f}s')
            else:
                comp_scores = score_neurons_by_crown(
                    bounds_np, ew_at_layer, nh)
            lp_m.dispose(); lp_e.dispose()
        else:
            comp_scores = score_neurons_by_crown(
                bounds_np, ew_at_layer, nh)

        per_comp_sorted[comp] = sorted(
            comp_scores.keys(), key=lambda k: comp_scores[k], reverse=True)

    # --- Phase 3b+3c: Escalation per comp ---
    n_binaries = 0
    for comp in sorted(remaining):
        if time_left() <= 0:
            break
        sorted_neurons = per_comp_sorted[comp]
        verified, n_bins = _racing_escalation(
            layers_np, x_lo_64, x_hi_64, bounds_np, pred, comp,
            sorted_neurons, n_cores, time_left, print_progress)
        if verified:
            remaining.discard(comp)
            n_binaries = max(n_binaries, n_bins)

    t_total = time.perf_counter() - (deadline - total_timeout)
    if not remaining:
        return 'verified', {'time': t_total, 'phase': 'spec_milp',
                            'n_binaries': n_binaries}
    return 'unknown', {'time': t_total, 'phase': 'spec_milp_timeout',
                        'remaining': len(remaining)}
