"""Graph verification mode: zonotope + CROWN + MILP on DAG-structured networks.

This module is a parallel implementation of the graph pipeline found in
`verify_milp._milp_verify_graph`, with two independent LP/MILP model builders
(a readable reference builder and a batched optimized builder) sharing exactly
the same op-walking, dead-neuron propagation, and variable-ref tracking.

Both builders preserve the Bug #1 fix: a conv/fc whose inputs are all dead
still outputs `bias[j]`, encoded as a fixed-bound Gurobi variable — never
returned as `None`.

All `optimize_checked(m)` calls go through `optimize_checked(m)` — they set
`DualReductions=0` (Bug #5), raise on unexpected Gurobi status codes, and
raise `GurobiNumericTrouble` if the solver emits any numeric-trouble
warnings (Markowitz / basis drops / quad precision).
"""

import os
import time
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F

from .settings import resolve_torch
from .gurobi_util import optimize_checked
from .verify_milp import (
    VerifyStats, _fire_callback,
    _compute_dead_at, _conv_sparse_matrix,
    _pgd_attack_general, _tighten_layer_parallel,
)
from .verify_zono_bnb import (
    _forward_zonotope_graph, _spec_backward_graph, _make_slopes,
    _find_shared_gens_count,
)
from .zonotope import TorchZonotope, make_input_zonotope
from . import verify_gen_lp


_OK_STATUSES_GLOBAL = None


def _ok_statuses():
    """Return the set of Gurobi status codes considered non-error."""
    global _OK_STATUSES_GLOBAL
    if _OK_STATUSES_GLOBAL is None:
        import gurobipy as grb
        _OK_STATUSES_GLOBAL = (
            grb.GRB.OPTIMAL, grb.GRB.INFEASIBLE, grb.GRB.TIME_LIMIT,
            grb.GRB.USER_OBJ_LIMIT, grb.GRB.SUBOPTIMAL,
            grb.GRB.INTERRUPTED, grb.GRB.NUMERIC, grb.GRB.ITERATION_LIMIT,
        )
    return _OK_STATUSES_GLOBAL


# ---------------------------------------------------------------------------
# Serialized op list
# ---------------------------------------------------------------------------

def _serialize_gg_ops(gg):
    """Strip torch tensors from gg['ops'], keep numpy-only fields.

    Returns a picklable list suitable for multiprocessing workers.
    """
    out = []
    for op in gg['ops']:
        d = {'name': op['name'], 'type': op['type'], 'inputs': list(op['inputs'])}
        t = op['type']
        if t == 'conv':
            d['kernel_np'] = op['kernel_np']
            d['bias_np'] = op['bias_np']
            d['in_shape'] = op['in_shape']
            d['out_shape'] = op['out_shape']
            d['stride'] = op['stride']
            d['padding'] = op['padding']
            d['n_out'] = op['n_out']
        elif t == 'fc':
            d['W_np'] = op['W_np']
            d['bias_np'] = op['bias_np']
        elif t == 'relu':
            if 'layer_idx' in op:
                d['layer_idx'] = op['layer_idx']
        elif t == 'add':
            d['is_merge'] = op.get('is_merge', False)
            if not d['is_merge']:
                d['bias'] = op.get('bias')
        elif t == 'sub':
            d['bias'] = op.get('bias')
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Reference builder: per-neuron addVar/addConstr (readable)
# ---------------------------------------------------------------------------

def _conv_bias_const(op, j):
    """Per-output bias for conv output neuron j."""
    spatial = op['out_shape'][1] * op['out_shape'][2]
    return float(op['bias_np'][j // spatial])


def _dead_constant(op, j):
    """Constant value of op's output neuron j when all inputs are dead.

    Only called for conv/fc ops (enforced by the builders). For both types
    the answer is the per-output bias.
    """
    if op['type'] == 'conv':
        return _conv_bias_const(op, j)
    return float(op['bias_np'][j])


def _build_reference(gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
                     *, target_layer_idx=None, use_milp=False,
                     milp_by_layer=None, n_threads=1):
    """Reference builder: per-neuron addVar + addConstr.

    Walks ops, creating one Gurobi variable per (non-dead) neuron and one
    equality constraint per linear output. Dead conv/fc outputs are encoded
    as fixed-bound variables carrying the bias (Bug #1 fix).

    Args:
        gg_ops: serialized op list (see _serialize_gg_ops)
        x_lo, x_hi: numpy input bounds
        bounds_by_relu: {layer_idx: (lo, hi)} pre-activation bounds
        input_name: gg['input_name']
        target_layer_idx: if set, stop after building the input-side linear
            op of that relu layer. Returns target_vars for that layer.
            If None, build the full model.
        use_milp: if True, unstable relus get binary encoding (MILP)
        milp_by_layer: optional {layer_idx: set(neuron_idx)} — only those
            neurons get MILP encoding; others use LP triangle
        n_threads: Gurobi threads

    Returns (model, env, op_var_refs, target_vars).
        target_vars is None unless target_layer_idx was given and matched.
    """
    import gurobipy as grb
    inf = grb.GRB.INFINITY

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    inp_vars = [m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
                for i in range(len(x_lo))]
    m.update()

    dead_at = _compute_dead_at(gg_ops, bounds_by_relu)

    target_input_name = None
    if target_layer_idx is not None:
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
                W_sp = op.get('W_sp')
                if W_sp is None:
                    W_sp = _conv_sparse_matrix(
                        op['kernel_np'], op['in_shape'],
                        op['stride'], op['padding'])
                    op['W_sp'] = W_sp
            else:
                n_out = op['W_np'].shape[0]

            my_dead = dead_at.get(nm)
            if my_dead is not None and len(my_dead) != n_out:
                my_dead = None
            all_prev_dead = all(p is None for p in prev)

            out = [None] * n_out
            for j in range(n_out):
                if my_dead is not None and my_dead[j]:
                    continue
                if all_prev_dead:
                    # Bug #1: output is the bias, not zero — fixed-bound var
                    c = _dead_constant(op, j)
                    v = m.addVar(lb=c, ub=c)
                    out[j] = v
                    continue

                expr = grb.LinExpr()
                if t == 'conv':
                    row = W_sp.getrow(j)
                    for fi, w in zip(row.indices, row.data):
                        if fi < n_prev and prev[fi] is not None:
                            expr.add(prev[fi], float(w))
                    b_j = _conv_bias_const(op, j)
                else:
                    W = op['W_np']
                    for k in range(n_prev):
                        wjk = W[j, k]
                        if wjk != 0 and prev[k] is not None:
                            expr.add(prev[k], float(wjk))
                    b_j = float(op['bias_np'][j])

                if expr.size() == 0:
                    # Bug #1: all live inputs have zero weight — constant bias
                    v = m.addVar(lb=b_j, ub=b_j)
                    out[j] = v
                    continue

                v = m.addVar(lb=-inf, ub=inf)
                out[j] = v
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
            ms = milp_by_layer.get(li, set()) if milp_by_layer else set()
            n = len(prev)
            out = [None] * n
            for j in range(n):
                if hi_r[j] <= 0:
                    continue
                z = prev[j]
                if z is None:
                    continue
                if lo_r[j] >= 0:
                    out[j] = z
                elif use_milp or j in ms:
                    a = m.addVar(lb=0.0, ub=float(hi_r[j]))
                    s = m.addVar(vtype=grb.GRB.BINARY)
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= float(hi_r[j]) * s)
                    m.addConstr(a <= z - float(lo_r[j]) * (1 - s))
                    out[j] = a
                else:
                    a = m.addVar(lb=0.0, ub=float(hi_r[j]))
                    m.addConstr(a >= z)
                    slope = float(hi_r[j]) / (float(hi_r[j]) - float(lo_r[j]))
                    m.addConstr(a <= slope * z - slope * float(lo_r[j]))
                    out[j] = a
            m.update()
            op_var_refs[nm] = out

        elif t == 'add':
            if op.get('is_merge'):
                va = op_var_refs[op['inputs'][0]]
                vb = op_var_refs[op['inputs'][1]]
                n = len(va)
                out = [None] * n
                for j in range(n):
                    if va[j] is None and vb[j] is None:
                        continue
                    if va[j] is None:
                        out[j] = vb[j]
                        continue
                    if vb[j] is None:
                        out[j] = va[j]
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == va[j] + vb[j])
                    out[j] = v
                m.update()
                op_var_refs[nm] = out
                if nm == target_input_name:
                    target_vars = out
                    break
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'sub':
            prev = op_var_refs[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bias_flat = bias.flatten().astype(np.float64)
                n = len(prev)
                out = [None] * n
                for j in range(n):
                    if prev[j] is None:
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == prev[j] - float(bias_flat[j]))
                    out[j] = v
                m.update()
                op_var_refs[nm] = out
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'reshape':
            op_var_refs[nm] = op_var_refs[op['inputs'][0]]

    m.update()
    return m, env, op_var_refs, target_vars


# ---------------------------------------------------------------------------
# Optimized builder: batched MVar + sparse matrices
# ---------------------------------------------------------------------------

def _build_optimized(gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
                     *, target_layer_idx=None, use_milp=False,
                     milp_by_layer=None, n_threads=1, compact_lp=False):
    """Batched builder using MVar + scipy.sparse for conv/fc layers.

    Produces identical constraints to _build_reference but with dramatically
    fewer Python<->C transitions. For conv layers the sparse weight matrix
    is applied as a single addConstr(out_mvar == W_live @ prev_mvar + b).

    compact_lp: when True, fold dead-branch constants through merge-Add
    ops instead of creating fixed-bound Gurobi variables. Produces fewer
    vars/constrs with identical LP results.

    Same signature and return shape as _build_reference.
    """
    import gurobipy as grb
    import scipy.sparse as sp
    inf = grb.GRB.INFINITY

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    inp_mv = m.addMVar(len(x_lo),
                       lb=x_lo.astype(np.float64),
                       ub=x_hi.astype(np.float64))
    m.update()
    inp_vars = list(inp_mv.tolist())

    dead_at = _compute_dead_at(gg_ops, bounds_by_relu)

    target_input_name = None
    if target_layer_idx is not None:
        for op in gg_ops:
            if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
                target_input_name = op['inputs'][0]
                break

    op_var_refs = {input_name: inp_vars}
    target_vars = None
    const_offset = {}

    for op in gg_ops:
        nm = op['name']
        t = op['type']

        if t in ('conv', 'fc'):
            prev = op_var_refs[op['inputs'][0]]
            n_prev = len(prev)
            if t == 'conv':
                n_out = op['n_out']
                W_sp = op.get('W_sp')
                if W_sp is None:
                    W_sp = _conv_sparse_matrix(
                        op['kernel_np'], op['in_shape'],
                        op['stride'], op['padding'])
                    op['W_sp'] = W_sp
                spatial = op['out_shape'][1] * op['out_shape'][2]
                bias_per_out = np.array(
                    [float(op['bias_np'][j // spatial]) for j in range(n_out)],
                    dtype=np.float64)
            else:
                W_np = op['W_np'].astype(np.float64)
                n_out = W_np.shape[0]
                W_sp = sp.csr_matrix(W_np)
                bias_per_out = op['bias_np'].astype(np.float64)

            my_dead = dead_at.get(nm)
            # _compute_dead_at may return a mask whose length doesn't match
            # n_out (e.g. through a reshape/flatten). Only trust it when
            # lengths agree — otherwise treat all outputs as live.
            if my_dead is not None and len(my_dead) != n_out:
                my_dead = None
            live_in_mask = np.array(
                [p is not None for p in prev], dtype=bool)

            out = [None] * n_out
            all_prev_dead = not live_in_mask.any()

            if all_prev_dead:
                input_offset = const_offset.get(op['inputs'][0])
                if compact_lp:
                    dead_mask = my_dead if my_dead is not None else np.zeros(n_out, dtype=bool)
                    cv = bias_per_out.copy()
                    if input_offset is not None:
                        cv = W_sp @ input_offset + cv
                    cv[dead_mask] = 0.0
                    const_offset[nm] = cv
                    op_var_refs[nm] = out
                    if nm == target_input_name:
                        target_vars = out
                        break
                    continue
                if input_offset is not None:
                    full_bias = W_sp @ input_offset + bias_per_out
                else:
                    full_bias = bias_per_out
                for j in range(n_out):
                    if my_dead is not None and my_dead[j]:
                        continue
                    c = float(full_bias[j])
                    out[j] = m.addVar(lb=c, ub=c)
                m.update()
                op_var_refs[nm] = out
                if nm == target_input_name:
                    target_vars = out
                    break
                continue

            # Fold any stored constant offset from the input into the bias
            input_offset = const_offset.get(op['inputs'][0])
            if compact_lp and input_offset is not None:
                bias_per_out = bias_per_out + np.asarray(
                    W_sp @ input_offset).flatten()

            # Extract submatrix for live predecessors and live outputs
            if my_dead is not None:
                live_out_mask = ~my_dead
            else:
                live_out_mask = np.ones(n_out, dtype=bool)

            W_live = W_sp[live_out_mask][:, live_in_mask]
            # Rows where all weights are zero (expr empty) — Bug #1: constants
            row_nnz = np.asarray(W_live.getnnz(axis=1)).flatten()

            live_out_idx = np.where(live_out_mask)[0]
            live_prev = [prev[i] for i in np.where(live_in_mask)[0]]

            # Rows with nonzero entries use a real constraint; rows with all-zero
            # entries become constant bias vars.
            has_terms_mask = row_nnz > 0
            constr_rows = live_out_idx[has_terms_mask]
            const_rows = live_out_idx[~has_terms_mask]

            if constr_rows.size > 0:
                W_constr = W_live[has_terms_mask]
                b_constr = bias_per_out[constr_rows]
                out_mv = m.addMVar(
                    constr_rows.size, lb=-inf, ub=inf)
                prev_mv = grb.MVar.fromlist(live_prev)
                m.addConstr(out_mv == W_constr @ prev_mv + b_constr)
                for local_j, j in enumerate(constr_rows):
                    out[int(j)] = out_mv[local_j].item()

            for j in const_rows:
                c = float(bias_per_out[int(j)])
                out[int(j)] = m.addVar(lb=c, ub=c)

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
            ms = milp_by_layer.get(li, set()) if milp_by_layer else set()
            n = len(prev)

            # Partition neurons into active / dead / unstable-lp / unstable-milp
            lo_arr = np.asarray(lo_r, dtype=np.float64)
            hi_arr = np.asarray(hi_r, dtype=np.float64)
            active_mask = (lo_arr >= 0) & (hi_arr > 0)
            dead_mask = hi_arr <= 0
            live_prev_mask = np.array(
                [p is not None for p in prev], dtype=bool)
            unstable_mask = (~active_mask) & (~dead_mask) & live_prev_mask
            # Active neurons with dead prev become dead
            active_mask &= live_prev_mask

            out = [None] * n

            # Active: pass through (ReLU is identity when stable-on)
            for j in np.where(active_mask)[0]:
                out[int(j)] = prev[int(j)]

            # Unstable: partition into MILP and LP
            if use_milp:
                milp_mask = unstable_mask.copy()
                lp_mask = np.zeros_like(unstable_mask)
            else:
                milp_sel = np.zeros(n, dtype=bool)
                for j in ms:
                    if 0 <= j < n and unstable_mask[j]:
                        milp_sel[j] = True
                milp_mask = unstable_mask & milp_sel
                lp_mask = unstable_mask & (~milp_sel)

            for j in np.where(milp_mask)[0]:
                z = prev[int(j)]
                a = m.addVar(lb=0.0, ub=float(hi_arr[j]))
                s = m.addVar(vtype=grb.GRB.BINARY)
                m.addConstr(a >= 0)
                m.addConstr(a >= z)
                m.addConstr(a <= float(hi_arr[j]) * s)
                m.addConstr(a <= z - float(lo_arr[j]) * (1 - s))
                out[int(j)] = a

            for j in np.where(lp_mask)[0]:
                z = prev[int(j)]
                a = m.addVar(lb=0.0, ub=float(hi_arr[j]))
                m.addConstr(a >= z)
                slope = float(hi_arr[j]) / (float(hi_arr[j]) - float(lo_arr[j]))
                m.addConstr(a <= slope * z - slope * float(lo_arr[j]))
                out[int(j)] = a

            m.update()
            op_var_refs[nm] = out

        elif t == 'add':
            if op.get('is_merge'):
                va = op_var_refs[op['inputs'][0]]
                vb = op_var_refs[op['inputs'][1]]
                n = len(va)
                off_a = const_offset.get(op['inputs'][0]) if compact_lp else None
                off_b = const_offset.get(op['inputs'][1]) if compact_lp else None
                out = [None] * n
                for j in range(n):
                    if va[j] is None and vb[j] is None:
                        continue
                    if va[j] is None:
                        out[j] = vb[j]
                        continue
                    if vb[j] is None:
                        out[j] = va[j]
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == va[j] + vb[j])
                    out[j] = v
                if off_a is not None or off_b is not None:
                    merged = np.zeros(n, dtype=np.float64)
                    if off_a is not None:
                        merged[:len(off_a)] += off_a
                    if off_b is not None:
                        merged[:len(off_b)] += off_b
                    if merged.any():
                        const_offset[nm] = merged
                m.update()
                op_var_refs[nm] = out
                if nm == target_input_name:
                    target_vars = out
                    break
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'sub':
            prev = op_var_refs[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bias_flat = bias.flatten().astype(np.float64)
                n = len(prev)
                out = [None] * n
                for j in range(n):
                    if prev[j] is None:
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == prev[j] - float(bias_flat[j]))
                    out[j] = v
                m.update()
                op_var_refs[nm] = out
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'reshape':
            op_var_refs[nm] = op_var_refs[op['inputs'][0]]
            if compact_lp and op['inputs'][0] in const_offset:
                const_offset[nm] = const_offset[op['inputs'][0]]

    m.update()
    return m, env, op_var_refs, target_vars


_BUILDERS = {
    'reference': _build_reference,
    'optimized': _build_optimized,
}


# ---------------------------------------------------------------------------
# Per-layer LP tightening using the builder
# ---------------------------------------------------------------------------

_graph_shared_model = None
_graph_shared_target_indices = None


def _tighten_neuron_graph(args):
    """Worker: copy shared model, solve one neuron's min/max bounds.

    Sets DualReductions=0 on the copied model (Bug #5).

    Args is (j, timeout, cur_lo, cur_hi) [legacy bound-asymmetry ordering]
    or (j, timeout, cur_lo, cur_hi, z_w_min, z_w_max) [witness-guided
    ordering, default per `tighten_witness_ordering`]. With witnesses, MIN
    runs first iff all witnesses give z_j ≥ 0 (only direction that can
    prove active); MAX first iff all witnesses ≤ 0; otherwise fall back
    to the |cur_lo|<|cur_hi| asymmetry. Both directions still run for
    genuinely unstable neurons — witness only changes ORDER.
    """
    import gurobipy as grb
    if len(args) >= 6:
        j, timeout, cur_lo, cur_hi, z_w_min, z_w_max = args[:6]
    else:
        j, timeout, cur_lo, cur_hi = args[:4]
        z_w_min = z_w_max = None

    m = _graph_shared_model.copy()
    m.setParam('DualReductions', 0)
    var_idx = _graph_shared_target_indices[j]

    lb, ub = cur_lo, cur_hi
    if var_idx < 0:
        m.dispose()
        return j, lb, ub

    tv = m.getVars()[var_idx]
    m.setParam('TimeLimit', timeout)
    # Pick "proving stable" direction first when witnesses are available.
    min_first = None
    if z_w_min is not None and z_w_max is not None:
        if z_w_min >= -1e-9:
            min_first = True       # witnesses all ≥ 0 → MIN can prove active
        elif z_w_max <= 1e-9:
            min_first = False      # witnesses all ≤ 0 → MAX can prove dead
    if min_first is None:
        min_first = abs(cur_lo) < abs(cur_hi)
    if min_first:
        m.setObjective(tv, grb.GRB.MINIMIZE)
        m.setParam('BestBdStop', 1e-6)
    else:
        m.setObjective(tv, grb.GRB.MAXIMIZE)
        m.setParam('BestBdStop', -1e-6)
    _OK_STATUSES = (
        grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT, grb.GRB.USER_OBJ_LIMIT,
        grb.GRB.SUBOPTIMAL, grb.GRB.INTERRUPTED, grb.GRB.NUMERIC,
        grb.GRB.ITERATION_LIMIT,
    )

    optimize_checked(m)
    status = m.Status
    if status not in _OK_STATUSES:
        raise RuntimeError(f'graph tighten: unexpected gurobi status {status}')
    try:
        b = m.ObjBound
        if m.ModelSense == 1:
            lb = max(lb, b)
        else:
            ub = min(ub, b)
    except Exception:
        pass

    if lb < 0 and ub > 0:
        m.reset()
        # Second direction is the opposite of `min_first`.
        if min_first:
            m.setObjective(tv, grb.GRB.MAXIMIZE)
            m.setParam('BestBdStop', -1e-6)
        else:
            m.setObjective(tv, grb.GRB.MINIMIZE)
            m.setParam('BestBdStop', 1e-6)
        optimize_checked(m)
        status = m.Status
        if status not in _OK_STATUSES:
            raise RuntimeError(
                f'graph tighten: unexpected gurobi status {status}')
        try:
            b = m.ObjBound
            if m.ModelSense == 1:
                lb = max(lb, b)
            else:
                ub = min(ub, b)
        except Exception:
            pass

    m.dispose()
    return j, lb, ub


def _forward_witnesses_graph(gg_ops, witnesses, target_layer_idx, input_name):
    """Forward a batch of witnesses (n_w, n_in_flat) through gg_ops up to
    the pre-activation of the ReLU at `target_layer_idx`. Returns a numpy
    array of shape (n_w, n_neur_at_target).

    Used by `_tighten_layer_graph` to pick MIN/MAX MILP/LP ordering per
    neuron — see `_tighten_neuron_graph`'s witness-guided rule.
    """
    import numpy as np
    n_w = witnesses.shape[0]
    state = {input_name: witnesses.astype(np.float64)}
    target_pre_op = None
    for op in gg_ops:
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            target_pre_op = op['inputs'][0]
            break
    assert target_pre_op is not None, \
        f'no ReLU op with layer_idx={target_layer_idx} in gg_ops'
    for op in gg_ops:
        nm = op['name']; t = op['type']
        if t == 'conv':
            a = state[op['inputs'][0]]
            C_in, H_in, W_in = op['in_shape']
            at = torch.from_numpy(a.reshape(n_w, C_in, H_in, W_in))
            k = op['kernel'].to(at.dtype)
            b = op['bias'].to(at.dtype)
            y = F.conv2d(at, k, bias=b, stride=op['stride'],
                          padding=op['padding']).reshape(n_w, -1)
            state[nm] = y.numpy()
        elif t == 'fc':
            a = state[op['inputs'][0]]
            W = op['W'].numpy().astype(np.float64)
            b = op['bias'].numpy().astype(np.float64)
            state[nm] = a @ W.T + b
        elif t == 'relu':
            a = state[op['inputs'][0]]
            state[nm] = np.maximum(a, 0.0)
        elif t == 'add':
            a = state[op['inputs'][0]]
            if op.get('is_merge'):
                b = state[op['inputs'][1]]
                state[nm] = a + b
            else:
                bias = op.get('bias')
                if bias is not None:
                    state[nm] = a + np.asarray(bias, dtype=np.float64).flatten()
                else:
                    state[nm] = a
        elif t == 'sub':
            a = state[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                state[nm] = a - np.asarray(bias, dtype=np.float64).flatten()
            else:
                state[nm] = a
        elif t == 'reshape':
            state[nm] = state[op['inputs'][0]]
        else:
            # Witness-skipping is fine for unknown ops — caller falls back
            # to the bound-asymmetry heuristic when no witness is provided.
            return None
        if nm == target_pre_op:
            return state[nm]
    return None


def _tighten_layer_graph(gg_ops, x_lo, x_hi, bounds_by_relu,
                          target_layer_idx, unstable, input_name,
                          build_fn, sample_timeout, n_cores, time_left_fn,
                          witness_n_random=8):
    """Tighten one ReLU layer's unstable neurons.

    Builds the graph LP up to the target layer once, then forks a worker
    pool to solve per-neuron min/max via model copy.

    Uses a single-neuron probe to estimate per-layer cost and skips the
    layer if the estimate exceeds half of remaining time.

    Returns dict with keys: new_lo, new_hi, build, probe, solve, skipped.
    """
    import gurobipy as grb
    global _graph_shared_model, _graph_shared_target_indices

    t_build = time.perf_counter()
    m_shared, env_shared, _, tvars = build_fn(
        gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
        target_layer_idx=target_layer_idx, use_milp=False)
    m_shared.setParam('DualReductions', 0)
    dt_build = time.perf_counter() - t_build

    lo, hi = bounds_by_relu[target_layer_idx]
    new_lo = lo.copy()
    new_hi = hi.copy()

    if tvars is None or dt_build > sample_timeout:
        m_shared.dispose()
        env_shared.dispose()
        return {'new_lo': new_lo, 'new_hi': new_hi,
                'build': dt_build, 'probe': 0.0, 'solve': 0.0,
                'skipped': True}

    probe_j = int(unstable[0])
    dt_probe = 0.0
    probe_timed_out = False
    if probe_j < len(tvars) and tvars[probe_j] is not None:
        cm = m_shared.copy()
        cm.setParam('DualReductions', 0)
        cm.setObjective(cm.getVars()[tvars[probe_j].index],
                         grb.GRB.MINIMIZE)
        cm.setParam('TimeLimit', sample_timeout)
        t_probe = time.perf_counter()
        optimize_checked(cm)
        dt_probe = time.perf_counter() - t_probe
        probe_status = cm.Status
        cm.dispose()
        # INFEASIBLE here means the LP's feasible region is empty, which
        # indicates unsound upstream bounds_by_relu (e.g., float32 zonotope
        # rounding errors that produce an under-approximation). This is a
        # soundness bug, not a numerical quirk — raise rather than hide it.
        if probe_status not in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT,
                                 grb.GRB.USER_OBJ_LIMIT, grb.GRB.SUBOPTIMAL,
                                 grb.GRB.INTERRUPTED, grb.GRB.NUMERIC,
                                 grb.GRB.ITERATION_LIMIT):
            m_shared.dispose()
            env_shared.dispose()
            raise RuntimeError(
                f'graph tighten probe: unexpected gurobi status {probe_status}')
        probe_timed_out = probe_status == grb.GRB.TIME_LIMIT

    est_time = (len(unstable) / max(1, n_cores)) * dt_probe * 2
    time_budget = time_left_fn()
    if probe_timed_out or est_time > time_budget * 0.5:
        m_shared.dispose()
        env_shared.dispose()
        return {'new_lo': new_lo, 'new_hi': new_hi,
                'build': dt_build, 'probe': dt_probe, 'solve': 0.0,
                'skipped': True}

    target_indices = {}
    for j in range(len(tvars)):
        target_indices[j] = tvars[j].index if tvars[j] is not None else -1
    _graph_shared_model = m_shared
    _graph_shared_target_indices = target_indices

    # Witness-guided ordering: forward witnesses through the actual ReLU
    # network to layer `target_layer_idx`'s pre-activation. Per-neuron
    # `(z_w_min[j], z_w_max[j])` lets the worker put the proving-stable
    # MILP/LP direction first (the only one BestBdStop can fire on early).
    z_w_min = z_w_max = None
    if witness_n_random > 0:
        rng = np.random.default_rng(7)
        x_lo_np = np.asarray(x_lo).flatten().astype(np.float64)
        x_hi_np = np.asarray(x_hi).flatten().astype(np.float64)
        rand = (x_lo_np[None, :] +
                rng.random((witness_n_random, x_lo_np.size)) *
                (x_hi_np - x_lo_np))
        witnesses = np.vstack([
            rand,
            x_lo_np[None, :], x_hi_np[None, :],
            ((x_lo_np + x_hi_np) / 2)[None, :],
        ])
        z_w = _forward_witnesses_graph(gg_ops, witnesses, target_layer_idx,
                                        input_name)
        if z_w is not None:
            z_w_min = z_w.min(axis=0); z_w_max = z_w.max(axis=0)

    if z_w_min is not None:
        tasks = [(int(j), sample_timeout, float(lo[j]), float(hi[j]),
                  float(z_w_min[j]), float(z_w_max[j]))
                 for j in unstable]
    else:
        tasks = [(int(j), sample_timeout, float(lo[j]), float(hi[j]))
                 for j in unstable]
    chunksize = max(1, len(tasks) // (n_cores * 4))
    t_solve = time.perf_counter()
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(_tighten_neuron_graph, tasks, chunksize=chunksize)
    dt_solve = time.perf_counter() - t_solve

    _graph_shared_model = None
    _graph_shared_target_indices = None
    m_shared.dispose()
    env_shared.dispose()

    for j, lb_j, ub_j in results:
        new_lo[j] = max(new_lo[j], lb_j)
        new_hi[j] = min(new_hi[j], ub_j)

    return {'new_lo': new_lo, 'new_hi': new_hi,
            'build': dt_build, 'probe': dt_probe, 'solve': dt_solve,
            'skipped': False}


def _has_merge_before(gg_ops, target_layer_idx):
    """True if any merge-Add op appears before the target relu layer."""
    for op in gg_ops:
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            return False
        if op['type'] == 'add' and op.get('is_merge'):
            return True
    return False


def _build_sequential_subgraph(gg_ops, target_layer_idx, bounds_by_relu):
    """Extract a sequential layers_np list from gg_ops for fast tightening.

    Returns (layers_np, seq_bounds, seq_li) suitable for the legacy
    _tighten_layer_parallel helper. Only valid when the target relu has
    no merge-Add ancestors (caller's responsibility).
    """
    target_name = None
    for op in gg_ops:
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            target_name = op['name']
            break
    assert target_name is not None

    op_by_name = {op['name']: op for op in gg_ops}
    ancestors = set()
    stack = [target_name]
    while stack:
        n = stack.pop()
        if n in ancestors:
            continue
        ancestors.add(n)
        op = op_by_name.get(n)
        if op:
            for inp in op['inputs']:
                stack.append(inp)

    layers_np = []
    for op in gg_ops:
        if op['name'] not in ancestors:
            continue
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            break
        if op['type'] == 'conv':
            layers_np.append({
                'type': 'conv', 'kernel': op['kernel_np'],
                'bias': op['bias_np'], 'in_shape': op['in_shape'],
                'stride': op['stride'], 'padding': op['padding']})
        elif op['type'] == 'fc':
            layers_np.append({
                'type': 'fc', 'W': op['W_np'], 'bias': op['bias_np']})
    seq_li = len(layers_np) - 1

    seq_bounds = {}
    rc = 0
    for op in gg_ops:
        if op['name'] not in ancestors:
            continue
        if op['type'] == 'relu' and 'layer_idx' in op:
            if op['layer_idx'] < target_layer_idx:
                seq_bounds[rc] = bounds_by_relu[op['layer_idx']]
                rc += 1
            elif op['layer_idx'] == target_layer_idx:
                seq_bounds[rc] = bounds_by_relu[target_layer_idx]
                break
    return layers_np, seq_bounds, seq_li


# ---------------------------------------------------------------------------
# Correct spec-verification worker (uses build_fn, Bug #1 preserved)
# ---------------------------------------------------------------------------

def _build_spec_expression(m, op_var_refs, gg_ops, query_w, query_bias):
    """Build the affine spec expression Σ w_j * y_j + bias from the final
    op's output variables. Respects Bug #1: dead outputs that were encoded
    as fixed-bound variables are kept in the sum (their constant value is
    carried by the variable itself).

    Returns (spec_expr, constant_term).
    """
    import gurobipy as grb
    last_name = gg_ops[-1]['name']
    last_vars = op_var_refs[last_name]
    spec_expr = grb.LinExpr()
    const = float(query_bias)
    for j in range(len(query_w)):
        w = float(query_w[j])
        if w == 0 or j >= len(last_vars):
            continue
        v = last_vars[j]
        if v is None:
            # Should never happen for the final output layer with the Bug #1
            # fix, but safeguard by dropping — the last layer has no
            # downstream consumers, so _compute_dead_at won't mark it.
            continue
        spec_expr.add(v, w)
    return spec_expr, const


def _solve_spec_worker_graph(args):
    """Subprocess worker that builds the full graph LP/MILP via a named
    builder and runs feasibility, optimize, or score mode.

    Respects Bug #1 (dead branches carry bias) and Bug #5 (DualReductions=0,
    status-code guards).

    args = (mode, impl, gg_ops, x_lo, x_hi, bounds_by_relu,
            query_w, query_bias, scored_keys, n_bins, n_threads, timeout,
            input_name)

    Returns (result_str, elapsed, payload) where payload is:
      - feasibility: info_dict (status, n_vars, n_constrs, n_bins)
      - optimize:    info_dict (lb, status, n_vars, n_constrs, n_bins)
      - score:       (scores_dict, lb_or_None)
    """
    import gurobipy as grb
    (mode, impl, gg_ops, x_lo, x_hi, bounds_by_relu, query_w, query_bias,
     scored_keys, n_bins, n_threads, timeout, input_name) = args

    build_fn = _BUILDERS[impl]
    milp_set = set(scored_keys[:n_bins]) if n_bins > 0 else set()
    milp_by_layer = {}
    for (li, ni) in milp_set:
        milp_by_layer.setdefault(li, set()).add(ni)

    t_build = time.perf_counter()
    m, env, op_var_refs, _ = build_fn(
        gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
        target_layer_idx=None,
        use_milp=False,
        milp_by_layer=milp_by_layer,
        n_threads=n_threads,
    )
    dt_build = time.perf_counter() - t_build
    m.setParam('DualReductions', 0)
    m.setParam('TimeLimit', float(timeout))

    spec_expr, const = _build_spec_expression(
        m, op_var_refs, gg_ops, query_w, query_bias)
    m.update()
    model_info = {'n_vars': m.NumVars, 'n_constrs': m.NumConstrs,
                  'n_bins': m.NumBinVars, 'build_time': dt_build}

    t0 = time.perf_counter()

    ok = _ok_statuses()

    if mode == 'feasibility':
        m.addConstr(spec_expr + const <= 0)
        m.setObjective(0, grb.GRB.MINIMIZE)
        optimize_checked(m)
        status = m.Status
        dt = time.perf_counter() - t0
        info = {**model_info, 'status': status}
        m.dispose(); env.dispose()
        assert status in ok, f'feasibility: unexpected status {status}'
        if status == grb.GRB.INFEASIBLE:
            return 'UNSAT', dt, info
        if status == grb.GRB.OPTIMAL:
            return 'SAT', dt, info
        return 'UNKNOWN', dt, info

    if mode == 'optimize':
        m.setParam('BestBdStop', 0.0)
        m.setObjective(spec_expr + const, grb.GRB.MINIMIZE)
        optimize_checked(m)
        status = m.Status
        lb = None
        try:
            lb = float(m.ObjBound)
        except Exception:
            pass
        n_sol = m.SolCount
        dt = time.perf_counter() - t0
        info = {**model_info, 'status': status, 'lb': lb}
        m.dispose(); env.dispose()
        assert status in ok, f'optimize: unexpected status {status}'
        if status in (grb.GRB.OPTIMAL, grb.GRB.USER_OBJ_LIMIT):
            return ('UNSAT' if lb is not None and lb > 0 else 'SAT'), dt, info
        if status == grb.GRB.TIME_LIMIT and n_sol > 0:
            return 'SAT', dt, info
        return 'UNKNOWN', dt, info

    # mode == 'score'
    m.setObjective(spec_expr + const, grb.GRB.MINIMIZE)
    optimize_checked(m)
    status = m.Status
    scores = {}
    lb = None
    if status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT,
                   grb.GRB.USER_OBJ_LIMIT, grb.GRB.SUBOPTIMAL):
        try:
            lb = float(m.ObjBound)
        except Exception:
            pass
        # Read fractional a/z values at each unstable ReLU
        if m.SolCount > 0:
            for op in gg_ops:
                if op['type'] != 'relu' or 'layer_idx' not in op:
                    continue
                li = op['layer_idx']
                lo_r, hi_r = bounds_by_relu[li]
                prev = op_var_refs.get(op['inputs'][0], [])
                relu_out = op_var_refs.get(op['name'], [])
                for j in range(len(relu_out)):
                    if relu_out[j] is None:
                        continue
                    if lo_r[j] >= 0 or hi_r[j] <= 0:
                        continue
                    try:
                        a_val = relu_out[j].X
                        z_val = prev[j].X if prev[j] is not None else 0.0
                        frac = abs(a_val - max(0.0, z_val))
                        scores[(li, j)] = frac
                    except Exception:
                        pass
    dt = time.perf_counter() - t0
    info = {**model_info, 'status': status, 'lb': lb, 'scores': scores}
    m.dispose(); env.dispose()
    assert status in ok, f'score: unexpected status {status}'
    if status == grb.GRB.OPTIMAL and lb is not None and lb > 0:
        return 'UNSAT', dt, info
    if status == grb.GRB.OPTIMAL:
        return 'SAT', dt, info
    return 'UNKNOWN', dt, info


def _racing_escalation_graph_correct(impl, gg_ops, x_lo, x_hi, bounds_by_relu,
                                       query_w, query_bias, scored_keys,
                                       n_cores, time_left_fn, input_name,
                                       print_progress=False):
    """Doubling-bin-schedule MILP escalation with three-way dispatch.

    Bug #4: explicit SAT / UNSAT / UNKNOWN branches, never collapses.
    Uses _solve_spec_worker_graph (Bug #1 safe).
    """
    bin_schedule = []
    b = 2
    while b <= len(scored_keys):
        bin_schedule.append(b)
        b *= 2
    if scored_keys and (not bin_schedule or bin_schedule[-1] < len(scored_keys)):
        bin_schedule.append(len(scored_keys))

    levels = []
    opt_threads = max(1, n_cores - 1)
    for n_bins in bin_schedule:
        tl = time_left_fn()
        if tl <= 0:
            break
        common = (impl, gg_ops, x_lo, x_hi, bounds_by_relu, query_w,
                  query_bias, scored_keys, n_bins)
        feas_args = ('feasibility',) + common + (1, tl, input_name)
        opt_args = ('optimize',) + common + (opt_threads, tl, input_name)

        pool = multiprocessing.Pool(2)
        async_feas = pool.apply_async(_solve_spec_worker_graph, (feas_args,))
        async_opt = pool.apply_async(_solve_spec_worker_graph, (opt_args,))
        winner = None
        while True:
            if async_feas.ready():
                feas_result, feas_dt, feas_info = async_feas.get()
                pool.terminate(); pool.join()
                rec = {'n_bins': n_bins, 'winner': 'feas',
                       'result': feas_result, 'time': feas_dt,
                       'info': feas_info}
                levels.append(rec)
                if feas_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'feas UNSAT ({feas_dt:.1f}s) → verified')
                    return True, n_bins, levels
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'feas {feas_result} ({feas_dt:.1f}s) → escalate')
                break
            if async_opt.ready():
                opt_result, opt_dt, opt_info = async_opt.get()
                pool.terminate(); pool.join()
                opt_lb = opt_info.get('lb') if isinstance(opt_info, dict) else opt_info
                lb_s = f'{opt_lb:.4f}' if opt_lb is not None else '?'
                rec = {'n_bins': n_bins, 'winner': 'opt',
                       'result': opt_result, 'time': opt_dt,
                       'lb': opt_lb, 'info': opt_info}
                levels.append(rec)
                if opt_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'opt lb={lb_s} ({opt_dt:.1f}s) → verified')
                    return True, n_bins, levels
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'opt lb={lb_s} ({opt_dt:.1f}s) → escalate')
                break
            time.sleep(0.05)

    return False, bin_schedule[-1] if bin_schedule else 0, levels


# ---------------------------------------------------------------------------
# Spec-level adaptive zonotope backward bound
# ---------------------------------------------------------------------------
# Same min-area triangle CROWN slopes as `_per_neuron_adaptive_bounds`,
# but the initial ew is the spec's linear weight vector at the final op's
# output (not an identity matrix at a middle layer). Produces a single
# scalar lower bound on `spec_ew @ y + spec_bias` for each query.
#
# This is mathematically *identical* to `verify_zono_bnb._spec_backward_graph`
# (both are CROWN-backward with min-area slopes). The point of having two
# implementations is that (a) they serve different entry points in the
# pipeline and (b) the equality check below is a regression guard: if
# someone changes one implementation's slope math, the other must match.

@torch.no_grad()
def _adaptive_spec_lb(gg, xl, xh, bounds_by_relu, spec_ew, spec_bias,
                       device, dtype):
    """Adaptive-zonotope lower bound on `spec_ew @ y + spec_bias`.

    Walks backward from the final op's output with `spec_ew` as the
    initial linear weight tensor. Uses the same min-area triangle
    slopes (`lo_s`, `up_s`, `up_t` from `_make_slopes`) that the
    per-neuron adaptive pass uses. Returns a Python float.
    """
    ops = gg['ops']
    input_name = gg['input_name']
    last_op_name = ops[-1]['name']

    ew_at = {last_op_name: spec_ew.to(dtype=dtype, device=device).clone()}
    acc = float(spec_bias)

    for op in reversed(ops):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]
        t = op['type']

        if t == 'conv':
            out_shape = op['out_shape']
            kernel = op['kernel'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            ew_4d = ew.reshape(1, *out_shape)
            acc += float(
                (ew_4d.reshape(out_shape[0], -1).sum(dim=-1) * bias).sum())
            ew_back = F.conv_transpose2d(
                ew_4d, kernel, stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).flatten()
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t == 'fc':
            W = op['W'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            acc += float(ew @ bias)
            ew_back = ew @ W
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t == 'relu':
            if 'layer_idx' in op:
                lo_k, hi_k = bounds_by_relu[op['layer_idx']]
                lo_k_t = torch.as_tensor(lo_k, dtype=dtype, device=device)
                hi_k_t = torch.as_tensor(hi_k, dtype=dtype, device=device)
                lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k_t, hi_k_t)
                ep = ew.clamp(min=0)
                en = ew.clamp(max=0)
                acc += float((en * up_t).sum())
                ew_back = ep * lo_s + en * up_s
            else:
                ew_back = ew
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
            else:
                bias = op.get('bias')
                if bias is not None:
                    b_t = torch.tensor(
                        bias.flatten(), dtype=dtype, device=device)
                    acc += float(ew @ b_t)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                b_t = torch.tensor(
                    bias.flatten(), dtype=dtype, device=device)
                acc -= float(ew @ b_t)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'reshape':
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

    ew_inp = ew_at.get(input_name)
    if ew_inp is None:
        return acc
    xl_t = xl.to(dtype=dtype, device=device)
    xh_t = xh.to(dtype=dtype, device=device)
    lb = acc + float(
        ew_inp.clamp(min=0) @ xl_t + ew_inp.clamp(max=0) @ xh_t)
    return lb


# ---------------------------------------------------------------------------
# Per-neuron adaptive zonotope backward bounds
# ---------------------------------------------------------------------------
# Port of the per-neuron adaptive zonotope backward pass from
# verify_zono_bnb._evaluate_region (the sequential-network analogue) to the
# graph/DAG case. For each target neuron at `target_layer_idx` we compute
# tight lower and upper pre-activation bounds by walking backward through
# the op DAG, applying the adaptive ReLU triangle slopes at every
# previously-tightened relu.
#
# Unlike a spec-level CROWN pass (one linear query), this tracks two
# independent linear functions per target neuron — one for the lower
# bound and one for the upper bound — because the sound choice of
# slopes differs by the sign of the weight at each relu.

@torch.no_grad()
def _per_neuron_adaptive_bounds(gg, xl, xh, bounds_by_relu, target_layer_idx,
                                 device, dtype, neuron_subset=None,
                                 return_ew=False):
    """Per-neuron pre-activation bounds at relu `target_layer_idx`.

    Uses the current `bounds_by_relu` for triangle slopes on *earlier*
    layers (so tightened upstream bounds feed forward) and maintains
    separate EW_lb / EW_ub matrices to get both sound lower and sound
    upper bounds in a single backward pass.

    `neuron_subset`: optional 1D numpy array of neuron indices to compute
    bounds for. If None, computes bounds for all neurons at the target
    layer. For wide layers with few unstable neurons, passing a subset
    keeps the EW tensor small.

    Returns (full_lb, full_ub) as numpy float64 arrays of length
    n_total; positions not in the subset keep their original bounds.
    """
    ops = gg['ops']

    target_relu_idx = None
    target_op_name = None
    for i, op in enumerate(ops):
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer_idx:
            target_relu_idx = i
            target_op_name = op['inputs'][0]
            break
    assert target_op_name is not None

    lo0, hi0 = bounds_by_relu[target_layer_idx]
    n_total = int(len(lo0))

    if neuron_subset is None:
        sub = np.arange(n_total)
    else:
        sub = np.asarray(neuron_subset, dtype=np.int64)
    n = int(len(sub))
    if n == 0:
        return lo0.copy(), hi0.copy()

    ew_init = torch.zeros(n, n_total, device=device, dtype=dtype)
    ew_init[torch.arange(n, device=device),
            torch.as_tensor(sub, device=device)] = 1.0
    # ew_at maps op_name → (EW_lb, EW_ub) in the post-op output space.
    ew_at_lb = {target_op_name: ew_init.clone()}
    ew_at_ub = {target_op_name: ew_init}
    acc_lb = torch.zeros(n, device=device, dtype=dtype)
    acc_ub = torch.zeros(n, device=device, dtype=dtype)
    ew_at_relu = {} if return_ew else None

    def _accum(store, inp, new_tensor):
        store[inp] = store.get(inp, torch.zeros_like(new_tensor)) + new_tensor

    for op in reversed(ops[:target_relu_idx]):
        name = op['name']
        if name not in ew_at_lb:
            continue
        ew_lb = ew_at_lb[name]
        ew_ub = ew_at_ub[name]
        t = op['type']

        if t == 'conv':
            out_shape = op['out_shape']
            bias = op['bias'].to(dtype=dtype, device=device)
            kernel = op['kernel'].to(dtype=dtype, device=device)
            acc_lb = acc_lb + (ew_lb.reshape(n, *out_shape).sum(
                dim=(-1, -2)) * bias).sum(dim=-1)
            acc_ub = acc_ub + (ew_ub.reshape(n, *out_shape).sum(
                dim=(-1, -2)) * bias).sum(dim=-1)
            ew_lb_back = F.conv_transpose2d(
                ew_lb.reshape(n, *out_shape), kernel,
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(n, -1)
            ew_ub_back = F.conv_transpose2d(
                ew_ub.reshape(n, *out_shape), kernel,
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(n, -1)
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb_back)
            _accum(ew_at_ub, inp, ew_ub_back)

        elif t == 'fc':
            W = op['W'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            acc_lb = acc_lb + ew_lb @ bias
            acc_ub = acc_ub + ew_ub @ bias
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb @ W)
            _accum(ew_at_ub, inp, ew_ub @ W)

        elif t == 'relu':
            if 'layer_idx' in op:
                if return_ew:
                    ew_at_relu[op['layer_idx']] = (
                        ew_lb.detach().cpu().numpy(),
                        ew_ub.detach().cpu().numpy())
                lo_k, hi_k = bounds_by_relu[op['layer_idx']]
                lo_k_t = torch.as_tensor(lo_k, dtype=dtype, device=device)
                hi_k_t = torch.as_tensor(hi_k, dtype=dtype, device=device)
                lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k_t, hi_k_t)
                ep_lb = ew_lb.clamp(min=0)
                en_lb = ew_lb.clamp(max=0)
                acc_lb = acc_lb + (en_lb * up_t).sum(dim=-1)
                ew_lb_back = ep_lb * lo_s + en_lb * up_s
                ep_ub = ew_ub.clamp(min=0)
                en_ub = ew_ub.clamp(max=0)
                acc_ub = acc_ub + (ep_ub * up_t).sum(dim=-1)
                ew_ub_back = ep_ub * up_s + en_ub * lo_s
            else:
                ew_lb_back = ew_lb
                ew_ub_back = ew_ub
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb_back)
            _accum(ew_at_ub, inp, ew_ub_back)

        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    _accum(ew_at_lb, inp, ew_lb)
                    _accum(ew_at_ub, inp, ew_ub)
            else:
                bias = op.get('bias')
                if bias is not None:
                    b_t = torch.tensor(bias.flatten(), dtype=dtype, device=device)
                    acc_lb = acc_lb + ew_lb @ b_t
                    acc_ub = acc_ub + ew_ub @ b_t
                inp = op['inputs'][0]
                _accum(ew_at_lb, inp, ew_lb)
                _accum(ew_at_ub, inp, ew_ub)

        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                b_t = torch.tensor(bias.flatten(), dtype=dtype, device=device)
                acc_lb = acc_lb - ew_lb @ b_t
                acc_ub = acc_ub - ew_ub @ b_t
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb)
            _accum(ew_at_ub, inp, ew_ub)

        elif t == 'reshape':
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb)
            _accum(ew_at_ub, inp, ew_ub)

    input_name = gg['input_name']
    ew_inp_lb = ew_at_lb.get(
        input_name, torch.zeros(n, len(xl), device=device, dtype=dtype))
    ew_inp_ub = ew_at_ub.get(
        input_name, torch.zeros(n, len(xl), device=device, dtype=dtype))
    xl_t = xl.to(dtype=dtype, device=device)
    xh_t = xh.to(dtype=dtype, device=device)
    lb = acc_lb + ew_inp_lb.clamp(min=0) @ xl_t + ew_inp_lb.clamp(max=0) @ xh_t
    ub = acc_ub + ew_inp_ub.clamp(min=0) @ xh_t + ew_inp_ub.clamp(max=0) @ xl_t
    lb_np = lb.cpu().numpy().astype(np.float64)
    ub_np = ub.cpu().numpy().astype(np.float64)
    full_lb = lo0.copy()
    full_ub = hi0.copy()
    full_lb[sub] = lb_np
    full_ub[sub] = ub_np
    if return_ew:
        return full_lb, full_ub, ew_at_relu
    return full_lb, full_ub


def _per_neuron_adaptive_bounds_chunked(gg, xl, xh, bounds_by_relu,
                                         target_layer_idx, device, dtype,
                                         neuron_subset=None,
                                         chunk_size=None):
    """Chunked wrapper around `_per_neuron_adaptive_bounds` to cap peak GPU
    memory. The inner pass materialises [n_unstable × layer_size × n_layers]
    of EW tensors; on wide nets (resnet_large L1 has 1687 unstable and
    16K-neuron layers) this hits 4+ GB. Chunking at ~256 keeps peak under
    ~600 MB on the same instance (matches α,β-CROWN's auto_LiRPA approach
    of splitting unstable rows across backward passes).

    The chunked result is identical to the single-call result: each chunk
    computes bounds for a disjoint neuron subset, and they're merged into
    the full arrays.
    """
    lo0, hi0 = bounds_by_relu[target_layer_idx]
    full_lb = lo0.copy()
    full_ub = hi0.copy()
    if neuron_subset is None or len(neuron_subset) == 0:
        return _per_neuron_adaptive_bounds(
            gg, xl, xh, bounds_by_relu, target_layer_idx,
            device, dtype, neuron_subset=neuron_subset)
    sub = np.asarray(neuron_subset, dtype=np.int64)
    n = len(sub)
    if chunk_size is None or n <= chunk_size:
        return _per_neuron_adaptive_bounds(
            gg, xl, xh, bounds_by_relu, target_layer_idx,
            device, dtype, neuron_subset=sub)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_sub = sub[start:end]
        lb_chunk, ub_chunk = _per_neuron_adaptive_bounds(
            gg, xl, xh, bounds_by_relu, target_layer_idx,
            device, dtype, neuron_subset=chunk_sub)
        # Each chunk's returned arrays have bounds set only at chunk_sub
        # (and the original lo0/hi0 elsewhere). Merge just those indices.
        full_lb[chunk_sub] = lb_chunk[chunk_sub]
        full_ub[chunk_sub] = ub_chunk[chunk_sub]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return full_lb, full_ub


# ---------------------------------------------------------------------------
# Sparse per-target-neuron graph model (generalization of the sequential
# sparse model to DAGs with merges)
# ---------------------------------------------------------------------------

from .verify_milp import _conv_connections, _conv_bias_idx


def _collect_backward_needs(gg_ops, target_op_name, target_neuron_idx,
                              bounds_by_relu):
    """Walk backward from (target_op, target_neuron) through the op DAG
    and collect, for every upstream op, the set of neuron indices whose
    value is actually needed to compute the target.

    Stops propagation through *dead* ReLU neurons (their output is
    identically zero, so upstream contributions are irrelevant). Live
    merge-Add ops propagate the need to both branches.

    Returns dict: op_name -> set(int neuron indices).
    """
    op_by_name = {op['name']: op for op in gg_ops}
    target_idx = None
    for i, op in enumerate(gg_ops):
        if op['name'] == target_op_name:
            target_idx = i
            break
    assert target_idx is not None

    needed_at = {target_op_name: {int(target_neuron_idx)}}

    for i in range(target_idx, -1, -1):
        op = gg_ops[i]
        name = op['name']
        if name not in needed_at:
            continue
        needed = needed_at[name]
        t = op['type']

        if t == 'conv':
            inp = op['inputs'][0]
            inp_set = needed_at.setdefault(inp, set())
            kernel = op['kernel_np']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            for j in needed:
                for fi, _ in _conv_connections(
                        j, kernel, in_shape, stride, padding):
                    inp_set.add(int(fi))

        elif t == 'fc':
            inp = op['inputs'][0]
            inp_set = needed_at.setdefault(inp, set())
            W = op['W_np']
            for j in needed:
                row = W[j]
                nz = np.nonzero(row)[0]
                inp_set.update(int(k) for k in nz)

        elif t == 'relu':
            inp = op['inputs'][0]
            inp_set = needed_at.setdefault(inp, set())
            if 'layer_idx' in op:
                li = op['layer_idx']
                _, hi = bounds_by_relu[li]
                for j in needed:
                    if j < len(hi) and hi[j] > 0:
                        inp_set.add(int(j))
            else:
                inp_set.update(int(j) for j in needed)

        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    needed_at.setdefault(inp, set()).update(
                        int(j) for j in needed)
            else:
                inp = op['inputs'][0]
                needed_at.setdefault(inp, set()).update(
                    int(j) for j in needed)

        elif t == 'sub':
            inp = op['inputs'][0]
            needed_at.setdefault(inp, set()).update(
                int(j) for j in needed)

        elif t == 'reshape':
            inp = op['inputs'][0]
            needed_at.setdefault(inp, set()).update(
                int(j) for j in needed)

    return needed_at


def _build_sparse_neuron_graph(gg_ops, x_lo, x_hi, bounds_by_relu,
                                 target_op_name, target_neuron_idx,
                                 input_name, use_milp=True, n_threads=1):
    """Build a sparse Gurobi model for a single target pre-activation
    neuron on a graph (possibly with merges).

    Only creates variables for neurons in the backward dependency cone
    of the target (dead ReLU neurons are skipped entirely — they
    contribute 0). Unstable neurons use binary encoding when
    `use_milp=True`, otherwise LP triangle.

    Returns (m, env, target_var). `target_var` is the Gurobi variable
    for the target neuron's pre-ReLU value, on which the caller should
    set its objective.
    """
    import gurobipy as grb

    needed_at = _collect_backward_needs(
        gg_ops, target_op_name, target_neuron_idx, bounds_by_relu)

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)
    m.setParam('DualReductions', 0)

    # Input variables (only the ones actually reached)
    inp_needed = needed_at.get(input_name, set())
    input_vars = {}
    for i in sorted(inp_needed):
        input_vars[i] = m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
    m.update()

    # var_refs[op_name] is a dict idx -> Gurobi var (sparse storage).
    var_refs = {input_name: input_vars}
    target_var = None

    is_target = lambda nm: nm == target_op_name

    for op in gg_ops:
        name = op['name']
        t = op['type']
        if name not in needed_at and not is_target(name):
            continue
        needed = set(needed_at.get(name, set()))
        # The target neuron must be created even if the backward
        # walker didn't put it in `needed` (target gets its own set).
        if is_target(name):
            needed.add(int(target_neuron_idx))

        if t in ('conv', 'fc'):
            prev = var_refs.get(op['inputs'][0], {})
            if t == 'conv':
                kernel = op['kernel_np']
                in_shape = op['in_shape']
                stride = op['stride']
                padding = op['padding']
                bias_np = op['bias_np']
                out_shape = op['out_shape']
                spatial = out_shape[1] * out_shape[2]
            else:
                W = op['W_np']
                bias_np = op['bias_np']

            out = {}
            for j in needed:
                expr = grb.LinExpr()
                if t == 'conv':
                    for fi, w in _conv_connections(
                            j, kernel, in_shape, stride, padding):
                        v = prev.get(int(fi))
                        if v is not None:
                            expr.add(v, float(w))
                    b_j = float(bias_np[j // spatial])
                else:
                    row = W[j]
                    nz = np.nonzero(row)[0]
                    for k in nz:
                        v = prev.get(int(k))
                        if v is not None:
                            expr.add(v, float(row[k]))
                    b_j = float(bias_np[j])

                if expr.size() == 0:
                    v = m.addVar(lb=b_j, ub=b_j)
                else:
                    v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                    m.addConstr(v == expr + b_j)
                out[int(j)] = v
            m.update()
            var_refs[name] = out

        elif t == 'relu':
            prev = var_refs.get(op['inputs'][0], {})
            if 'layer_idx' not in op:
                var_refs[name] = prev
            else:
                li = op['layer_idx']
                lo_r, hi_r = bounds_by_relu[li]
                out = {}
                for j in needed:
                    if j >= len(hi_r) or hi_r[j] <= 0:
                        continue
                    z = prev.get(int(j))
                    if z is None:
                        continue
                    lo_j = float(lo_r[j])
                    hi_j = float(hi_r[j])
                    if lo_j >= 0:
                        a = m.addVar(lb=lo_j, ub=hi_j)
                        m.addConstr(a == z)
                    elif use_milp:
                        a = m.addVar(lb=0.0, ub=hi_j)
                        s = m.addVar(vtype=grb.GRB.BINARY)
                        m.addConstr(a >= 0)
                        m.addConstr(a >= z)
                        m.addConstr(a <= hi_j * s)
                        m.addConstr(a <= z - lo_j * (1 - s))
                    else:
                        a = m.addVar(lb=0.0, ub=hi_j)
                        m.addConstr(a >= z)
                        slope = hi_j / (hi_j - lo_j)
                        m.addConstr(a <= slope * z - slope * lo_j)
                    out[int(j)] = a
                m.update()
                var_refs[name] = out

        elif t == 'add':
            if op.get('is_merge'):
                va = var_refs.get(op['inputs'][0], {})
                vb = var_refs.get(op['inputs'][1], {})
                out = {}
                for j in needed:
                    v_a = va.get(int(j))
                    v_b = vb.get(int(j))
                    if v_a is None and v_b is None:
                        continue
                    expr = grb.LinExpr()
                    if v_a is not None:
                        expr.add(v_a, 1.0)
                    if v_b is not None:
                        expr.add(v_b, 1.0)
                    v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                    m.addConstr(v == expr)
                    out[int(j)] = v
                m.update()
                var_refs[name] = out
            else:
                prev = var_refs.get(op['inputs'][0], {})
                bias = op.get('bias')
                if bias is None:
                    var_refs[name] = prev
                else:
                    bias_flat = bias.flatten().astype(np.float64)
                    out = {}
                    for j in needed:
                        v_prev = prev.get(int(j))
                        if v_prev is None:
                            continue
                        v = m.addVar(
                            lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                        m.addConstr(v == v_prev + float(bias_flat[j]))
                        out[int(j)] = v
                    m.update()
                    var_refs[name] = out

        elif t == 'sub':
            prev = var_refs.get(op['inputs'][0], {})
            bias = op.get('bias')
            if bias is None:
                var_refs[name] = prev
            else:
                bias_flat = bias.flatten().astype(np.float64)
                out = {}
                for j in needed:
                    v_prev = prev.get(int(j))
                    if v_prev is None:
                        continue
                    v = m.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                    m.addConstr(v == v_prev - float(bias_flat[j]))
                    out[int(j)] = v
                m.update()
                var_refs[name] = out

        elif t == 'reshape':
            var_refs[name] = var_refs.get(op['inputs'][0], {})

        if is_target(name):
            target_var = var_refs[name].get(int(target_neuron_idx))
            break

    m.update()
    assert target_var is not None, (
        f'sparse graph build: target neuron {target_op_name}:'
        f'{target_neuron_idx} not reached')
    return m, env, target_var


# Module globals for multiprocessing worker (COW via fork)
_sparse_graph_args = None


def _solve_sparse_neuron_graph_worker(args):
    """Worker: build the sparse graph model for one neuron, solve
    min & max. Uses module-global _sparse_graph_args via COW fork.

    args is `(j, timeout, cur_lo, cur_hi, use_milp)` for the legacy
    bound-asymmetry ordering, or `(j, timeout, cur_lo, cur_hi, use_milp,
    z_w_min, z_w_max)` for witness-guided ordering (default per
    `tighten_witness_ordering=True`).
    """
    if len(args) >= 7:
        j, timeout, cur_lo, cur_hi, use_milp, z_w_min, z_w_max = args[:7]
    else:
        j, timeout, cur_lo, cur_hi, use_milp = args[:5]
        z_w_min = z_w_max = None
    import gurobipy as grb

    (gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, input_name) = \
        _sparse_graph_args

    m, env, tv = _build_sparse_neuron_graph(
        gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, j,
        input_name, use_milp=use_milp, n_threads=1)
    m.setParam('TimeLimit', float(timeout))

    lb, ub = float(cur_lo), float(cur_hi)
    any_timeout = False

    # Pick "proving stable" direction first when witnesses available.
    min_first = None
    if z_w_min is not None and z_w_max is not None:
        if z_w_min >= -1e-9: min_first = True
        elif z_w_max <= 1e-9: min_first = False
    if min_first is None:
        min_first = abs(cur_lo) < abs(cur_hi)
    if min_first:
        first, second = grb.GRB.MINIMIZE, grb.GRB.MAXIMIZE
    else:
        first, second = grb.GRB.MAXIMIZE, grb.GRB.MINIMIZE

    def _record(b, direction):
        nonlocal lb, ub
        if b is None:
            return
        if direction == grb.GRB.MINIMIZE:
            lb = max(lb, b)
        else:
            ub = min(ub, b)

    m.setObjective(tv, first)
    m.setParam('BestBdStop',
                1e-6 if first == grb.GRB.MINIMIZE else -1e-6)
    optimize_checked(m)
    if m.Status == grb.GRB.TIME_LIMIT:
        any_timeout = True
    try:
        _record(m.ObjBound, first)
    except Exception:
        pass

    if lb < 0 and ub > 0:
        m.reset()
        m.setObjective(tv, second)
        m.setParam('BestBdStop',
                    1e-6 if second == grb.GRB.MINIMIZE else -1e-6)
        optimize_checked(m)
        if m.Status == grb.GRB.TIME_LIMIT:
            any_timeout = True
        try:
            _record(m.ObjBound, second)
        except Exception:
            pass

    m.dispose()
    env.dispose()
    return int(j), lb, ub, any_timeout


def _probe_sparse_neuron(gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name,
                           probe_j, input_name, use_milp, sample_timeout):
    """Build and solve a probe sparse model for one neuron.

    Returns (dt_build, dt_probe, status). status is the Gurobi status
    code after optimize, or None on build timeout.
    """
    import gurobipy as grb
    t_build0 = time.perf_counter()
    m, env, tv = _build_sparse_neuron_graph(
        gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, probe_j,
        input_name, use_milp=use_milp, n_threads=1)
    dt_build = time.perf_counter() - t_build0
    if dt_build > sample_timeout:
        m.dispose()
        env.dispose()
        return dt_build, 0.0, None
    m.setObjective(tv, grb.GRB.MINIMIZE)
    m.setParam('TimeLimit', sample_timeout)
    t_probe0 = time.perf_counter()
    optimize_checked(m)
    dt_probe = time.perf_counter() - t_probe0
    status = m.Status
    m.dispose()
    env.dispose()
    return dt_build, dt_probe, status


def _tighten_layer_graph_sparse(gg_ops, x_lo, x_hi, bounds_by_relu,
                                  target_layer_idx, unstable, input_name,
                                  mode, sample_timeout, n_cores, time_left,
                                  witness_n_random=8):
    """Per-target-neuron sparse tightening on a graph with merges.

    Each unstable neuron gets its own sparse Gurobi model covering only
    its backward dependency cone (dead neurons dropped). Models are
    solved in parallel by `_solve_sparse_neuron_graph_worker`.

    `mode` controls solver choice, mirroring
    `_tighten_sequential_with_probe`:
      - 'probe': probe MILP; on failure fall through to LP probe
      - 'milp' : probe MILP only; on timeout return 'skip' (no LP
        fallback — caller explicitly asked for MILP)
      - 'lp'   : skip MILP, probe LP only
      - 'skip' : noop

    Returns (new_lo, new_hi, method, dt_build, dt_probe, dt_solve).
    method is one of {'milp', 'lp', 'skip', 'milp-partial', 'lp-partial'}.
    """
    import gurobipy as grb
    global _sparse_graph_args

    lo0, hi0 = bounds_by_relu[target_layer_idx]
    if mode == 'skip' or len(unstable) == 0:
        return lo0.copy(), hi0.copy(), 'skip', 0.0, 0.0, 0.0

    target_op_name = None
    for op in gg_ops:
        if (op['type'] == 'relu'
                and op.get('layer_idx') == target_layer_idx):
            target_op_name = op['inputs'][0]
            break
    assert target_op_name is not None

    probe_j = int(unstable[0])
    dt_build_total = 0.0
    dt_probe_total = 0.0
    per_neuron_est = 0.0  # winning-method-only cost for budget estimate
    use_milp = False

    if mode in ('probe', 'milp'):
        # Try MILP.
        dt_b, dt_p, status = _probe_sparse_neuron(
            gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name,
            probe_j, input_name, use_milp=True,
            sample_timeout=sample_timeout)
        dt_build_total += dt_b
        dt_probe_total += dt_p
        if status is not None and status != grb.GRB.TIME_LIMIT:
            use_milp = True
            per_neuron_est = dt_b + dt_p
        elif mode == 'milp':
            # No LP fallback requested — bail.
            return (lo0.copy(), hi0.copy(), 'skip',
                    dt_build_total, dt_probe_total, 0.0)
        # Else (mode=='probe'): fall through to LP probe below.

    if not use_milp:
        dt_b, dt_p, status = _probe_sparse_neuron(
            gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name,
            probe_j, input_name, use_milp=False,
            sample_timeout=sample_timeout)
        dt_build_total += dt_b
        dt_probe_total += dt_p
        if status is None or status == grb.GRB.TIME_LIMIT:
            return (lo0.copy(), hi0.copy(), 'skip',
                    dt_build_total, dt_probe_total, 0.0)
        per_neuron_est = dt_b + dt_p

    # Budget guard: estimate from the *winning* solver's cost only.
    est_full = (len(unstable) / max(1, n_cores)) * per_neuron_est * 2
    if est_full > time_left() * 0.5:
        return (lo0.copy(), hi0.copy(), 'skip',
                dt_build_total, dt_probe_total, 0.0)

    _sparse_graph_args = (
        gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, input_name)

    # Witness-guided ordering — see `_solve_sparse_neuron_graph_worker`.
    z_w_min = z_w_max = None
    if witness_n_random > 0:
        rng = np.random.default_rng(7)
        x_lo_np = np.asarray(x_lo).flatten().astype(np.float64)
        x_hi_np = np.asarray(x_hi).flatten().astype(np.float64)
        rand = (x_lo_np[None, :] +
                rng.random((witness_n_random, x_lo_np.size)) *
                (x_hi_np - x_lo_np))
        witnesses = np.vstack([
            rand,
            x_lo_np[None, :], x_hi_np[None, :],
            ((x_lo_np + x_hi_np) / 2)[None, :],
        ])
        z_w = _forward_witnesses_graph(gg_ops, witnesses,
                                        target_layer_idx, input_name)
        if z_w is not None:
            z_w_min = z_w.min(axis=0); z_w_max = z_w.max(axis=0)

    if z_w_min is not None:
        tasks = [(int(j), sample_timeout, float(lo0[j]), float(hi0[j]),
                  use_milp, float(z_w_min[j]), float(z_w_max[j]))
                 for j in unstable]
    else:
        tasks = [(int(j), sample_timeout, float(lo0[j]), float(hi0[j]),
                  use_milp) for j in unstable]
    chunksize = max(1, len(tasks) // (n_cores * 4))
    t_solve0 = time.perf_counter()
    any_timeout = False
    new_lo = lo0.copy()
    new_hi = hi0.copy()
    with multiprocessing.Pool(n_cores) as pool:
        for j, lb_j, ub_j, to_j in pool.imap_unordered(
                _solve_sparse_neuron_graph_worker, tasks,
                chunksize=chunksize):
            if to_j:
                any_timeout = True
            new_lo[j] = max(new_lo[j], lb_j)
            new_hi[j] = min(new_hi[j], ub_j)
    dt_solve = time.perf_counter() - t_solve0
    _sparse_graph_args = None

    method = 'milp' if use_milp else 'lp'
    if any_timeout:
        method = method + '-partial'
    return (new_lo, new_hi, method,
            dt_build_total, dt_probe_total, dt_solve)


# ---------------------------------------------------------------------------
# Sequential-layer tightening with MILP/LP sampling probe
# ---------------------------------------------------------------------------

def _tighten_sequential_with_probe(
        layers_np_seq, x_lo, x_hi, seq_bounds, seq_li, unstable,
        lp_per_worker, sample_timeout, n_cores, time_left,
        mode='probe'):
    """Sampling-probe tightening for a sequential subgraph.

    `mode` controls which solvers we try:
      - 'probe': sample half MILP + half LP; prefer MILP if it survives,
        else LP, else skip.
      - 'milp': probe MILP only. If MILP times out, return 'skip' (no
        automatic LP fallback — the caller explicitly asked for MILP).
      - 'lp': skip the MILP probe entirely (a previous layer's MILP
        already timed out, or the caller asked for LP). Probe only LP;
        if LP survives, solve all with LP, else skip.
      - 'skip': do nothing and return the original bounds.

    Conv layers use sparse per-neuron MILP models inside
    `_tighten_layer_parallel` (much faster than LP for conv), so 'probe'
    is worth trying at L1/L2 where receptive fields are small.

    Returns (new_lo, new_hi, method, dt_probe, dt_solve). method is
    one of {'milp', 'lp', 'skip'}.
    """
    lo0, hi0 = seq_bounds[seq_li]
    if mode == 'skip' or len(unstable) == 0:
        return lo0.copy(), hi0.copy(), 'skip', 0.0, 0.0

    n_sample = min(n_cores, len(unstable))
    rng = np.random.RandomState(int(seq_li))
    sample_idx = rng.choice(unstable, n_sample, replace=False)

    is_fc = layers_np_seq[seq_li]['type'] == 'fc'
    # FC layers' MILP probes tend to blow up; follow milp_verify's
    # heuristic of skipping the MILP sample on FC layers past the first.
    milp_allowed = not (is_fc and seq_li > 0)
    try_milp = (mode in ('probe', 'milp')) and milp_allowed
    try_lp = mode in ('probe', 'lp')
    # 'probe' splits sample half-half; 'milp'/'lp' use the full sample.
    half = (n_sample // 2) if mode == 'probe' else n_sample

    t0 = time.perf_counter()
    milp_any_timeout = not try_milp
    if try_milp and half > 0:
        _, _, milp_any_timeout = _tighten_layer_parallel(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
            use_milp=True, timeout=sample_timeout,
            n_cores=n_cores, neuron_subset=sample_idx[:half])

    lp_any_timeout = not try_lp
    if try_lp:
        lp_probe_start = half if mode == 'probe' else 0
        if n_sample - lp_probe_start > 0:
            _, _, lp_any_timeout = _tighten_layer_parallel(
                layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
                use_milp=False, timeout=sample_timeout,
                n_cores=n_cores, neuron_subset=sample_idx[lp_probe_start:],
                lp_per_worker=lp_per_worker)
    dt_probe = time.perf_counter() - t0

    # Budget guard: if the estimated full-layer time exceeds the
    # remaining budget, don't bother.
    est_full = (len(unstable) / max(1, n_cores)) * (dt_probe / max(1, n_sample))
    if est_full > time_left() * 0.5:
        return lo0.copy(), hi0.copy(), 'skip', dt_probe, 0.0

    t_solve0 = time.perf_counter()
    if try_milp and not milp_any_timeout:
        new_lo, new_hi, _ = _tighten_layer_parallel(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
            use_milp=True, timeout=sample_timeout,
            n_cores=n_cores, neuron_subset=unstable)
        new_lo = np.maximum(new_lo, lo0)
        new_hi = np.minimum(new_hi, hi0)
        return new_lo, new_hi, 'milp', dt_probe, time.perf_counter() - t_solve0
    if try_lp and not lp_any_timeout:
        new_lo, new_hi, _ = _tighten_layer_parallel(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
            use_milp=False, timeout=sample_timeout,
            n_cores=n_cores, lp_per_worker=lp_per_worker,
            neuron_subset=unstable)
        new_lo = np.maximum(new_lo, lo0)
        new_hi = np.minimum(new_hi, hi0)
        return new_lo, new_hi, 'lp', dt_probe, time.perf_counter() - t_solve0
    return lo0.copy(), hi0.copy(), 'skip', dt_probe, 0.0


# ---------------------------------------------------------------------------
# G-cone sparse per-neuron tightening (tighten_formulation='gen_cone')
# ---------------------------------------------------------------------------

def _gen_cone_state(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                    pre_relu_op_name, *, device, dtype, conv_chunk_size=None,
                    g_storage='dense', oom_log=None):
    """Run gen-LP-style forward propagation up to `pre_relu_op_name`.

    Returns the state dict from `verify_gen_lp.precompute_gen_state` whose
    `obj_c_out` / `obj_G_out_csr` represent the pre-activation linear form
    at the target layer. Uses formulation='dense' so stable-on neurons
    pass their full pre-relu G rows through — exactly the recursion the
    cone walker depends on.

    `conv_chunk_size`: optional row-chunk for the conv/fc gen-image
    propagation. None = legacy un-chunked path (no overhead). When set,
    the conv2d/matmul over the n_gens dim is chunked with OOM-halve-retry
    — see `_conv2d_chunked_with_oom_halving` in verify_gen_lp.
    """
    truncated = []
    for op in gg_ops_ser:
        truncated.append(op)
        if op['name'] == pre_relu_op_name:
            break
    return verify_gen_lp.precompute_gen_state(
        truncated, x_lo, x_hi, bounds_by_relu, input_name,
        pre_relu_op_name, device=device, dtype=dtype,
        formulation='dense', conv_chunk_size=conv_chunk_size,
        g_storage=g_storage, _oom_log=oom_log)


def _build_gen_rows_reverse_map(state):
    """From a gen-LP state, produce `gen_rows_by_layer` and `col_origin`.

    - `gen_rows_by_layer`: `{li: {neuron_idx: unstable_entry}}` keyed by
      the gen-LP entry layout (row_indices, row_values, e_new_col, c_in,
      lo, hi).
    - `col_origin`: `{e_new_col_int: (li, neuron_idx)}` reverse lookup for
      cone walking. Input cols (< n_input) are not in this map.
    """
    gen_rows_by_layer = {}
    col_origin = {}
    for ul in state.get('unstable_list', ()):
        li = ul['layer_idx']
        j = ul['neuron_idx']
        gen_rows_by_layer.setdefault(li, {})[j] = ul
        col_origin[int(ul['e_new_col'])] = (li, j)
    return gen_rows_by_layer, col_origin


def _dependency_cone(target_row_indices, target_row_values,
                     gen_rows_by_layer, col_origin, n_input):
    """BFS over the recorded gen-LP rows to find the target's cone.

    Starts from the target row's nonzeros and walks each upstream
    unstable neuron's own stored row recursively. Dead ends at input
    cols (added to `input_cols`) or at cols not in `col_origin` (e.g.
    stable-group cols in a sparse formulation — absent here since we
    use 'dense').

    Returns `(input_cols_sorted_list, upstream_topo_list)` where the
    upstream list is sorted ascending by (layer_idx, neuron_idx).
    """
    input_cols = set()
    visited_upstream = {}  # e_new_col -> entry
    stack = [int(c) for c in target_row_indices]
    # target_row_values unused here; sparsity is what matters.
    _ = target_row_values
    seen = set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        if c < n_input:
            input_cols.add(c)
            continue
        origin = col_origin.get(c)
        if origin is None:
            continue
        if c in visited_upstream:
            continue
        li_c, j_c = origin
        entry = gen_rows_by_layer[li_c][j_c]
        visited_upstream[c] = entry
        for c2 in entry['row_indices']:
            c2_int = int(c2)
            if c2_int not in seen:
                stack.append(c2_int)
    upstream_topo = sorted(
        visited_upstream.values(),
        key=lambda e: (e['layer_idx'], e['neuron_idx']))
    return sorted(input_cols), upstream_topo


def _build_gen_cone_lp(target_c, target_row_indices, target_row_values,
                      input_cols, upstream_topo, milp_neurons, sense):
    """Gurobi LP/MILP over the dependency cone for one target neuron.

    Formulation mirrors `verify_gen_lp.build_gen_lp_from_state` restricted
    to the cone: `e_in[c] ∈ [-1, 1]` for input cols and `a_{li,k} ∈ [0, hi]`
    for each upstream unstable neuron, with triangle relaxation (or big-M
    if `(li,k)` is in `milp_neurons`). The objective is
    `target_c + sum_{c in row} target_row_val[c] * var(c)`.

    Returns `(model, env)`. Caller is responsible for dispose.
    """
    import gurobipy as grb
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    e_in_var = {c: m.addVar(lb=-1.0, ub=1.0, name=f'e_in_{c}')
                for c in input_cols}
    a_var = {}
    m.update()

    for entry in upstream_topo:
        # Soundness guard: this builder uses ALPHA-form vars
        # (a ∈ [0, hi]) and assumes the row coefficient for the e_new
        # column is the literal next-layer weight (NOT μ-scaled). Rows
        # tagged 'phase1' or 'alpha_zono' are zono-parallelogram form
        # — interpreting them here yields a different relaxation that
        # is sound but materially looser. Dispatch to the matching
        # builder instead. See the regression test in
        # tests/test_gen_cone_form_dispatch.py.
        _entry_form = entry.get('form', 'alpha')
        assert _entry_form == 'alpha', (
            f'_build_gen_cone_lp got entry with form={_entry_form!r}; '
            'expected "alpha". Use _build_gen_cone_lp_phase1 for '
            'zono-form rows.')
        li = entry['layer_idx']
        j = entry['neuron_idx']
        c_in = float(entry['c_in'])
        lo_j = float(entry['lo'])
        hi_j = float(entry['hi'])
        z_vars, z_coefs = [], []
        for idx, val in zip(entry['row_indices'], entry['row_values']):
            c = int(idx)
            v = e_in_var.get(c)
            if v is None:
                v = a_var.get(c)
            if v is None:
                continue
            z_vars.append(v)
            z_coefs.append(float(val))

        a = m.addVar(lb=0.0, ub=hi_j, name=f'a_L{li}_{j}')
        a_var[int(entry['e_new_col'])] = a
        m.update()
        # Triangle lower: a >= z  <=>  z - a <= -c_in
        m.addLConstr(grb.LinExpr(z_coefs, z_vars) - a <= -c_in,
                     name=f'tri_lo_L{li}_{j}')
        if (li, j) in milp_neurons:
            s = m.addVar(vtype=grb.GRB.BINARY, name=f's_L{li}_{j}')
            m.update()
            m.addLConstr(a - hi_j * s <= 0, name=f'bigM_hi_L{li}_{j}')
            m.addLConstr(
                a - grb.LinExpr(z_coefs, z_vars) - lo_j * s
                <= c_in - lo_j, name=f'bigM_z_L{li}_{j}')
        else:
            slope = hi_j / (hi_j - lo_j)
            m.addLConstr(
                a - grb.LinExpr([slope * w for w in z_coefs], z_vars)
                <= slope * (c_in - lo_j), name=f'tri_up_L{li}_{j}')

    obj_vars, obj_coefs = [], []
    for idx, val in zip(target_row_indices, target_row_values):
        c = int(idx)
        v = e_in_var.get(c)
        if v is None:
            v = a_var.get(c)
        if v is None:
            continue
        obj_vars.append(v)
        obj_coefs.append(float(val))
    grb_sense = grb.GRB.MINIMIZE if sense == 'min' else grb.GRB.MAXIMIZE
    m.setObjective(grb.LinExpr(obj_coefs, obj_vars) + float(target_c),
                   grb_sense)
    if not milp_neurons:
        m.setParam('Method', 1)
    m.update()
    return m, env


def _build_gen_cone_lp_phase1(target_c, target_row_indices, target_row_values,
                               input_cols, upstream_topo, milp_neurons, sense):
    """Like `_build_gen_cone_lp` but for ZONO-form rows (e_new ∈ [-1, 1]
    with μ-scaled coefficients on the new column).

    Mirrors `verify_gen_lp._build_phase1_lp` restricted to a target
    neuron's dependency cone: variables are `e_in[i] ∈ [-1, 1]` for
    input cols and `e_new_k ∈ [-1, 1]` for each upstream unstable.
    Per upstream the post-relu `y_k` is the parallelogram expression
    `λ_k·z_k + μ_k·(1 + e_new_k)`, and the LP triangle floor is
    recovered via two extra constraints (`y_k ≥ 0`, `y_k ≥ z_k`).
    For binarized neurons four big-M constraints are added.

    Compatible with rec_zono entries — `_record_zono_pre_relu_rows`
    extracts these rows in-place from the live forward zonotope, so
    no second gen-LP conv pass is needed (the piggyback fast path).

    Returns `(model, env)`. Caller is responsible for dispose.
    """
    import gurobipy as grb
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)

    e_in_var = {c: m.addVar(lb=-1.0, ub=1.0, name=f'e_in_{c}')
                for c in input_cols}
    e_new_var = {}
    m.update()

    for entry in upstream_topo:
        # Soundness guard: this builder uses ZONO-form vars
        # (e_new ∈ [-1, 1]) and the parallelogram constraints
        # `y_k = λ·z + μ·(1+e_new)`. Rows tagged 'alpha' have
        # coefficient 1.0 (not μ) on the e_new column and pair with
        # `a ∈ [0, hi]` — incompatible coordinate system. See
        # tests/test_gen_cone_form_dispatch.py.
        _entry_form = entry.get('form', 'phase1')
        assert _entry_form in ('phase1', 'alpha_zono'), (
            f'_build_gen_cone_lp_phase1 got entry with form='
            f'{_entry_form!r}; expected "phase1" or "alpha_zono". '
            'Use _build_gen_cone_lp for alpha-form rows.')
        li = entry['layer_idx']
        j = entry['neuron_idx']
        c_in = float(entry['c_in'])
        lo_j = float(entry['lo'])
        hi_j = float(entry['hi'])
        gap = hi_j - lo_j
        # Phase 1 must have classified this neuron as unstable, so gap > 0.
        lam = hi_j / gap
        mu = -hi_j * lo_j / (2.0 * gap)

        z_vars, z_coefs = [], []
        for idx, val in zip(entry['row_indices'], entry['row_values']):
            c = int(idx)
            v = e_in_var.get(c)
            if v is None:
                v = e_new_var.get(c)
            if v is None:
                continue
            z_vars.append(v)
            z_coefs.append(float(val))

        e_new = m.addVar(lb=-1.0, ub=1.0, name=f'e_new_L{li}_{j}')
        e_new_var[int(entry['e_new_col'])] = e_new
        m.update()

        # y_k ≥ 0:  Σ(λ·row)·vars + μ·e_new ≥ -λ·c_in - μ
        lin_y0 = grb.LinExpr([lam * w for w in z_coefs], z_vars)
        lin_y0.add(e_new, mu)
        m.addLConstr(lin_y0 >= -lam * c_in - mu, name=f'tri_lo_L{li}_{j}')
        # y_k ≥ z_k:  Σ((λ-1)·row)·vars + μ·e_new ≥ -(λ-1)·c_in - μ
        lin_yz = grb.LinExpr([(lam - 1.0) * w for w in z_coefs], z_vars)
        lin_yz.add(e_new, mu)
        m.addLConstr(lin_yz >= -(lam - 1.0) * c_in - mu,
                     name=f'tri_up_L{li}_{j}')

        if (li, j) in milp_neurons:
            s = m.addVar(vtype=grb.GRB.BINARY, name=f's_L{li}_{j}')
            m.update()
            # y_k ≤ hi·s:  Σ(λ·row)·vars + μ·e_new - hi·s ≤ -μ - λ·c_in
            lin_hi = grb.LinExpr([lam * w for w in z_coefs], z_vars)
            lin_hi.add(e_new, mu)
            lin_hi.add(s, -hi_j)
            m.addLConstr(lin_hi <= -mu - lam * c_in,
                         name=f'bigM_hi_L{li}_{j}')
            # y_k ≤ z - lo·(1-s):
            #   Σ((λ-1)·row)·vars + μ·e_new - lo·s ≤ -lo - μ - (λ-1)·c_in
            lin_lo = grb.LinExpr([(lam - 1.0) * w for w in z_coefs], z_vars)
            lin_lo.add(e_new, mu)
            lin_lo.add(s, -lo_j)
            m.addLConstr(lin_lo <= -lo_j - mu - (lam - 1.0) * c_in,
                         name=f'bigM_z_L{li}_{j}')

    obj_vars, obj_coefs = [], []
    for idx, val in zip(target_row_indices, target_row_values):
        c = int(idx)
        v = e_in_var.get(c)
        if v is None:
            v = e_new_var.get(c)
        if v is None:
            continue
        obj_vars.append(v)
        obj_coefs.append(float(val))
    grb_sense = grb.GRB.MINIMIZE if sense == 'min' else grb.GRB.MAXIMIZE
    m.setObjective(grb.LinExpr(obj_coefs, obj_vars) + float(target_c),
                   grb_sense)
    if not milp_neurons:
        m.setParam('Method', 1)
    m.update()
    return m, env


def precompute_gen_rows_all_layers(gg_ops_ser, x_lo, x_hi,
                                    bounds_by_relu, input_name, *,
                                    device='cuda',
                                    dtype=torch.float64):
    """Single gen-LP forward over the full network; returns the
    per-ReLU unstable rows plus the `e_new_col → (li, k)` reverse map.

    This is the `reuse_gen_rows` primitive: one forward pass records
    everything `_tighten_layer_gen_cone` needs for *any* target layer,
    eliminating the per-layer redundant forward. Built by reusing
    `verify_gen_lp.precompute_gen_state` (which already accumulates
    unstable entries at every ReLU) with the network's final op as the
    output target — the `obj_*` output arrays are discarded.

    Returns `(gen_rows_by_layer, col_origin, n_input)`.
    """
    assert len(gg_ops_ser) > 0
    output_op = gg_ops_ser[-1]['name']
    state = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, device=device, dtype=dtype,
        formulation='dense')
    gen_rows_by_layer, col_origin = _build_gen_rows_reverse_map(state)
    return gen_rows_by_layer, col_origin, state['n_input']


_gen_cone_args = None  # shared state for pool workers


def _solve_gen_cone_neuron_worker(task):
    """Pool worker: min and max LP/MILP for one target neuron over its cone.

    Reads gg-level state from module global `_gen_cone_args` (set by
    `_tighten_layer_gen_cone`) to avoid pickling a large state per task.
    If `use_milp` is True, the full cone of upstream unstable neurons
    gets big-M binary encoding; otherwise pure LP-triangle.
    The target's own pre-relu row is looked up in
    `gen_rows_by_layer[target_li][j]` (populated by the shared
    precompute pass — no per-worker forward).
    Returns `(j, lb, ub, any_timeout, dt_build, dt_solve)`.
    """
    import gurobipy as grb
    j, timeout, lo_init, hi_init, use_milp, target_li = task
    # _gen_cone_args is either (rows, col_origin, n_input) — alpha form,
    # the legacy default — or (rows, col_origin, n_input, form) where
    # form ∈ {'alpha', 'phase1'}. 'phase1' selects the zono-form builder
    # so the rec_zono piggyback path can be reused without re-deriving
    # rows in alpha form.
    if len(_gen_cone_args) == 3:
        (gen_rows_by_layer, col_origin, n_input) = _gen_cone_args
        form = 'alpha'
    else:
        (gen_rows_by_layer, col_origin, n_input, form) = _gen_cone_args
    entry = gen_rows_by_layer[target_li][j]
    nz = entry['row_indices']
    vals = entry['row_values']
    c_in = float(entry['c_in'])
    input_cols, upstream_topo = _dependency_cone(
        nz, vals, gen_rows_by_layer, col_origin, n_input)
    # Sliding window à la α,β-CROWN's bab-refine: only binarize the
    # upstream unstables in the last `_window` layers BEFORE target_li.
    # Older layers stay LP-triangle (sound, looser). Caps the cone
    # binary count from O(layers × n_unstable) down to O(K × n_unstable).
    # Set via VC_TIGHTEN_WINDOW=K env var (None = full cone, default).
    _window_env = os.environ.get('VC_TIGHTEN_WINDOW', '')
    _window = None
    if _window_env:
        try:
            _window = int(_window_env)
        except ValueError:
            _window = None
    if use_milp:
        if _window is None:
            milp_set = frozenset(
                (e['layer_idx'], e['neuron_idx']) for e in upstream_topo)
        else:
            cutoff = int(target_li) - int(_window)
            milp_set = frozenset(
                (e['layer_idx'], e['neuron_idx'])
                for e in upstream_topo
                if e['layer_idx'] >= cutoff)
    else:
        milp_set = frozenset()

    lb = lo_init
    ub = hi_init
    any_timeout = False
    dt_build = 0.0
    dt_solve = 0.0
    from vibecheck.gurobi_util import GurobiNumericTrouble
    _builder = (_build_gen_cone_lp_phase1
                if form == 'phase1' else _build_gen_cone_lp)
    for sense in ('min', 'max'):
        t_b = time.perf_counter()
        m, env = _builder(
            c_in, nz, vals, input_cols, upstream_topo,
            milp_neurons=milp_set, sense=sense)
        m.setParam('TimeLimit', float(timeout))
        # BestBdStop: terminate the per-neuron MILP as soon as Gurobi
        # proves the bound flips sign (LB > 0 → neuron active, UB < 0
        # → dead). Mirrors AB-CROWN's `lp_mip_solver.py:259/253` —
        # avoids spending time grinding to optimum once stability is
        # established. Direction-specific:
        #   sense='min' is computing LB on z_j: stop when LB > 0
        #     (proves neuron active for ANY input in the cone).
        #   sense='max' is computing UB on z_j: stop when UB < 0
        #     (proves neuron dead).
        # Use a small tolerance to avoid premature stops at numerical
        # noise; AB-CROWN uses ±1e-5.
        if use_milp:
            if sense == 'min':
                m.setParam('BestBdStop', 1e-5)
            else:
                m.setParam('BestBdStop', -1e-5)
        # Optional Gurobi tuning controlled per-call by env var so it
        # can be A/B'd without touching settings. Values are applied
        # as-is; missing keys leave Gurobi defaults in place.
        _tune = os.environ.get('VC_GUROBI_TUNE', '')
        _trouble_log = os.environ.get('VC_LOG_TROUBLE', '')
        if use_milp and _tune:
            for spec_kv in _tune.split(','):
                if '=' in spec_kv:
                    k, val = spec_kv.split('=', 1)
                    try:
                        v_num = float(val)
                        if v_num.is_integer():
                            v_num = int(v_num)
                        m.setParam(k.strip(), v_num)
                    except ValueError:
                        m.setParam(k.strip(), val)
        dt_build += time.perf_counter() - t_b
        # MILP dump hook — writes .lp + sidecar .pkl per (layer, neuron, sense)
        # when VC_DUMP_MILP_DIR is set. Optional VC_DUMP_MILP_LAYERS=1,2 (CSV
        # of target_li values) restricts which layers get dumped (default:
        # all). Used to capture phase-1 bound-tightening MILPs for
        # standalone Gurobi profiling and dual-BnB experiments.
        _dump_dir = os.environ.get('VC_DUMP_MILP_DIR', '')
        if _dump_dir and use_milp:
            _dump_layers = os.environ.get('VC_DUMP_MILP_LAYERS', '')
            if _dump_layers:
                _allowed = {int(s) for s in _dump_layers.split(',') if s}
                _do_dump = int(target_li) in _allowed
            else:
                _do_dump = True
            if _do_dump:
                import pickle as _pkl
                os.makedirs(_dump_dir, exist_ok=True)
                _stem = f'L{int(target_li)}_n{int(j)}_{sense}'
                m.write(f'{_dump_dir}/{_stem}.lp')
                _meta = {
                    'form': form, 'c_in': float(c_in),
                    'nz': list(nz), 'vals': list(vals),
                    'input_cols': list(input_cols),
                    'upstream_topo': list(upstream_topo),
                    'milp_neurons': list(milp_set),
                    'sense': sense, 'timeout': float(timeout),
                    'best_bd_stop': (1e-5 if sense == 'min' else -1e-5),
                    'target_li': int(target_li), 'neuron_j': int(j),
                    'lo_init': float(lo_init), 'hi_init': float(hi_init),
                    'tune_env': _tune,
                }
                with open(f'{_dump_dir}/{_stem}.pkl', 'wb') as _f:
                    _pkl.dump(_meta, _f)
        t_s = time.perf_counter()
        try:
            optimize_checked(m)
        except GurobiNumericTrouble as _gn:
            # Per-neuron fallback: keep the original lo/hi, mark as timeout so
            # caller can log a partial run. This localizes numeric fragility
            # to the one neuron instead of crashing the whole layer pass.
            dt_solve += time.perf_counter() - t_s
            any_timeout = True
            if _trouble_log:
                with open(_trouble_log, 'a') as _f:
                    _f.write(f'TROUBLE,L{target_li},j{j},sense={sense},'
                             f'msg={str(_gn)[:80]}\n')
            m.dispose()
            env.dispose()
            continue
        dt_solve += time.perf_counter() - t_s
        status = m.Status
        # Use ObjBound on both OPTIMAL and TIME_LIMIT — for both statuses
        # ObjBound is a SOUND lower bound on min (or upper on max). When
        # OPTIMAL, ObjBound = ObjVal (gap closed, tight). When TIME_LIMIT,
        # ObjBound is the best LP-relaxation bound found in the search
        # tree — looser but still sound. Previously we discarded the
        # TIME_LIMIT bound, leaving the looser `lo_init` from upstream
        # adapt; this caused gen_cone-MILP to report bounds 1.7+ looser
        # than the true MILP optimum on cases where Gurobi proved a
        # tighter relaxation bound but didn't close the integer gap.
        if status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT,
                       grb.GRB.USER_OBJ_LIMIT):
            try:
                v = float(m.ObjBound)
                if sense == 'min':
                    lb = max(lb, v)
                else:
                    ub = min(ub, v)
            except (AttributeError, grb.GurobiError):
                # No valid ObjBound (e.g. SUBOPTIMAL with no relaxation
                # solved yet) — keep `lo_init`/`hi_init`.
                pass
            if status == grb.GRB.TIME_LIMIT:
                any_timeout = True
        # Append solve result to dumped sidecar (if dump hook fired above).
        if _dump_dir and use_milp:
            _dump_layers2 = os.environ.get('VC_DUMP_MILP_LAYERS', '')
            if _dump_layers2:
                _allowed2 = {int(s) for s in _dump_layers2.split(',') if s}
                _do_dump2 = int(target_li) in _allowed2
            else:
                _do_dump2 = True
            if _do_dump2:
                import pickle as _pkl2
                _stem2 = f'L{int(target_li)}_n{int(j)}_{sense}'
                _pp = f'{_dump_dir}/{_stem2}.pkl'
                try:
                    with open(_pp, 'rb') as _rf:
                        _meta2 = _pkl2.load(_rf)
                except FileNotFoundError:
                    _meta2 = {}
                _obj_bound = None; _obj_val = None
                try:
                    _obj_bound = float(m.ObjBound)
                except (AttributeError, grb.GurobiError):
                    pass
                try:
                    _obj_val = float(m.ObjVal)
                except (AttributeError, grb.GurobiError):
                    pass
                _meta2['result'] = {
                    'status': int(status),
                    'obj_bound': _obj_bound,
                    'obj_val': _obj_val,
                    'solve_time_s': time.perf_counter() - t_s,
                    'build_time_s': dt_build,
                    'mip_gap': float(getattr(m, 'MIPGap', float('nan'))),
                    'node_count': int(getattr(m, 'NodeCount', 0)),
                }
                with open(_pp, 'wb') as _wf:
                    _pkl2.dump(_meta2, _wf)
        m.dispose()
        env.dispose()
    return int(j), lb, ub, any_timeout, dt_build, dt_solve


def _tighten_layer_gen_cone(gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                            target_layer_idx, unstable, input_name,
                            sample_timeout, n_cores, time_left, *,
                            mode='probe', device, dtype,
                            use_milp=False, precomputed=None,
                            gpu_lock=None):
    """G-cone per-target-neuron tightening for one ReLU layer.

    Obtains the gen-LP-style per-ReLU rows either from `precomputed`
    (tuple `(gen_rows_by_layer, col_origin, n_input)` for the alpha
    form, or `(gen_rows_by_layer, col_origin, n_input, 'phase1')` for
    the zono form populated by `_record_zono_pre_relu_rows` —
    the rec_zono **piggyback** path that amortizes one forward across
    every tightened layer) or by running `_gen_cone_state` for this
    layer alone (always alpha form). Dispatches one LP/MILP per
    unstable target neuron over its dependency cone via
    `multiprocessing.Pool`.

    Returns `(new_lo, new_hi, method, dt_build, dt_probe, dt_solve)`.
    `method` ∈ {'lp','milp','skip','lp-partial','milp-partial'}. `mode`
    is read but only 'skip' is honored as a no-op short-circuit.
    """
    lo0, hi0 = bounds_by_relu[target_layer_idx]
    if mode == 'skip' or len(unstable) == 0:
        return lo0.copy(), hi0.copy(), 'skip', 0.0, 0.0, 0.0

    t_build0 = time.perf_counter()
    form = 'alpha'
    if precomputed is not None:
        if len(precomputed) == 4:
            gen_rows_by_layer, col_origin, n_input, form = precomputed
        else:
            gen_rows_by_layer, col_origin, n_input = precomputed
    else:
        # Find the target layer's RELU op itself (not its pre-relu op)
        # so the forward walks through its relu, recording its unstable
        # rows into gen_rows_by_layer[target_layer_idx].
        target_relu_op_name = None
        for op in gg_ops_ser:
            if (op['type'] == 'relu'
                    and op.get('layer_idx') == target_layer_idx):
                target_relu_op_name = op['name']
                break
        assert target_relu_op_name is not None, \
            f'no relu op for layer_idx={target_layer_idx}'
        # Per-call chunking of the gen-LP conv2d propagation. Defaults to
        # 256 (chunked + OOM-halve-retry); set env VC_GEN_LP_CONV_CHUNK=0
        # to force the un-chunked legacy path. Settings-aware callers can
        # also override via the env var.
        _conv_chunk = 256
        try:
            import os as _os_local
            _env = _os_local.environ.get('VC_GEN_LP_CONV_CHUNK', '')
            if _env:
                _v = int(_env)
                _conv_chunk = _v if _v > 0 else None
        except (ValueError, TypeError):
            _conv_chunk = 256
        _g_storage = os.environ.get('VC_GEN_LP_G_STORAGE', 'sparse')
        if _g_storage not in ('dense', 'sparse'):
            _g_storage = 'sparse'
        state = _gen_cone_state(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            target_relu_op_name, device=device, dtype=dtype,
            conv_chunk_size=_conv_chunk, g_storage=_g_storage)
        gen_rows_by_layer, col_origin = \
            _build_gen_rows_reverse_map(state)
        n_input = state['n_input']
    dt_build = time.perf_counter() - t_build0

    global _gen_cone_args
    _gen_cone_args = (gen_rows_by_layer, col_origin, n_input, form)
    tasks = [(int(j), sample_timeout, float(lo0[j]), float(hi0[j]),
              bool(use_milp), int(target_layer_idx))
             for j in unstable]

    t_solve0 = time.perf_counter()
    any_timeout = False
    new_lo = lo0.copy()
    new_hi = hi0.copy()
    per_neuron_build = 0.0
    per_neuron_solve = 0.0
    if n_cores <= 1 or len(tasks) <= 1:
        for task in tasks:
            j, lb_j, ub_j, to_j, db_j, ds_j = \
                _solve_gen_cone_neuron_worker(task)
            if to_j:
                any_timeout = True
            new_lo[j] = max(new_lo[j], lb_j)
            new_hi[j] = min(new_hi[j], ub_j)
            per_neuron_build += db_j
            per_neuron_solve += ds_j
    else:
        # chunksize=1 + tasks sorted by score (caller's responsibility)
        # means workers pick off the highest-impact neurons first.
        # Per-neuron `sample_timeout` caps individual MILP cost; we
        # also enforce a hard wall-clock cap via `time_left()` so the
        # cascade doesn't overrun `total_timeout` when a layer has
        # hundreds of unstables (observed: oval21 deep_kw img7358 had
        # L2=654 unstables × 5s/MIP, and the layer alone consumed 90s+
        # past deadline before this check). On deadline-hit we
        # `pool.terminate()` — losing in-flight result deliveries — and
        # return whatever bounds we've already aggregated.
        chunksize = 1
        n_timeouts = 0; n_solved = 0
        pool = multiprocessing.Pool(n_cores)
        # GPU is truly idle here — only Gurobi CPU work in worker
        # subprocesses. Signal `milp_active` so a parallel PGD worker
        # thread knows it can launch CUDA kernels without contending
        # with main. Main never blocks; the thread polls this flag.
        if gpu_lock is not None:
            gpu_lock.set()  # gpu_lock is a threading.Event named milp_active
        try:
            for j, lb_j, ub_j, to_j, db_j, ds_j in pool.imap_unordered(
                    _solve_gen_cone_neuron_worker, tasks,
                    chunksize=chunksize):
                if time_left() <= 0:
                    break
                if to_j:
                    any_timeout = True
                    n_timeouts += 1
                n_solved += 1
                new_lo[j] = max(new_lo[j], lb_j)
                new_hi[j] = min(new_hi[j], ub_j)
                per_neuron_build += db_j
                per_neuron_solve += ds_j
        finally:
            pool.terminate()
            pool.join()
            if gpu_lock is not None:
                gpu_lock.clear()
        if os.environ.get('VC_LOG_MILP_TIMEOUTS', '') == '1' and n_solved > 0:
            print(f'    [tighten L{target_layer_idx}] {n_timeouts}/{n_solved} '
                  f'neurons hit MILP timeout '
                  f'({100*n_timeouts/n_solved:.0f}%)', flush=True)
    _ = time.perf_counter() - t_solve0  # wall time of solve phase (unused)
    _gen_cone_args = None

    if use_milp:
        method = 'milp-partial' if any_timeout else 'milp'
    else:
        method = 'lp-partial' if any_timeout else 'lp'
    # Return:
    #   dt_build: shared precompute_gen_state setup
    #   dt_probe: summed per-neuron Gurobi model-build (construction cost)
    #   dt_solve: summed per-neuron Gurobi optimize (solve cost)
    return (new_lo, new_hi, method,
            dt_build, per_neuron_build, per_neuron_solve)


# ---------------------------------------------------------------------------
# Interleaved zonotope forward with per-layer tightening
# ---------------------------------------------------------------------------

def _record_zono_pre_relu_rows(z, li, bounds, rec_zono):
    """Extract zonotope pre-ReLU G rows for every unstable neuron at
    layer `li` and populate `rec_zono`. Called right before
    `z.apply_relu(...)` so the row values are pre-activation (before λ
    scaling and new-col addition).

    Dispatches to ``z.nonzero_rows(unstable)`` — both ``TorchZonotope``
    and ``PatchesZonotope`` provide it. The patches implementation
    avoids materialising the full ``(n_flat, K)`` dense G; the dense
    implementation slices rows directly. This is the **reuse_gen_rows**
    primitive piggybacking on the existing zonotope forward — no
    separate gen-LP conv pass needed.
    """
    lo_arr, hi_arr = bounds
    unstable = np.where((lo_arr < 0) & (hi_arr > 0))[0]
    if hasattr(z, '_mode') and z._mode == 'patches':
        n_gens_now = z._patches.shape[0]
        dev = z._patches.device
    else:
        # Dense path (TorchZonotope OR a materialised PatchesZonotope).
        inner = z._dense if hasattr(z, '_mode') else z
        if inner._gen_4d is not None:
            n_gens_now = inner._gen_4d.shape[0]
            dev = inner._gen_4d.device
        else:
            n_gens_now = inner._gen_2d.shape[1]
            dev = inner._gen_2d.device

    layer_dict = {}
    if len(unstable) > 0:
        idx_t = torch.as_tensor(unstable, dtype=torch.long, device=dev)
        # Use the polymorphic nonzero_rows API. Returns row_ids indexing
        # into `unstable` (NOT into flat neuron space).
        rid_d, cid_d, val_d = z.nonzero_rows(idx_t)
        # Sort by row_id then col_id for stable downstream indexing.
        # nonzero_rows already returns in row-major scan order for the
        # dense path; for the chunked patches path we sort to be safe.
        order = torch.argsort(rid_d * (n_gens_now + 1) + cid_d)
        rid_d = rid_d[order]
        cid_d = cid_d[order]
        val_d = val_d[order]
        rid_np = rid_d.cpu().numpy()
        cid_np = cid_d.cpu().numpy().astype(np.int32)
        val_np = val_d.cpu().numpy()
        split = np.searchsorted(rid_np, np.arange(len(unstable) + 1))
        center_np = z.center.cpu().numpy()
        for local_idx, k in enumerate(unstable):
            start = int(split[local_idx])
            end = int(split[local_idx + 1])
            e_col = n_gens_now + local_idx
            layer_dict[int(k)] = {
                'layer_idx': li,
                'neuron_idx': int(k),
                'c_in': float(center_np[int(k)]),
                'lo': float(lo_arr[int(k)]),
                'hi': float(hi_arr[int(k)]),
                'e_new_col': e_col,
                'row_indices': cid_np[start:end],
                'row_values': val_np[start:end],
                # Tag for downstream LP/MILP builders. The live
                # zonotope's pre-ReLU row coefficients are in ZONO
                # (μ-scaled) form: the e_new column carries μ_k and is
                # paired with `e_new_k ∈ [-1, 1]`. Builders that consume
                # this entry MUST interpret the new column as e_new
                # (not as the alpha-form a_k ∈ [0, hi]). Only
                # `_build_gen_cone_lp_phase1` / `_build_phase1_lp` are
                # safe; `_build_gen_cone_lp` (alpha-form vars) would
                # produce a different — sound but materially looser —
                # bound. See test_gen_cone_form_dispatch.py for the
                # regression guard.
                'form': 'phase1',
            }
            rec_zono['col_origin'][e_col] = (li, int(k))
    rec_zono['gen_rows_by_layer'][li] = layer_dict


@torch.no_grad()
def _forward_zonotope_interleaved(
        xl, xh, gg, gg_ops_ser, x_lo_64, x_hi_64,
        build_fn, sample_timeout, n_cores, time_left,
        device, dtype, settings, print_progress=False, verbose_cb=None,
        rec_zono=None):
    """Zonotope forward that stops at every ReLU, tightens bounds via
    per-neuron adaptive backward + optional LP probe, and feeds the
    tightened bounds into the ReLU relaxation so subsequent layers'
    zonotope propagation benefits from the tightening.

    This merges the old phase 1 (zono forward) and phase 4 (per-layer
    tightening) into a single pass, as requested.

    Args:
        xl, xh: input bound tensors (torch)
        gg: gpu_graph dict (dtype-matched)
        gg_ops_ser: serialized ops for LP building
        x_lo_64, x_hi_64: numpy input bounds for LP
        build_fn: builder for LP probe
        sample_timeout: per-neuron LP timeout
        n_cores: cores for LP pool
        time_left: callable returning seconds remaining
        verbose_cb: optional (li, name, record_dict) callback for
            per-layer timing telemetry
        rec_zono: optional dict to populate with `{gen_rows_by_layer,
            col_origin, n_input}` harvested at each layer's pre-ReLU.
            When provided, extracts the zonotope's G rows for every
            unstable neuron into a cone-walker-compatible cache (the
            `reuse_gen_rows` primitive). When None, behaves identically
            to the pre-reuse implementation.

    Returns:
        sb: {layer_idx: (lo_tensor, hi_tensor)} of the *tightened*
            pre-activation bounds actually used in the relu relaxation
        bounds_by_relu: {layer_idx: (lo_np64, hi_np64)} same bounds as np
        z_final: the final zonotope after the last op
    """
    input_name = gg['input_name']
    forks = gg['fork_points']

    z_init = make_input_zonotope(
        settings, xl, xh, device, dtype, in_shape=gg.get('input_shape'))
    zono_state = {input_name: z_init}
    gen_count = {input_name: z_init.n_gens}
    sb = {}
    bounds_by_relu = {}
    if rec_zono is not None:
        rec_zono.setdefault('gen_rows_by_layer', {})
        rec_zono.setdefault('col_origin', {})
        rec_zono['n_input'] = gen_count[input_name]

    # Phase-1 tightening is two orthogonal axes:
    #   formulation: 'weight_walk' (backward ONNX weight walk), 'gen_cone'
    #                (reuse the zonotope G rows), or 'skip' (adapt-only).
    #   solver:      'lp', 'milp', or 'probe' (MILP→LP auto-fallback).
    # Sticky downgrade applies only to the weight_walk path: once its
    # chosen solver times out at any layer, we downshift and never probe
    # that solver again on later layers.
    formulation = str(getattr(settings, 'tighten_formulation', 'weight_walk'))
    solver = str(getattr(settings, 'tighten_solver', 'probe'))
    assert formulation in ('weight_walk', 'gen_cone', 'skip'), formulation
    assert solver in ('lp', 'milp', 'probe'), solver
    # gen_cone doesn't implement probe fallback; treat probe as milp.
    if formulation == 'gen_cone' and solver == 'probe':
        solver = 'milp'
    lp_per_worker = bool(getattr(settings, 'milp_lp_per_worker', True))

    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    # Track remaining consumers per fork point so the LAST consumer can take
    # ownership of the original state (no clone). Saves the ~1 GB fork-copy
    # peak on resnet-sized nets.
    remaining_consumers = {
        fn: sum(1 for op2 in gg['ops'] if fn in op2['inputs'])
        for fn in forks}

    def _get(name):
        if name in forks:
            remaining_consumers[name] -= 1
            if remaining_consumers[name] > 0:
                return zono_state[name].copy()
        return zono_state[name]

    # Phase 1 timing breakdown: which categories of work took how long.
    tb = {'conv': 0.0, 'fc': 0.0, 'add': 0.0, 'sub': 0.0, 'reshape': 0.0,
          'relu_bounds': 0.0, 'relu_adapt': 0.0, 'relu_tighten': 0.0,
          'relu_apply': 0.0}

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']

        if t == 'conv':
            _ts = time.perf_counter()
            z = _get(op['inputs'][0])
            z.propagate_conv(op['kernel'], op['bias'], op['in_shape'],
                             op['stride'], op['padding'])
            zono_state[name] = z
            tb['conv'] += time.perf_counter() - _ts

        elif t == 'fc':
            _ts = time.perf_counter()
            z = _get(op['inputs'][0])
            z.propagate_fc(op['W'], op['bias'])
            zono_state[name] = z
            tb['fc'] += time.perf_counter() - _ts

        elif t == 'relu':
            z = _get(op['inputs'][0])
            if 'layer_idx' not in op:
                # Output relu (rare) — just apply with internal bounds.
                _ts = time.perf_counter()
                z.apply_relu()
                zono_state[name] = z
                gen_count[name] = z.n_gens
                for inp in op['inputs']:
                    if last_use.get(inp) == op_idx and inp in zono_state:
                        del zono_state[inp]
                tb['relu_apply'] += time.perf_counter() - _ts
                continue

            li = op['layer_idx']

            # Step 1: seed bounds_by_relu from the zonotope's internal bounds.
            _ts = time.perf_counter()
            z_lo, z_hi = z.bounds()
            lo_np = z_lo.cpu().numpy().astype(np.float64)
            hi_np = z_hi.cpu().numpy().astype(np.float64)
            bounds_by_relu[li] = (lo_np, hi_np)
            unstable_initial = np.where((lo_np < 0) & (hi_np > 0))[0]
            tb['relu_bounds'] += time.perf_counter() - _ts

            dt_adapt = dt_build = dt_probe = dt_solve = 0.0
            crown_fixed = lp_fixed = 0
            method = 'zono-only'
            # Track mean width (hi - lo) over the initial unstable set
            # at each pass so we can see the progression zono → adapt → LP.
            def _mean_w(lo_a, hi_a, idx):
                if len(idx) == 0:
                    return 0.0
                return float((hi_a[idx] - lo_a[idx]).mean())

            width_zono = _mean_w(lo_np, hi_np, unstable_initial)
            width_adapt = width_zono
            width_lp = width_zono

            if li > 0 and len(unstable_initial) > 0 and time_left() > 1:
                # Step 2: per-neuron adaptive zonotope backward bounds,
                # using already-tightened bounds_by_relu[0..li-1]. Chunked
                # so wide layers (resnet_large has 1687 unstable at L1)
                # don't hit 4 GB peak — α,β-CROWN pattern. Chunk size
                # tunable via `adapt_chunk_size` setting (None = no chunk).
                # `adapt_topk` (None = all): only tighten the top-K most
                # impactful unstable neurons by `|center|×width`. On
                # cifar_biasfield (1500+ unstable per layer), the all-
                # neurons sweep took ~24 s and fixed only ~12%; the
                # remaining 88% can be left to Phase 2.5's spec-aware
                # α-CROWN cascade. `adapt_enabled=False` skips entirely.
                _adapt_enabled = bool(getattr(
                    settings, 'phase1_adapt_enabled', True))
                _adapt_topk = getattr(settings, 'phase1_adapt_topk', None)
                if not _adapt_enabled:
                    new_lo, new_hi = lo_np.copy(), hi_np.copy()
                    after_adapt = unstable_initial
                    width_adapt = width_zono
                    width_lp = width_adapt
                    dt_adapt = 0.0
                else:
                    t_a = time.perf_counter()
                    _chunk = getattr(settings, 'adapt_chunk_size', 256)
                    if _adapt_topk is not None and len(unstable_initial) > int(
                            _adapt_topk):
                        # Score by |center|×width; pick top-K.
                        un = unstable_initial
                        score = (np.abs((lo_np[un] + hi_np[un]) / 2.0)
                                 * (hi_np[un] - lo_np[un]))
                        top = un[np.argsort(-score)[:int(_adapt_topk)]]
                        adapt_subset = np.sort(top)
                    else:
                        adapt_subset = unstable_initial
                    adapt_lo, adapt_hi = _per_neuron_adaptive_bounds_chunked(
                        gg, xl, xh, bounds_by_relu, li, device, dtype,
                        neuron_subset=adapt_subset,
                        chunk_size=_chunk)
                    dt_adapt = time.perf_counter() - t_a
                    new_lo = np.maximum(lo_np, adapt_lo)
                    new_hi = np.minimum(hi_np, adapt_hi)
                    after_adapt = np.where((new_lo < 0) & (new_hi > 0))[0]
                    crown_fixed = len(unstable_initial) - len(after_adapt)
                    width_adapt = _mean_w(new_lo, new_hi, unstable_initial)
                    width_lp = width_adapt

                # Step 3: LP probe for neurons the adaptive pass couldn't
                # prove stable. Write the (partially-tightened) bounds
                # back so the LP builder sees them at this layer.
                bounds_by_relu[li] = (new_lo, new_hi)

                max_layer = getattr(settings, 'max_tighten_layer', None)
                max_layer_lp = getattr(settings, 'max_tighten_layer_lp', None)
                # Effective upper bound: either layer ≤ max_tighten_layer
                # (MILP allowed) OR layer ≤ max_tighten_layer_lp (LP-only).
                _in_milp_range = (max_layer is None
                                   or li <= int(max_layer))
                _in_lp_range = (max_layer_lp is not None
                                 and max_layer is not None
                                 and int(max_layer) < li
                                 <= int(max_layer_lp))
                if (len(after_adapt) > 0 and time_left() > 2
                        and formulation != 'skip'
                        and (_in_milp_range or _in_lp_range)):
                    if formulation == 'gen_cone':
                        # G-cone per-target-neuron LP/MILP. Two routes
                        # depending on whether `rec_zono` is populated
                        # by the forward:
                        #   rec_zono is not None → **piggyback path**:
                        #     upstream rows come from the existing
                        #     zonotope forward (no second gen-LP conv
                        #     pass). Target layer's rows are extracted
                        #     inline from the current zono state and
                        #     composed with rec_zono's upstream entries
                        #     into a temporary cache for dispatch.
                        #   rec_zono is None → **dedicated-precompute**
                        #     path: `_tighten_layer_gen_cone` runs its
                        #     own truncated `precompute_gen_state`.
                        # Both produce bound-equivalent LP/MILP
                        # formulations (LP triangle ≡ gen-LP triangle;
                        # MILP big-M ≡ probe sparse MILP, modulo
                        # Gurobi feasibility tolerance).
                        # Layers in the LP-extension range (above
                        # max_tighten_layer but ≤ max_tighten_layer_lp)
                        # force LP regardless of `solver`.
                        use_milp = (solver == 'milp') and not _in_lp_range
                        precomputed = None
                        # Piggyback path: rec_zono records ZONO-form
                        # rows (μ-scaled e_new column). Tagged 'phase1'
                        # so the worker dispatches to the matching
                        # `_build_gen_cone_lp_phase1` builder. Available
                        # for both LP and MILP — the phase1 form is
                        # sound (it uses Phase-1 forward bbr for the
                        # parallelogram, which is the form contract);
                        # the trade vs an alpha-form rebuild is fresh
                        # big-M (alpha; tighter LP relaxation, more
                        # MILP optimality on a per-neuron basis) versus
                        # one extra forward pass per layer. On the
                        # CIFAR100 v10 sample, alpha rebuild caused a
                        # net regression (idx=13 verified→timeout) so
                        # we keep the piggyback default. Use
                        # `tighten_use_piggyback_milp=False` to opt
                        # back into the alpha rebuild for tighter
                        # bounds on harder networks (relusplitter
                        # mnist_fc 256x6 prefers alpha).
                        _piggyback_milp = bool(getattr(
                            settings,
                            'tighten_use_piggyback_milp',
                            True))
                        if rec_zono is not None and (
                                not use_milp or _piggyback_milp):
                            tmp_rec = {
                                'gen_rows_by_layer': dict(
                                    rec_zono['gen_rows_by_layer']),
                                'col_origin': dict(
                                    rec_zono['col_origin']),
                                'n_input': rec_zono['n_input'],
                            }
                            _record_zono_pre_relu_rows(
                                z, li, bounds_by_relu[li], tmp_rec)
                            precomputed = (
                                tmp_rec['gen_rows_by_layer'],
                                tmp_rec['col_origin'],
                                tmp_rec['n_input'],
                                'phase1')
                        legacy_mode = ('gen_cone_milp' if use_milp
                                       else 'gen_cone')
                        gc_lo, gc_hi, gc_method, dt_build, dt_probe, dt_solve = \
                            _tighten_layer_gen_cone(
                                gg_ops_ser, x_lo_64, x_hi_64,
                                bounds_by_relu, li, after_adapt,
                                gg['input_name'],
                                sample_timeout=sample_timeout,
                                n_cores=n_cores, time_left=time_left,
                                mode=legacy_mode,
                                device=device, dtype=dtype,
                                use_milp=use_milp,
                                precomputed=precomputed)
                        new_lo = np.maximum(new_lo, gc_lo)
                        new_hi = np.minimum(new_hi, gc_hi)
                        method = f'adapt+{legacy_mode}-{gc_method}'
                    elif _has_merge_before(gg_ops_ser, li):
                        # Weight-walk per-target-neuron model: one small
                        # Gurobi model per unstable neuron, covering
                        # only that neuron's backward dependency cone.
                        # The dependency walker follows both branches of
                        # a merge-Add. Solver controlled by `solver`;
                        # sticky downgrade happens below.
                        _wnr = (
                            int(getattr(settings,
                                        'tighten_witness_n_random', 8))
                            if getattr(settings,
                                       'tighten_witness_ordering', True)
                            else 0)
                        gs_lo, gs_hi, gs_method, dt_build, dt_probe, dt_solve = \
                            _tighten_layer_graph_sparse(
                                gg_ops_ser, x_lo_64, x_hi_64,
                                bounds_by_relu, li, after_adapt,
                                gg['input_name'], mode=solver,
                                sample_timeout=sample_timeout,
                                n_cores=n_cores, time_left=time_left,
                                witness_n_random=_wnr)
                        new_lo = np.maximum(new_lo, gs_lo)
                        new_hi = np.minimum(new_hi, gs_hi)
                        method = f'adapt+graph-{gs_method}'
                        if gs_method == 'skip':
                            # Downgrade: the LP probe (under 'probe'
                            # mode) or explicit solver also timed out,
                            # so disable weight-walk tightening for
                            # later layers.
                            formulation = 'skip'
                        elif (gs_method.startswith('lp')
                              and solver == 'probe'):
                            solver = 'lp'
                    else:
                        # Sequential subgraph (no merge upstream): probe
                        # MILP and LP on a sample, then solve all with
                        # whichever survived. Conv layers use sparse
                        # per-neuron MILP — typically faster *and*
                        # tighter than LP at L1/L2. The `solver` is
                        # sticky: in probe mode, once MILP times out we
                        # drop to 'lp', once LP times out we drop to
                        # 'skip'. In explicit milp/lp mode we drop
                        # straight to skip.
                        layers_np_seq, seq_bounds, seq_li = \
                            _build_sequential_subgraph(
                                gg_ops_ser, li, bounds_by_relu)
                        seq_lo, seq_hi, seq_method, dt_probe, dt_solve = \
                            _tighten_sequential_with_probe(
                                layers_np_seq, x_lo_64, x_hi_64,
                                seq_bounds, seq_li, after_adapt,
                                lp_per_worker=lp_per_worker,
                                sample_timeout=sample_timeout,
                                n_cores=n_cores, time_left=time_left,
                                mode=solver)
                        new_lo = np.maximum(new_lo, seq_lo)
                        new_hi = np.minimum(new_hi, seq_hi)
                        method = f'adapt+seq-{seq_method}'
                        if seq_method == 'lp' and solver == 'probe':
                            solver = 'lp'
                        elif seq_method == 'skip':
                            formulation = 'skip'
                    after_lp = np.where((new_lo < 0) & (new_hi > 0))[0]
                    lp_fixed = len(after_adapt) - len(after_lp)
                    width_lp = _mean_w(new_lo, new_hi, unstable_initial)
                else:
                    # Adaptive ran (above) but LP/MILP probing didn't —
                    # either because formulation is 'skip' (a prior
                    # layer's probe failed), or adapt already closed all
                    # unstable neurons, or the time budget is used up.
                    # The adaptive pass still tightened bounds even when
                    # crown_fixed == 0, so 'adapt-only' is the right
                    # label either way.
                    method = 'adapt-only'
                    if formulation == 'skip':
                        method = 'adapt-only (skip-mode)'
                    elif not len(after_adapt):
                        method = 'adapt-only (all-stable)'

                bounds_by_relu[li] = (new_lo, new_hi)
                tight_lo_t = torch.as_tensor(
                    new_lo, dtype=dtype, device=device)
                tight_hi_t = torch.as_tensor(
                    new_hi, dtype=dtype, device=device)
            else:
                tight_lo_t = None
                tight_hi_t = None

            tb['relu_adapt'] += dt_adapt
            tb['relu_tighten'] += dt_build + dt_probe + dt_solve

            # Step 3.5: record zonotope pre-ReLU rows (reuse_gen_rows)
            # if the caller asked for it. Must happen before apply_relu
            # touches z's G matrix; the unstable set matches apply_relu's
            # own classification (same lo/hi after tight-bound intersection
            # is used to build bounds_by_relu[li] above).
            if rec_zono is not None:
                _record_zono_pre_relu_rows(
                    z, li, bounds_by_relu[li], rec_zono)

            # Step 4: apply the relu using the tightest available bounds.
            # The zonotope's internal bounds are intersected with
            # (tight_lo_t, tight_hi_t) inside apply_relu itself.
            _ts = time.perf_counter()
            lo_used, hi_used = z.apply_relu(
                tight_lo=tight_lo_t, tight_hi=tight_hi_t)
            sb[li] = (lo_used.clone(), hi_used.clone())
            lo_final = lo_used.cpu().numpy().astype(np.float64)
            hi_final = hi_used.cpu().numpy().astype(np.float64)
            bounds_by_relu[li] = (lo_final, hi_final)
            zono_state[name] = z
            tb['relu_apply'] += time.perf_counter() - _ts
            # Width after intersecting the LP/adapt bounds with the
            # zonotope's own internal bounds inside apply_relu.
            width_final = _mean_w(lo_final, hi_final, unstable_initial)

            if print_progress:
                new_ust = int(((lo_final < 0) & (hi_final > 0)).sum())
                print(f'  li={li} ({gg["relu_names"][li]}): '
                      f'{len(unstable_initial)} → {new_ust} unstable '
                      f'[{method}] widths zono={width_zono:.3f} '
                      f'adapt={width_adapt:.3f} lp={width_lp:.3f} '
                      f'final={width_final:.3f} '
                      f'(adapt={dt_adapt:.2f}s fixed {crown_fixed}, '
                      f'lp={dt_solve:.2f}s fixed {lp_fixed})')
            if verbose_cb is not None:
                verbose_cb(li, gg['relu_names'][li], {
                    'adapt': dt_adapt,
                    'build': dt_build,
                    'probe': dt_probe,
                    'solve': dt_solve,
                    'width_zono': width_zono,
                    'width_adapt': width_adapt,
                    'width_lp': width_lp,
                    'width_final': width_final,
                })

        elif t == 'add':
            _ts = time.perf_counter()
            if op.get('is_merge'):
                z_a = _get(op['inputs'][0])
                z_b = _get(op['inputs'][1])
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                zono_state[name] = z_a.add(z_b, shared)
            else:
                z = _get(op['inputs'][0])
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.tensor(
                        bias.flatten(), dtype=dtype, device=device)
                    z = z.copy()
                    z.center = z.center + bt
                zono_state[name] = z
            tb['add'] += time.perf_counter() - _ts

        elif t == 'sub':
            _ts = time.perf_counter()
            z = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(
                    bias.flatten(), dtype=dtype, device=device)
                z = z.copy()
                z.center = z.center - bt
            zono_state[name] = z
            tb['sub'] += time.perf_counter() - _ts

        elif t == 'reshape':
            _ts = time.perf_counter()
            zono_state[name] = _get(op['inputs'][0])
            tb['reshape'] += time.perf_counter() - _ts

        gen_count[name] = zono_state[name].n_gens
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]

    z_final = zono_state[gg['ops'][-1]['name']]
    return sb, bounds_by_relu, z_final, tb


def _zono_spec_lbs_and_open_qis(center, generators, queries_flat,
                                   w_qs=None, b_qs=None):
    """Compute per-query zonotope spec lower bounds + open-query list.

    The unsafe region is OR-of-disjuncts of AND-of-queries (DNF). A
    disjunct is "open" (potentially SAT, needing further refinement) iff
    EVERY query in it has lb ≤ 0 — one provably-safe query closes the
    whole conjunct (AND fails if any operand fails).

    Returns (spec_lbs_dict, open_qis) where:
      - spec_lbs_dict: {query_index: zono_lb_float}. KEYED BY QUERY INDEX,
        not disjunct id. For multi-query conjuncts (e.g. cersyve has 2
        queries per disjunct), disjunct-id keying silently overwrites
        (last query wins), losing per-query info. Downstream consumers
        read `spec_lbs.get(qi, 0.0)` expecting query-indexed dict; the
        mismatch had previously masked a soundness-relevant bug where the
        pre-cascade PGD sorted disjuncts by garbage `min` over partially-
        missing lbs.
      - open_qis: list of query indices belonging to still-open disjuncts.

    Pure numpy / lightweight — kept testable independent of GPU graph.
    """
    c = np.asarray(center, dtype=np.float64).flatten()
    G = np.asarray(generators, dtype=np.float64)
    if G.size > 0 and G.shape[0] != c.size:
        G = G.reshape(c.size, -1)
    if w_qs is None:
        w_qs = np.stack([np.asarray(q[1], dtype=np.float64)
                          for q in queries_flat])
    if b_qs is None:
        b_qs = np.asarray([float(q[2]) for q in queries_flat],
                           dtype=np.float64)
    spec_lbs = {}
    for qi in range(len(queries_flat)):
        wq = w_qs[qi]
        bq = float(b_qs[qi])
        # `as_linear_queries` convention: verified safe iff
        # min(w·y + bias) > 0. For zono y = c + G·e, e∈[-1,1]^k:
        #   lb(w·y + bias) = w·c + bias - |w·G|_1.
        # The bias is ADDED, not subtracted. Bias is 0 for pairwise
        # constraints (cifar/tinyimagenet) AND for cersyve pendulum
        # (threshold value=0), so a wc - bq sign error is silent on
        # current benchmarks — but produces sort-order garbage on any
        # threshold spec with value≠0.
        wc = float(wq @ c) + bq
        wG = wq @ G if G.size > 0 else np.zeros(0)
        spec_lbs[qi] = wc - float(np.sum(np.abs(wG)))
    disj_of_qi = {qi: queries_flat[qi][0]
                   for qi in range(len(queries_flat))}
    disj_open = set()
    for di in {q[0] for q in queries_flat}:
        qids_in_disj = [qi for qi in range(len(queries_flat))
                         if disj_of_qi[qi] == di]
        if all(spec_lbs[qi] <= 0 for qi in qids_in_disj):
            disj_open.add(di)
    open_qis = [qi for qi in range(len(queries_flat))
                 if disj_of_qi[qi] in disj_open]
    return spec_lbs, open_qis


def _phase1_bab_refine(
        xl, xh, gg, gg_ops_ser, x_lo_64, x_hi_64,
        sample_timeout, n_cores, time_left,
        device, dtype, settings, spec=None,
        print_progress=False, verbose_cb=None,
        rec_zono=None, pre_cascade_hook=None, gpu_lock=None,
        parallel_pgd_ctx=None):
    """α,β-CROWN-style bab-refine cascade Phase 1.

    Algorithm:
      1. Forward zono once → initial bbr at every ReLU.
      2. for L in 0..max_tighten_layer:
           a. MILP-tighten z_L with sliding window K (only L's last K
              upstream layers are binarized; older layers stay LP).
           b. Run batched α-CROWN with current bbr → tightened
              intermediate bounds at every layer (computed via CROWN
              backward from each start node, sound for the input box).
           c. Merge α-CROWN's `best_bounds` into the global bbr
              (element-wise max on lo, min on hi).

    Returns (sb, bounds_by_relu, z_final, tb) — same signature as
    `_forward_zonotope_interleaved`. When ``rec_zono`` is provided, the
    initial forward populates it AND ``z_final`` is captured from that
    same forward, so the downstream ``state_from_phase1`` path can
    skip the multi-GB ``precompute_gen_state`` allocation. When
    ``rec_zono is None`` (legacy), ``z_final`` is None and Phase 7
    falls back to the dense precompute.
    """
    from . import alpha_crown as ac
    tb = {'phase1_bab_refine': 0.0, 'phase1_alpha_total': 0.0,
          'phase1_milp_total': 0.0}
    t_total = time.perf_counter()

    # 1. Initial forward zono (no grad — pure tensor math).
    # When rec_zono is provided, also harvest pre-ReLU gen rows here
    # so Phase 7 can reuse them via state_from_phase1 (avoids the
    # dense precompute that OOMs on TinyImageNet).
    with torch.no_grad():
        sb_init, z_final_initial = _forward_zonotope_graph(
            xl, xh, gg, device, dtype, settings=settings, rec_zono=rec_zono)
    bounds_by_relu = {
        L: (lo.cpu().numpy().astype(np.float64),
            hi.cpu().numpy().astype(np.float64))
        for L, (lo, hi) in sb_init.items()}

    # 1b. Per-neuron adaptive CROWN backward at every (non-L0) ReLU.
    # The legacy `_forward_zonotope_interleaved` did this interleaved
    # with the forward at each ReLU; bab_refine silently dropped it.
    # That cost us ~50% extra unstable at the deepest ReLU layer (L9
    # on TinyImageNet ResNet: forward-zono 58 vs CROWN-backward 37) —
    # which directly translated to needing ~2× more Phase 8 binaries.
    # L0 is exact under forward zono (pre-act = affine of input box),
    # so skip it. Use `bab_refine_adapt_enabled` knob to disable.
    _adapt_enabled = bool(getattr(
        settings, 'bab_refine_adapt_enabled', True))
    if _adapt_enabled:
        _adapt_chunk = getattr(settings, 'adapt_chunk_size', 256)
        _adapt_topk = getattr(settings, 'phase1_adapt_topk', None)
        t_adapt = time.perf_counter()
        for L in sorted(bounds_by_relu.keys()):
            if L == 0:
                continue
            lo_np, hi_np = bounds_by_relu[L]
            unstable = np.where((lo_np < 0) & (hi_np > 0))[0]
            if len(unstable) == 0:
                continue
            if (_adapt_topk is not None
                    and len(unstable) > int(_adapt_topk)):
                un = unstable
                score = (np.abs((lo_np[un] + hi_np[un]) / 2.0)
                         * (hi_np[un] - lo_np[un]))
                top = un[np.argsort(-score)[:int(_adapt_topk)]]
                neuron_subset = np.sort(top)
            else:
                neuron_subset = unstable
            try:
                adapt_lo, adapt_hi = _per_neuron_adaptive_bounds_chunked(
                    gg, xl, xh, bounds_by_relu, L, device, dtype,
                    neuron_subset=neuron_subset, chunk_size=_adapt_chunk)
                new_lo = np.maximum(lo_np, adapt_lo)
                new_hi = np.minimum(hi_np, adapt_hi)
                bounds_by_relu[L] = (new_lo, new_hi)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                # Skip this layer if backward CROWN OOMs; downstream
                # cascade still uses whatever we have.
                pass
        tb['phase1_per_neuron_adapt'] = time.perf_counter() - t_adapt

    # 2. Build query directions from spec for α-CROWN refresh.
    # gg_ops_ser is the *serialized* op list — fc uses 'W_np', conv
    # uses 'kernel_np'. The gpu_graph live form (used in the prototype
    # script) uses 'W' / 'kernel' instead.
    last_op = gg_ops_ser[-1]
    if last_op['type'] == 'fc':
        if 'W_np' in last_op:
            n_output = int(last_op['W_np'].shape[0])
        else:
            n_output = int(last_op['W'].shape[0])
    elif last_op['type'] == 'conv':
        out_shape = last_op.get('out_shape')
        n_output = int(np.prod(out_shape)) if out_shape else 10
    else:
        n_output = sb_init[max(sb_init.keys())][0].numel()
    queries_flat = spec.as_linear_queries(n_output) if spec is not None else []
    if queries_flat:
        w_qs = np.stack([np.asarray(q[1], dtype=np.float64)
                         for q in queries_flat])
        b_qs = np.asarray([float(q[2]) for q in queries_flat],
                          dtype=np.float64)
    else:
        w_qs = np.zeros((1, n_output), dtype=np.float64)
        b_qs = np.zeros(1, dtype=np.float64)

    # 2b. Pre-α-CROWN hook: zono-based spec lbs from z_final_initial give
    # an *order* of disjuncts by closeness-to-SAT before paying for the
    # α-CROWN intermediate refresh + spec backward. If PGD finds SAT here
    # we skip α-CROWN entirely. For zonotope y = c + G·e, e∈[-1,1]^k:
    #   spec_lb = w_q·c - b_q - ||w_q·G||_1
    if pre_cascade_hook is not None and queries_flat:
        _c = z_final_initial.center.flatten().detach().cpu().numpy(
            ).astype(np.float64)
        _G = z_final_initial.generators
        _G_np = (_G.detach().cpu().numpy().astype(np.float64)
                  if _G.numel() > 0
                  else np.zeros((_c.size, 0), dtype=np.float64))
        spec_lbs_zono_dict, _open_qis_zono = _zono_spec_lbs_and_open_qis(
            _c, _G_np, queries_flat, w_qs=w_qs, b_qs=b_qs)
        tb['spec_lbs_zono'] = dict(spec_lbs_zono_dict)
        sat_witness = pre_cascade_hook(
            spec_lbs_zono_dict, bounds_by_relu, _open_qis_zono)
        if sat_witness is not None:
            sb = {L: (torch.tensor(lo, dtype=dtype, device=device),
                      torch.tensor(hi, dtype=dtype, device=device))
                  for L, (lo, hi) in bounds_by_relu.items()}
            tb['phase1_bab_refine'] = time.perf_counter() - t_total
            tb['pre_cascade_sat'] = sat_witness
            return sb, bounds_by_relu, z_final_initial, tb

    # 3. Window override (env var > setting).
    win = int(getattr(settings, 'bab_refine_window', 1))
    win_env = os.environ.get('VC_TIGHTEN_WINDOW', '')
    if win_env:
        try:
            win = int(win_env)
        except ValueError:
            pass
    os.environ['VC_TIGHTEN_WINDOW'] = str(win)

    # 4. Cascade.
    max_layer = getattr(settings, 'max_tighten_layer', None)
    if max_layer is None:
        max_layer = max(bounds_by_relu.keys())
    max_layer = min(int(max_layer), max(bounds_by_relu.keys()))
    alpha_iters = int(getattr(settings, 'zono_lift_alpha_iters', 10))
    alpha_lr = float(getattr(settings, 'zono_lift_alpha_lr', 0.25))
    # Per-layer time budget. `bab_refine_layer_budget_frac > 0` means use
    # FRACTION of total_timeout (more robust than the absolute-seconds cap
    # for short timeouts). Otherwise fall back to the absolute knob.
    _budget_frac = float(getattr(settings, 'bab_refine_layer_budget_frac', 0.0))
    if _budget_frac > 0.0:
        layer_budget = _budget_frac * float(getattr(settings,
                                                     'total_timeout', 120.0))
    else:
        layer_budget = float(getattr(settings, 'bab_refine_layer_budget', 30.0))
    # Topk filter (AB-CROWN's `topk_filter`) — only tighten the most
    # impactful K unstable neurons per layer.
    topk = int(getattr(settings, 'bab_refine_topk', 0))
    score_mode = str(getattr(settings, 'bab_refine_score', 'center_width'))
    short_circuit = bool(getattr(settings, 'bab_refine_short_circuit', True))
    n_passes = max(1, int(getattr(settings, 'bab_refine_passes', 1)))
    phase05_alpha_iters = int(
        getattr(settings, 'bab_refine_phase05_alpha_iters', 10))
    phase05_spec_iters = int(
        getattr(settings, 'bab_refine_phase05_spec_iters', 20))

    def _alpha_refresh_best_bounds(bbr_now, S_nodes, un_idx, n_iters):
        """Dispatch α-CROWN intermediate-bound refresh.

        Returns `best_bounds: dict {L: (lo_t, hi_t)}` from
        `run_alpha_crown_batched` (shared α across queries with a
        spec-LB loss).
        """
        _dir_mode = str(settings.alpha_crown_dir_mode
                         if 'alpha_crown_dir_mode' in settings
                         else 'auto')
        _s_split = int(settings.alpha_crown_s_split_n
                        if 'alpha_crown_s_split_n' in settings
                        else 1)
        _, _, best_bounds, _ = ac.run_alpha_crown_batched(
            gg, xl, xh, bbr_now, w_qs, b_qs, S_nodes, un_idx,
            device, dtype, n_iters=n_iters, lr=alpha_lr,
            lr_decay=0.98, early_stop_on_positive=False,
            dir_mode=_dir_mode, s_split_n=_s_split,
            time_left_fn=time_left)
        return best_bounds

    ew_score_per_layer = None
    min_ew_per_layer = None
    need_ew = score_mode == 'ew_frac' and topk > 0
    spec_ew_local = {
        qi: (torch.as_tensor(np.asarray(queries_flat[qi][1],
                                          dtype=np.float64),
                              dtype=dtype, device=device),
             float(queries_flat[qi][2]))
        for qi in range(len(queries_flat))
    }

    # ---------- Phase 0.5: α-CROWN open-spec detection ----------
    # 1. α-CROWN joint intermediate tightening (Adam over per-layer α)
    # 2. α-CROWN spec direction with frozen intermediate (Adam over spec α)
    # 3. open_qis = queries with spec_lb still <= 0 after step 2
    # The post-Phase-0.5 bbr feeds the ew sweep, and `open_qis` restricts
    # the AB-CROWN `remove_unstable_neurons` filter to un-proved queries.
    intermediate_start_nodes_init = [
        Lk for Lk in bounds_by_relu if Lk > 0 and
        ((bounds_by_relu[Lk][0] < 0)
         & (bounds_by_relu[Lk][1] > 0)).any()]
    unstable_indices_init = {
        Lk: np.where((bounds_by_relu[Lk][0] < 0)
                      & (bounds_by_relu[Lk][1] > 0))[0].tolist()
        for Lk in bounds_by_relu}
    if intermediate_start_nodes_init and queries_flat:
        t_a = time.perf_counter()
        best_bounds = _alpha_refresh_best_bounds(
            bounds_by_relu, intermediate_start_nodes_init,
            unstable_indices_init, phase05_alpha_iters)
        for Lk in best_bounds:
            lo_t, hi_t = best_bounds[Lk]
            lo_a = lo_t.detach().cpu().numpy().astype(np.float64)
            hi_a = hi_t.detach().cpu().numpy().astype(np.float64)
            lo_g, hi_g = bounds_by_relu[Lk]
            bounds_by_relu[Lk] = (
                np.maximum(lo_g, lo_a),
                np.minimum(hi_g, hi_a))
        tb['phase1_alpha_total'] += time.perf_counter() - t_a

    cascade_start_layer = 1

    # Spec α-CROWN with frozen (just-tightened) intermediate bounds.
    open_qis = list(range(len(queries_flat)))
    spec_lbs_phase05 = None
    spec_alpha_phase05 = None  # captured for downstream ew sweep
    if queries_flat:
        t_a = time.perf_counter()
        spec_lbs_phase05, alpha_dict_p05, _, _ = (
            ac.run_alpha_crown_fixed_intermediate_batched(
                gg, xl, xh, bounds_by_relu, w_qs, b_qs,
                device, dtype, n_iters=phase05_spec_iters, lr=alpha_lr,
                lr_decay=0.98, early_stop_on_positive=True,
                per_spec_alpha=False, time_left_fn=time_left))
        spec_alpha_phase05 = alpha_dict_p05.get('spec')
        open_qis = [qi for qi in range(len(queries_flat))
                    if float(spec_lbs_phase05[qi]) <= 0]
        tb['phase1_alpha_total'] += time.perf_counter() - t_a
        # Stash spec lbs (keyed by spec qi) so Phase 2 CROWN can be skipped
        # when phase2_crown_enabled=False — α-CROWN already gave tighter lbs.
        tb['spec_lbs_phase05'] = {
            queries_flat[i][0]: float(spec_lbs_phase05[i])
            for i in range(len(queries_flat))
        }
        if print_progress:
            n_closed = len(queries_flat) - len(open_qis)
            print(f'  [phase0.5] α-CROWN closed {n_closed}/'
                  f'{len(queries_flat)} specs', flush=True)

    # If Phase 0.5 closed every spec, exit Phase 1 immediately.
    if not open_qis:
        sb = {L: (torch.tensor(lo, dtype=dtype, device=device),
                  torch.tensor(hi, dtype=dtype, device=device))
              for L, (lo, hi) in bounds_by_relu.items()}
        tb['phase1_bab_refine'] = time.perf_counter() - t_total
        return sb, bounds_by_relu, z_final_initial, tb

    # ---------- Parallel PGD worker (only open disjuncts, MILP-window) ----------
    # Spawn AFTER Phase 0.5 so we target only the queries Phase 0.5
    # couldn't close. Thread runs PGD on each open disjunct (sorted by
    # spec_lb ascending = most-likely-SAT first) during cascade's MILP
    # windows (signaled via `gpu_lock` which is a threading.Event used
    # as `milp_active`). Thread stops + joins at end of bab_refine so it
    # never runs concurrently with Phase 8.
    parallel_pgd_thread = None
    parallel_pgd_state = None
    if (parallel_pgd_ctx is not None and gpu_lock is not None
            and open_qis and not bool(getattr(
                settings, 'disable_sat_finding', False))):
        import threading as _threading
        # Map open queries → disjuncts, sorted by spec_lb ascending.
        _open_disjs = {}
        for _qi in open_qis:
            _di = queries_flat[_qi][0]
            _lb = float(spec_lbs_phase05[_qi])
            if _di not in _open_disjs or _lb < _open_disjs[_di]:
                _open_disjs[_di] = _lb
        _ordered_disjs = sorted(_open_disjs.keys(),
                                  key=lambda _d: _open_disjs[_d])
        _per_b = (float(settings.phase26_pgd_per_spec_min_per_spec)
                   if 'phase26_pgd_per_spec_min_per_spec' in settings
                   else 0.5)
        parallel_pgd_state = {
            'witness': None, 'di': None,
            'stop': _threading.Event(), 'n': 0}
        _xl_pgd = parallel_pgd_ctx['xl_pgd']
        _xh_pgd = parallel_pgd_ctx['xh_pgd']
        _gg_pgd = parallel_pgd_ctx['gg_pgd']
        _spec_pgd = parallel_pgd_ctx['spec']
        milp_active = gpu_lock  # alias for clarity (it's a threading.Event)

        def _pgd_thread_main():
            for _di in _ordered_disjs:
                # Poll for milp_active in short ticks; don't break on
                # any single timeout — pre-cascade or α-CROWN may run
                # many seconds before the first MILP window opens.
                while True:
                    if parallel_pgd_state['stop'].is_set(): return
                    if time_left() <= 1.0: return
                    if milp_active.wait(timeout=0.5):
                        break
                if parallel_pgd_state['stop'].is_set(): return
                try:
                    _ok, _w = _pgd_attack_general(
                        _xl_pgd, _xh_pgd, _spec_pgd, _gg_pgd, settings,
                        restrict_disj={_di},
                        time_budget=min(_per_b,
                                          max(time_left() - 0.5, 0.1)))
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    _ok, _w = False, None
                parallel_pgd_state['n'] += 1
                if _ok:
                    parallel_pgd_state['witness'] = _w
                    parallel_pgd_state['di'] = _di
                    return

        parallel_pgd_thread = _threading.Thread(
            target=_pgd_thread_main, daemon=True)
        parallel_pgd_thread.start()
        if print_progress:
            print(f'  [parallel-PGD] thread started (open-only, '
                  f'{len(_ordered_disjs)} disj, per={_per_b}s)',
                  flush=True)

    # ---------- ew sweep restricted to OPEN queries ----------
    # min_ew_per_layer[L][j] = min over open specs of ew[i, j]. The
    # AB-CROWN `remove_unstable_neurons` filter skips neurons whose
    # min_ew > 0 (no open spec uses y_j's upper-bound ReLU relaxation).
    if need_ew:
        from .alpha_crown import _make_slopes
        # Use the OPTIMIZED spec α from Phase 0.5 when available — its
        # ew sign agrees with the actual α-CROWN's spec-direction
        # backward, which trivial α only approximates. Trivial α
        # disagreement with optimized α grows with depth and on
        # mnist 256x6 prop_5_0.05 the trivial-α filter wrongly skips
        # genuinely-helpful neurons (regresses 74.4s → 302s timeout).
        # Fall back to trivial α when Phase 0.5 ran with per-spec α
        # (capture_ew_per_relu expects (n_neurons,) shape) or didn't run.
        alpha_for_ew = {}
        for L in bounds_by_relu:
            lo_t = torch.as_tensor(bounds_by_relu[L][0], dtype=dtype,
                                    device=device)
            hi_t = torch.as_tensor(bounds_by_relu[L][1], dtype=dtype,
                                    device=device)
            lo_s, _, _, active, dead, unstable = _make_slopes(lo_t, hi_t)
            if (spec_alpha_phase05 is not None
                    and L in spec_alpha_phase05
                    and spec_alpha_phase05[L].numel() == lo_t.numel()):
                # Use optimized α from Phase 0.5's spec α-CROWN call.
                a = spec_alpha_phase05[L].detach().clone()
            else:
                # Trivial α (lower-edge slope) fallback.
                a = torch.zeros_like(lo_t)
                a[active] = 1.0
                a[unstable] = lo_s[unstable]
            alpha_for_ew[L] = a
        ew_score_per_layer = {L: np.zeros(bounds_by_relu[L][0].size,
                                           dtype=np.float64)
                               for L in bounds_by_relu}
        min_ew_per_layer = {L: np.full(bounds_by_relu[L][0].size,
                                        np.inf, dtype=np.float64)
                             for L in bounds_by_relu}
        for qi in open_qis:
            _, ew_at = ac.capture_ew_per_relu(
                gg, xl, xh, alpha_for_ew, bounds_by_relu,
                w_qs[qi], float(b_qs[qi]), device, dtype)
            for L, ew_t in ew_at.items():
                ew_np = ew_t.detach().cpu().numpy().astype(np.float64)
                lo_arr, hi_arr = bounds_by_relu[L]
                width = (hi_arr - lo_arr).clip(min=1e-12)
                frac = np.clip(-lo_arr / width, 0.0, 1.0)
                score = np.abs(ew_np) * frac
                ew_score_per_layer[L] = np.maximum(
                    ew_score_per_layer[L], score)
                min_ew_per_layer[L] = np.minimum(
                    min_ew_per_layer[L], ew_np)

    for pass_idx in range(n_passes):
        if time_left() <= 1:
            break
        if print_progress and n_passes > 1:
            print(f'  [bab_refine] pass {pass_idx+1}/{n_passes}', flush=True)
        # Skip L=0: zonotope forward is exact at the first hidden layer
        # (z_0 = W_0 x + b_0 over a box-shaped input has tight closed-form
        # interval bounds), so per-neuron MILP at L=0 has no binaries to
        # branch on (nothing upstream is unstable) and produces the same
        # bound as the box. Saves the ~0.2s no-op and matches AB-CROWN's
        # `if relu_idx >= 1` guard at lp_mip_solver.py:1849.
        for L in range(cascade_start_layer, max_layer + 1):
            if time_left() <= 1:
                break
            lo_l, hi_l = bounds_by_relu[L]
            unstable = np.where((lo_l < 0) & (hi_l > 0))[0]
            if len(unstable) == 0:
                continue
            # Compute per-neuron score for ordering (highest-impact first).
            # The pool worker gets tasks in this order, so the layer-budget
            # cap (below) cuts off the LOW-impact tail rather than random.
            if (score_mode == 'ew_frac' and ew_score_per_layer is not None
                    and L in ew_score_per_layer):
                score = ew_score_per_layer[L][unstable]
            else:
                # Legacy default — |c_in| × width
                score = np.abs((lo_l[unstable] + hi_l[unstable]) / 2.0) * (
                    hi_l[unstable] - lo_l[unstable])
            # Sort unstables in DESCENDING score order. Pool's imap_unordered
            # picks tasks off in submission order; with this sort, workers
            # process the highest-impact neurons first.
            order = np.argsort(-score)
            unstable_sorted = unstable[order]
            # Topk filter — when explicitly requested via `bab_refine_topk>0`
            # we still hard-cap the count after sorting. With topk=0 (default)
            # the budget alone gates how many run.
            if topk > 0 and len(unstable_sorted) > topk:
                unstable_for_milp = unstable_sorted[:topk]
            else:
                unstable_for_milp = unstable_sorted
            # MILP-tighten z_L (sliding window applied via env var).
            t_m = time.perf_counter()
            # Cap total time for this layer's MILP — wraps the time_left
            # function passed to _tighten_layer_gen_cone so the worker pool
            # gives up after `layer_budget` seconds, not after the global
            # remaining time.
            layer_start = time.perf_counter()
            gtl = time_left
            _layer_lb = layer_budget
            def _layer_time_left(start=layer_start, lb=_layer_lb,
                                  gtl=gtl):
                return min(gtl(), lb - (time.perf_counter() - start))
            new_lo, new_hi, _meth, _db, _dp, _ds = _tighten_layer_gen_cone(
                gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu, L,
                unstable_for_milp,
                gg['input_name'],
                sample_timeout=sample_timeout, n_cores=n_cores,
                time_left=_layer_time_left, mode='gen_cone_milp',
                device=device, dtype=dtype, use_milp=True, precomputed=None,
                gpu_lock=gpu_lock)
            bounds_by_relu[L] = (np.maximum(lo_l, new_lo),
                                  np.minimum(hi_l, new_hi))
            tb['phase1_milp_total'] += time.perf_counter() - t_m
            # Hard early-exit hook: terminate the whole verifier process
            # after a specific layer's MILPs are done. Used by the
            # MILP-dump scratch script to capture L1+L2 problems without
            # paying for L3+ work or the full verify pipeline.
            _stop_after = os.environ.get('VC_STOP_AFTER_LAYER', '')
            if _stop_after:
                try:
                    if L >= int(_stop_after):
                        if print_progress:
                            print(f'  [bab_refine] stopping after L{L} '
                                  f'(VC_STOP_AFTER_LAYER={_stop_after}) — '
                                  f'terminating process for MILP dump',
                                  flush=True)
                        os._exit(0)
                except ValueError:
                    pass
            # α-CROWN refresh.
            if time_left() <= 1:
                break
            t_a = time.perf_counter()
            intermediate_start_nodes = [
                Lk for Lk in bounds_by_relu if Lk > 0 and
                ((bounds_by_relu[Lk][0] < 0)
                 & (bounds_by_relu[Lk][1] > 0)).any()]
            unstable_indices = {
                Lk: np.where((bounds_by_relu[Lk][0] < 0)
                              & (bounds_by_relu[Lk][1] > 0))[0].tolist()
                for Lk in bounds_by_relu}
            if intermediate_start_nodes and queries_flat:
                best_bounds = _alpha_refresh_best_bounds(
                    bounds_by_relu, intermediate_start_nodes,
                    unstable_indices, alpha_iters)
                for Lk in best_bounds:
                    lo_t, hi_t = best_bounds[Lk]
                    lo_a = lo_t.detach().cpu().numpy().astype(np.float64)
                    hi_a = hi_t.detach().cpu().numpy().astype(np.float64)
                    lo_g, hi_g = bounds_by_relu[Lk]
                    bounds_by_relu[Lk] = (
                        np.maximum(lo_g, lo_a),
                        np.minimum(hi_g, hi_a))
            tb['phase1_alpha_total'] += time.perf_counter() - t_a
            if print_progress:
                counts = ','.join(
                    f'L{Lk}={int(((bounds_by_relu[Lk][0]<0)&(bounds_by_relu[Lk][1]>0)).sum())}'
                    for Lk in sorted(bounds_by_relu.keys()))
                print(f'  [bab_refine L{L}] tighten+α: {counts}', flush=True)
            # Short-circuit: quick CROWN spec check on still-open specs. If
            # all closed, exit Phase 1 immediately — saves work on cases where
            # an early layer's tightening already verifies the spec.
            if short_circuit and len(queries_flat) > 0:
                sb_now = {Lk: (torch.as_tensor(lo, dtype=dtype, device=device),
                                torch.as_tensor(hi, dtype=dtype, device=device))
                           for Lk, (lo, hi) in bounds_by_relu.items()}
                spec_lbs_now, _ = _spec_backward_graph(
                    sb_now, xl, xh, gg, spec_ew_local,
                    list(range(len(queries_flat))),
                    len(bounds_by_relu), device, dtype)
                if all(lb > 0 for lb in spec_lbs_now.values()):
                    if print_progress:
                        print(f'  [bab_refine L{L}] all specs closed (short-circuit)',
                              flush=True)
                    break
    # Materialise sb (torch tensors) from updated bounds_by_relu.
    sb = {L: (torch.tensor(lo, dtype=dtype, device=device),
              torch.tensor(hi, dtype=dtype, device=device))
          for L, (lo, hi) in bounds_by_relu.items()}
    # If rec_zono was requested, redo the forward with the cascade-
    # tightened bounds_by_relu fed into apply_relu via tight_lo/tight_hi.
    # This recomputes (μ, λ) from the tighter (lo, hi) so that
    # state_from_phase1's LP triangle constraints reflect the cascade
    # gain. Without this we'd ship Phase 1's looser bounds into the LP
    # and pay multi-second Phase 8 MILP escalation that the cascade
    # already obviated. One extra forward cost (small relative to the
    # cascade itself).
    if rec_zono is not None:
        rec_zono.clear()
        rec_zono.setdefault('gen_rows_by_layer', {})
        rec_zono.setdefault('col_origin', {})
        with torch.no_grad():
            _, z_final_initial = _forward_zonotope_graph(
                xl, xh, gg, device, dtype, settings=settings,
                rec_zono=rec_zono, tight_bounds=bounds_by_relu)

    # Stop + join parallel PGD thread (if spawned). MUST happen before
    # return so the thread never runs concurrently with Phase 7/8.
    if parallel_pgd_thread is not None:
        parallel_pgd_state['stop'].set()
        if gpu_lock is not None:
            gpu_lock.set()  # wake any pending .wait()
        parallel_pgd_thread.join(timeout=2.0)
        tb['parallel_pgd_attacks'] = parallel_pgd_state['n']
        if parallel_pgd_state['witness'] is not None:
            tb['parallel_pgd_sat'] = parallel_pgd_state['witness']
            tb['parallel_pgd_di'] = parallel_pgd_state['di']
        if print_progress:
            _alive = parallel_pgd_thread.is_alive()
            print(f'  [parallel-PGD] joined: '
                  f'n_attacks={parallel_pgd_state["n"]}, '
                  f'sat={parallel_pgd_state["witness"] is not None}, '
                  f'alive_after_join={_alive}', flush=True)

    tb['phase1_bab_refine'] = time.perf_counter() - t_total
    return sb, bounds_by_relu, z_final_initial, tb


# ---------------------------------------------------------------------------
# Phase 2.5: zonotope-lift tightening with closed-form box-halfspace LP
# ---------------------------------------------------------------------------
#
# A cheap per-query tightening pass that sits between Phase 2 (CROWN) and
# Phase 7 (triangle LP). Uses the forward zonotope's generator matrix G and
# the lifted spec halfspace (G^T w · e ≤ −w·c − b) to tighten every
# unstable pre-ReLU bound via a closed-form 1-halfspace LP
# (see `vibecheck.box_halfspace`). Iterates until CROWN LB crosses 0 or
# no further flips occur. Bounds stay query-local (sound only for the
# counterexample region); they're NOT merged into the shared
# `bounds_by_relu`.

@torch.no_grad()
def _forward_keep_pre_gpu(xl, xh, gg, device, dtype, override_tight=None,
                          settings=None, unstable_per_layer=None):
    """Forward zono keeping pre-ReLU (c, G_unstable_rows) per layer_idx.

    Mirrors `_forward_zonotope_graph` but:
      - Records the pre-ReLU (center, G) at every ``layer_idx`` ReLU so
        the box-halfspace LP can read G rows for each unstable neuron.
      - Accepts ``override_tight={layer_idx: (lo_np, hi_np)}`` bounds to
        feed into ``apply_relu(tight_lo, tight_hi)``.

    When ``unstable_per_layer`` is provided (dict
    ``{layer_idx: LongTensor of neuron indices}``), only those rows of
    G are retained in ``pre_relu_gpu[L]``. This avoids materialising a
    dense ``(n_flat, K)`` G per layer (10 GB+ on resnet_large); the
    halfspace-LP only reads unstable rows anyway.

    When ``unstable_per_layer`` is None, falls back to storing full G
    per layer — legacy behaviour.

    Returns ``(z_final, pre_relu_gpu)`` with
    ``pre_relu_gpu[L] = (c_slim, G_slim)`` either slim (unstable-only)
    or full, depending on ``unstable_per_layer``.
    """
    z_init = make_input_zonotope(
        settings, xl, xh, device, dtype, in_shape=gg.get('input_shape'))
    zono_state = {gg['input_name']: z_init}
    gen_count = {gg['input_name']: z_init.n_gens}
    forks = gg['fork_points']
    pre_relu_gpu = {}
    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    # Track remaining consumers per fork point so the LAST consumer can take
    # ownership of the original state (no clone). Saves the ~1 GB fork-copy
    # peak on resnet-sized nets.
    remaining_consumers = {
        fn: sum(1 for op2 in gg['ops'] if fn in op2['inputs'])
        for fn in forks}

    def _get(name):
        if name in forks:
            remaining_consumers[name] -= 1
            if remaining_consumers[name] > 0:
                return zono_state[name].copy()
        return zono_state[name]

    def _snapshot_pre_relu(z, L):
        """Snapshot (center, G) pre-ReLU at layer L. Uses nonzero_rows to
        avoid materialising the full dense G when ``unstable_per_layer``
        is provided."""
        if (unstable_per_layer is not None
                and L in unstable_per_layer
                and unstable_per_layer[L].numel() > 0):
            un = unstable_per_layer[L].to(device)
            c_slim = z.center[un].clone()
            rid, cid, val = z.nonzero_rows(un)
            K = z.n_gens
            G_slim = torch.zeros(
                un.numel(), K, dtype=z.center.dtype, device=device)
            if rid.numel() > 0:
                G_slim[rid, cid] = val
            pre_relu_gpu[L] = (c_slim, G_slim)
        elif (unstable_per_layer is not None
              and L in unstable_per_layer
              and unstable_per_layer[L].numel() == 0):
            # No unstable rows at this layer — empty snapshot.
            pre_relu_gpu[L] = (
                torch.empty(0, dtype=z.center.dtype, device=device),
                torch.empty(0, 0, dtype=z.center.dtype, device=device))
        else:
            # Legacy full-G path (no unstable_per_layer given).
            pre_relu_gpu[L] = (z.center.clone(), z.generators.clone())

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']; t = op['type']
        if t == 'conv':
            z = _get(op['inputs'][0])
            z.propagate_conv(op['kernel'], op['bias'], op['in_shape'],
                             op['stride'], op['padding'])
            zono_state[name] = z
        elif t == 'fc':
            z = _get(op['inputs'][0])
            z.propagate_fc(op['W'], op['bias'])
            zono_state[name] = z
        elif t == 'relu':
            z = _get(op['inputs'][0])
            if 'layer_idx' in op:
                _snapshot_pre_relu(z, op['layer_idx'])
                tl = th = None
                if override_tight and op['layer_idx'] in override_tight:
                    lo_np, hi_np = override_tight[op['layer_idx']]
                    tl = torch.as_tensor(lo_np, dtype=dtype, device=device)
                    th = torch.as_tensor(hi_np, dtype=dtype, device=device)
                z.apply_relu(tight_lo=tl, tight_hi=th)
            else:
                z.apply_relu()
            zono_state[name] = z
        elif t == 'add':
            if op.get('is_merge'):
                z_a = _get(op['inputs'][0]); z_b = _get(op['inputs'][1])
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                zono_state[name] = z_a.add(z_b, shared)
            else:
                z = _get(op['inputs'][0]); bias = op.get('bias')
                if bias is not None:
                    bt = torch.tensor(
                        bias.flatten(), dtype=dtype, device=device)
                    z = z.copy()
                    z.center = z.center + bt
                zono_state[name] = z
        elif t == 'sub':
            z = _get(op['inputs'][0]); bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(
                    bias.flatten(), dtype=dtype, device=device)
                z = z.copy()
                z.center = z.center - bt
            zono_state[name] = z
        elif t == 'reshape':
            zono_state[name] = _get(op['inputs'][0])
        gen_count[name] = zono_state[name].n_gens
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]

    last_name = gg['ops'][-1]['name']
    return zono_state[last_name], pre_relu_gpu


def _phase2p5_zono_lift(
        gg, xl_g, xh_g, bounds_by_relu, disj_queries, spec_ew, spec_lbs,
        still_open_disj, device, dtype, settings, print_progress,
        mem_audit=None):
    """Iterative zono-lift tightening for still-open disjuncts.

    For each query qi in a still-open disjunct:
      1. Forward zono with `bounds_by_relu` → (c_out, G_out).
      2. Lift spec_qi halfspace into generator space.
      3. Tighten all unstable pre-ReLU bounds via closed-form box+halfspace LP.
      4. Re-forward with tightened bounds, recompute CROWN LB on spec_qi.
      5. Iterate up to `zono_lift_max_passes`. Stop when CROWN LB > 0
         (query closed) or LB improvement < `zono_lift_tolerance`.

    Updates `spec_lbs[qi]` with the best LB found; tightening bounds
    remain query-local (never merged into `bounds_by_relu`).

    Returns (still_open_disj_updated, info_dict).
    """
    from . import box_halfspace
    from . import alpha_crown as ac

    # Phase 1+2 can leave multi-GB of cached GPU allocator chunks; force
    # a fresh malloc pool before the cascade starts.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    max_passes = int(getattr(settings, 'zono_lift_max_passes', 3))
    tol = float(getattr(settings, 'zono_lift_tolerance', 1e-4))
    layers_cfg = getattr(settings, 'zono_lift_layers', None)
    use_alpha = bool(getattr(settings, 'zono_lift_alpha_crown', True))
    alpha_iters = int(getattr(settings, 'zono_lift_alpha_iters', 20))
    alpha_lr = float(getattr(settings, 'zono_lift_alpha_lr', 0.25))
    early_stop_on_positive = bool(
        getattr(settings, 'alpha_crown_early_stop_on_positive', True))
    alpha_impl = str(getattr(settings, 'alpha_crown_impl', 'legacy'))
    # Auto-switch to v2_fixed_intermediate on big networks where the
    # joint-α 138-conv-transpose-backwards-per-iter cost is prohibitive
    # (cifar_biasfield: 416 ms/iter on RTX 3080 vs 4.2 ms/iter for
    # fixed_intermediate). Threshold = total unstable neurons across all
    # hidden ReLUs.
    _switch_thr = getattr(
        settings, 'alpha_crown_impl_auto_switch_threshold', None)
    if (alpha_impl == 'legacy' and _switch_thr is not None):
        _total_unstable = 0
        for _L in bounds_by_relu:
            _lo, _hi = bounds_by_relu[_L]
            _total_unstable += int(((np.asarray(_lo) < 0)
                                    & (np.asarray(_hi) > 0)).sum())
        if _total_unstable > int(_switch_thr):
            alpha_impl = 'v2_fixed_intermediate'
            if print_progress:
                print(f'  α-CROWN: auto-switched to fixed_intermediate '
                      f'(total unstable {_total_unstable} > {_switch_thr})',
                      flush=True)
    alpha_lr_decay = float(getattr(settings, 'alpha_crown_lr_decay', 0.98))
    sparse_alpha = bool(
        getattr(settings, 'alpha_crown_sparse_alpha', False))
    cascade_skip_on_close = bool(
        getattr(settings, 'zono_lift_cascade_skip_on_close', True))
    # Cascade: re-run α-CROWN INSIDE the loop on the tightened bounds so
    # each pass sees better α slopes. Enabling this turns Phase 2.5 into
    # the full α-CROWN ↔ spec-halfspace cascade (step67 / step68 results).
    cascade_per_iter_alpha = bool(
        getattr(settings, 'zono_lift_cascade_alpha', True))
    # Plateau detection: stop after K consecutive passes with no bound
    # tightening. (Independent of `tol` which uses CROWN LB delta.)
    plateau_patience = int(
        getattr(settings, 'zono_lift_plateau_patience', 2))

    info = {'per_query': {}, 'n_closed': 0}
    nh = gg['n_relu']

    still_open = set(still_open_disj)
    verified_here = set()

    # Time-budget for the whole Phase 2.5 cascade. None = unlimited.
    time_budget = getattr(settings, 'zono_lift_time_budget', None)
    time_budget = float(time_budget) if time_budget is not None else None
    _t_phase25_start = time.perf_counter()
    info['time_budget'] = time_budget
    info['time_budget_hit'] = False
    info['disjuncts_skipped_hopeless'] = 0
    info['disjuncts_skipped_time'] = 0
    info['queries_skipped_time'] = 0

    # Disjunct-level early-abort threshold (after batched pass-0 α-CROWN).
    # If the worst lb_alpha across queries in a disjunct is below this,
    # the disjunct cannot close in 2.5 (closure requires ALL queries to
    # close); skip the per-query cascade entirely.
    hopeless_disjunct_lb = getattr(
        settings, 'zono_lift_disjunct_hopeless_lb', None)
    hopeless_disjunct_lb = (float(hopeless_disjunct_lb)
                            if hopeless_disjunct_lb is not None else None)

    promising_first = bool(
        getattr(settings, 'zono_lift_promising_first', True))

    # Identify unstable neurons per layer for α-CROWN start_node selection.
    unstable_indices = {
        L: np.where(
            (np.asarray(bounds_by_relu[L][0]) < 0) &
            (np.asarray(bounds_by_relu[L][1]) > 0))[0].tolist()
        for L in bounds_by_relu
    }
    intermediate_start_nodes = [
        L for L, un in unstable_indices.items() if len(un) > 0 and L > 0]

    batch_queries = bool(
        getattr(settings, 'zono_lift_batch_queries', True))

    # --- Global batched pass-0 α-CROWN across all open queries in all
    # still-open disjuncts. One shared α-Adam graph; spec backward batched
    # (n_q_total, n_out). Matches α,β-CROWN's (1, n_spec) batching.
    # Per-query cascade still runs for anything not closed here.
    global_qi_to_batched_idx = {}
    alpha_params_shared = None
    best_bounds_shared = None
    best_lbs_batch = None
    if batch_queries and use_alpha and (
            intermediate_start_nodes or alpha_impl == 'v2_fixed_intermediate'):
        all_open_q = []
        for di in list(still_open):
            for qi, w, b in disj_queries[di]:
                if spec_lbs.get(qi, -1.0) <= 0:
                    all_open_q.append((qi, w, b))
        if len(all_open_q) >= 2:
            w_qs = np.stack([np.asarray(w, dtype=np.float64)
                             for _, w, _ in all_open_q])
            b_qs = np.asarray([float(b) for _, _, b in all_open_q],
                              dtype=np.float64)
            # Hopeless-bound thresholds: stop α-CROWN early when bound
            # is far negative AND not improving. None disables.
            _hopeless_lb = getattr(settings, 'alpha_crown_hopeless_lb', None)
            _hopeless_delta = float(getattr(
                settings, 'alpha_crown_hopeless_delta', 0.5))
            if alpha_impl == 'v2_fixed_intermediate':
                best_lbs_batch, alpha_params_shared, best_bounds_shared, _ = (
                    ac.run_alpha_crown_fixed_intermediate_batched(
                        gg, xl_g, xh_g, bounds_by_relu, w_qs, b_qs,
                        device, dtype, n_iters=alpha_iters, lr=alpha_lr,
                        lr_decay=alpha_lr_decay,
                        early_stop_on_positive=early_stop_on_positive,
                        sparse_alpha=sparse_alpha,
                        hopeless_lb=_hopeless_lb,
                        hopeless_delta=_hopeless_delta))
            else:
                best_lbs_batch, alpha_params_shared, best_bounds_shared, _ = (
                    ac.run_alpha_crown_batched(
                        gg, xl_g, xh_g, bounds_by_relu, w_qs, b_qs,
                        intermediate_start_nodes, unstable_indices,
                        device, dtype, n_iters=alpha_iters, lr=alpha_lr,
                        lr_decay=alpha_lr_decay,
                        early_stop_on_positive=early_stop_on_positive,
                        sparse_alpha=sparse_alpha,
                        hopeless_lb=_hopeless_lb,
                        hopeless_delta=_hopeless_delta))
            for idx, (qi, _, _) in enumerate(all_open_q):
                global_qi_to_batched_idx[qi] = idx

            # Optional: merge α-CROWN's per-layer best_bounds into the
            # GLOBAL bounds_by_relu so the downstream Phase 7/8 spec
            # MILP sees the tighter intermediate bounds. α-CROWN's
            # `best_bounds` are sound for the input box (the chosen α
            # gives a specific valid CROWN relaxation; bounds derived
            # from any α are sound — α optimization just picks better
            # ones). Disabled by default because the merge can shift
            # the LP big-M of phase1-form rec_zono (which captured
            # forward-zono μ, λ); enable explicitly when running with
            # max_tighten_layer ≤ K and you want the K+1..nh layers
            # tightened by α-CROWN globally. Mirrors α,β-CROWN's
            # bab-refine pattern of "tighten K layers with MILP, then
            # re-run α-CROWN to refresh deeper layers."
            if best_bounds_shared is not None:
                for L in best_bounds_shared:
                    lo_t, hi_t = best_bounds_shared[L]
                    lo_a = lo_t.detach().cpu().numpy().astype(np.float64)
                    hi_a = hi_t.detach().cpu().numpy().astype(np.float64)
                    lo_g, hi_g = bounds_by_relu[L]
                    bounds_by_relu[L] = (
                        np.maximum(lo_g, lo_a),
                        np.minimum(hi_g, hi_a))

    # Per-disjunct min(lb_alpha) over still-open queries — used for both
    # disjunct ordering (most-promising first) and hopeless skip.
    disjunct_min_lb = {}
    if best_lbs_batch is not None:
        for di in list(still_open):
            lbs = []
            for qi, _, _ in disj_queries[di]:
                if spec_lbs.get(qi, -1.0) > 0:
                    continue
                if qi in global_qi_to_batched_idx:
                    lbs.append(float(
                        best_lbs_batch[global_qi_to_batched_idx[qi]]))
            if lbs:
                disjunct_min_lb[di] = min(lbs)

    # Hopeless skip: disjuncts whose worst pass-0 lb_alpha is too negative
    # cannot close in Phase 2.5 (closure requires ALL queries to close).
    skip_hopeless = set()
    if hopeless_disjunct_lb is not None:
        for di, mlb in disjunct_min_lb.items():
            if mlb < hopeless_disjunct_lb:
                skip_hopeless.add(di)
        info['disjuncts_skipped_hopeless'] = len(skip_hopeless)

    # Most-promising-first ordering across disjuncts.
    if promising_first and disjunct_min_lb:
        disjunct_order = sorted(
            list(still_open),
            key=lambda d: -disjunct_min_lb.get(d, -float('inf')))
    else:
        disjunct_order = list(still_open)

    for di in disjunct_order:
        # Time-budget check at the disjunct boundary.
        if (time_budget is not None
                and (time.perf_counter() - _t_phase25_start) >= time_budget):
            info['time_budget_hit'] = True
            info['disjuncts_skipped_time'] += 1
            continue
        if di in skip_hopeless:
            continue
        q_list_full = [(qi, w, b) for qi, w, b in disj_queries[di]
                        if spec_lbs.get(qi, -1.0) <= 0]
        if not q_list_full:
            continue
        # Within-disjunct order: bottleneck (lowest lb_alpha) first so a
        # hopeless query is detected before time is spent on the rest.
        if promising_first and best_lbs_batch is not None:
            def _qlb(t):
                qi = t[0]
                if qi in global_qi_to_batched_idx:
                    return float(best_lbs_batch[global_qi_to_batched_idx[qi]])
                return -float('inf')
            q_list = sorted(q_list_full, key=_qlb)
        else:
            q_list = q_list_full
        q_info = {}
        any_open_after = False

        for qi_idx, (qi, w_np, b_q) in enumerate(q_list):
            # Per-query time-budget check at the inner loop boundary.
            if (time_budget is not None
                    and (time.perf_counter() - _t_phase25_start)
                    >= time_budget):
                info['time_budget_hit'] = True
                info['queries_skipped_time'] += 1
                any_open_after = True
                continue
            _t_qstart = time.perf_counter()
            w_np = np.asarray(w_np, dtype=np.float64)
            b_q = float(b_q)
            w_t = torch.as_tensor(w_np, dtype=dtype, device=device)
            override = {}
            lb_prev = float(spec_lbs.get(qi, -1.0))
            pass_records = []
            closed_here = False

            # --- Optional: α-CROWN optimization once per query. ---
            alpha_per_layer = None
            bbr_alpha = None
            lb_alpha = None
            q_info_ew = None  # α-CROWN's ew per ReLU (populated below)
            _use_alpha_here = use_alpha and (
                intermediate_start_nodes
                or alpha_impl == 'v2_fixed_intermediate')
            if _use_alpha_here:
                if (best_lbs_batch is not None
                        and qi in global_qi_to_batched_idx):
                    # Reuse batched pass-0 result — no extra run_alpha_crown.
                    bidx = global_qi_to_batched_idx[qi]
                    lb_alpha = float(best_lbs_batch[bidx])
                    alpha_params = alpha_params_shared
                    best_bounds = best_bounds_shared
                elif alpha_impl == 'v2_fixed_intermediate':
                    lb_alpha, alpha_params, best_bounds, _hist = (
                        ac.run_alpha_crown_fixed_intermediate(
                            gg, xl_g, xh_g, bounds_by_relu, w_np, b_q,
                            device, dtype, n_iters=alpha_iters, lr=alpha_lr,
                            lr_decay=alpha_lr_decay,
                            early_stop_on_positive=early_stop_on_positive))
                else:
                    # Route through the batched α-CROWN with n_q=1 so the
                    # dir_mode/s_split_n OOM fallback ladder applies (the
                    # legacy single-query `run_alpha_crown` has no such
                    # ladder and OOMs on cifar100_resnet_large prop_7866).
                    _dm = str(settings.alpha_crown_dir_mode
                               if 'alpha_crown_dir_mode' in settings else 'split')
                    _ss = int(settings.alpha_crown_s_split_n
                               if 'alpha_crown_s_split_n' in settings else 2)
                    _w_qs = np.asarray(w_np, dtype=np.float64).reshape(1, -1)
                    _b_qs = np.asarray([float(b_q)], dtype=np.float64)
                    _lbs, alpha_params, best_bounds, _hist = (
                        ac.run_alpha_crown_batched(
                            gg, xl_g, xh_g, bounds_by_relu, _w_qs, _b_qs,
                            intermediate_start_nodes, unstable_indices,
                            device, dtype, n_iters=alpha_iters, lr=alpha_lr,
                            lr_decay=alpha_lr_decay,
                            early_stop_on_positive=early_stop_on_positive,
                            dir_mode=_dm, s_split_n=_ss))
                    lb_alpha = float(_lbs[0])
                # If α-CROWN already verifies (common for easy cases that
                # α,β-CROWN closes via "verified with init bound!"), record
                # and skip Phase 2.5 entirely for this query. Skipping the
                # ew-capture + direction-adaptive α construction below saves
                # ~1 extra CROWN backward per already-closed query.
                if (cascade_skip_on_close and lb_alpha is not None
                        and lb_alpha > 0):
                    spec_lbs[qi] = lb_alpha
                    q_info[qi] = {'passes': [{
                        'it': 0, 'lb_alpha_crown': lb_alpha,
                        'closed_by': 'alpha_crown'}],
                        'closed': True,
                        't_total': time.perf_counter() - _t_qstart}
                    continue
                bbr_alpha = {
                    L: (lo_t.detach().cpu().numpy().astype(np.float64),
                         hi_t.detach().cpu().numpy().astype(np.float64))
                    for L, (lo_t, hi_t) in best_bounds.items()
                }
                alpha_spec = {
                    L: alpha_params['spec'][L].detach()
                    for L in alpha_params['spec']
                }
                # Direction-adaptive reconstruction via one CROWN backward pass.
                _, ew_at_relu = ac.capture_ew_per_relu(
                    gg, xl_g, xh_g, alpha_spec, bbr_alpha, w_np, b_q,
                    device, dtype)
                alpha_per_layer = ac.build_dir_adaptive_alpha(
                    alpha_spec, ew_at_relu, bbr_alpha, device, dtype)
                # Stash α-CROWN ew_at_relu per-query so Phase 8 can score
                # binarization candidates without a redundant CROWN backward.
                q_info_ew = {
                    L: ew.detach().cpu().numpy().astype(np.float64)
                    for L, ew in ew_at_relu.items()}

            no_change_passes = 0
            for it in range(max_passes):
                _t_pass_start = time.perf_counter()
                # Cascade: re-run α-CROWN on current tightened bounds (from
                # prior pass's override). Gives fresh, tighter α slopes that
                # make the next forward-zono and halfspace-LP tighter.
                if (cascade_per_iter_alpha and use_alpha and it > 0
                        and (intermediate_start_nodes
                             or alpha_impl == 'v2_fixed_intermediate')
                        and override):
                    # Merge override into current bbr_alpha for α-CROWN init.
                    bbr_for_ac = {
                        L: (np.maximum(bbr_alpha[L][0], override[L][0]),
                            np.minimum(bbr_alpha[L][1], override[L][1]))
                        if L in override else bbr_alpha[L]
                        for L in bbr_alpha}
                    un_idx_cur = {
                        L: np.where(
                            (np.asarray(bbr_for_ac[L][0]) < 0) &
                            (np.asarray(bbr_for_ac[L][1]) > 0))[0].tolist()
                        for L in bbr_for_ac}
                    isn_cur = [L for L, un in un_idx_cur.items()
                               if len(un) > 0 and L > 0]
                    _run_cascade = isn_cur or alpha_impl == 'v2_fixed_intermediate'
                    if _run_cascade:
                        if alpha_impl == 'v2_fixed_intermediate':
                            lb_ac_it, alpha_params_it, best_bounds_it, _ = (
                                ac.run_alpha_crown_fixed_intermediate(
                                    gg, xl_g, xh_g, bbr_for_ac, w_np, b_q,
                                    device, dtype, n_iters=alpha_iters,
                                    lr=alpha_lr, lr_decay=alpha_lr_decay,
                                    early_stop_on_positive=early_stop_on_positive))
                        else:
                            # Route through batched α-CROWN with n_q=1 for the
                            # OOM-fallback ladder (see comment in Phase 2.5
                            # initial α-CROWN call).
                            _dm_c = str(settings.alpha_crown_dir_mode
                                         if 'alpha_crown_dir_mode' in settings
                                         else 'split')
                            _ss_c = int(settings.alpha_crown_s_split_n
                                         if 'alpha_crown_s_split_n' in settings
                                         else 2)
                            _w_qs_c = np.asarray(w_np, dtype=np.float64).reshape(1, -1)
                            _b_qs_c = np.asarray([float(b_q)], dtype=np.float64)
                            _lbs_c, alpha_params_it, best_bounds_it, _ = (
                                ac.run_alpha_crown_batched(
                                    gg, xl_g, xh_g, bbr_for_ac, _w_qs_c, _b_qs_c,
                                    isn_cur, un_idx_cur,
                                    device, dtype,
                                    n_iters=alpha_iters, lr=alpha_lr,
                                    lr_decay=alpha_lr_decay,
                                    early_stop_on_positive=early_stop_on_positive,
                                    dir_mode=_dm_c, s_split_n=_ss_c))
                            lb_ac_it = float(_lbs_c[0])
                        bbr_alpha = {
                            L: (lo_t.detach().cpu().numpy().astype(np.float64),
                                hi_t.detach().cpu().numpy().astype(np.float64))
                            for L, (lo_t, hi_t) in best_bounds_it.items()}
                        alpha_spec = {
                            L: alpha_params_it['spec'][L].detach()
                            for L in alpha_params_it['spec']}
                        _, ew_at_relu_it = ac.capture_ew_per_relu(
                            gg, xl_g, xh_g, alpha_spec, bbr_alpha,
                            w_np, b_q, device, dtype)
                        alpha_per_layer = ac.build_dir_adaptive_alpha(
                            alpha_spec, ew_at_relu_it, bbr_alpha,
                            device, dtype)
                        # Refresh stashed α-CROWN ew from the cascade pass —
                        # later passes use tighter α's so ew is more accurate.
                        q_info_ew = {
                            L: ew.detach().cpu().numpy().astype(np.float64)
                            for L, ew in ew_at_relu_it.items()}
                        lb_alpha = lb_ac_it
                        if lb_alpha > 0:
                            spec_lbs[qi] = lb_alpha
                            closed_here = True
                            pass_records.append({
                                'it': it + 1, 'lb_alpha_crown': lb_alpha,
                                'closed_by': 'alpha_crown_cascade'})
                            break
                # Forward zono: α-CROWN-direction-adaptive if enabled, else
                # min-area. Override_tight from prior Phase 2.5 passes.
                # Pre-compute per-layer unstable indices for this query so
                # pre_relu_gpu can store only those rows of G — avoids the
                # (n_flat × K × 4 bytes × n_layers) full-G materialisation
                # that OOMs on resnet_large (see _forward_keep_pre_gpu).
                _bbr_for_unstable = (bbr_alpha if alpha_per_layer is not None
                                     else (override or bounds_by_relu))
                unstable_per_layer_q = {}
                for L, (lo_np, hi_np) in _bbr_for_unstable.items():
                    un = np.where((np.asarray(lo_np) < 0)
                                  & (np.asarray(hi_np) > 0))[0]
                    unstable_per_layer_q[L] = torch.as_tensor(
                        un, dtype=torch.long, device=device)
                # Effective bbr for the forward (α-tightened ∧ override).
                _bbr_for_fwd = bbr_alpha if not override else {
                    L: (np.maximum(bbr_alpha[L][0], override[L][0]),
                        np.minimum(bbr_alpha[L][1], override[L][1]))
                    if L in override else bbr_alpha[L]
                    for L in bbr_alpha}
                if alpha_per_layer is not None:
                    z_final, pre_relu_gpu = ac.forward_zono_dir_adaptive(
                        xl_g, xh_g, gg, alpha_per_layer, _bbr_for_fwd,
                        device, dtype, settings=settings,
                        unstable_per_layer=unstable_per_layer_q)
                else:
                    z_final, pre_relu_gpu = _forward_keep_pre_gpu(
                        xl_g, xh_g, gg, device, dtype,
                        override_tight=(override if override else None),
                        settings=settings,
                        unstable_per_layer=unstable_per_layer_q)
                c_out = z_final.center.detach().cpu().numpy().astype(np.float64)
                G_out = z_final.generators.detach().cpu().numpy().astype(np.float64)
                lb_zono = float(
                    w_np @ c_out + b_q - np.sum(np.abs(G_out.T @ w_np)))

                if layers_cfg is None:
                    layers = [L for L in range(nh)
                               if L in bounds_by_relu
                               and ((np.asarray(bounds_by_relu[L][0]) < 0) &
                                    (np.asarray(bounds_by_relu[L][1]) > 0)).any()]
                else:
                    layers = list(layers_cfg)

                # Base bbr for tightening: α-tightened if α-CROWN ran, else
                # the global bbr. Combine with override.
                base_bbr = bbr_alpha if bbr_alpha is not None else bounds_by_relu
                working_bbr = {}
                for L in base_bbr:
                    lo0 = np.asarray(base_bbr[L][0], dtype=np.float64).copy()
                    hi0 = np.asarray(base_bbr[L][1], dtype=np.float64).copy()
                    if L in override:
                        lo0 = np.maximum(lo0, override[L][0])
                        hi0 = np.minimum(hi0, override[L][1])
                    working_bbr[L] = (lo0, hi0)

                result, stats = box_halfspace.tighten_all_layers(
                    pre_relu_gpu, c_out, G_out, w_np, b_q, working_bbr,
                    layers, device, dtype)
                for L, (lo_n, hi_n) in result.items():
                    if L in override:
                        ol, oh = override[L]
                        override[L] = (np.maximum(ol, lo_n),
                                        np.minimum(oh, hi_n))
                    else:
                        override[L] = (lo_n, hi_n)
                q_bbr = {
                    L: (np.maximum(base_bbr[L][0], override[L][0]),
                         np.minimum(base_bbr[L][1], override[L][1]))
                    if L in override else base_bbr[L]
                    for L in base_bbr}
                lb_crown = float(_adaptive_spec_lb(
                    gg, xl_g, xh_g, q_bbr, w_t, b_q, device, dtype))
                _dt_pass = time.perf_counter() - _t_pass_start
                pass_records.append({
                    'it': it + 1, 'lb_zono': lb_zono, 'lb_crown': lb_crown,
                    'n_flipped': stats['n_flipped'],
                    'Δwidth': stats['total_shrink'],
                    'lb_alpha_crown': lb_alpha,
                    't_pass': _dt_pass,
                })
                if print_progress:
                    _lbac_str = (f'{float(lb_alpha):+.4f}'
                                  if lb_alpha is not None else '  n/a ')
                    print(f'  q{qi} pass{it}: '
                          f'lb_crown={lb_crown:+.4f}  α-LB={_lbac_str}  '
                          f'flipped={stats["n_flipped"]:3d}  '
                          f'{_dt_pass*1000:.0f}ms')
                if lb_crown > 0:
                    spec_lbs[qi] = lb_crown
                    closed_here = True
                    break
                # Plateau: if tightening produced no bound change this pass,
                # count it; after `plateau_patience` consecutive no-change
                # passes, give up and fall through to MILP.
                if stats.get('total_shrink', 0.0) < 1e-9:
                    no_change_passes += 1
                    if no_change_passes >= plateau_patience:
                        spec_lbs[qi] = max(
                            spec_lbs.get(qi, -1e30), lb_crown)
                        break
                else:
                    no_change_passes = 0
                if abs(lb_crown - lb_prev) < tol:
                    spec_lbs[qi] = max(spec_lbs.get(qi, -1e30), lb_crown)
                    break
                lb_prev = lb_crown
                # Free GPU copies before next pass's forward — but keep them
                # on the LAST iteration so the BaB-split block below can
                # reuse the spec-adaptive zonotope state.
                if it < max_passes - 1:
                    del z_final, pre_relu_gpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            # Compute combined bounds (α-CROWN ∧ halfspace-LP override).
            tightened = None
            if bbr_alpha is not None:
                tightened = {}
                for L, (lo_a, hi_a) in bbr_alpha.items():
                    lo_f = np.asarray(lo_a, dtype=np.float64).copy()
                    hi_f = np.asarray(hi_a, dtype=np.float64).copy()
                    if L in override:
                        lo_f = np.maximum(lo_f, override[L][0])
                        hi_f = np.minimum(hi_f, override[L][1])
                    tightened[L] = (lo_f, hi_f)

            q_info[qi] = {'passes': pass_records, 'closed': closed_here}
            _dt_q = time.perf_counter() - _t_qstart
            q_info[qi]['t_total'] = _dt_q
            # Stash α-CROWN ew per-query for Phase 8 scoring (only if it
            # wasn't closed here — closed queries don't need MILP).
            if not closed_here and q_info_ew is not None:
                q_info[qi]['ew_at_relu'] = q_info_ew
            # Stash the (possibly BaB-tightened) combined bounds for Phase 8.
            if not closed_here and tightened is not None:
                q_info[qi]['tightened_bounds'] = tightened
            if print_progress:
                _final_lb = spec_lbs.get(qi, lb_prev)
                _n_passes = len(pass_records)
                _status = 'closed' if closed_here else 'open'
                print(f'  q{qi}: {_status} after {_n_passes} pass(es)  '
                      f'LB={_final_lb:+.4f}  {_dt_q*1000:.0f}ms')
            if not closed_here:
                any_open_after = True
            # End of this query: explicitly drop the GPU tensors from the
            # LAST pass. The `break` path (closed / plateau / tol) leaves
            # `z_final` and `pre_relu_gpu` alive in the function's locals.
            # Without this, q50's forward state leaks into q11's allocation
            # and OOMs on resnet-sized nets.
            try: del z_final
            except NameError: pass
            try: del pre_relu_gpu
            except NameError: pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        info['per_query'][di] = q_info
        if not any_open_after:
            verified_here.add(di)
            info['n_closed'] += 1

    still_open_after = still_open - verified_here
    if print_progress:
        print(f'Phase 2.5 (zono-lift): closed {info["n_closed"]} disjuncts  '
              f'still_open={len(still_open_after)}')
    return still_open_after, info


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _compute_n_splits(gg, bounds_by_relu):
    """Count unstable ReLU neurons per layer, keyed by ONNX op name."""
    n_splits = {}
    relu_names = gg['relu_names']
    for li, name in enumerate(relu_names):
        if li not in bounds_by_relu:
            n_splits[name] = 0
            continue
        lo, hi = bounds_by_relu[li]
        lo_np = lo if isinstance(lo, np.ndarray) else lo.cpu().numpy()
        hi_np = hi if isinstance(hi, np.ndarray) else hi.cpu().numpy()
        n_splits[name] = int(((lo_np < 0) & (hi_np > 0)).sum())
    return n_splits


def _compute_avg_layer_width(gg, bounds_by_relu):
    """Mean (hi - lo) over unstable neurons per layer, keyed by ONNX op name."""
    out = {}
    relu_names = gg['relu_names']
    for li, name in enumerate(relu_names):
        if li not in bounds_by_relu:
            out[name] = 0.0
            continue
        lo, hi = bounds_by_relu[li]
        lo_np = lo if isinstance(lo, np.ndarray) else lo.cpu().numpy()
        hi_np = hi if isinstance(hi, np.ndarray) else hi.cpu().numpy()
        mask = (lo_np < 0) & (hi_np > 0)
        out[name] = float((hi_np - lo_np)[mask].mean()) if mask.any() else 0.0
    return out


def _validate_sat_witness(onnx_path, spec, witness, atol=1e-4):
    """Run a SAT witness through ONNXRuntime + check it actually violates
    the spec. Catches spurious counterexamples from PGD/MILP bugs OR from
    graph-builder bugs (vibecheck's internal forward might compute a
    different value than the original ONNX). Mirrors VNNCOMP scoring's
    counterexample-validation step (COUNTEREXAMPLE_ATOL=1e-4).

    Returns (ok, info_dict). `ok=True` iff witness is in the input box
    (within atol) AND its ORT output satisfies the unsafe condition
    (i.e., `spec.check(out, out)` returns 'unknown', within atol on
    constraint margins).
    """
    info = {'ok': False, 'reason': None}
    if onnx_path is None:
        info['reason'] = 'no onnx_path stashed on graph; skipping validation'
        return True, info
    w = np.asarray(witness).flatten().astype(np.float64)
    if w.shape != spec.x_lo.shape:
        info['reason'] = (f'witness shape {w.shape} != x_lo shape '
                           f'{spec.x_lo.shape}')
        return False, info
    if (np.any(w < spec.x_lo - atol)
            or np.any(w > spec.x_hi + atol)):
        info['reason'] = (f'witness outside input box (atol={atol})')
        info['out_of_box'] = (
            float((spec.x_lo - w).max()), float((w - spec.x_hi).max()))
        return False, info
    try:
        import onnxruntime as ort
    except ImportError:
        info['reason'] = 'onnxruntime not installed; skipping validation'
        return True, info
    # ORT can't read .gz natively. Decompress to a bytes buffer first so
    # the validator works on the VNNCOMP-shipped `.onnx.gz` files. A
    # prior version called `ort.InferenceSession(path)` directly and
    # rejected every PGD witness on cersyve as "spurious" because the
    # session-load itself raised. Validation failures must come from
    # the witness being wrong, not from file format.
    try:
        if onnx_path.endswith('.gz'):
            import gzip
            with gzip.open(onnx_path, 'rb') as _f:
                _model_bytes = _f.read()
            sess = ort.InferenceSession(
                _model_bytes, providers=['CPUExecutionProvider'])
        else:
            sess = ort.InferenceSession(
                onnx_path, providers=['CPUExecutionProvider'])
        in_meta = sess.get_inputs()[0]
        in_shape = [d if isinstance(d, int) and d > 0 else 1
                    for d in in_meta.shape]
        x = w.reshape(in_shape).astype(np.float32)
        out = sess.run(None, {in_meta.name: x})[0]
        out_flat = np.asarray(out).flatten().astype(np.float64)
    except Exception as e:
        info['reason'] = f'ORT forward failed: {type(e).__name__}: {e}'
        return False, info
    info['out'] = out_flat
    # spec.check returns 'unknown' iff worst margin <= 0. Apply atol slack
    # by shifting output: pass (out-atol, out+atol) so margins computed
    # against a generous output band — a witness near the boundary counts
    # as a valid SAT if any output in the +/-atol envelope violates.
    check_res, check_info = spec.check(out_flat - atol, out_flat + atol)
    info['spec_check'] = check_res
    info['worst_margin'] = check_info.get('worst_margin')
    if check_res == 'unknown':
        info['ok'] = True
        return True, info
    info['reason'] = (f'ORT output does not violate spec '
                       f'(worst_margin={info["worst_margin"]:.4g}, '
                       f'atol={atol})')
    return False, info


def _run_pipeline(graph, spec, settings, build_fn, impl):
    """Run the 9-phase graph verification pipeline.

    `build_fn` is used for in-process LP builds (phase 4 tightening);
    `impl` is the string name ('reference' or 'optimized') used to
    dispatch the subprocess worker (phases 7/8).

    Returns (result_str, details_dict) per the contract in the module plan.
    """
    device, dtype = resolve_torch(settings)
    torch.set_num_threads(1)
    total_timeout = float(settings.total_timeout)
    print_progress = bool(settings.print_progress)
    verbose = print_progress

    t_start = time.perf_counter()
    deadline = t_start + total_timeout
    # Optional guarantee: Phase 8 gets at least this fraction of total_timeout.
    # Pre-Phase-8 phases see a wrapped time_left that returns
    # min(remaining, phase8_min_deadline - now). When `_phase8_started` flips
    # True (right before Phase 8 BaB dispatch), time_left returns the full
    # remaining budget.
    _phase8_min_frac = float(getattr(settings, 'phase8_min_budget_frac', 0.0))
    _phase8_min_deadline = (t_start + total_timeout * (1.0 - _phase8_min_frac)
                             if _phase8_min_frac > 0 else deadline)
    _phase8_started = [False]

    def time_left():
        now = time.perf_counter()
        full = max(0.0, deadline - now)
        if not _phase8_started[0] and _phase8_min_frac > 0:
            return max(0.0, min(full, _phase8_min_deadline - now))
        return full

    stats = VerifyStats()
    timing = {}
    details = {'timing': timing}
    if verbose:
        per_layer_timing = {}
        avg_layer_width = {}
        details['per_layer_timing'] = per_layer_timing
        details['avg_layer_width'] = avg_layer_width
        details['build_time_total'] = 0.0
        details['gpu_mem'] = []

    def _mem_audit(tag):
        """Record (tag, alloc_gb, reserved_gb, peak_alloc_since_reset_gb).
        Called at phase boundaries when verbose; also resets the peak
        tracker so the NEXT call's peak covers only the in-between region.
        """
        if not (verbose and torch.cuda.is_available()):
            return
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        resv_gb = torch.cuda.memory_reserved() / 1e9
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        rec = {'tag': tag, 'alloc_gb': alloc_gb,
                'reserved_gb': resv_gb, 'peak_alloc_gb': peak_alloc_gb}
        details['gpu_mem'].append(rec)
        if print_progress:
            print(f'  [GPU {tag}: alloc {alloc_gb:.2f} GB, '
                  f'peak {peak_alloc_gb:.2f} GB, '
                  f'reserved {resv_gb:.2f} GB]')
        torch.cuda.reset_peak_memory_stats()

    def _finalize(result_str, phase, **extra):
        # Defense-in-depth: every SAT verdict must pass ONNXRuntime
        # validation against the spec (within COUNTEREXAMPLE_ATOL).
        # Catches spurious witnesses from PGD/MILP bugs AND graph-
        # builder bugs that would otherwise leak through silently.
        # Spurious witnesses get downgraded to 'unknown' with
        # `details['spurious_witness']` set and logged.
        if (result_str == 'sat' and extra.get('witness') is not None
                and not bool(getattr(
                    settings, 'skip_sat_validation', False))):
            _atol = float(settings.sat_validate_atol
                           if 'sat_validate_atol' in settings else 1e-4)
            _onnx_path = getattr(graph, 'onnx_path', None)
            _ok, _info = _validate_sat_witness(
                _onnx_path, spec, extra['witness'], atol=_atol)
            if not _ok:
                if print_progress:
                    print(f'  [validate] SPURIOUS SAT from {phase}: '
                          f'{_info.get("reason")}', flush=True)
                # Stash for diagnostics + downgrade verdict.
                extra = dict(extra)
                extra['spurious_witness'] = _info
                extra['original_phase'] = phase
                result_str = 'unknown'
                phase = f'spurious_sat_{phase}'
        details['result'] = result_str
        details['phase'] = phase
        details['time'] = time.perf_counter() - t_start
        details['n_splits'] = _compute_n_splits(gg, bounds_by_relu)
        # Surface the per-query spec lower bounds for diagnostics —
        # callers often want to see how close we got to verification
        # even on 'unknown' verdicts (closer-to-0 = better strategy).
        try:
            details['spec_lbs'] = dict(spec_lbs)
        except Exception:
            pass
        # Surface intermediate pre-ReLU bounds for diagnostic comparisons
        # against reference verifiers (e.g. α,β-CROWN's per-layer widths).
        try:
            bbr_out = {}
            for L, (lo, hi) in bounds_by_relu.items():
                lo_np = lo.cpu().numpy() if hasattr(lo, 'cpu') else np.asarray(lo)
                hi_np = hi.cpu().numpy() if hasattr(hi, 'cpu') else np.asarray(hi)
                bbr_out[L] = (lo_np, hi_np)
            details['bounds_by_relu'] = bbr_out
        except Exception:
            pass
        if verbose:
            details['avg_layer_width'] = _compute_avg_layer_width(
                gg, bounds_by_relu)
        for k, v in extra.items():
            details[k] = v
        details['neuron_stats'] = stats.neuron_stats
        return result_str, details

    gg = graph.gpu_graph(device, dtype)
    nh = gg['n_relu']
    relu_names = gg['relu_names']

    # Output size
    n_output = None
    for op in reversed(gg['ops']):
        if op['type'] == 'fc':
            n_output = op['W'].shape[0]
            break
        if op['type'] == 'conv':
            n_output = op['n_out']
            break
    assert n_output is not None

    queries = spec.as_linear_queries(n_output)
    disj_queries = {}
    for qi, (di, w, bias) in enumerate(queries):
        disj_queries.setdefault(di, []).append((qi, w, bias))

    spec_ew = {}
    for qi, (di, w, bias) in enumerate(queries):
        spec_ew[qi] = (
            torch.tensor(w, dtype=dtype, device=device), float(bias))

    xl_g = torch.tensor(spec.x_lo.astype(np.float64), dtype=dtype, device=device)
    xh_g = torch.tensor(spec.x_hi.astype(np.float64), dtype=dtype, device=device)

    # PGD only needs approximate gradients — run it in float32 on a
    # dedicated float32 gpu_graph so the attack stays cheap even when the
    # main verification is float64. When `bits=32`, `gg` is already fp32,
    # so reuse it (saves ~10ms + avoids duplicating weight tensors on GPU).
    if dtype == torch.float32:
        gg_pgd = gg
    else:
        gg_pgd = graph.gpu_graph(device, torch.float32)
    xl_pgd = xl_g.to(torch.float32)
    xh_pgd = xh_g.to(torch.float32)

    # Pre-seed bounds_by_relu as empty so _finalize never fails
    bounds_by_relu = {}

    # Prepare serialized ops. `W_sp` (dense-unrolled sparse conv weight
    # matrix) is built LAZILY per layer on first access — every consumer
    # already has the `op.get('W_sp'); if None: build+cache` pattern.
    # Eagerly building all 20 matrices up front on resnet_large cost
    # ~5.5 s of wall time and was wasted on layers that end up
    # `adapt-only` / `zono-only` (no LP probe, no MILP tighten).
    gg_ops_ser = _serialize_gg_ops(gg)
    x_lo_64 = spec.x_lo.astype(np.float64)
    x_hi_64 = spec.x_hi.astype(np.float64)
    n_cores = multiprocessing.cpu_count()
    sample_timeout = float(getattr(settings, 'milp_sample_timeout', 5.0))

    def _verbose_cb(li, name, rec):
        if verbose:
            details['per_layer_timing'][name] = {
                'layer_total': rec['adapt'] + rec['build'] + rec['probe']
                    + rec['solve'],
                'adapt': rec['adapt'],
                'build': rec['build'],
                'probe': rec['probe'],
                'solve': rec['solve'],
                'width_zono': rec.get('width_zono', 0.0),
                'width_adapt': rec.get('width_adapt', 0.0),
                'width_lp': rec.get('width_lp', 0.0),
                'width_final': rec.get('width_final', 0.0),
            }
            details['build_time_total'] += rec['build']

    # --- Phase 0: PGD attack BEFORE Phase 1 cascade. Mirrors α,β-CROWN's
    # pgd_order='before' default: try to find a counter-example with attack
    # first; if SAT, return immediately and skip the entire Phase 1 cascade
    # (which would otherwise burn 30-60s of MIP work that's wasted on a
    # SAT case). On the 5 mnist 256x4 SAT regressions (prop_4/0/1/7/2 at
    # eps 0.05) the cascade ran 30s+ before PGD got 5s; with Phase 0 PGD
    # at 10-15s budget + adam_clipping the SAT cases close in <15s.
    _disable_sat = bool(getattr(settings, 'disable_sat_finding', False))
    if (getattr(settings, 'pgd_phase0_enabled', True)
            and not _disable_sat):
        t0 = time.perf_counter()
        _pgd_budget_phase0 = float(
            getattr(settings, 'pgd_time_budget_phase0', 10.0))
        # Phase 0 can use a lighter PGD config (fewer restarts/iter) than
        # Phase 2.6 (per-spec). Swap settings in place around the call so
        # subsequent _pgd_attack_general invocations (Phase 2.6, Phase 3.5)
        # see the default pgd_iter/pgd_restarts.
        _p0_iter = (int(settings.pgd_phase0_iter)
                     if 'pgd_phase0_iter' in settings else None)
        _p0_restarts = (int(settings.pgd_phase0_restarts)
                         if 'pgd_phase0_restarts' in settings else None)
        _orig_iter = settings.pgd_iter
        _orig_restarts = settings.pgd_restarts
        if _p0_iter is not None:
            settings.pgd_iter = _p0_iter
        if _p0_restarts is not None:
            settings.pgd_restarts = _p0_restarts
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg_pgd, settings,
                time_budget=_pgd_budget_phase0)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        finally:
            settings.pgd_iter = _orig_iter
            settings.pgd_restarts = _orig_restarts
        timing['phase0_pgd'] = time.perf_counter() - t0
        if print_progress:
            print(f'Phase 0 (PGD before cascade): '
                  f'{timing["phase0_pgd"]:.2f}s  sat={pgd_sat}', flush=True)
        if pgd_sat:
            return _finalize('sat', 'pgd', witness=pgd_witness)

    # --- Parallel PGD context (spawned inside Phase 1 after Phase 0.5). ---
    # Pass the float32 GPU graph + bounds; _phase1_bab_refine spawns the
    # worker thread AFTER Phase 0.5 closes the easy specs so the thread
    # only attacks actually-open disjuncts. `milp_active` is a
    # threading.Event toggled by main inside the cascade's MILP loop.
    milp_active = None
    parallel_pgd_ctx = None
    _parallel_pgd = (
        bool(settings.parallel_pgd_enabled)
        if 'parallel_pgd_enabled' in settings else False)
    if (_parallel_pgd and not _disable_sat
            and disj_queries and time_left() > 5):
        import threading as _threading
        milp_active = _threading.Event()
        parallel_pgd_ctx = dict(
            xl_pgd=xl_pgd, xh_pgd=xh_pgd, gg_pgd=gg_pgd, spec=spec)

    # --- Phase 1: interleaved zonotope forward + per-layer tightening ---
    # At every ReLU we pause, run the per-neuron adaptive backward and
    # an optional LP probe, then apply the ReLU using the tightened
    # pre-activation bounds. This lets subsequent layers' zonotope
    # propagation benefit from the tightening (smaller new generators),
    # so deeper layers start with tighter bounds for free.
    #
    # When tighten_formulation='gen_cone', we pass an empty rec_zono
    # dict so the forward populates per-ReLU gen-LP-style rows from
    # the zonotope's own G matrix. The gen_cone dispatch inside the
    # forward then consumes these rows via its `precomputed=` path,
    # avoiding the separate precompute_gen_state conv pass.
    _tf = str(getattr(settings, 'tighten_formulation', 'weight_walk'))
    rec_zono = {} if _tf == 'gen_cone' else None

    t0 = time.perf_counter()
    _phase1_method = str(getattr(settings, 'phase1_method', 'legacy'))
    # Pre-cascade per-spec PGD hook (run right after Phase 0.5 spec α-CROWN,
    # before the expensive MILP cascade). Catches SAT cheaply on cases the
    # cascade-then-Phase-8-witness path would only catch after a full
    # cascade burn. Sorted by spec_lb ascending (most-likely-SAT first).
    def _pre_cascade_pgd_hook(spec_lbs_dict, _bbr, _open_qis):
        # DotMap returns empty DotMap (not the default!) for missing keys,
        # so use `in` check explicitly.
        _enabled = (settings.phase26_pre_cascade_enabled
                     if 'phase26_pre_cascade_enabled' in settings else True)
        if _disable_sat or not bool(_enabled):
            return None
        t_hook = time.perf_counter()
        # Build sorted disjunct list by spec_lb (lowest first).
        disj_lb = {}
        for di, qlist in disj_queries.items():
            lbs = [spec_lbs_dict.get(qi, 0.0) for qi, _, _ in qlist]
            if lbs:
                disj_lb[di] = min(lbs)
        # Restrict to OPEN disjuncts (spec_lb ≤ 0).
        ordered = sorted([d for d in disj_lb if disj_lb[d] <= 0],
                          key=lambda d: disj_lb[d])
        # DotMap returns empty DotMap (not the default!) for missing keys.
        _per_min = (float(settings.phase26_pgd_per_spec_min_per_spec)
                     if 'phase26_pgd_per_spec_min_per_spec' in settings
                     else 0.5)
        # Uniform-split: total = max(n×per_min, time_left×frac). When few
        # open specs (e.g. case 22: 2 open), each gets a much bigger slice
        # without exceeding a small frac of the remaining wall budget.
        _frac = (float(settings.phase26_pre_cascade_total_frac)
                  if 'phase26_pre_cascade_total_frac' in settings
                  else 0.10)
        _n = max(len(ordered), 1)
        _total = max(_n * _per_min, time_left() * _frac)
        _per = _total / _n
        # Per-spec hard cap so a single spec never gobbles the budget.
        _per_cap = (float(settings.phase26_pre_cascade_per_spec_cap)
                     if 'phase26_pre_cascade_per_spec_cap' in settings
                     else 5.0)
        _per = min(_per, _per_cap)
        if print_progress:
            print(f'  [pre-cascade PGD] {len(ordered)} open disjuncts, '
                  f'{_per:.2f}s each (sorted by spec_lb asc; '
                  f'frac={_frac:.2f}, per_min={_per_min:.2f}, '
                  f'cap={_per_cap:.1f})', flush=True)
        n_attacked = 0
        for di in ordered:
            if time_left() <= 0.5: break
            _budget = min(_per, max(time_left() - 0.5, 0.1))
            try:
                _ok, _w = _pgd_attack_general(
                    xl_pgd, xh_pgd, spec, gg_pgd, settings,
                    restrict_disj={di}, time_budget=_budget)
            except RuntimeError:
                _ok, _w = False, None
            n_attacked += 1
            if _ok:
                if print_progress:
                    print(f'  [pre-cascade PGD] SAT on disjunct {di} '
                          f'(attacked {n_attacked}/{len(ordered)}; '
                          f'{time.perf_counter()-t_hook:.2f}s)', flush=True)
                return _w
        if print_progress:
            print(f'  [pre-cascade PGD] no SAT, '
                  f'attacked {n_attacked}/{len(ordered)} '
                  f'({time.perf_counter()-t_hook:.2f}s)', flush=True)
        return None

    try:
        if _phase1_method == 'bab_refine':
            # Pass rec_zono into bab_refine so its initial forward
            # populates the per-layer pre-ReLU rows. Phase 7 then takes
            # the state_from_phase1 fast path and avoids the dense
            # precompute_gen_state allocation (which OOMs the RTX 3080
            # on TinyImageNet's 14×14 stage).
            sb, bounds_by_relu, z_final_phase1, phase1_tb = \
                _phase1_bab_refine(
                    xl_g, xh_g, gg, gg_ops_ser, x_lo_64, x_hi_64,
                    sample_timeout, n_cores, time_left, device, dtype,
                    settings, spec=spec,
                    print_progress=print_progress, verbose_cb=_verbose_cb,
                    rec_zono=rec_zono,
                    pre_cascade_hook=_pre_cascade_pgd_hook,
                    gpu_lock=milp_active,
                    parallel_pgd_ctx=parallel_pgd_ctx)
        else:
            sb, bounds_by_relu, z_final_phase1, phase1_tb = \
                _forward_zonotope_interleaved(
                    xl_g, xh_g, gg, gg_ops_ser, x_lo_64, x_hi_64, build_fn,
                    sample_timeout, n_cores, time_left, device, dtype, settings,
                    print_progress=print_progress, verbose_cb=_verbose_cb,
                    rec_zono=rec_zono)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # CPU fallback is silently 10-100× slower — masks regressions
        # (a forward that used to fit suddenly doesn't) and hides memory
        # blow-ups from the user. Require BOTH `allow_cpu_fallback=True`
        # AND `raise_on_oom=False` (explicit opt-in, two knobs). Default
        # behaviour is to re-raise so the user sees the OOM.
        cpu_ok = (device.type != 'cpu'
                  and getattr(settings, 'allow_cpu_fallback', False)
                  and not getattr(settings, 'raise_on_oom', True))
        if cpu_ok:
            if print_progress:
                print(f'  GPU OOM/runtime ({e!s:.60}); falling back to CPU '
                      '(allow_cpu_fallback=True, raise_on_oom=False)')
            device = torch.device('cpu')
            gg = graph.gpu_graph(device, dtype)
            if dtype == torch.float32:
                gg_pgd = gg
            else:
                gg_pgd = graph.gpu_graph(device, torch.float32)
            xl_g = xl_g.cpu(); xh_g = xh_g.cpu()
            xl_pgd = xl_pgd.cpu(); xh_pgd = xh_pgd.cpu()
            spec_ew = {qi: (w.cpu(), b) for qi, (w, b) in spec_ew.items()}
            rec_zono = {} if _tf == 'gen_cone' else None
            sb, bounds_by_relu, z_final_phase1, phase1_tb = \
                _forward_zonotope_interleaved(
                    xl_g, xh_g, gg, gg_ops_ser, x_lo_64, x_hi_64, build_fn,
                    sample_timeout, n_cores, time_left, device, dtype,
                    settings, print_progress=print_progress,
                    verbose_cb=_verbose_cb, rec_zono=rec_zono)
        else:
            raise
    timing['phase1_zono_tighten'] = time.perf_counter() - t0
    timing['phase1_breakdown'] = phase1_tb
    _mem_audit('after phase1')

    # --- Check parallel PGD result (thread joined inside Phase 1) ---
    if isinstance(phase1_tb, dict):
        _pp_attacks = phase1_tb.get('parallel_pgd_attacks')
        if _pp_attacks is not None:
            timing['parallel_pgd_attacks'] = _pp_attacks
        _pp_witness = phase1_tb.get('parallel_pgd_sat')
        if _pp_witness is not None:
            return _finalize('sat', 'parallel_pgd', witness=_pp_witness)

    # Phase 1 done — drop the pre-Phase-8 budget cap so Phase 2/2.5/2.6/7/8
    # all see the full remaining time_left. The cap's purpose is to stop
    # the cascade from eating Phase 8's budget; once the cascade has
    # returned, everything downstream should use the real deadline.
    _phase8_started[0] = True

    # Pre-cascade hook may have caught SAT before the cascade ran.
    _pre_cascade_witness = (phase1_tb.get('pre_cascade_sat')
                              if isinstance(phase1_tb, dict) else None)
    if _pre_cascade_witness is not None:
        if print_progress:
            print(f'Phase 1 pre-cascade PGD found SAT — skipping cascade',
                  flush=True)
        return _finalize('sat', 'pgd_pre_cascade',
                          witness=_pre_cascade_witness)

    # --- Phase 2: CROWN backward ---
    # Skippable when Phase 1 already produced tighter α-CROWN spec lbs
    # (Phase 0.5 step 2). Setting `phase2_crown_enabled=False` reuses those
    # lbs instead of recomputing a (looser) plain-CROWN backward.
    t0 = time.perf_counter()
    all_qids = set(spec_ew.keys())
    _phase2_crown = bool(getattr(settings, 'phase2_crown_enabled', True))
    _ph1_spec_lbs = phase1_tb.get('spec_lbs_phase05') if isinstance(phase1_tb, dict) else None
    if _phase2_crown or not _ph1_spec_lbs:
        with torch.no_grad():
            spec_lbs, _ = _spec_backward_graph(
                sb, xl_g, xh_g, gg, spec_ew, all_qids, nh, device, dtype)
    else:
        # Reuse Phase 0.5 α-CROWN spec lbs (already tighter than plain CROWN).
        spec_lbs = {qi: _ph1_spec_lbs.get(qi, -1.0) for qi in all_qids}
    timing['phase2_crown'] = time.perf_counter() - t0
    stats.record_bounds(sb)

    verified_disj = {di for di, qlist in disj_queries.items()
                      if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist)}
    still_open_disj = set(disj_queries.keys()) - verified_disj

    if print_progress:
        worst = min(spec_lbs.values()) if spec_lbs else 0.0
        print(f'Phase 1+2 (interleaved zono-tighten + CROWN): '
              f'{timing["phase1_zono_tighten"] + timing["phase2_crown"]:.2f}s  '
              f'verified={len(verified_disj)}/{len(disj_queries)}  '
              f'worst={worst:.4f}')
    _fire_callback(settings, 'phase_done', {'phase': 'crown',
                    'elapsed': timing['phase2_crown']})

    if not still_open_disj:
        return _finalize('verified', 'crown')

    # --- Phase 3: initial PGD (α,β-CROWN pgd_order="before"). Disabled
    # by default (α,β-CROWN's cifar100 yaml uses `pgd_order="middle"`,
    # i.e. no pre-CROWN PGD). Phase 3.5 below handles attack post-CROWN
    # with the spec set already pruned. ---
    if (getattr(settings, 'pgd_before_enabled', False)
            and not _disable_sat):
        t0 = time.perf_counter()
        _pgd_budget_before = float(
            getattr(settings, 'pgd_time_budget_before', 5.0))
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg_pgd, settings,
                time_budget=_pgd_budget_before)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        timing['phase3_pgd'] = time.perf_counter() - t0
        if print_progress:
            print(f'Phase 3 (PGD before): '
                  f'{timing["phase3_pgd"]:.2f}s  sat={pgd_sat}')
        if pgd_sat:
            return _finalize('sat', 'pgd', witness=pgd_witness)

    # --- Phase 3.5: middle PGD — re-attack disjuncts that survived CROWN.
    # α,β-CROWN's `pgd_order="middle"` trick: after Phase 2's intermediate
    # bounds have pruned easy specs, concentrate all restarts on the hard
    # ones. Huge win on deep-ResNet nets with many OR-spec clauses.
    if (still_open_disj and getattr(settings, 'pgd_middle_enabled', True)
            and not _disable_sat):
        t0 = time.perf_counter()
        _pgd_budget_middle = float(
            getattr(settings, 'pgd_time_budget_middle', 5.0))
        # Count surviving spec rows (constraints) being attacked. With
        # restrict_disj, only constraints inside still-open disjuncts are
        # in the loss — same pruning α,β-CROWN does via prune_after_crown.
        n_specs_attacked = sum(
            len(spec.disjuncts[di].constraints)
            for di in still_open_disj)
        n_specs_total = sum(
            len(d.constraints) for d in spec.disjuncts)
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg_pgd, settings,
                restrict_disj=still_open_disj,
                time_budget=_pgd_budget_middle)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        timing['phase3p5_pgd_middle'] = time.perf_counter() - t0
        if print_progress:
            print(f'Phase 3.5 (PGD middle, '
                   f'{len(still_open_disj)} open disjuncts, '
                   f'{n_specs_attacked}/{n_specs_total} specs): '
                   f'{timing["phase3p5_pgd_middle"]:.2f}s  sat={pgd_sat}')
        if pgd_sat:
            return _finalize('sat', 'pgd_middle', witness=pgd_witness)

    if time_left() <= 0:
        return _finalize('unknown', 'timeout',
                          remaining=len(still_open_disj))

    stats.record_bounds(sb)

    # Phases 5 (CROWN recheck) and 6 (post-tighten PGD) removed:
    # phase 2 already used the interleaved-tightened bounds for its
    # CROWN spec backward, so phase 5 would compute the exact same
    # numbers; phase 3 PGD already ran on the tightened network, so
    # phase 6 would too.

    if time_left() <= 0:
        return _finalize('unknown', 'timeout',
                          remaining=len(still_open_disj))

    # Refresh bounds_by_relu from tightened sb
    for li in range(nh):
        lo_t, hi_t = sb[li]
        bounds_by_relu[li] = (lo_t.cpu().numpy().astype(np.float64),
                               hi_t.cpu().numpy().astype(np.float64))

    _mem_audit('before phase2p5')

    # --- Phase 2.5: iterative zono-lift tightening ---
    if (getattr(settings, 'zono_lift_enabled', True) and still_open_disj
            and time_left() > 0):
        t0 = time.perf_counter()
        still_open_disj, phase25_info = _phase2p5_zono_lift(
            gg, xl_g, xh_g, bounds_by_relu, disj_queries, spec_ew,
            spec_lbs, still_open_disj, device, dtype, settings,
            print_progress=print_progress,
            mem_audit=_mem_audit if verbose else None)
        timing['phase2p5_zono_lift'] = time.perf_counter() - t0
        details['phase2p5'] = phase25_info

        if not still_open_disj:
            return _finalize('verified', 'zono_lift')

    # --- Phase 2.6: per-spec targeted PGD on still-open disjuncts. ---
    # Iterates per-disjunct so each spec's margin gradient is isolated.
    # Each open spec is guaranteed at least `phase26_pgd_per_spec_min_per_spec`
    # seconds (default 0.5s) — the total budget cap is honored only when
    # the per-spec quota fits inside it. When `phase26_pgd_per_spec_strict_min`
    # is True (default), the min wins even at the cost of overrunning the
    # total budget — useful on many-open-spec cases (e.g. tinyimagenet's 30
    # open disjuncts at 0.2s each was too little to find SAT).
    # Order: most-likely-SAT first (lowest spec_lb across the disjunct's
    # queries) so the budget is spent where SAT is most plausible.
    if (still_open_disj and not _disable_sat
            and getattr(settings, 'phase26_pgd_per_spec_enabled', True)
            and parallel_pgd_ctx is None    # skip when parallel ran inside Phase 1
            and time_left() > 0):
        t0 = time.perf_counter()
        _p26_total = float(getattr(
            settings, 'phase26_pgd_per_spec_time_budget', 3.0))
        _p26_min = float(getattr(
            settings, 'phase26_pgd_per_spec_min_per_spec', 0.5))
        _p26_strict = bool(getattr(
            settings, 'phase26_pgd_per_spec_strict_min', True))
        _n_open = len(still_open_disj)
        if _p26_strict:
            _per = _p26_min   # ignore total budget; each spec gets min
            _p26_remaining = float('inf')
        else:
            _per = max(_p26_min, _p26_total / max(_n_open, 1))
            _p26_remaining = _p26_total
        _p26_n_attacked = 0
        _p26_sat = False
        # Sort by closeness to SAT: lowest spec_lb first.
        def _disj_score(di):
            qids = [qi for qi, _, _ in disj_queries.get(di, [])]
            if not qids: return float('inf')
            return min(spec_lbs.get(qi, 0.0) for qi in qids)
        _ordered_disj = sorted(still_open_disj, key=_disj_score)
        for _di in _ordered_disj:
            if _p26_remaining <= 0 or time_left() <= 0:
                break
            _budget = min(_per, _p26_remaining, max(time_left() - 0.5, 0.1))
            try:
                _ok, _w = _pgd_attack_general(
                    xl_pgd, xh_pgd, spec, gg_pgd, settings,
                    restrict_disj={_di}, time_budget=_budget)
            except RuntimeError:
                _ok, _w = False, None
            _p26_n_attacked += 1
            _p26_remaining -= _budget
            if _ok:
                _p26_sat = True
                pgd_witness = _w
                break
        timing['phase2p6_pgd_per_spec'] = time.perf_counter() - t0
        details['phase2p6'] = {
            'n_attacked': _p26_n_attacked,
            'n_open': _n_open,
            'sat': _p26_sat,
            'per_spec_budget': _per}
        if print_progress:
            print(f'Phase 2.6 (per-spec PGD, {_p26_n_attacked}/{_n_open} '
                  f'attacked, {_per:.2f}s each): '
                  f'{timing["phase2p6_pgd_per_spec"]:.2f}s  '
                  f'sat={_p26_sat}', flush=True)
        if _p26_sat:
            return _finalize('sat', 'pgd_per_spec', witness=pgd_witness)

    # --- Phase 7: LP score+verify (replaces 7a feasibility + 7b CROWN) ---
    # One LP solve per query in 'score' mode: minimizes spec objective,
    # returns (status, lb, scores). If lb > 0 → verified. Otherwise use
    # the fractional-a scores to seed MILP racing. Fallback to CROWN-based
    # scoring (ew * frac) if LP times out / no solution.
    remaining_qids = set()
    for di in still_open_disj:
        for qi, _, _ in disj_queries[di]:
            if spec_lbs.get(qi, -1) <= 0:
                remaining_qids.add(qi)

    t0 = time.perf_counter()
    per_query_scored = {}
    still_needs_milp = set()
    lp_scores_by_query = {}
    spec_impl = str(getattr(settings, 'spec_impl', 'gen_lp'))
    skip_phase7_lp = bool(getattr(settings, 'gen_lp_skip_phase7_lp', False))
    # Unsafe-halfspace mode: when active, forces Phase 7 LP to run so its
    # duals (with the halfspace constraint) can be used for neuron scoring.
    # `phase8_milp_mode` is the unified mode switch.
    _milp_mode = str(getattr(settings, 'phase8_milp_mode', 'alpha_zono_bnb'))
    if _milp_mode in ('find_sat', 'alpha_zono_bnb'):
        _hs_mode = 'none'
        _hs_in_milp = False
    elif _milp_mode in ('infeasibility', 'alpha_zono_infeasibility'):
        # Use inequality halfspace `qw·y + qb ≤ 0` — sound: relaxation∩
        # {≤0}=∅ ⇔ relaxation min > 0, which proves the spec.
        _hs_mode = 'inequality'
        _hs_in_milp = True
    else:
        raise ValueError(
            f'unknown phase8_milp_mode {_milp_mode!r}; valid: '
            "'find_sat' | 'infeasibility' | 'alpha_zono_bnb' "
            "| 'alpha_zono_infeasibility'")
    if _hs_mode != 'none':
        skip_phase7_lp = False
    gen_lp_device = ('cuda' if (spec_impl == 'gen_lp'
                                 and torch.cuda.is_available())
                     else 'cpu')
    if spec_impl == 'gen_lp' and not torch.cuda.is_available():
        # gen_lp on CPU works but is slow; fall back to monolithic
        spec_impl = 'monolithic'
    gen_lp_state = None
    # Always precompute gen_lp_state if spec_impl is 'gen_lp' — Phase 8 needs it.
    # Phase 7 LP (per-query) is optional if skip_phase7_lp=True.
    if remaining_qids and time_left() > 2:
        if spec_impl == 'gen_lp':
            # Precompute the gen LP state once — G matrices and centers as
            # numpy arrays — reused across all Phase 7 and Phase 8 solves.
            # This avoids GPU non-determinism causing bit-different constraint
            # coefficients between calls (the soundness bug source).
            # empty_cache before the GPU forward pass: Phase 2.5's α-CROWN
            # backward + halfspace-LP cascade leaves multi-GB reserved in
            # the caching allocator on resnet_large; without reclaim, the
            # gen-LP forward runs out of contiguous GPU memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_name = gg_ops_ser[-1]['name']
            t_pre = time.perf_counter()
            _reuse_phase1 = bool(getattr(
                settings, 'phase7_reuse_phase1_zono', True))
            if _reuse_phase1 and rec_zono is not None and rec_zono.get(
                    'gen_rows_by_layer'):
                # Reuse Phase 1's already-propagated z_final + rec_zono.
                # Skips the dense (n_layer × n_gens) conv allocation that
                # OOMs on resnet_large CIFAR100. Phase-1 form constraints
                # use Phase 1's bounds for soundness.
                gen_lp_state = verify_gen_lp.state_from_phase1(
                    z_final_phase1, rec_zono, x_lo_64, x_hi_64,
                    gg_ops_ser, gg['input_name'], last_name)
                _state_src = 'phase1_reuse'
            else:
                gen_lp_state = verify_gen_lp.precompute_gen_state(
                    gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
                    gg['input_name'], last_name,
                    device=gen_lp_device, dtype=torch.float64,
                    formulation=str(getattr(
                        settings, 'gen_lp_formulation', 'dense')))
                _state_src = 'precompute'
            if print_progress:
                print(f'  Gen LP state precomputed [{_state_src}]: '
                      f'n_gens={gen_lp_state["n_gens"]}, '
                      f'unstable={len(gen_lp_state["unstable_list"])} '
                      f'({time.perf_counter()-t_pre:.2f}s)')
            # Serial gen_lp solves (GPU can't be shared across procs)
            if skip_phase7_lp:
                if print_progress:
                    print(f'  Phase 7 LP skipped (skip_phase7_lp=True); '
                          f'using ew*frac scoring for MILP binaries')
                still_needs_milp = set(remaining_qids)
            else:
                lp_tl = min(120.0, time_left() * 0.5)
                # With the unsafe halfspace in the LP, the objective min
                # is ≤ 0 by construction; use 'lp_dual' scoring so the
                # triangle-constraint duals (binding under the halfspace)
                # drive neuron ranking.
                _phase7_score_method = ('lp_dual' if _hs_mode != 'none'
                                         else 'lp_ew_frac')
                # Per-spec state_from_phase1 with that spec's Phase 2.5
                # tightened bounds. Mirrors the existing Phase 8
                # `phase8_per_query_tightened_bounds` path but for the
                # Phase 7 LP triangle constraints. Same soundness story:
                # rebuild rec_zono via a fresh forward with the spec's
                # bounds (apply_relu intersects), so the (μ, λ) inside
                # rec_zono and the (lo, hi) used in triangle constraints
                # are consistent.
                # Default OFF: per-spec rebuild uses state_from_phase1's
                # closed-form min-area (μ, λ). When Phase 2.5 narrows
                # (hi - lo) on already-near-stable neurons, μ = -hi·lo /
                # (2·(hi-lo)) blows up — Gurobi simplex hits "drop
                # variables from basis / switch to quad precision" and
                # `optimize_checked` raises GurobiNumericTrouble (a real
                # soundness signal we MUST NOT swallow). Phase 8's
                # alpha_zono path uses α-CROWN-optimized (μ, λ) which
                # doesn't have this pathology and already incorporates
                # per-spec Phase 2.5 bounds (`merged_bbr` at the
                # alpha_zono builder). So this knob is only useful on
                # nets where state_from_phase1 stays well-conditioned.
                _per_q_p7 = bool(getattr(
                    settings, 'phase7_per_query_tightened_bounds', False))
                _p25_per_q = (details.get('phase2p5', {})
                              .get('per_query', {}))
                # Cache per-qi tightened bounds for fast lookup
                _tb_for_qi = {}
                for _di, _qmap in _p25_per_q.items():
                    for _qi_p, _qinfo_p in _qmap.items():
                        _tb_p = _qinfo_p.get('tightened_bounds')
                        if _tb_p is not None:
                            _tb_for_qi[_qi_p] = _tb_p
                for qi in sorted(remaining_qids):
                    _, q_w, q_bias = queries[qi]
                    _state_for_qi = gen_lp_state
                    if _per_q_p7 and qi in _tb_for_qi:
                        _tb_q = _tb_for_qi[qi]
                        _rec_q = {}
                        with torch.no_grad():
                            _, _zf_q = _forward_zonotope_graph(
                                xl_g, xh_g, gg, device, dtype,
                                settings=settings,
                                rec_zono=_rec_q,
                                tight_bounds=_tb_q)
                        _state_for_qi = (
                            verify_gen_lp.state_from_phase1(
                                _zf_q, _rec_q, x_lo_64, x_hi_64,
                                gg_ops_ser, gg['input_name'],
                                last_name))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    res, dt_lp, info = verify_gen_lp.solve_spec(
                        gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
                        gg['input_name'], last_name, q_w, q_bias,
                        milp_set=None, time_limit=lp_tl,
                        n_threads=n_cores, device=gen_lp_device,
                        state=_state_for_qi,
                        score_method=_phase7_score_method,
                        unsafe_halfspace=_hs_mode)
                    details.setdefault('phase7', {})[qi] = {
                        'result': res, 'time': dt_lp, 'info': info}
                    if print_progress:
                        lb = info.get('lb')
                        lb_s = f'{lb:+.4f}' if isinstance(lb, float) else 'n/a'
                        print(f'  Phase 7 query {qi}: {res} ({dt_lp:.1f}s) '
                              f'lb={lb_s} vars={info["n_vars"]}')
                    if res == 'UNSAT':
                        spec_lbs[qi] = 1.0
                        continue
                    still_needs_milp.add(qi)
                    if info.get('scores'):
                        lp_scores_by_query[qi] = info['scores']
        else:
            # Monolithic (old behavior)
            n_q = len(remaining_qids)
            per_worker_threads = n_cores if n_q == 1 else 1
            lp_tl = min(120.0, time_left() * 0.5)
            tasks = []
            for qi in sorted(remaining_qids):
                _, q_w, q_bias = queries[qi]
                tasks.append((
                    'score', impl, gg_ops_ser, x_lo_64, x_hi_64,
                    bounds_by_relu, q_w, q_bias, [], 0, per_worker_threads,
                    lp_tl, gg['input_name']))
            pool_size = min(n_q, n_cores)
            with multiprocessing.Pool(pool_size) as pool:
                results = pool.map(_solve_spec_worker_graph, tasks)
            for qi, (res, dt_lp, info) in zip(sorted(remaining_qids), results):
                details.setdefault('phase7', {})[qi] = {
                    'result': res, 'time': dt_lp, 'info': info}
                if print_progress:
                    lb = info.get('lb') if isinstance(info, dict) else None
                    lb_s = f'{lb:+.4f}' if isinstance(lb, float) else 'n/a'
                    st = info.get('status', '?') if isinstance(info, dict) else '?'
                    print(f'  Phase 7 query {qi}: {res} ({dt_lp:.1f}s) '
                          f'lb={lb_s} status={st}')
                if res == 'UNSAT':
                    spec_lbs[qi] = 1.0
                    continue
                still_needs_milp.add(qi)
                if isinstance(info, dict) and info.get('scores'):
                    lp_scores_by_query[qi] = info['scores']
    else:
        still_needs_milp = set(remaining_qids)
    timing['phase7_lp_score'] = time.perf_counter() - t0

    # Fallback ew*frac scoring for queries that didn't get LP scores. Prefer
    # α-CROWN's captured ew (from Phase 2.5) when available — those slopes
    # are direction-optimized and give sharper scores than a fresh min-area
    # CROWN backward. Fall back to `_spec_backward_graph` only for queries
    # that never ran through Phase 2.5's α-CROWN capture.
    _use_alpha_ew = bool(getattr(settings, 'phase8_use_alpha_ew', True))
    phase25_info = details.get('phase2p5', {}) if _use_alpha_ew else {}
    alpha_ew_by_qi = {}
    if _use_alpha_ew:
        for _di_info in phase25_info.get('per_query', {}).values():
            for _qi, _q in _di_info.items():
                if 'ew_at_relu' in _q:
                    alpha_ew_by_qi[_qi] = _q['ew_at_relu']
    need_ew = [qi for qi in still_needs_milp
                if qi not in lp_scores_by_query
                and qi not in alpha_ew_by_qi]
    if need_ew:
        with torch.no_grad():
            _, _, ew_at_relu = _spec_backward_graph(
                sb, xl_g, xh_g, gg, spec_ew, set(need_ew), nh,
                device, dtype, return_ew=True)
    else:
        ew_at_relu = {}
    # Merge α-CROWN's cached ew into the same dict structure expected below
    # (keyed by qi → {L: np.ndarray}).
    for _qi, _ew_by_L in alpha_ew_by_qi.items():
        if _qi not in ew_at_relu:
            ew_at_relu[_qi] = _ew_by_L

    for qi in sorted(still_needs_milp):
        if qi in lp_scores_by_query:
            scores = lp_scores_by_query[qi]
        else:
            scores = {}
            q_ew = ew_at_relu.get(qi, {})
            for li in range(nh):
                lo_l, hi_l = bounds_by_relu[li]
                unstable = np.where((lo_l < 0) & (hi_l > 0))[0]
                if li in q_ew:
                    ew = np.abs(q_ew[li])
                    for i in unstable:
                        ew_i = float(ew[i]) if i < len(ew) else 1.0
                        frac = float(hi_l[i]) * abs(float(lo_l[i])) / float(
                            hi_l[i] - lo_l[i])
                        scores[(li, int(i))] = ew_i * frac
                else:
                    for i in unstable:
                        scores[(li, int(i))] = float(hi_l[i]) * abs(float(lo_l[i])) / 2
        per_query_scored[qi] = sorted(
            scores.keys(), key=lambda k: scores[k], reverse=True)

    # --- Phase 8: MILP racing escalation ---
    t_phase8 = time.perf_counter()
    milp_witness = None
    _skip_milp = bool(getattr(settings, 'skip_phase8_milp', False))
    # Save hook runs BEFORE the skip check so exploration scripts can
    # harvest state without waiting out Phase 8.
    if spec_impl == 'gen_lp' and still_needs_milp:
        query_specs = []
        for qi in sorted(still_needs_milp):
            _, q_w, q_bias = queries[qi]
            scored_keys = per_query_scored.get(qi, [])
            query_specs.append((qi, q_w, q_bias, scored_keys))
        import os as _os
        _dbg_save = _os.environ.get('GEN_LP_SAVE_STATE')
        if _dbg_save:
            import pickle
            with open(_dbg_save, 'wb') as _f:
                pickle.dump({'state': gen_lp_state,
                             'query_specs': query_specs,
                             'gg_ops_ser': gg_ops_ser,
                             'x_lo_64': x_lo_64,
                             'x_hi_64': x_hi_64,
                             'bounds_by_relu': bounds_by_relu,
                             'input_name': gg['input_name'],
                             'last_name': gg_ops_ser[-1]['name']}, _f)
            print(f'  [debug] saved gen_lp_state + query_specs to {_dbg_save}')
    if _skip_milp and still_needs_milp:
        if print_progress:
            print(f'  Phase 8 skipped (skip_phase8_milp=True); '
                  f'{len(still_needs_milp)} quer{"y" if len(still_needs_milp)==1 else "ies"} '
                  f'left unknown')
        timing['phase8_milp'] = 0.0
        return _finalize('unknown', 'spec_lp',
                          remaining=len(still_needs_milp))
    if spec_impl == 'gen_lp' and still_needs_milp:
        # Parallel racing across open queries; within each query,
        # sequential bin escalation with early termination on UNSAT.
        # Optional: rescore each query's binaries via the LP duals of the
        # gen-LP (ranks triangle-binding constraints; catches L5-dominant
        # killers that kfsb/lp-fractional miss).
        _score_method = getattr(settings, 'gen_lp_score_method', 'lp_ew_frac')
        if _score_method == 'lp_dual':
            import gurobipy as grb
            new_query_specs = []
            for (qi, qw_s, qb_s, scored_keys) in query_specs:
                m_rs, env_rs, uinfo_rs, _ = verify_gen_lp.build_gen_lp_from_state(
                    gen_lp_state, qw_s, float(qb_s), milp_set=set(),
                    n_threads=1)
                m_rs.setParam('OutputFlag', 0); m_rs.setParam('Method', 1)
                m_rs.setParam('TimeLimit', 10.0)
                from vibecheck.gurobi_util import (
                    optimize_checked, GurobiNumericTrouble)
                try:
                    optimize_checked(m_rs)
                    dual_scores = verify_gen_lp.compute_scores(
                        m_rs, uinfo_rs, None, method='lp_dual')
                    new_keys = sorted(dual_scores.keys(),
                                      key=lambda k: dual_scores[k],
                                      reverse=True)
                except GurobiNumericTrouble:
                    new_keys = scored_keys
                m_rs.dispose(); env_rs.dispose()
                new_query_specs.append((qi, qw_s, qb_s, new_keys))
            query_specs = new_query_specs
            if print_progress:
                print(f'  [dual-score] reranked {len(query_specs)} queries '
                      f'by |tri_lo|+|tri_up| duals')
        _parallel_race = bool(getattr(settings, 'gen_lp_parallel_racing',
                                       True))
        _grb_threads = int(getattr(settings, 'gen_lp_gurobi_threads', 1))
        _leave_open = int(getattr(settings, 'phase8_leave_cores_open', 1))
        _n_workers = max(1, n_cores - _leave_open)
        # Per-query rebuild: if Phase 2.5 tightened the intermediate bounds
        # for this query (α-CROWN best_bounds ∧ halfspace-LP override),
        # rebuild a per-query gen_lp_state so the MILP triangles + stable
        # classifications reflect those tighter bounds. New λ slopes fall
        # out automatically from the ReLU (lo, hi) → triangle reconstruction
        # inside `precompute_gen_state`.
        # (Flag was already flipped right after Phase 1; this is redundant.)
        _phase8_started[0] = True
        state_by_qi = {}
        _per_qi_rebuild = bool(getattr(
            settings, 'phase8_per_query_tightened_bounds', True))

        # alpha_zono modes: per-query α-zono state. Run α-CROWN per still-
        # open query (re-using Phase 2.5's tightened bounds when available),
        # build the direction-adaptive forward zonotope, then convert to a
        # state_from_alpha_zono dict. The MILP encodes the parallelogram-
        # only relaxation (no triangle floor) for non-binarized neurons.
        # Both 'alpha_zono_bnb' (feasibility proof) and
        # 'alpha_zono_infeasibility' (halfspace + INFEASIBLE proof) share
        # this state-build path; the only difference is the
        # `_hs_mode`/`_hs_in_milp` flags (resolved earlier) which decide
        # whether the equality halfspace is added inside `_build_alpha_zono_lp`.
        if _milp_mode in ('alpha_zono_bnb', 'alpha_zono_infeasibility'):
            from . import alpha_crown as ac
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for qi in sorted(still_needs_milp):
                _, qw_q, qb_q = queries[qi]
                # Resolve per-query bbr: prefer Phase 2.5's tightened
                # bounds when present, else the global bbr.
                merged_bbr = dict(bounds_by_relu)
                for di_info in details.get('phase2p5', {}).get(
                        'per_query', {}).values():
                    q_p = di_info.get(qi)
                    if q_p is None:
                        continue
                    tb = q_p.get('tightened_bounds')
                    if tb is not None:
                        for L, (lo_t, hi_t) in tb.items():
                            merged_bbr[L] = (
                                np.maximum(bounds_by_relu[L][0], lo_t),
                                np.minimum(bounds_by_relu[L][1], hi_t))
                # α-CROWN with fixed intermediate bounds; spec-only α.
                t_ac = time.perf_counter()
                _, alpha_params, _, _ = (
                    ac.run_alpha_crown_fixed_intermediate(
                        gg, xl_g, xh_g, merged_bbr, qw_q, float(qb_q),
                        device, dtype,
                        n_iters=int(getattr(
                            settings, 'zono_lift_alpha_iters', 10)),
                        lr=float(getattr(
                            settings, 'zono_lift_alpha_lr', 0.25)),
                        lr_decay=float(getattr(
                            settings, 'alpha_crown_lr_decay', 0.98)),
                        early_stop_on_positive=bool(getattr(
                            settings,
                            'alpha_crown_early_stop_on_positive', True))))
                _, ew_at_relu_q = ac.capture_ew_per_relu(
                    gg, xl_g, xh_g, alpha_params['spec'], merged_bbr,
                    qw_q, float(qb_q), device, dtype)
                alpha_per_layer_q = ac.build_dir_adaptive_alpha(
                    alpha_params['spec'], ew_at_relu_q, merged_bbr,
                    device, dtype)
                # Build per-layer unstable index tensors so the forward
                # only materialises the unstable rows of the pre-ReLU G.
                unstable_per_layer_q = {}
                for L, (lo_np, hi_np) in merged_bbr.items():
                    un = np.where((np.asarray(lo_np) < 0)
                                   & (np.asarray(hi_np) > 0))[0]
                    unstable_per_layer_q[L] = torch.as_tensor(
                        un, dtype=torch.long, device=device)
                z_alpha, pre_relu_gpu_q = ac.forward_zono_dir_adaptive(
                    xl_g, xh_g, gg, alpha_per_layer_q, merged_bbr,
                    device, dtype, settings=settings,
                    unstable_per_layer=unstable_per_layer_q)
                state_by_qi[qi] = verify_gen_lp.state_from_alpha_zono(
                    z_alpha, pre_relu_gpu_q, alpha_per_layer_q,
                    merged_bbr, x_lo_64, x_hi_64,
                    gg_ops_ser, gg['input_name'],
                    gg_ops_ser[-1]['name'],
                    unstable_per_layer=unstable_per_layer_q)
                # Box+halfspace per-neuron delta_LB scoring — closed-form
                # Lagrangian, ~10ms/1k neurons. Picks kfsb-quality
                # binarisation candidates by simulating off-side
                # (y_k=0) and on-side (y_k=z_k) substitutions and
                # taking BaB worst-child LB. Overrides ew*frac for
                # this query — measured 24/30 overlap with AB-CROWN's
                # actual splits on tinyimagenet prop_7575 (vs 19/30
                # for ew*frac). Gated by `phase8_score_box_halfspace`.
                if bool(getattr(
                        settings, 'phase8_score_box_halfspace', True)):
                    ew_np = {
                        L: (t.detach().cpu().numpy().astype(np.float64)
                            if hasattr(t, 'detach')
                            else np.asarray(t, dtype=np.float64))
                        for L, t in ew_at_relu_q.items()}
                    bh_scores = verify_gen_lp.score_box_halfspace_delta_lb(
                        state_by_qi[qi], qw_q, float(qb_q), ew_np)
                    per_query_scored[qi] = sorted(
                        bh_scores.keys(),
                        key=lambda k: bh_scores[k], reverse=True)
                if print_progress:
                    print(f'  Per-query α-zono state q{qi}: '
                          f'n_gens={state_by_qi[qi]["n_gens"]}, '
                          f'unstable={len(state_by_qi[qi]["unstable_list"])} '
                          f'({time.perf_counter()-t_ac:.2f}s)')
                # Free GPU tensors before next query.
                del z_alpha, pre_relu_gpu_q, alpha_per_layer_q
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif _per_qi_rebuild:
            # Reclaim caching-allocator memory from Phase 2.5's GPU work
            # before allocating fresh tensors for the per-query rebuild.
            # On resnet_large (10 GB GPU) the global gen_lp_state already
            # leaves ~9 GB reserved; without this, the per-query rebuild
            # OOMs even though the SUM of allocations would fit.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for di_info in details.get('phase2p5', {}).get(
                    'per_query', {}).values():
                for qi_p, q_p in di_info.items():
                    if ('tightened_bounds' in q_p
                            and qi_p in still_needs_milp):
                        tb = q_p['tightened_bounds']
                        merged_bbr = dict(bounds_by_relu)
                        for L, (lo_t, hi_t) in tb.items():
                            merged_bbr[L] = (
                                np.maximum(bounds_by_relu[L][0], lo_t),
                                np.minimum(bounds_by_relu[L][1], hi_t))
                        t_prq = time.perf_counter()
                        state_by_qi[qi_p] = verify_gen_lp.precompute_gen_state(
                            gg_ops_ser, x_lo_64, x_hi_64, merged_bbr,
                            gg['input_name'], gg_ops_ser[-1]['name'],
                            device=gen_lp_device, dtype=torch.float64,
                            formulation=str(getattr(
                                settings, 'gen_lp_formulation', 'sparse')))
                        if print_progress:
                            print(f'  Per-query gen_lp_state q{qi_p}: '
                                  f'n_gens={state_by_qi[qi_p]["n_gens"]}, '
                                  f'unstable={len(state_by_qi[qi_p]["unstable_list"])} '
                                  f'({time.perf_counter()-t_prq:.2f}s)')
                        # Free the transient GPU tensors used inside this
                        # rebuild before the next iter — the returned state
                        # is numpy-only, so this is sound.
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        # In infeasibility modes pass the unsafe-halfspace mode into the
        # worker init so every MILP solve adds the halfspace constraint.
        # Otherwise workers see 'none' — standard BestBdStop MILP.
        _milp_hs = _hs_mode if _hs_in_milp else 'none'

        # Optional debug dump AFTER state_by_qi is populated, so external
        # tooling (e.g. BaB-cost probes) can pickle the per-query α-zono
        # states alongside the shared gen_lp_state.
        _dbg_save_after = _os.environ.get('GEN_LP_SAVE_STATE_AFTER')
        if _dbg_save_after:
            import pickle
            with open(_dbg_save_after, 'wb') as _f:
                pickle.dump({'state': gen_lp_state,
                             'state_by_qi': state_by_qi,
                             'query_specs': query_specs,
                             'gg_ops_ser': gg_ops_ser,
                             'x_lo_64': x_lo_64,
                             'x_hi_64': x_hi_64,
                             'bounds_by_relu': bounds_by_relu,
                             'input_name': gg['input_name'],
                             'last_name': gg_ops_ser[-1]['name']}, _f)
            print(f'  [debug] saved gen_lp_state + state_by_qi + '
                  f'query_specs to {_dbg_save_after}')

        # Main-thread MILP-seeded PGD refinement: for every worker that
        # returns with a feasible MILP solution but no real counterexample,
        # seed a short PGD from the MILP's e_in (plus small perturbations).
        # Runs on GPU in the main thread — no contention with the CPU
        # multiprocessing pool. Doesn't wait for all workers; each worker's
        # result triggers its own PGD as it comes back via imap_unordered.
        _pgd_seed_enabled = bool(getattr(
            settings, 'phase8_pgd_seed_from_milp', True)) and not _disable_sat
        _pgd_n_iters = int(getattr(settings, 'phase8_pgd_seed_iters', 20))
        _pgd_n_perts = int(getattr(settings, 'phase8_pgd_seed_perts', 8))
        _pgd_pert_noise = float(getattr(
            settings, 'phase8_pgd_seed_noise', 0.01))

        def _pgd_refine(qi_ref, e_in_ref):
            state_for_qi = (state_by_qi.get(qi_ref, gen_lp_state)
                             if state_by_qi else gen_lp_state)
            c_in = (state_for_qi['x_hi'] + state_for_qi['x_lo']) / 2.0
            half_w = (state_for_qi['x_hi'] - state_for_qi['x_lo']) / 2.0
            x_np = c_in + half_w * e_in_ref
            x_np = x_np.astype(np.float32)
            x_t = torch.as_tensor(x_np, device=xl_pgd.device,
                                    dtype=xl_pgd.dtype).reshape(xl_pgd.shape)
            # Batch: 1 clean init + n_perts small-noise perturbations.
            inits = [x_t.unsqueeze(0)]
            if _pgd_n_perts > 0:
                eps_box = (xh_pgd - xl_pgd)
                noise = (_pgd_pert_noise * eps_box *
                          (torch.rand(_pgd_n_perts, *xl_pgd.shape,
                                       device=xl_pgd.device,
                                       dtype=xl_pgd.dtype) * 2 - 1))
                inits.append(torch.clamp(
                    x_t.unsqueeze(0) + noise, xl_pgd, xh_pgd))
            x_init_batch = torch.cat(inits, dim=0)
            # Which disjunct does qi belong to?
            di_for_q = next(
                (di for di, qs in disj_queries.items()
                 if any(q[0] == qi_ref for q in qs)), None)
            restrict = {di_for_q} if di_for_q is not None else None
            try:
                from .pgd import pgd_attack_from_init
                is_sat, w = pgd_attack_from_init(
                    x_init_batch, xl_pgd, xh_pgd, spec, gg_pgd, settings,
                    restrict_disj=restrict, n_iter=_pgd_n_iters)
            except RuntimeError:
                return None
            return w if is_sat else None

        _refine_arg = _pgd_refine if _pgd_seed_enabled else None
        # Triangle-top-K: only meaningful for alpha_zono modes (the
        # triangle floor is already always present in triangle-LP
        # formulations). Read the setting unconditionally; the LP
        # builder ignores it for non-alpha_zono states.
        _triangle_top_k = int(getattr(
            settings, 'phase8_alpha_zono_triangle_top_k', 0))
        if _milp_mode not in ('alpha_zono_bnb', 'alpha_zono_infeasibility'):
            _triangle_top_k = 0
        # Per-setting bin schedule override. Default schedule is
        # [8, 16, 24, ..., 8*n_workers]. For tinyimagenet our
        # box+halfspace scoring close cases at K=18/22/30 that the
        # stride-8 schedule skips between 16 and 32; supplying a
        # custom list catches those.
        _bin_sched = getattr(settings, 'phase8_bin_schedule', None)
        if _bin_sched is not None:
            try:
                _bin_sched = [int(x) for x in _bin_sched]
            except (TypeError, ValueError):
                _bin_sched = None
        _use_dual_gpu = bool(getattr(
            settings, 'phase8_use_dual_ascent_gpu', False))
        if _use_dual_gpu:
            from .dual_ascent_bab import verify_query_dual_ascent_bab
            _da_K = int(getattr(settings, 'phase8_dual_ascent_max_iter', 1))
            _da_rep = int(getattr(
                settings, 'phase8_dual_ascent_repair_steps', 5))
            raw = []
            # Witness-attack callback: maps each LP-relaxation primal witness
            # (e ∈ [-1,1]^n_input) back to real input space and runs the NN
            # forward to test if it actually falsifies the spec. Real
            # counterexamples found this way short-circuit BaB with 'sat'.
            x_lo_t = torch.as_tensor(x_lo_64, device=device, dtype=dtype)
            x_hi_t = torch.as_tensor(x_hi_64, device=device, dtype=dtype)
            _disable_sat_local = bool(getattr(settings, 'disable_sat_finding', False))
            def _da_witness_check(w_e_input):
                """w_e_input: [N, n_input] in [-1,1] (top-K worst LP witnesses).
                For each: map to real input space, batch-forward, then PGD-refine
                from any near-boundary witnesses. Returns counterexample np array
                or None.
                """
                if _disable_sat_local: return None
                w_t = torch.as_tensor(w_e_input, device=device, dtype=dtype)
                center = (x_lo_t + x_hi_t) * 0.5
                halfwidth = (x_hi_t - x_lo_t) * 0.5
                x_real = center.unsqueeze(0) + w_t * halfwidth.unsqueeze(0)
                x_real = x_real.reshape(w_t.shape[0], *spec.x_lo.shape)
                # First: cheap point-forward check (most LP witnesses ARE
                # already adversarial since we sort by primal value).
                x_np_batch = x_real.cpu().numpy()
                for bi in range(x_np_batch.shape[0]):
                    y_np = verify_gen_lp.forward_point(
                        gg_ops_ser, x_np_batch[bi],
                        gg['input_name'], gg_ops_ser[-1]['name'])
                    _, ck = spec.check(y_np, y_np)
                    if ck.get('worst_margin', 0.0) < 0:
                        return x_np_batch[bi]
                # If raw witnesses don't falsify, run a SHORT PGD from them.
                # The LP witness is near the spec boundary; PGD nudges it
                # toward a real adversarial. Cheap: ~20 iter on B=5 batch ~ms.
                try:
                    from .pgd import pgd_attack_from_init
                    _xl = xl_pgd.to(device) if xl_pgd.device != device else xl_pgd
                    _xh = xh_pgd.to(device) if xh_pgd.device != device else xh_pgd
                    is_sat, w = pgd_attack_from_init(
                        x_real, _xl, _xh, spec, gg_pgd, settings,
                        n_iter=20)
                    if is_sat: return w
                except Exception:
                    pass
                return None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            for (qi, qw_q, qb_q, scored_keys_q) in query_specs:
                if time_left() <= 0:
                    raw.append((qi, 'unknown', [], None))
                    continue
                state_q = (state_by_qi.get(qi, gen_lp_state)
                           if state_by_qi else gen_lp_state)
                # BUGFIX: query_specs was built with lp_ew_frac scoring at
                # phase 8 start; bh_scores reranking later updates
                # per_query_scored but query_specs holds stale reference.
                # Re-fetch the latest per_query_scored (which IS bh_scores
                # if `phase8_score_box_halfspace=True`).
                _latest_keys = per_query_scored.get(qi, scored_keys_q)
                if list(_latest_keys) != list(scored_keys_q):
                    scored_keys_q = _latest_keys
                # DEBUG: dump state passed to dual-ascent BaB if env var set
                import os as _os
                _dump = _os.environ.get('DA_BAB_DUMP_DIR')
                if _dump:
                    _os.makedirs(_dump, exist_ok=True)
                    import pickle as _pkl
                    _path = f'{_dump}/dump_q{qi}.pkl'
                    # Also extract ew_per_relu if available in scope
                    try:
                        _ew = ew_at_relu_q
                        _ew_dump = {L: (t.detach().cpu().numpy().astype(np.float64)
                                        if hasattr(t, 'detach')
                                        else np.asarray(t, dtype=np.float64))
                                    for L, t in _ew.items()}
                    except (NameError, AttributeError):
                        _ew_dump = None
                    with open(_path, 'wb') as _f:
                        _pkl.dump({'state': state_q,
                                    'qw': np.asarray(qw_q),
                                    'qb': float(qb_q),
                                    'scored_keys': list(scored_keys_q),
                                    'ew_at_relu': _ew_dump}, _f)
                    print(f'  [DA-DUMP] q{qi} → {_path}')
                vd, info = verify_query_dual_ascent_bab(
                    state_q, qw_q, qb_q,
                    [k for k in scored_keys_q],
                    time_limit=time_left(),
                    max_iter=_da_K, repair_steps=_da_rep,
                    print_progress=False, time_left_fn=time_left,
                    witness_check_fn=_da_witness_check)
                if print_progress:
                    print(f'  [dual-ascent-gpu] query {qi}: {vd} '
                          f'nodes={info["nodes"]} '
                          f'wall={info["wall"]:.3f}s '
                          f'(safe={info["exit_counts"]["dual_safe"]} '
                          f'uns={info["exit_counts"]["primal_unsafe"]} '
                          f'cap={info["exit_counts"]["safety_cap"]})')
                # CRITICAL: when DA-BaB finds SAT via witness-attack,
                # info['sat_witness'] is the real-input counterexample.
                # Bug 2026-05-15: was passing witness=None, dropping all
                # SAT discoveries from BaB. 8 missed AB-SAT cases on
                # tinyimagenet were almost all caused by this.
                _da_witness = info.get('sat_witness') if vd == 'sat' else None
                race_levels = [{'n_bins': 0, 'result': vd,
                                'lb': 1.0 if vd == 'unsat' else 0.0,
                                'time': info['wall'],
                                'source': 'dual_ascent',
                                'nodes': info.get('nodes', 0),
                                'witness_source': 'dual_ascent_bab' if vd == 'sat' else None}]
                raw.append((qi, vd, race_levels, _da_witness))
        elif _parallel_race:
            raw = verify_gen_lp.parallel_query_racing(
                gen_lp_state, query_specs,
                time_left_fn=time_left,
                n_threads_total=_n_workers,
                print_progress=False,
                gurobi_threads=_grb_threads,
                state_by_qi=state_by_qi if state_by_qi else None,
                unsafe_halfspace=_milp_hs,
                triangle_top_k=_triangle_top_k,
                witness_refine_fn=_refine_arg,
                bin_schedule_override=_bin_sched)
        else:
            raw = verify_gen_lp.sequential_query_racing(
                gen_lp_state, query_specs,
                time_left_fn=time_left,
                n_threads=_grb_threads,
                print_progress=print_progress)
        for qi, verdict, race_levels, witness in raw:
            if print_progress:
                # Dual-ascent reuses the race_levels container with a single
                # synthetic entry — print a dual-ascent-friendly summary
                # instead of the MILP/racing template.
                src = race_levels[0].get('source') if race_levels else None
                if src == 'dual_ascent':
                    lv = race_levels[0]
                    nd = lv.get('nodes', 0)
                    print(f'  [dual-ascent] query {qi} '
                          f'(disjunct {queries[qi][0]}): '
                          f'{lv["result"]}  nodes={nd}  '
                          f'({lv["time"]:.3f}s)')
                else:
                    print(f'  MILP query {qi} (disjunct {queries[qi][0]}):')
                    for lv in race_levels:
                        lb = lv.get('lb')
                        lb_s = f'{lb:+.4f}' if isinstance(lb, float) else 'n/a'
                        wsrc = lv.get('witness_source')
                        src_s = f' witness={wsrc}' if wsrc else ''
                        print(f'    Racing bins={lv["n_bins"]}: '
                              f'{lv["result"]} lb={lb_s} '
                              f'({lv["time"]:.1f}s){src_s}')
                n_bins_used = (race_levels[-1]['n_bins']
                               if race_levels else 0)
                details.setdefault('racing', {})[qi] = {
                    'verified': verdict == 'unsat', 'n_bins': n_bins_used,
                    'levels': race_levels}
            if verdict == 'unsat':
                spec_lbs[qi] = 1.0
            elif (verdict == 'sat' and witness is not None
                    and milp_witness is None
                    and not _disable_sat):
                # Validate witness against the full spec.
                y = verify_gen_lp.forward_point(
                    gg_ops_ser, witness, gg['input_name'],
                    gg_ops_ser[-1]['name'])
                _, check_details = spec.check(y, y)
                worst = check_details.get('worst_margin', 0.0)
                if worst < 0:
                    milp_witness = witness
                    # Identify which level produced the witness so the
                    # user can see whether the MILP integer solution was
                    # itself a counterexample (`milp_direct`) or whether
                    # the main-thread PGD seeded from MILP's e_in had
                    # to walk to the violation (`pgd_seeded`). A purely
                    # 'unknown' source means the level info wasn't
                    # tagged — shouldn't happen, but kept defensive.
                    src = next(
                        (lv.get('witness_source') for lv in race_levels
                         if lv.get('witness_source')),
                        'unknown')
                    if print_progress:
                        print(f'  MILP query {qi} found SAT witness '
                              f'(worst margin {worst:+.6f}, '
                              f'source={src})')
    else:
        for qi in sorted(still_needs_milp):
            if time_left() <= 0:
                break
            _, q_w, q_bias = queries[qi]
            scored_keys = per_query_scored.get(qi, [])
            if print_progress:
                print(f'  MILP query {qi} (disjunct {queries[qi][0]}):')
            verified, n_bins_used, race_levels = \
                _racing_escalation_graph_correct(
                    impl, gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
                    q_w, q_bias, scored_keys, n_cores, time_left,
                    gg['input_name'], print_progress)
            if print_progress:
                details.setdefault('racing', {})[qi] = {
                    'verified': verified, 'n_bins': n_bins_used,
                    'levels': race_levels}
            if verified:
                spec_lbs[qi] = 1.0

    # High-bin infeasibility fallback: each still-open query gets one more
    # MILP solve at bins=phase8_high_bin_count + inequality halfspace.
    # Empirically closes hard cases (oval21 deep_kw img3039 q8) where
    # racing maxes at 8*n_workers and the LP relaxation needs more bins
    # to push ObjBound > 0. Sound: Gurobi.INFEASIBLE under the halfspace
    # constraint ⇔ standard LP min > 0 (LP feasibility theorem).
    _hi_bin_fallback = bool(getattr(
        settings, 'phase8_high_bin_fallback', True))
    _hi_bin_count = int(getattr(settings, 'phase8_high_bin_count', 200))
    _hi_bin_time = float(getattr(
        settings, 'phase8_high_bin_time_limit', 60.0))
    if (_hi_bin_fallback and _milp_mode in ('alpha_zono_bnb',
                                              'alpha_zono_infeasibility')
            and time_left() > 5.0):
        for qi, qw_q, qb_q, scored_keys_q in query_specs:
            if spec_lbs.get(qi, -1.0) > 0:
                continue
            if time_left() <= 5.0:
                break
            state_q = (state_by_qi.get(qi, gen_lp_state)
                       if state_by_qi else gen_lp_state)
            if state_q is None or not scored_keys_q:
                continue
            n_bins_fb = min(_hi_bin_count, len(scored_keys_q))
            tl = min(_hi_bin_time, time_left())
            # Fallback runs sequentially — use ALL cores for one Gurobi
            # solve. Bypass solve_spec because that path disables cuts/
            # heuristics/presolve for fast small-bin tasks; at high bins
            # we WANT Gurobi's defaults (cuts ON, presolve ON) to reach
            # INFEASIBLE quickly. Empirically (img3039 q8): 200 bins +
            # default Gurobi closes in ~19s; bins=200 with cuts off
            # times out at 60s.
            import gurobipy as _grb
            try:
                m_fb, env_fb, _, _ = verify_gen_lp.build_gen_lp_from_state(
                    state_q, qw_q, qb_q,
                    milp_set=set(scored_keys_q[:n_bins_fb]),
                    n_threads=n_cores, unsafe_halfspace='inequality')
                m_fb.setParam('TimeLimit', float(tl))
                m_fb.setParam('BestBdStop', _grb.GRB.INFINITY)
                from .gurobi_util import (
                    optimize_checked, GurobiNumericTrouble)
                try:
                    optimize_checked(m_fb)
                except (GurobiNumericTrouble, _grb.GurobiError):
                    m_fb.dispose(); env_fb.dispose()
                    continue
                fb_status = m_fb.Status
                try:
                    fb_obj_bound = float(m_fb.ObjBound)
                except Exception:
                    fb_obj_bound = None
                m_fb.dispose(); env_fb.dispose()
            except Exception:
                continue
            if fb_status == _grb.GRB.INFEASIBLE:
                # Sound: relaxation∩{qw·y+qb≤0}=∅ ⇔ relaxation min > 0.
                # By the same LP feasibility-theorem reasoning the racing
                # path uses; standard cross-check unnecessary because we
                # restored Gurobi's default numerics (cuts/presolve ON).
                spec_lbs[qi] = 1.0
                if print_progress:
                    print(f'  Fallback q{qi} bins={n_bins_fb} '
                          f'+halfspace: INFEASIBLE → q{qi} CLOSED',
                          flush=True)
            elif print_progress:
                status_name = {_grb.GRB.OPTIMAL: 'OPTIMAL',
                                _grb.GRB.TIME_LIMIT: 'TIMEOUT',
                                _grb.GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT'}.get(
                    fb_status, f'STATUS_{fb_status}')
                ob_s = (f'{fb_obj_bound:+.4f}'
                        if isinstance(fb_obj_bound, float) else 'n/a')
                print(f'  Fallback q{qi} bins={n_bins_fb} '
                      f'+halfspace: {status_name} lb={ob_s}', flush=True)
    timing['phase8_milp'] = time.perf_counter() - t_phase8

    # If the gen_lp path found a true counterexample from a MILP integer
    # solution, short-circuit to SAT before running final PGD.
    if milp_witness is not None:
        return _finalize('sat', 'spec_milp', witness=milp_witness)

    verified_disj = {di for di, qlist in disj_queries.items()
                      if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist)}
    still_open_disj = set(disj_queries.keys()) - verified_disj

    # --- Phase 9: final PGD ---
    t0 = time.perf_counter()
    if still_open_disj and time_left() > 0 and not _disable_sat:
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg_pgd, settings)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        if pgd_sat:
            timing['phase9_pgd'] = time.perf_counter() - t0
            return _finalize('sat', 'pgd', witness=pgd_witness)
    timing['phase9_pgd'] = time.perf_counter() - t0

    if not still_open_disj:
        return _finalize('verified', 'spec_milp')
    return _finalize('unknown', 'timeout', remaining=len(still_open_disj))


def verify_graph(graph, spec, settings):
    """Graph verification mode entry point.

    Dispatches to the reference or optimized builder based on
    settings.graph_impl ('reference' or 'optimized', default 'optimized').

    Optionally wraps the pipeline in an input-space BaB when the input
    dimension is small (≤ settings.input_split_max_dims). Splitting the
    input box on the widest dim keeps each sub-region's zono enclosure
    tighter, which often closes spec gaps that a single-shot Phase 1+2
    can't (especially on cifar_biasfield-style 16-dim parameters).

    Returns (result_str, details_dict).
    """
    # Route conv-heavy nets (e.g. oval21 cifar_base/deep/wide_kw) to
    # the historical milp_verify pipeline. The bab_refine cascade +
    # alpha-zono Phase 8 are tuned for FC nets like mnist_fc and
    # underperform on conv ResNets where the alpha-zono LP relaxation
    # is too loose at the spec layer. milp_verify uses a per-neuron
    # layer-wise MILP encoding + Phase 5 racing escalation which
    # empirically closes oval21 medium-eps cases (e.g. img1204
    # eps=0.025: milp_verify 36s vs bab_refine 181s timeout).
    #
    # Trigger condition (all four must hold):
    #   1. settings.auto_route_milp_for_conv is True
    #   2. graph has at least one Conv node (not a pure FC net)
    #   3. no fork points (milp_verify's non-graph path requires this)
    #   4. input dimension > `input_split_max_dims` (without this guard
    #      cifar_biasfield (input_dim=16) was misrouted to milp_verify
    #      where the joint α-CROWN tightening OOMs the 10 GB GPU; with
    #      it the input-split BaB / fast-leaf path takes over and
    #      verifies the same case in ~55 s)
    n_in = (int(np.prod(spec.x_lo.shape))
            if hasattr(spec, 'x_lo') else 10**9)
    _split_max = int(getattr(settings, 'input_split_max_dims', 20))
    if (bool(getattr(settings, 'auto_route_milp_for_conv', True))
            and bool(getattr(settings, 'input_split_enabled', True))
            and n_in > _split_max):
        has_conv = any(getattr(n, 'op_type', '') == 'Conv'
                       for n in graph.nodes.values())
        if has_conv and not graph.fork_points():
            from .verify_milp import milp_verify
            if getattr(settings, 'print_progress', False):
                print('[verify_graph] auto-routing conv net to milp_verify '
                      '(historical pipeline tuned for oval21/cifar conv)',
                      flush=True)
            return milp_verify(graph, spec, settings)

    impl = str(getattr(settings, 'graph_impl', 'optimized'))
    assert impl in _BUILDERS, f'unknown graph_impl: {impl!r}'
    build_fn = _BUILDERS[impl]
    n_in = int(np.prod(spec.x_lo.shape)) if hasattr(spec, 'x_lo') else 10**9
    if (getattr(settings, 'input_split_enabled', True)
            and n_in <= int(getattr(settings, 'input_split_max_dims', 20))):
        if bool(getattr(settings, 'input_split_batched_enabled', False)):
            # Build the GPU graph once at the top — the batched driver
            # reuses it across all iterations.
            from .settings import resolve_torch
            _dev, _dtype = resolve_torch(settings)
            try:
                _gg = graph.gpu_graph(_dev, _dtype)
            except (ValueError, NotImplementedError, KeyError, RuntimeError) as _e:
                # gpu_graph can't represent this model (transformer
                # attention, etc.). Fall back to raw-ONNX PGD as a SAT
                # finder; if that fails the case is reported unknown.
                _onnx_p = getattr(graph, 'onnx_path', None)
                if _onnx_p is not None and not bool(getattr(
                        settings, 'disable_sat_finding', False)):
                    from .onnx_torch_runner import pgd_via_onnx
                    try:
                        _sat, _w = pgd_via_onnx(
                            _onnx_p, spec,
                            n_restarts=int(settings.pgd_phase0_restarts
                                if 'pgd_phase0_restarts' in settings
                                else 256),
                            n_iter=int(settings.pgd_phase0_iters
                                if 'pgd_phase0_iters' in settings
                                else 100))
                        if _sat:
                            return 'sat', {
                                'phase': 'onnx_pgd_unsupported_gpu_graph',
                                'witness': _w}
                    except Exception as _pe:
                        if getattr(settings, 'print_progress', False):
                            print(f'  [onnx-pgd] failed: {type(_pe).__name__}: '
                                  f'{_pe}', flush=True)
                return 'unknown', {
                    'phase': 'gpu_graph_build_failed',
                    'reason': f'{type(_e).__name__}: {_e}'}
            try:
                result, details = _input_split_batched(
                    graph, spec, settings, _gg, _dev, _dtype)
            except (ValueError, NotImplementedError, KeyError, RuntimeError) as _e:
                # batched zono/CROWN can't handle some op (transformer
                # attention's max_pool/softmax/matmul_bilinear). Fall
                # back to raw-ONNX PGD as a SAT finder.
                _onnx_p = getattr(graph, 'onnx_path', None)
                if _onnx_p is not None and not bool(getattr(
                        settings, 'disable_sat_finding', False)):
                    from .onnx_torch_runner import pgd_via_onnx
                    try:
                        _sat, _w = pgd_via_onnx(
                            _onnx_p, spec,
                            n_restarts=int(settings.pgd_phase0_restarts
                                if 'pgd_phase0_restarts' in settings
                                else 256),
                            n_iter=int(settings.pgd_phase0_iters
                                if 'pgd_phase0_iters' in settings
                                else 100))
                        if _sat:
                            return 'sat', {
                                'phase': 'onnx_pgd_unsupported_batched',
                                'witness': _w}
                    except Exception as _pe:
                        if getattr(settings, 'print_progress', False):
                            print(f'  [onnx-pgd] failed: {type(_pe).__name__}: '
                                  f'{_pe}', flush=True)
                return 'unknown', {
                    'phase': 'batched_forward_failed',
                    'reason': f'{type(_e).__name__}: {_e}'}
        else:
            result, details = _input_split_verify(
                graph, spec, settings, build_fn, impl)
    else:
        result, details = _run_pipeline(
            graph, spec, settings, build_fn, impl)
    # Top-level SAT witness validation — defense-in-depth against
    # spurious counterexamples from any code path (input_split bypasses
    # _finalize). Idempotent: if _finalize already validated, the
    # witness is already a real one and the second check just confirms.
    if (result == 'sat' and isinstance(details, dict)
            and details.get('witness') is not None
            and not bool(getattr(settings, 'skip_sat_validation', False))
            and not details.get('witness_validated_at_topvel', False)):
        _atol = float(settings.sat_validate_atol
                       if 'sat_validate_atol' in settings else 1e-4)
        _ok, _info = _validate_sat_witness(
            getattr(graph, 'onnx_path', None), spec, details['witness'],
            atol=_atol)
        details['witness_validated_at_topvel'] = True
        if not _ok:
            if getattr(settings, 'print_progress', False):
                print(f'  [validate-top] SPURIOUS SAT from '
                      f'{details.get("phase")}: '
                      f'{_info.get("reason")}', flush=True)
            details['spurious_witness'] = _info
            details['original_phase'] = details.get('phase')
            details['phase'] = f'spurious_sat_{details.get("phase")}'
            result = 'unknown'
    return result, details


def _score_input_axes(graph, spec, gg=None, device=None, dtype=None):
    """Score each input axis by its contribution to the worst spec margin.

    Runs a single fast zono forward pass through the network from the
    current input box, then for each input column `k` of the output
    zonotope generators, computes the spec-side weight `|qw·G[:, k]|`
    summed across all disjuncts (averaging out per-disjunct sign).
    Larger score = splitting that axis tightens the spec bound more.

    Returns ``(scores, axis_for_col)`` where ``scores[i]`` is the score
    for original input axis ``i``, and axes with zero radius get score 0.

    `gg` / `device` / `dtype`: when provided, the cached GPU graph is
    reused. Without these, the function rebuilds `gpu_graph(...)` from
    scratch which is hundreds of milliseconds per call on
    cifar_biasfield-class networks. Passing the cached `fast_gg` from
    `_input_split_verify` cut per-BaB-node cost from ~1 s to ~580 ms in
    profiling.
    """
    import torch
    if device is None or dtype is None:
        device = 'cuda' if (str(getattr(graph, 'device', 'gpu')) == 'gpu'
                             and torch.cuda.is_available()) else 'cpu'
        dtype = torch.float32
    x_lo = torch.tensor(spec.x_lo.flatten(), dtype=dtype, device=device)
    x_hi = torch.tensor(spec.x_hi.flatten(), dtype=dtype, device=device)
    radii = (x_hi - x_lo) / 2
    nz = torch.nonzero(radii).squeeze(1)
    if nz.numel() == 0:
        return np.zeros(x_lo.numel()), np.array([], dtype=np.int64)

    if gg is None:
        gg = graph.gpu_graph(device=device, dtype=dtype)
    _, z_final = _forward_zonotope_graph(x_lo, x_hi, gg, device, dtype)
    G = z_final.generators  # (n_out, n_gens)
    K_in = nz.numel()
    G_in = G[:, :K_in]  # input columns come first in make_input_zonotope

    queries = spec.as_linear_queries(int(G.shape[0]))
    if not queries:
        return np.zeros(x_lo.numel()), nz.cpu().numpy()
    score_per_col = torch.zeros(K_in, dtype=dtype, device=device)
    for _, qw, _ in queries:
        qw_t = torch.tensor(qw, dtype=dtype, device=device)
        score_per_col = score_per_col + (qw_t @ G_in).abs()

    score_per_axis = np.zeros(x_lo.numel())
    nz_np = nz.cpu().numpy()
    score_per_axis[nz_np] = score_per_col.cpu().numpy()
    return score_per_axis, nz_np


def _joint_and_infeasible_in_box(linears, xl_np, xh_np):
    """Return True if the system {A_i·x + b_i ≤ 0 for all i, x in box}
    is INFEASIBLE — i.e., the AND-conjunct of unsafe halfspaces does
    NOT intersect the input box.

    `linears` is a list of (A, b) numpy pairs. A small Gurobi LP. For
    tiny input dims (cersyve: 2-D / 4-D inputs) this is <1 ms per leaf.
    """
    import gurobipy as gp
    from .gurobi_util import optimize_checked
    n = len(xl_np)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    x_vars = m.addMVar(n, lb=xl_np, ub=xh_np, name='x')
    for A, b in linears:
        # A·x + b ≤ 0
        m.addConstr(A @ x_vars + b <= 0)
    m.setObjective(0.0)
    optimize_checked(m)
    # Status 3 = INFEASIBLE. Any FEASIBLE status means the joint
    # halfspace AND intersects the box — leaf can't be closed by this.
    is_infeasible = (m.Status == 3)
    m.dispose()
    env.dispose()
    return is_infeasible


def _joint_and_infeasible_triangle_lp(gg_ops_ser, x_lo, x_hi,
                                        bounds_by_relu, qlists_disj,
                                        time_limit=2.0):
    """Full triangle-LP joint feasibility check at a leaf.

    Builds Gurobi LP with:
      - Triangle relaxations for every unstable ReLU
      - Linear pass-through for stable ones
      - Linear layer constraints (Conv/Gemm)
      - Input box [xl, xh]
      - ALL queries of a disjunct as constraints: w_q · y + b_q ≤ 0
    Then checks feasibility. Infeasible → the AND-conjunct is empty in
    the leaf → leaf safe.

    Tighter than CROWN's input-space hyperplane bound because the
    triangle LP allows the ReLU error to be distributed *jointly* across
    constraints, rather than independently per query (the CROWN bound
    integrates the ReLU error per query).

    Returns set of disjunct ids closed.
    """
    import gurobipy as gp
    from .gurobi_util import optimize_checked
    closed = set()
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    try:
        for di, qlist in qlists_disj.items():
            if len(qlist) < 2:
                continue
            m = _build_leaf_triangle_lp(
                env, gg_ops_ser, x_lo, x_hi, bounds_by_relu)
            if m is None:
                continue
            x_vars = m._x_vars
            y_vars = m._y_vars
            n_out = len(y_vars)
            for w_q, b_q in qlist:
                w_q = np.asarray(w_q, dtype=np.float64)
                expr = gp.LinExpr()
                for j in range(n_out):
                    if w_q[j] != 0.0:
                        expr.addTerms(float(w_q[j]), y_vars[j])
                expr.addConstant(float(b_q))
                m.addConstr(expr <= 0)
            m.setObjective(0.0)
            m.Params.TimeLimit = time_limit
            optimize_checked(m)
            if m.Status == 3:  # INFEASIBLE
                closed.add(di)
            m.dispose()
    finally:
        env.dispose()
    return closed


def _build_leaf_triangle_lp(env, gg_ops_ser, x_lo, x_hi, bounds_by_relu):
    """Build a triangle-LP encoding of the network for a single leaf.

    Returns the Gurobi Model with `m._x_vars` (input vars) and
    `m._y_vars` (output vars) stashed, or None if the network has ops
    the simple builder doesn't handle. Skips graph-specific cases (add,
    sub, reshape with non-trivial pass-through) — the caller should
    catch None and fall back.
    """
    import gurobipy as gp
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    x_lo_np = np.asarray(x_lo, dtype=np.float64).flatten()
    x_hi_np = np.asarray(x_hi, dtype=np.float64).flatten()
    n_in = len(x_lo_np)
    x_vars = [m.addVar(lb=x_lo_np[i], ub=x_hi_np[i], name=f'x_{i}')
              for i in range(n_in)]
    # Track per-op output variables.
    var_map = {}
    input_name = gg_ops_ser[0].get('inputs', [None])[0] if gg_ops_ser else None
    # gg_ops_ser ops use op['name'] for output, op['inputs'] for input refs.
    # Find the model input name.
    for op in gg_ops_ser:
        for inp in op.get('inputs', []):
            if inp not in var_map and inp != op.get('name'):
                # Treat any unresolved input as the model input on first sight.
                if input_name is None or inp == input_name:
                    var_map[inp] = x_vars
                    input_name = inp
                    break
        if input_name and input_name in var_map:
            break
    if input_name is None:
        return None
    if input_name not in var_map:
        var_map[input_name] = x_vars

    relu_idx = 0
    for op_i, op in enumerate(gg_ops_ser):
        name = op['name']
        t = op['type']
        if t == 'fc':
            W = op.get('W_np')
            b = op.get('bias_np')
            if W is None or b is None:
                m.dispose()
                return None
            W = np.asarray(W, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64).flatten()
            in_vars = var_map[op['inputs'][0]]
            if len(in_vars) != W.shape[1]:
                m.dispose()
                return None
            out_vars = []
            for i in range(W.shape[0]):
                v = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                              name=f'fc{op_i}_{i}')
                expr = gp.LinExpr()
                for j in range(W.shape[1]):
                    if W[i, j] != 0.0:
                        expr.addTerms(float(W[i, j]), in_vars[j])
                expr.addConstant(float(b[i]))
                m.addConstr(v == expr)
                out_vars.append(v)
            var_map[name] = out_vars
        elif t == 'relu':
            in_vars = var_map[op['inputs'][0]]
            L = op.get('layer_idx')
            if L is None or L not in bounds_by_relu:
                m.dispose()
                return None
            lo_arr, hi_arr = bounds_by_relu[L]
            if len(in_vars) != len(lo_arr):
                m.dispose()
                return None
            out_vars = []
            for j in range(len(in_vars)):
                lo_j = float(lo_arr[j])
                hi_j = float(hi_arr[j])
                if lo_j >= 0:
                    out_vars.append(in_vars[j])
                elif hi_j <= 0:
                    out_vars.append(m.addVar(lb=0, ub=0,
                                              name=f'relu{L}_{j}_zero'))
                else:
                    # Triangle: a >= 0, a >= z, a <= slope·(z - lo)
                    a = m.addVar(lb=0, ub=hi_j, name=f'relu{L}_{j}')
                    m.addConstr(a >= in_vars[j])
                    slope = hi_j / (hi_j - lo_j)
                    m.addConstr(a <= slope * (in_vars[j] - lo_j))
                    out_vars.append(a)
            var_map[name] = out_vars
            relu_idx += 1
        elif t == 'add':
            ins = op.get('inputs', [])
            if len(ins) != 2 or ins[0] not in var_map or ins[1] not in var_map:
                m.dispose()
                return None
            a_vars = var_map[ins[0]]
            b_vars = var_map[ins[1]]
            if len(a_vars) != len(b_vars):
                m.dispose()
                return None
            out_vars = []
            for j in range(len(a_vars)):
                v = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                              name=f'add{op_i}_{j}')
                m.addConstr(v == a_vars[j] + b_vars[j])
                out_vars.append(v)
            var_map[name] = out_vars
        elif t == 'sub':
            in_vars = var_map[op['inputs'][0]]
            bias = op.get('bias')
            if bias is None:
                var_map[name] = in_vars
                continue
            b_arr = np.asarray(bias, dtype=np.float64).flatten()
            if len(in_vars) != len(b_arr):
                m.dispose()
                return None
            out_vars = []
            for j in range(len(in_vars)):
                v = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                              name=f'sub{op_i}_{j}')
                m.addConstr(v == in_vars[j] - float(b_arr[j]))
                out_vars.append(v)
            var_map[name] = out_vars
        elif t == 'reshape':
            var_map[name] = var_map[op['inputs'][0]]
        else:
            # Unsupported op for this minimalist builder.
            m.dispose()
            return None

    out_name = gg_ops_ser[-1]['name']
    if out_name not in var_map:
        m.dispose()
        return None
    m._x_vars = x_vars
    m._y_vars = var_map[out_name]
    return m


def _joint_and_infeasible_zono(z_center_np, z_gens_np, qlists_disj):
    """Joint-AND infeasibility check using the OUTPUT ZONOTOPE.

    The output zonotope Y(e) = c + G·e for e ∈ [-1,1]^k captures more
    correlations between outputs than a single CROWN hyperplane in
    input space (which already integrated out e via |·|_1 over each
    column). For a multi-query AND-conjunct, the unsafe AND region in
    OUTPUT space is empty within the zonotope iff:

      ∄ e ∈ [-1,1]^k :  ∀q ∈ disj  w_q · (c + G·e) + b_q ≤ 0

    i.e., the LP:
      minimize 0
      s.t.    w_q · G · e ≤ -(w_q · c + b_q)   for all q
              -1 ≤ e_i ≤ 1
    is infeasible.

    `qlists_disj` is a dict {disjunct_id: [(w_q_np, b_q_float), ...]}.
    Returns set of disjunct ids closed via this check.

    Compared to `_joint_and_infeasible_in_box` (input-space LP), this
    uses the zonotope's full generator structure — captures dependence
    via internal e variables — so it's strictly tighter on cases where
    CROWN over-approximates the input→output map.
    """
    import gurobipy as gp
    from .gurobi_util import optimize_checked
    if not qlists_disj or z_gens_np.size == 0:
        return set()
    k = z_gens_np.shape[1]
    closed = set()
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    try:
        for di, qlist in qlists_disj.items():
            if len(qlist) < 2:
                continue
            m = gp.Model(env=env)
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            e = m.addMVar(k, lb=-1.0, ub=1.0, name='e')
            for w_q, b_q in qlist:
                wG = (w_q @ z_gens_np).astype(np.float64)
                wc = float(w_q @ z_center_np) + float(b_q)
                # w_q · (c + G·e) + b_q ≤ 0  ⇔  wG · e ≤ -wc
                m.addConstr(wG @ e <= -wc)
            m.setObjective(0.0)
            optimize_checked(m)
            if m.Status == 3:
                closed.add(di)
            m.dispose()
    finally:
        env.dispose()
    return closed


def _input_split_fast_leaf(graph, spec, settings, gg, device, dtype):
    """Minimal per-leaf bound check for input-split BaB.

    Bypasses _run_pipeline (which has 60+ s of overhead from Phase 0
    PGD, multiprocessing pool startup, Phase 8 MILP setup, etc.). For
    a single input box, just runs:
      1. Forward zono (~200-400ms)
      2. CROWN backward to spec (~20-50ms)
      3. α-CROWN with 3 iters if any spec is still open (~3s)

    Returns (verdict, details). Verdict is 'verified' iff every spec
    disjunct has all its queries with lb > 0 — same logic as the full
    pipeline. No PGD, no MILP, no Phase 7/8.

    On cifar_biasfield_0 a leaf takes ~3.4s here vs ~65s via the full
    _run_pipeline path (19× faster).
    """
    from .verify_zono_bnb import _forward_zonotope_graph, _spec_backward_graph
    from . import alpha_crown as ac
    xl = torch.tensor(spec.x_lo.flatten(), device=device, dtype=dtype)
    xh = torch.tensor(spec.x_hi.flatten(), device=device, dtype=dtype)
    # Capture z_final too — caller can reuse it for the BaB axis-score
    # forward pass instead of doing a third redundant forward zono per
    # node. On cifar_biasfield_70 the per-leaf forward zono is ~250 ms;
    # `_score_input_axes` was doing an identical pass per BaB node so
    # ~25 s of every 60 s budget went to redundant kernel launches.
    sb, z_final_leaf = _forward_zonotope_graph(
        xl, xh, gg, device, dtype, settings=settings)
    bbr = {L: (lo.cpu().numpy().astype(np.float64),
                hi.cpu().numpy().astype(np.float64))
           for L, (lo, hi) in sb.items()}

    # Output size — match the existing _run_pipeline logic: pull from
    # the LAST fc/conv op (n_out for conv = filter count = class count
    # before reshape).
    n_output = None
    for op in reversed(gg['ops']):
        if op.get('type') == 'fc':
            W = op.get('W_np') if 'W_np' in op else op.get('W')
            if W is not None:
                n_output = int(W.shape[0]); break
        if op.get('type') == 'conv':
            n_output = int(op.get('n_out', 0))
            if n_output > 0: break
    if n_output is None:
        return 'unknown', {'phase': 'fast_leaf_no_output'}
    queries = spec.as_linear_queries(n_output)
    if not queries:
        return 'verified', {'phase': 'fast_leaf_no_queries'}
    disj_queries = {}
    for qi, (di, w, b) in enumerate(queries):
        disj_queries.setdefault(di, []).append((qi, w, b))
    w_qs = np.stack([np.asarray(q[1], dtype=np.float64)
                      for q in queries])
    b_qs = np.array([float(q[2]) for q in queries], dtype=np.float64)

    spec_ew = {qi: (torch.as_tensor(w, dtype=dtype, device=device),
                      float(b))
                for qi, (w, b) in enumerate(zip(w_qs, b_qs))}
    spec_lbs, _ = _spec_backward_graph(
        sb, xl, xh, gg, spec_ew, list(range(len(queries))),
        len(sb), device, dtype)

    def _all_verified():
        # A conjunct (AND of queries) is safe iff ANY single query has
        # lb > 0 — because the unsafe region requires ALL constraints to
        # hold, and one provably-violated constraint makes the unsafe
        # region unreachable. The old code required ALL queries to have
        # lb > 0 — wrong semantics for AND; on cersyve made every leaf
        # return 'unknown' even on a point-width input box. Mirrors the
        # Conjunct.margin fix in spec.py.
        return all(
            any(spec_lbs.get(qi, -1.0) > 0 for qi, _, _ in qlist)
            for qlist in disj_queries.values())

    if _all_verified():
        return 'verified', {'phase': 'fast_leaf_crown',
                            'spec_lbs': dict(spec_lbs)}

    # AND-conjunct JOINT input-space LP check. For multi-query disjuncts
    # (cersyve), per-query CROWN often can't close because the AND-region
    # in OUTPUT space (Y_0 ≤ 0 AND Y_1 ≥ 0) is non-empty BUT its preimage
    # in the LEAF's input box is empty. CROWN gives linear lower-bound
    # coefficients in input space for each query:
    #   for x in leaf:  spec_q(x) >= A_q·x + b_q
    # The AND-region is INFEASIBLE iff no x in leaf satisfies BOTH
    # A_0·x + b_0 ≤ 0 AND A_1·x + b_1 ≤ 0 — a small LP feasibility on
    # the input box. For 2-D / 4-D inputs (cersyve) this is trivial.
    # Catches cases that single-query CROWN and λ-combo CROWN both
    # miss because the joint infeasibility comes from CURVED separation
    # in input space, not a single hyperplane. Disabled by default.
    if (bool(getattr(settings, 'input_split_leaf_joint_input_lp', False))
            and any(len(ql) > 1 for ql in disj_queries.values())):
        _, _, input_linear = _spec_backward_graph(
            sb, xl, xh, gg, spec_ew, list(range(len(queries))),
            len(sb), device, dtype, return_input_linear=True)
        xl_np = xl.detach().cpu().numpy().astype(np.float64)
        xh_np = xh.detach().cpu().numpy().astype(np.float64)
        closed_via_joint = set()
        for di, qlist in disj_queries.items():
            if len(qlist) < 2:
                continue
            # Build LP: find x in [xl, xh] with A_q·x + b_q ≤ 0 for
            # ALL q in this disjunct. If infeasible, conjunct is
            # UNSAT in this leaf.
            if _joint_and_infeasible_in_box(
                    [input_linear[qi] for qi, _, _ in qlist],
                    xl_np, xh_np):
                closed_via_joint.add(di)
        # Stronger fallback: when CROWN-input-space LP doesn't close a
        # disjunct, retry with the OUTPUT ZONOTOPE's joint feasibility
        # LP. Captures correlations between outputs that the single-
        # hyperplane CROWN bound integrates away. Tighter on boundary
        # leaves of UNSAT cases where CROWN's input-space bound is
        # one-hyperplane-too-loose. ~1-2ms extra per leaf for tiny
        # zonotopes.
        if bool(getattr(settings, 'input_split_leaf_joint_zono_lp', True)):
            remaining = [di for di, qlist in disj_queries.items()
                          if len(qlist) > 1 and di not in closed_via_joint]
            if remaining:
                z_c = z_final_leaf.center.detach().cpu().numpy().astype(
                    np.float64).flatten()
                _G = z_final_leaf.generators
                z_G = (_G.detach().cpu().numpy().astype(np.float64)
                        if _G.numel() > 0
                        else np.zeros((z_c.size, 0), dtype=np.float64))
                if z_G.size > 0 and z_G.shape[0] != z_c.size:
                    z_G = z_G.reshape(z_c.size, -1)
                qlists_for_zono = {
                    di: [(w_qs[qi], float(b_qs[qi]))
                          for qi, _, _ in disj_queries[di]]
                    for di in remaining}
                closed_via_joint |= _joint_and_infeasible_zono(
                    z_c, z_G, qlists_for_zono)
        # Strongest fallback: full per-leaf triangle-LP that builds the
        # entire network's LP relaxation with BOTH unsafe constraints
        # added jointly. Captures correlations across the network that
        # neither CROWN linearization nor the output zonotope can. For
        # tiny cersyve networks (~10 ReLU, 64 neurons total) the LP is
        # ~5-20 ms per leaf. Only used when prior checks fail. Opt-in
        # via `input_split_leaf_joint_triangle_lp`.
        if bool(getattr(settings,
                          'input_split_leaf_joint_triangle_lp', False)):
            remaining_tri = [di for di, qlist in disj_queries.items()
                              if len(qlist) > 1 and di not in closed_via_joint]
            if remaining_tri:
                qlists_for_tri = {
                    di: [(w_qs[qi], float(b_qs[qi]))
                          for qi, _, _ in disj_queries[di]]
                    for di in remaining_tri}
                try:
                    closed_via_joint |= _joint_and_infeasible_triangle_lp(
                        gg['ops'], spec.x_lo, spec.x_hi, bbr,
                        qlists_for_tri)
                except Exception:
                    pass
        if closed_via_joint:
            # Mark first query of each closed disjunct as positive
            # so `_all_verified` passes.
            for di in closed_via_joint:
                first_qi = disj_queries[di][0][0]
                spec_lbs[first_qi] = max(spec_lbs.get(first_qi, -1e9), 1.0)
            if _all_verified():
                return 'verified', {'phase': 'fast_leaf_joint_input_lp',
                                     'spec_lbs': dict(spec_lbs),
                                     'closed_via_joint': list(closed_via_joint)}

    # AND-conjunct joint check via λ-combo CROWN. For a multi-query
    # conjunct, per-query lbs may all be ≤ 0 even on an UNSAT leaf
    # because closing it requires JOINT reasoning over the queries.
    # Concretely: if (w_a·y > 0) AND (w_b·y > 0) is the SAFE-side
    # encoding, the unsafe AND-region is empty iff some convex combo
    # `λ·w_a + (1-λ)·w_b` has lb > 0 — because any unsafe point has
    # both w_a·y ≤ 0 and w_b·y ≤ 0, hence λ·w_a·y + (1-λ)·w_b·y ≤ 0
    # for any λ ∈ [0,1], contradicting lb > 0. Cheap: one extra CROWN
    # backward per (λ × disjunct), grid-searched. Disabled by default;
    # `input_split_leaf_joint_lambdas` enables it. Setting `[0.5]`
    # already covers most cersyve UNSAT leaves at +1 backward pass.
    _lambdas = getattr(settings, 'input_split_leaf_joint_lambdas', None)
    if (_lambdas
            and any(len(ql) > 1 for ql in disj_queries.values())):
        # For each multi-query disjunct, try linear combos.
        joint_qid = max(spec_ew.keys()) + 1  # synthetic qid for combos
        joint_ew = {}
        joint_for_disj = {}  # synth_qid → disjunct id
        for di, qlist in disj_queries.items():
            if len(qlist) < 2:
                continue
            for lam in _lambdas:
                w_combo = np.zeros_like(w_qs[qlist[0][0]])
                b_combo = 0.0
                # 2-query: λ·q0 + (1-λ)·q1. For ≥3 queries, evenly
                # distribute (1-λ) across remaining queries.
                wts = [lam] + [(1 - lam) / (len(qlist) - 1)] * (len(qlist) - 1)
                for wt, (qi, w, b) in zip(wts, qlist):
                    w_combo += wt * w_qs[qi]
                    b_combo += wt * b_qs[qi]
                joint_ew[joint_qid] = (
                    torch.as_tensor(w_combo, dtype=dtype, device=device),
                    float(b_combo))
                joint_for_disj[joint_qid] = di
                joint_qid += 1
        if joint_ew:
            joint_lbs, _ = _spec_backward_graph(
                sb, xl, xh, gg, joint_ew, list(joint_ew.keys()),
                len(sb), device, dtype)
            # Mark a disjunct closed if any λ-combo gives lb > 0.
            closed_disj = {di for sqid, di in joint_for_disj.items()
                            if joint_lbs.get(sqid, -1.0) > 0}
            if closed_disj:
                # Synthesize a positive lb on the FIRST query of each
                # closed disjunct so `_all_verified` sees it.
                for di in closed_disj:
                    first_qi = disj_queries[di][0][0]
                    spec_lbs[first_qi] = max(
                        spec_lbs.get(first_qi, -1e9),
                        max(joint_lbs[sqid] for sqid, _di in
                             joint_for_disj.items() if _di == di))
                if _all_verified():
                    return 'verified', {'phase': 'fast_leaf_joint_crown',
                                         'spec_lbs': dict(spec_lbs)}

    # α-CROWN tightening — only run if it can fit in memory.
    #
    # joint α-CROWN (`run_alpha_crown_batched`) keeps a backward graph
    # per (start_node, layer) pair × per Adam iter. On cifar_biasfield
    # leaves with ~7000 unstable across 7 hidden layers, the peak
    # working set exceeds the 10 GB RTX 3080. Pre-2026-05-10 we ran α
    # speculatively and caught the resulting OOM with `pass` — that
    # masked real memory regressions and burned ~3 s per leaf on
    # quickly-swallowed errors.
    #
    # New strategy: count total_unstable upfront. If it exceeds
    # `alpha_crown_impl_auto_switch_threshold`, the joint path can't
    # fit. Fall back to `run_alpha_crown_fixed_intermediate_batched`
    # (sparse_alpha=True), which only optimises spec α (no per-(S, L)
    # tensors) and has no compounding backward-graph cost. If even
    # the lightweight variant fails, the OOM propagates so the user
    # sees the regression.
    isn = [Lk for Lk in bbr if Lk > 0 and
            ((bbr[Lk][0] < 0) & (bbr[Lk][1] > 0)).any()]
    un = {Lk: np.where((bbr[Lk][0] < 0) & (bbr[Lk][1] > 0))[0].tolist()
          for Lk in bbr}
    if isn and any(un.values()):
        n_iters = int(getattr(settings,
                                'input_split_alpha_iters', 3))
        # `... or 10**9` would WRONGLY treat 0 as "no cap" because
        # `0 or 10**9 == 10**9` — explicit `is None` check instead.
        _thr_raw = getattr(
            settings, 'alpha_crown_impl_auto_switch_threshold', 5000)
        switch_thr = 10**9 if _thr_raw is None else int(_thr_raw)
        total_unstable = sum(len(un[L]) for L in un)
        if total_unstable <= switch_thr:
            # Truncate `intermediate_start_nodes` to the deepest
            # `input_split_alpha_max_start_nodes` layers. The joint
            # α-CROWN cost grows roughly linearly in `len(isn)` because
            # each start node spawns a backward chain through every
            # earlier layer (per-iter conv-transpose count is
            # ~Σ|chunks(S)|). Profiling cifar_biasfield_70: full isn
            # (6 layers) takes 800 ms and closes 4/9 specs; last 2
            # layers take 491 ms and close 3/9. The split itself is
            # already shrinking the input box per BaB iter, so we
            # don't actually need joint α to RE-tighten every
            # intermediate — only the deepest few layers (which
            # accumulate the most spec-direction error) materially
            # help. Default 0 = use ALL start nodes (legacy
            # behaviour). Setting e.g. 2 gives ~40 % per-leaf
            # speedup on cifar_biasfield at the cost of
            # marginally looser bounds; the BaB compensates by
            # splitting one more level deeper.
            _max_isn = int(getattr(
                settings, 'input_split_alpha_max_start_nodes', 0))
            if _max_isn > 0 and len(isn) > _max_isn:
                isn = isn[-_max_isn:]
            best_lbs, _, _, _ = ac.run_alpha_crown_batched(
                gg, xl, xh, bbr, w_qs, b_qs, isn, un,
                device, dtype, n_iters=n_iters, lr=0.25, lr_decay=0.98,
                early_stop_on_positive=True)
            for qi, lb in enumerate(best_lbs):
                spec_lbs[qi] = max(spec_lbs.get(qi, -1e9), float(lb))
        else:
            # Lightweight fallback: spec-only α, fixed intermediate
            # bounds, sparse α tensors. Doesn't tighten intermediates,
            # so the leaf often returns 'unknown' and BaB splits
            # further — but it costs ~100 ms instead of ~3 s of
            # OOM-attempt-and-discard.
            best_lbs, _, _, _ = ac.run_alpha_crown_fixed_intermediate_batched(
                gg, xl, xh, bbr, w_qs, b_qs,
                device, dtype, n_iters=n_iters, lr=0.25, lr_decay=0.98,
                early_stop_on_positive=True, sparse_alpha=True)
            for qi, lb in enumerate(best_lbs):
                spec_lbs[qi] = max(spec_lbs.get(qi, -1e9), float(lb))

    if _all_verified():
        del sb, spec_ew
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 'verified', {'phase': 'fast_leaf_alpha',
                            'spec_lbs': dict(spec_lbs)}

    # Compute the axis-score from z_final_leaf BEFORE releasing GPU
    # tensors. This is the same quantity `_score_input_axes` would
    # compute via a *third* forward zono in `_bab` — by stashing it in
    # the details dict the caller can reuse our forward pass and avoid
    # ~250 ms of redundant work per BaB node on cifar_biasfield-class
    # graphs (~25 s saved across the 60 s budget).
    score_per_axis = None
    try:
        radii = (xh - xl) / 2
        nz = torch.nonzero(radii).squeeze(1)
        if nz.numel() > 0:
            G = z_final_leaf.generators  # (n_out, n_gens)
            K_in = nz.numel()
            G_in = G[:, :K_in]
            score_per_col = torch.zeros(K_in, dtype=dtype, device=device)
            for w, _ in spec_ew.values():
                score_per_col = score_per_col + (w @ G_in).abs()
            score_per_axis = np.zeros(xl.numel())
            score_per_axis[nz.cpu().numpy()] = score_per_col.cpu().numpy()
    except Exception:
        score_per_axis = None  # caller falls back to width-based score

    # Release GPU tensors held by sb / spec_ew before next leaf.
    # On deep BaB trees (15+ leaves on cifar_biasfield) the cached
    # allocator fills up if we don't drop refs explicitly.
    del sb, spec_ew, z_final_leaf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 'unknown', {'phase': 'fast_leaf_open',
                       'spec_lbs': dict(spec_lbs),
                       'axis_scores': score_per_axis}


# Module-level worker state: persistent Gurobi env per multiprocessing
# worker (init once, reuse across many LP solves). Reduces per-LP cost
# from ~5 ms (env init dominates) to ~0.2 ms (LP-only).
_GRB_WORKER_ENV = None


def _grb_worker_init():
    """Init a persistent Gurobi env in this worker process."""
    global _GRB_WORKER_ENV
    import gurobipy as gp
    _GRB_WORKER_ENV = gp.Env(empty=True)
    _GRB_WORKER_ENV.setParam('OutputFlag', 0)
    _GRB_WORKER_ENV.start()


def _grb_lp_clip_worker(args):
    """Per-leaf full-LP clipping: tightest box-projection bounds on each
    input dim, OR return infeasible.

    Args:
      xl, xh: (n_in,) numpy float64 — input box
      A, b: (Q, n_in) and (Q,) — query linear-lb coeffs

    Returns:
      (xl_new, xh_new, feasible):
        feasible=False → polytope ∩ box empty → leaf verified
        else: tight projection bounds (xl_new ≥ xl, xh_new ≤ xh)
    """
    import gurobipy as gp
    global _GRB_WORKER_ENV
    if _GRB_WORKER_ENV is None:
        _grb_worker_init()
    xl, xh, A, b = args
    n_in = len(xl)
    Q = A.shape[0]
    m = gp.Model(env=_GRB_WORKER_ENV)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.Method = 0  # primal simplex — small problems, fast warm-start
    x_vars = m.addMVar(n_in, lb=xl, ub=xh)
    for q in range(Q):
        m.addConstr(A[q] @ x_vars + float(b[q]) <= 0)
    # Quick infeasibility check first: solve with no objective.
    m.setObjective(0)
    m.optimize()
    if m.Status == 3:  # INFEASIBLE
        m.dispose()
        return (xl, xh, False)
    # Feasible: solve 2 LPs per dim for tight projection bounds.
    xl_new = xl.copy()
    xh_new = xh.copy()
    for i in range(n_in):
        m.setObjective(x_vars[i], gp.GRB.MINIMIZE)
        m.optimize()
        if m.Status == 2:
            xl_new[i] = max(xl_new[i], m.ObjBound)
        m.setObjective(x_vars[i], gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == 2:
            xh_new[i] = min(xh_new[i], m.ObjBound)
    m.dispose()
    return (xl_new, xh_new, True)


def _clip_box_by_full_lp_batched(xl, xh, A, b, pool=None):
    """Parallel full-LP clipping across leaves via multiprocessing pool.

    Strictly tighter than `_clip_box_by_halfspaces_batched` (uses joint
    constraints in one LP per leaf), at the cost of CPU-bound LP solves.
    For 5-D input / 4-query constraints (acasxu prop_2): ~0.5-1 ms per
    LP × 11 LPs per leaf (1 feasibility + 2×n_in projection) = ~10 ms
    per leaf serial; with `pool` of N workers, ~10 ms × B / N parallel.

    Args:
      xl, xh: (B, n_in) torch tensors
      A: (B, Q, n_in), b: (B, Q) torch tensors
      pool: optional `multiprocessing.Pool` (with `_grb_worker_init`
        initializer for persistent Gurobi env). If None, runs serial
        on the calling process.

    Returns:
      xl_new, xh_new, feasible: same contract as the per-halfspace
      variant.
    """
    import torch as _t
    B = xl.shape[0]
    xl_np = xl.detach().cpu().numpy().astype(np.float64)
    xh_np = xh.detach().cpu().numpy().astype(np.float64)
    A_np = A.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    args_list = [(xl_np[i], xh_np[i], A_np[i], b_np[i]) for i in range(B)]
    if pool is not None:
        results = pool.map(_grb_lp_clip_worker, args_list,
                            chunksize=max(1, B // (4 * pool._processes)))
    else:
        results = [_grb_lp_clip_worker(a) for a in args_list]
    xl_out = np.stack([r[0] for r in results])
    xh_out = np.stack([r[1] for r in results])
    feasible = np.array([r[2] for r in results], dtype=bool)
    device = xl.device
    return (_t.tensor(xl_out, dtype=xl.dtype, device=device),
            _t.tensor(xh_out, dtype=xh.dtype, device=device),
            _t.tensor(feasible, dtype=_t.bool, device=device))


def _clip_box_by_halfspaces_batched(xl, xh, A, b):
    """Sound bounding-box clip of `{x in [xl, xh] : A_q·x + b_q ≤ 0 for all q}`.

    Inputs:
      xl, xh: (B, n_in)
      A: (B, Q, n_in), b: (B, Q) — per-query linear lower-bound coeffs
        from CROWN backward. The query is interpreted as a SAFE-side
        constraint `A_q·x + b_q ≤ 0` representing the possibly-unsafe
        region for that query.

    Returns:
      xl_new, xh_new: (B, n_in) — tightened bounds (≥ xl, ≤ xh).
      feasible: (B,) bool — False means the polytope ∩ box is empty
        in this leaf, so the leaf is SAFE (verified by clipping).

    Method: per (query, dim) independently. For halfspace `A·x ≤ -b`
    with box [xl, xh], the projection of the polytope onto x_i is:
      x_i ≤ (-b - min_other A·x_other) / A_i      if A_i > 0
      x_i ≥ (-b - min_other A·x_other) / A_i      if A_i < 0
      x_i unconstrained                            if A_i == 0
    where `min_other A·x_other = pos(A[~i])·xl[~i] + neg(A[~i])·xh[~i]`.
    Take the intersection across queries (tightest ub, tightest lb per
    dim). Approximation: the true polytope bounding box can be tighter
    by joint LP across queries; we trade tightness for ~free cost.
    """
    B, Q, n_in = A.shape
    A_pos = A.clamp(min=0)
    A_neg = A.clamp(max=0)
    # total_min[b,q] = pos(A[b,q]) · xl[b] + neg(A[b,q]) · xh[b]   (B, Q)
    total_min = ((A_pos * xl.unsqueeze(1)).sum(dim=-1)
                  + (A_neg * xh.unsqueeze(1)).sum(dim=-1))
    # contrib[b,q,i] = i-th term of total_min                       (B, Q, n_in)
    contrib = A_pos * xl.unsqueeze(1) + A_neg * xh.unsqueeze(1)
    L_other = total_min.unsqueeze(-1) - contrib  # min_{x_~i in box} A·x_~i
    # threshold[b,q,i] = (-b[b,q] - L_other[b,q,i]) / A[b,q,i]
    A_nz = A.abs() > 1e-12
    A_safe = torch.where(A_nz, A, torch.ones_like(A))
    threshold = (-b.unsqueeze(-1) - L_other) / A_safe  # (B, Q, n_in)

    pos_mask = (A > 1e-12)
    neg_mask = (A < -1e-12)

    # Per-query candidate ub/lb. For positions where the query doesn't
    # contribute (A==0, or wrong sign), use the original bound — so the
    # cross-query min/max effectively ignores those queries on that dim.
    xh_orig = xh.unsqueeze(1).expand(B, Q, n_in)
    xl_orig = xl.unsqueeze(1).expand(B, Q, n_in)
    xh_per_q = torch.where(pos_mask, threshold, xh_orig)
    xl_per_q = torch.where(neg_mask, threshold, xl_orig)

    xh_new = xh_per_q.min(dim=1).values  # tightest UB across queries
    xl_new = xl_per_q.max(dim=1).values  # tightest LB
    xh_new = torch.minimum(xh_new, xh)
    xl_new = torch.maximum(xl_new, xl)

    # Feasibility: tighter check via per-query halfspace feasibility too —
    # if for some q, total_min + b > 0, polytope ∩ that halfspace empty.
    # (Equivalent to "spec_lb_q > 0" — caller already screens those, but
    # box-projection slack can hide it; cheap to recheck.)
    per_q_infeasible = (total_min + b > 0)
    halfspace_empty = per_q_infeasible.any(dim=1)
    box_empty = (xl_new > xh_new).any(dim=-1)
    feasible = ~(halfspace_empty | box_empty)
    return xl_new, xh_new, feasible


def _milp_escalate_worker(args):
    """Per-leaf MILP escalation: for each disjunct, try queries in
    best-CROWN-lb order; first MILP-UNSAT closes that disjunct. Leaf
    closes iff every disjunct closes.
    """
    (ci, gg_ops_ser, xl_l_np, xh_l_np, bbr_l, last_op_name,
     per_disj_queries, per_query_tl) = args
    import torch as _torch
    from . import verify_gen_lp as _glp
    milp_set = set()
    for L in bbr_l:
        if L > 0:
            lo, hi = bbr_l[L]
            for j in np.where((lo < 0) & (hi > 0))[0]:
                milp_set.add((L, int(j)))
    for ordered_queries in per_disj_queries:
        disj_closed = False
        for qw_np, qb_v in ordered_queries:
            try:
                res, _, _ = _glp.solve_spec(
                    gg_ops_ser, xl_l_np, xh_l_np, bbr_l, 'input',
                    last_op_name, qw_np, qb_v, device='cpu',
                    dtype=_torch.float64, time_limit=per_query_tl,
                    n_threads=1, milp_set=milp_set)
                if res == 'UNSAT':
                    disj_closed = True
                    break
            except Exception:
                pass
        if not disj_closed:
            return (ci, False)
    return (ci, True)


def _input_split_batched(graph, spec, settings, gg, device, dtype):
    """Worklist-based input-split BaB with batched leaf evaluation.

    Each iteration pops up to `input_split_batch_size` boxes from the
    worklist, stacks them into (B, n_in) tensors, runs ONE batched
    forward zono + spec backward, identifies verified vs unclosed
    leaves, splits unclosed leaves on widest axis, pushes children
    back. AB-CROWN-style; mirrors `_input_split_fast_leaf` semantics
    but vectorized.

    Closure semantics: a disjunct is verified at a leaf iff ANY of its
    queries has lb > 0 (AND-conjunct: one provably-safe operand closes).

    No per-leaf joint LP, no α-CROWN, no MILP — pure batched per-query
    CROWN. The order-of-magnitude throughput gain (1000s of leaves /
    second vs ~30 leaves / second sequential) compensates by going
    deeper. For 4-D cersyve `_finetune_inv` cases the throughput jump
    is the whole point.
    """
    from .verify_zono_bnb import (
        _forward_zonotope_graph_batched, _spec_backward_graph_batched)
    t_start = time.perf_counter()
    total_budget = float(getattr(settings, 'total_timeout', 60.0))
    batch_size = int(getattr(settings, 'input_split_batch_size', 4096))
    # Persistent multiprocessing pool for full-LP clipping (one Gurobi
    # env per worker, reused across all iterations). Init cost ~50 ms
    # once vs ~5 ms per LP forever; pays off after ~10 LPs per worker.
    _lp_pool = None
    if bool(getattr(settings, 'input_split_batched_clip_full_lp', False)):
        import multiprocessing as _mp
        _w = getattr(settings, 'input_split_batched_clip_lp_workers', None)
        # DotMap can return empty DotMap for missing keys.
        if not isinstance(_w, (int, float)) or _w is None or _w <= 0:
            _n_lp_workers = max(1, _mp.cpu_count() - 1)
        else:
            _n_lp_workers = int(_w)
        _lp_pool = _mp.Pool(_n_lp_workers, initializer=_grb_worker_init)
    try:
        return _input_split_batched_inner(
            graph, spec, settings, gg, device, dtype, _lp_pool, t_start)
    finally:
        if _lp_pool is not None:
            _lp_pool.close(); _lp_pool.join()


def _input_split_batched_inner(graph, spec, settings, gg, device, dtype,
                                  _lp_pool, t_start):
    """Inner driver — split out so pool cleanup happens via try/finally."""
    from .verify_zono_bnb import (
        _forward_zonotope_graph_batched, _spec_backward_graph_batched)
    total_budget = float(getattr(settings, 'total_timeout', 60.0))
    batch_size = int(getattr(settings, 'input_split_batch_size', 4096))
    max_worklist = int(getattr(
        settings, 'input_split_batched_max_worklist', 200_000))
    clip_enabled = bool(getattr(
        settings, 'input_split_batched_clip_enabled', True))
    # Serialize ops once (for MILP escalation; cheap, no torch tensors).
    gg_ops_ser = []
    for op in gg['ops']:
        d = {'name': op['name'], 'type': op['type'],
             'inputs': op['inputs']}
        if op['type'] == 'fc':
            d['W_np'] = op['W_np']; d['bias_np'] = op['bias_np']
        elif op['type'] == 'relu' and 'layer_idx' in op:
            d['layer_idx'] = op['layer_idx']
        elif op['type'] == 'add':
            d['is_merge'] = op.get('is_merge', False)
            d['bias'] = op.get('bias')
        elif op['type'] == 'sub':
            d['bias'] = op.get('bias')
        gg_ops_ser.append(d)

    # Root-level PGD attack — same as `_input_split_verify`.
    if (not bool(getattr(settings, 'disable_sat_finding', False))
            and bool(getattr(settings, 'pgd_phase0_enabled', True))):
        try:
            xl_pgd = torch.tensor(spec.x_lo, dtype=dtype, device=device)
            xh_pgd = torch.tensor(spec.x_hi, dtype=dtype, device=device)
            pgd_budget = float(getattr(
                settings, 'pgd_time_budget_phase0', 5.0))
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg, settings, time_budget=pgd_budget)
            if pgd_sat and pgd_witness is not None:
                return 'sat', {'phase': 'batched_pgd', 'witness': pgd_witness}
        except Exception:
            pass
        # Also try the simpler multi-disjunct-aware PGD (verify_hybrid_acasxu
        # `_simple_pgd`): straight sign-gradient over the input box, no
        # OSI/multi-α, 10K restarts × 50 iters. Catches multi-disjunct DNF
        # SAT cases that _pgd_attack_general misses on cgan because the
        # latter doesn't reduce loss across disjuncts.
        try:
            from .verify_hybrid_acasxu import _simple_pgd
            n_out_pgd = None
            for op in reversed(gg['ops']):
                if op.get('type') == 'fc':
                    n_out_pgd = int(op['W'].shape[0]); break
                if op.get('type') in ('conv', 'conv_transpose'):
                    n_out_pgd = op['n_out']; break
            if n_out_pgd is not None:
                xl_pgd2 = xl_pgd.flatten().unsqueeze(0)
                xh_pgd2 = xh_pgd.flatten().unsqueeze(0)
                sat_simple, w_simple = _simple_pgd(
                    xl_pgd2, xh_pgd2, spec, gg, n_out_pgd, device, dtype,
                    n_restarts=10000, n_iter=50)
                if sat_simple:
                    return 'sat', {'phase': 'batched_simple_pgd',
                                    'witness': w_simple}
        except Exception:
            pass
        # Last-resort: PGD via raw-ONNX interpreter. Catches models the
        # gpu_graph forward can't handle (transformer attention, etc.).
        try:
            _onnx_p = getattr(graph, 'onnx_path', None)
            if _onnx_p is not None:
                from .onnx_torch_runner import pgd_via_onnx
                sat_or, w_or = pgd_via_onnx(
                    _onnx_p, spec,
                    n_restarts=int(settings.pgd_phase0_restarts
                        if 'pgd_phase0_restarts' in settings else 256),
                    n_iter=int(settings.pgd_phase0_iters
                        if 'pgd_phase0_iters' in settings else 100))
                if sat_or:
                    return 'sat', {'phase': 'batched_onnx_pgd',
                                    'witness': w_or}
        except Exception:
            pass

    # Find n_output (mirror `_input_split_fast_leaf`).
    n_output = None
    for op in reversed(gg['ops']):
        if op.get('type') == 'fc':
            W = op.get('W_np') if 'W_np' in op else op.get('W')
            if W is not None:
                n_output = int(W.shape[0]); break
        if op.get('type') == 'conv':
            n_output = int(op.get('n_out', 0))
            if n_output > 0:
                break
    if n_output is None:
        return 'unknown', {'phase': 'batched_no_output'}

    queries = spec.as_linear_queries(n_output)
    if not queries:
        return 'verified', {'phase': 'batched_no_queries'}
    disj_queries = {}
    for qi, (di, w, b) in enumerate(queries):
        disj_queries.setdefault(di, []).append((qi, w, b))
    spec_ew = {qi: (torch.as_tensor(w, dtype=dtype, device=device).flatten(),
                      float(b))
                for qi, (di, w, b) in enumerate(queries)}
    qids_sorted = sorted(spec_ew.keys())
    # Index of each query within the sorted batch order.
    q_index = {qi: i for i, qi in enumerate(qids_sorted)}
    # Pre-build per-disjunct query-index sets for the batched closure.
    disj_q_idx = {di: [q_index[qi] for qi, _, _ in qlist]
                   for di, qlist in disj_queries.items()}

    # Build per-disjunct query-index tensors for multi-disjunct clipping.
    # Single-disjunct case: clip via AND of halfspaces directly.
    # Multi-disjunct: clip each disjunct's halfspaces separately and
    # take the union's bounding box.
    disj_q_idx_tensors = {
        di: torch.tensor(qixs, device=device, dtype=torch.long)
        for di, qixs in disj_q_idx.items()}
    multi_disjunct = len(disj_q_idx) > 1

    # Initialize worklist with the root box.
    xl0 = torch.as_tensor(spec.x_lo, dtype=dtype, device=device).flatten()
    xh0 = torch.as_tensor(spec.x_hi, dtype=dtype, device=device).flatten()
    n_in = xl0.numel()
    worklist_xl = [xl0]
    worklist_xh = [xh0]

    n_leaves_visited = 0
    n_iters = 0
    n_open_at_timeout = 0

    while worklist_xl:
        if time.perf_counter() - t_start > total_budget - 0.5:
            n_open_at_timeout = len(worklist_xl)
            break
        # Pop a batch (LIFO — depth-first; cheap as Python list ops).
        B = min(batch_size, len(worklist_xl))
        xl_list = worklist_xl[-B:]
        xh_list = worklist_xh[-B:]
        del worklist_xl[-B:]; del worklist_xh[-B:]
        xl_batch = torch.stack(xl_list)  # (B, n_in)
        xh_batch = torch.stack(xh_list)

        # Batched bound.
        try:
            sb_b, _ = _forward_zonotope_graph_batched(
                xl_batch, xh_batch, gg, device, dtype)
            if clip_enabled:
                spec_lbs_b, A_lin, b_lin = _spec_backward_graph_batched(
                    sb_b, xl_batch, xh_batch, gg, spec_ew, device, dtype,
                    return_input_linear=True)
            else:
                spec_lbs_b = _spec_backward_graph_batched(
                    sb_b, xl_batch, xh_batch, gg, spec_ew, device, dtype)
                A_lin, b_lin = None, None
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
            import os
            if os.environ.get('VIBECHECK_DEBUG'):
                import traceback
                traceback.print_exc()
                print(f'  [debug] caught {type(_e).__name__} at batch_size={batch_size}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_bs = max(1, batch_size // 2)
            if new_bs == batch_size:
                return 'unknown', {'phase': 'batched_oom',
                                    'batch_size': batch_size}
            worklist_xl.extend(xl_list); worklist_xh.extend(xh_list)
            batch_size = new_bs
            continue

        # Per-batch closure: every disjunct must have at least one
        # query with lb > 0.
        disj_closed_masks = []
        for di, q_idxs in disj_q_idx.items():
            disj_closed_masks.append(
                (spec_lbs_b[:, q_idxs] > 0).any(dim=1))
        if disj_closed_masks:
            all_disj_closed = torch.stack(disj_closed_masks, dim=1).all(dim=1)
        else:
            all_disj_closed = torch.ones(B, dtype=torch.bool, device=device)

        # Selective α-CROWN on boundary leaves: per-query CROWN gives
        # spec_lbs that plateau just below 0 (e.g., -0.018 on
        # `1_8 prop_2`). α-CROWN with a few iters can push them
        # positive, closing the leaf without splitting. Running it on
        # ALL leaves would dominate cost; instead, target only the
        # leaves where some query's lb is within `boundary_eps` of 0.
        # Serial per leaf (~5-20 ms each). Win condition: closes
        # enough boundary leaves to cancel the BaB explosion.
        alpha_eps = float(getattr(
            settings, 'input_split_batched_alpha_boundary_eps', 0.0))
        alpha_max_iters = int(getattr(
            settings, 'input_split_batched_alpha_iters', 0))
        alpha_max_leaves = int(getattr(
            settings, 'input_split_batched_alpha_max_leaves', 200))
        if alpha_eps > 0 and alpha_max_iters > 0 and not all_disj_closed.all():
            from .verify_zono_bnb import _run_alpha_crown_inputsplit_batched
            best_per_disj_lb = []
            for di, q_idxs in disj_q_idx.items():
                best_per_disj_lb.append(spec_lbs_b[:, q_idxs].max(dim=1).values)
            worst_disj_best = torch.stack(best_per_disj_lb, dim=1).min(dim=1).values
            close_mask = (~all_disj_closed) & (worst_disj_best > -alpha_eps)
            close_idx = close_mask.nonzero(as_tuple=True)[0]
            if close_idx.numel() > 0:
                order = worst_disj_best[close_idx].argsort(descending=True)
                close_idx = close_idx[order[:alpha_max_leaves]]
                # BATCHED α-CROWN on the boundary leaves' boxes. The
                # whole batch is optimized on GPU in one Adam loop —
                # massively faster than per-leaf serial. Per-iter cost
                # is one forward zono + one batched spec backward
                # (~5-20 ms for B=200) vs ~10-50 ms per leaf serial.
                xl_close = xl_batch[close_idx]
                xh_close = xh_batch[close_idx]
                try:
                    new_lbs_batch = _run_alpha_crown_inputsplit_batched(
                        xl_close, xh_close, gg, spec_ew, device, dtype,
                        n_iters=alpha_max_iters, lr=0.25, lr_decay=0.98)
                    # Scatter back: spec_lbs_b[close_idx] = max(old, new)
                    spec_lbs_b[close_idx] = torch.maximum(
                        spec_lbs_b[close_idx], new_lbs_batch)
                except Exception:
                    pass
                disj_closed_masks = []
                for di, q_idxs in disj_q_idx.items():
                    disj_closed_masks.append(
                        (spec_lbs_b[:, q_idxs] > 0).any(dim=1))
                all_disj_closed = torch.stack(
                    disj_closed_masks, dim=1).all(dim=1)

        unclosed = (~all_disj_closed).nonzero(as_tuple=True)[0]
        n_leaves_visited += B
        n_verified_by_crown_iter = int(all_disj_closed.sum().item())

        # Domain clipping: shrink unclosed leaves to the bounding box of
        # the intersection of unsafe halfspaces (CROWN's input-space
        # linearization). For AND-conjuncts where per-query CROWN can't
        # close but the intersection of halfspaces is empty in input
        # space, clipping verifies the leaf. For non-empty intersections,
        # the smaller box gives tighter next CROWN bounds, propagating
        # to faster convergence. Mirrors AB-CROWN's `clip_input_domain:
        # complete`.
        n_verified_by_clip_iter = 0
        if clip_enabled and unclosed.numel() > 0 and A_lin is not None:
            xl_u = xl_batch[unclosed]
            xh_u = xh_batch[unclosed]
            A_u = A_lin[unclosed]
            b_u = b_lin[unclosed]
            # Iterate per-halfspace clipping until no further shrinkage.
            clip_iters = int(getattr(
                settings, 'input_split_batched_clip_iters', 1))
            xl_c, xh_c, feasible_c = xl_u, xh_u, None
            for _ci in range(clip_iters):
                if multi_disjunct:
                    # Per-disjunct clip; union the resulting bboxes.
                    # Leaf is infeasible iff EVERY disjunct's polytope is
                    # empty (no point can satisfy any disjunct).
                    B_u = xl_c.shape[0]
                    per_disj_xl = []
                    per_disj_xh = []
                    per_disj_feas = []
                    for di, q_t in disj_q_idx_tensors.items():
                        A_d = A_u.index_select(1, q_t)
                        b_d = b_u.index_select(1, q_t)
                        xl_d, xh_d, feas_d = _clip_box_by_halfspaces_batched(
                            xl_c, xh_c, A_d, b_d)
                        # For infeasible disjuncts: clamp xl_d > xh_d to a
                        # null contribution to the union — equivalent to
                        # excluding that disjunct from the union.
                        per_disj_xl.append(torch.where(
                            feas_d.unsqueeze(-1), xl_d,
                            torch.full_like(xl_d, float('+inf'))))
                        per_disj_xh.append(torch.where(
                            feas_d.unsqueeze(-1), xh_d,
                            torch.full_like(xh_d, float('-inf'))))
                        per_disj_feas.append(feas_d)
                    # Per-dim union: take min of mins and max of maxes
                    # over disjuncts (only feasible ones contribute).
                    xl_new = torch.stack(per_disj_xl, dim=0).min(dim=0).values
                    xh_new = torch.stack(per_disj_xh, dim=0).max(dim=0).values
                    feasible_new = torch.stack(per_disj_feas, dim=0).any(dim=0)
                    # For leaves where all disjuncts are infeasible, the
                    # min/max are inf/-inf — replace with original box so
                    # downstream doesn't blow up; feasible_new masks them.
                    xl_new = torch.where(
                        feasible_new.unsqueeze(-1), xl_new, xl_c)
                    xh_new = torch.where(
                        feasible_new.unsqueeze(-1), xh_new, xh_c)
                else:
                    xl_new, xh_new, feasible_new = _clip_box_by_halfspaces_batched(
                        xl_c, xh_c, A_u, b_u)
                shrunk = ((xl_new > xl_c) | (xh_new < xh_c)).any().item()
                xl_c, xh_c, feasible_c = xl_new, xh_new, feasible_new
                if not shrunk:
                    break
            # Optional second stage: full-LP clip on the SURVIVING
            # leaves. Strictly tighter than per-halfspace (joint
            # constraints in one LP); catches infeasibility cases that
            # cheap clip misses. Parallelized across CPU cores via
            # persistent-env Gurobi workers. Roughly +10 ms per leaf
            # per iter; only enable when batched_clip + per-halfspace
            # isn't converging (acasxu prop_1/5/6/9 etc).
            if (bool(getattr(settings,
                              'input_split_batched_clip_full_lp', False))
                    and feasible_c.any().item()):
                surviving = feasible_c.nonzero(as_tuple=True)[0]
                xl_s = xl_c[surviving]
                xh_s = xh_c[surviving]
                A_s = A_u[surviving]
                b_s = b_u[surviving]
                xl_lp, xh_lp, feas_lp = _clip_box_by_full_lp_batched(
                    xl_s, xh_s, A_s, b_s, pool=_lp_pool)
                # Write back into the per-halfspace results.
                xl_c[surviving] = xl_lp
                xh_c[surviving] = xh_lp
                # Mark infeasibles via the LP.
                inf_s = ~feas_lp
                if inf_s.any().item():
                    inf_global = surviving[inf_s]
                    feasible_c[inf_global] = False
            n_verified_by_clip_iter = int((~feasible_c).sum().item())
            # Only feasible-after-clip leaves continue to split.
            feasible_idx = feasible_c.nonzero(as_tuple=True)[0]
            if feasible_idx.numel() > 0:
                xl_split = xl_c[feasible_idx]
                xh_split = xh_c[feasible_idx]
            else:
                xl_split = xl_u[:0]
                xh_split = xh_u[:0]

            # Clip → re-CROWN cycles: the old CROWN bounds were
            # computed on the LARGER pre-clip box, so they're still
            # valid (just loose). Re-running CROWN on the clipped box
            # gives TIGHTER per-query lbs that may close the leaf
            # without splitting. AB-CROWN-style inner iteration.
            # Each cycle is one extra forward zono + spec backward
            # (~5-20 ms per iter for the whole batch); pays off if it
            # closes leaves that would otherwise become 2 children +
            # next-iter CROWN.
            n_recrown_cycles = int(getattr(
                settings, 'input_split_batched_clip_recrown_cycles', 0))
            for _rc in range(n_recrown_cycles):
                if xl_split.numel() == 0:
                    break
                sb_r, _ = _forward_zonotope_graph_batched(
                    xl_split, xh_split, gg, device, dtype)
                spec_lbs_r, A_r, b_r = _spec_backward_graph_batched(
                    sb_r, xl_split, xh_split, gg, spec_ew, device, dtype,
                    return_input_linear=True)
                # Closure on re-CROWN
                closed_r = []
                for di, q_idxs in disj_q_idx.items():
                    closed_r.append((spec_lbs_r[:, q_idxs] > 0).any(dim=1))
                if closed_r:
                    all_closed_r = torch.stack(closed_r, dim=1).all(dim=1)
                else:
                    all_closed_r = torch.ones(
                        xl_split.shape[0], dtype=torch.bool, device=device)
                n_verified_by_clip_iter += int(all_closed_r.sum().item())
                # Drop closed leaves
                still_open_r = (~all_closed_r).nonzero(as_tuple=True)[0]
                if still_open_r.numel() == 0:
                    xl_split = xl_split[:0]
                    xh_split = xh_split[:0]
                    break
                # Clip the still-open leaves with the new tighter A, b
                xl_o = xl_split[still_open_r]
                xh_o = xh_split[still_open_r]
                A_o = A_r[still_open_r]
                b_o = b_r[still_open_r]
                xl_r2, xh_r2, feas_r2 = _clip_box_by_halfspaces_batched(
                    xl_o, xh_o, A_o, b_o)
                n_verified_by_clip_iter += int((~feas_r2).sum().item())
                feas_r2_idx = feas_r2.nonzero(as_tuple=True)[0]
                if feas_r2_idx.numel() == 0:
                    xl_split = xl_split[:0]
                    xh_split = xh_split[:0]
                    break
                xl_new = xl_r2[feas_r2_idx]
                xh_new = xh_r2[feas_r2_idx]
                # Stop if no significant shrinkage on this cycle.
                old_widths = (xh_split[still_open_r][feas_r2_idx]
                              - xl_split[still_open_r][feas_r2_idx])
                new_widths = xh_new - xl_new
                if (new_widths / (old_widths + 1e-12)).min().item() > 0.99:
                    xl_split = xl_new; xh_split = xh_new
                    break
                xl_split = xl_new
                xh_split = xh_new
        else:
            xl_split = xl_batch[unclosed]
            xh_split = xh_batch[unclosed]

        # MILP escalation on stuck boundary leaves. After CROWN +
        # α-CROWN + clipping fail to close a leaf, if its unstable-
        # neuron count is low enough, the full triangle MILP (exact ReLU
        # encoding via binaries) often closes in <50 ms. Only escalate
        # leaves where the input box has been split enough that
        # unstable ≤ `milp_max_unstable` (~80). At root, ACASXU has
        # ~270 unstable and MILP times out; at 1/32 sub-box it drops
        # to ~60 and MILP closes in 0 s. Per-leaf serial Gurobi (small
        # MILPs solve fast), parallelizable across leaves via mp pool
        # in future. Mirrors AB-CROWN's MIP attack on deep leaves.
        milp_escalate = bool(getattr(
            settings, 'input_split_batched_milp_escalate', False))
        if (milp_escalate and xl_split.numel() > 0 and gg_ops_ser):
            from . import verify_gen_lp as _glp
            max_un = int(getattr(
                settings, 'input_split_batched_milp_max_unstable', 80))
            max_per_iter = int(getattr(
                settings, 'input_split_batched_milp_max_leaves', 20))
            per_query_tl = float(getattr(
                settings, 'input_split_batched_milp_tl', 2.0))
            # Per-leaf bbr from the batched sb_b. Get the indices in
            # the FULL batch that correspond to xl_split (i.e., the
            # `unclosed[feasible_idx]` mapping).
            if clip_enabled and unclosed.numel() > 0 and A_lin is not None:
                xl_split_idx = unclosed[feasible_idx]
            else:
                xl_split_idx = unclosed
            # Compute per-leaf unstable count + worst-disjunct lb
            n_un_per_leaf = torch.zeros(
                xl_split.shape[0], dtype=torch.int32, device=device)
            for L in sb_b:
                lo, hi = sb_b[L]
                lo_s = lo[xl_split_idx]; hi_s = hi[xl_split_idx]
                n_un_per_leaf += ((lo_s < 0) & (hi_s > 0)).sum(dim=1).int()
            # Pick leaves to escalate: low unstable count + worst lb
            # close to 0.
            best_lb_per_leaf = spec_lbs_b[xl_split_idx].max(dim=1).values
            cand_mask = (n_un_per_leaf <= max_un) & (best_lb_per_leaf > -1.0)
            cand_idx = cand_mask.nonzero(as_tuple=True)[0]
            if cand_idx.numel() > 0:
                # Sort by best_lb descending (closest to closing first).
                order = best_lb_per_leaf[cand_idx].argsort(descending=True)
                cand_idx = cand_idx[order[:max_per_iter]]
                # Build per-leaf MILP tasks and dispatch in parallel
                # via multiprocessing. Each task tries queries in
                # best-CROWN-lb order; first MILP-UNSAT per disjunct
                # closes that disjunct; leaf closes when every disjunct
                # is MILP-closed.
                qids_list = sorted(spec_ew.keys())
                last_op_name = gg['ops'][-1]['name']
                tasks = []
                for ci in cand_idx.cpu().numpy():
                    full_idx = int(xl_split_idx[ci].item())
                    xl_l_np = xl_split[ci].cpu().numpy().astype(np.float64)
                    xh_l_np = xh_split[ci].cpu().numpy().astype(np.float64)
                    bbr_l = {L: (sb_b[L][0][full_idx].cpu().numpy().astype(np.float64),
                                  sb_b[L][1][full_idx].cpu().numpy().astype(np.float64))
                             for L in sb_b}
                    # Per-disjunct best-lb query ordering.
                    per_disj_queries = []
                    for di, qlist in disj_queries.items():
                        q_idxs = [qids_list.index(qi) for qi, _, _ in qlist]
                        q_order = sorted(q_idxs,
                                         key=lambda qix: -spec_lbs_b[full_idx, qix].item())
                        ordered = []
                        for qix in q_order:
                            qi = qids_list[qix]
                            ordered.append((
                                spec_ew[qi][0].cpu().numpy().astype(np.float64),
                                float(spec_ew[qi][1])))
                        per_disj_queries.append(ordered)
                    tasks.append((int(ci), gg_ops_ser, xl_l_np, xh_l_np,
                                   bbr_l, last_op_name, per_disj_queries,
                                   per_query_tl))
                import multiprocessing as _mp
                closed_via_milp = torch.zeros(
                    xl_split.shape[0], dtype=torch.bool, device=device)
                n_workers = min(_mp.cpu_count() - 1, len(tasks), 8)
                if n_workers <= 1:
                    results = [_milp_escalate_worker(t) for t in tasks]
                else:
                    with _mp.Pool(n_workers) as pool:
                        results = pool.map(_milp_escalate_worker, tasks)
                for ci, closed in results:
                    if closed:
                        closed_via_milp[ci] = True
                n_milp_closed = int(closed_via_milp.sum().item())
                if n_milp_closed > 0:
                    n_verified_by_clip_iter += n_milp_closed
                    # Drop closed leaves from xl_split.
                    keep = ~closed_via_milp
                    xl_split = xl_split[keep]
                    xh_split = xh_split[keep]
                    if A_lin is not None and clip_enabled and feasible_idx.numel() > 0:
                        A_u = A_u[keep] if A_u.shape[0] == keep.shape[0] else A_u

        # SB (smart branching) axis selection — pick the dim that most
        # tightens the worst CROWN bound when split. Score_i = width_i ×
        # |A_q*[i]| where q* is the worst query (closest-to-zero lb).
        # Falls back to widest-axis if A_lin not available.
        # AB-CROWN's `branching.method: sb` does similar; ours is a
        # simpler form (sum |A_q[i]| across queries instead of full
        # gradient on worst bound).
        sb_enabled = bool(getattr(
            settings, 'input_split_batched_branch_sb', True))
        if xl_split.numel() > 0:
            widths = (xh_split - xl_split).cpu()
            if sb_enabled and A_lin is not None:
                # Recover surviving indices in xl_split's coordinate
                # frame. `unclosed`→A_u; clip restricts to feasible_idx
                # of that; xl_split lines up with feasible_idx.
                if clip_enabled and unclosed.numel() > 0 and A_lin is not None:
                    A_split = A_u[feasible_idx]
                else:
                    A_split = A_lin[unclosed]
                # Per-leaf sensitivity: sum_q |A_q[i]| × width_i
                sens = A_split.abs().sum(dim=1).cpu()  # (B, n_in)
                scores = widths * sens
                ax = scores.argmax(dim=1)
            else:
                ax = widths.argmax(dim=1)
            xl_cpu = xl_split.cpu()
            xh_cpu = xh_split.cpu()
            n = xl_split.shape[0]
            for i in range(n):
                xl_i = xl_cpu[i]
                xh_i = xh_cpu[i]
                a = int(ax[i].item())
                if float(widths[i, a]) < 1e-12:
                    n_open_at_timeout += 1
                    continue
                mid = float((xl_i[a] + xh_i[a]) / 2)
                xh_a = xh_i.clone(); xh_a[a] = mid
                xl_b_v = xl_i.clone(); xl_b_v[a] = mid
                worklist_xl.append(xl_i.to(device, dtype=dtype))
                worklist_xh.append(xh_a.to(device, dtype=dtype))
                worklist_xl.append(xl_b_v.to(device, dtype=dtype))
                worklist_xh.append(xh_i.to(device, dtype=dtype))
            if len(worklist_xl) > max_worklist:
                return 'unknown', {
                    'phase': 'batched_worklist_overflow',
                    'batched_n_leaves': n_leaves_visited,
                    'batched_n_iters': n_iters,
                    'worklist_size': len(worklist_xl)}
        n_iters += 1

    if not worklist_xl and n_open_at_timeout == 0:
        return 'verified', {
            'phase': 'batched_verified',
            'batched_n_leaves': n_leaves_visited,
            'batched_n_iters': n_iters,
            'batched_batch_size': batch_size}
    return 'unknown', {
        'phase': 'batched_timeout',
        'batched_n_leaves': n_leaves_visited,
        'batched_n_iters': n_iters,
        'batched_worklist_left': len(worklist_xl),
        'batched_open_degenerate': n_open_at_timeout,
        'batched_batch_size': batch_size}


def _input_split_verify(graph, spec, settings, build_fn, impl):
    """Recursive input-space BaB with sensitivity-scored axis selection.

    Score ``|qw_disj · G[:, k]|`` per input column from one fast zono
    forward pass — the column ``k`` of ``G`` is exactly the output's
    sensitivity to input axis ``k``, so splitting the highest-scored
    axis maximally tightens the spec bound. Recompute scores at each
    node (cheap; ~50 ms per zono pass on small-input benchmarks).

    With `input_split_fast_leaf=True` (default), each leaf runs only
    forward zono + α-CROWN(3 iters) + spec check (~3.4s on
    cifar_biasfield_0), bypassing the full pipeline's 60+ s of overhead.
    With `input_split_fast_leaf=False`, falls back to the legacy
    `_run_pipeline`-per-leaf which has heavy startup overhead but
    supports modes like 'medium' that re-enable Phase 2.5 in leaves.
    """
    import copy
    t_start = time.perf_counter()
    total_budget = float(getattr(settings, 'total_timeout', 60.0))
    # Time is the budget; depth cap (when set) is just a safety net.
    # `None` (default) means no cap — runs until time_left exhausted.
    _md = getattr(settings, 'input_split_max_depth', None)
    if isinstance(_md, (int, float)):
        max_depth = int(_md)
    else:
        max_depth = 10**9
    # Bump Python recursion limit so DFS recursion doesn't hit it at
    # deep BaB. CPython default is 1000; raise to 10**6 (no practical
    # downside other than later StackOverflow on truly pathological
    # cases, which we'd hit time first anyway).
    import sys
    if sys.getrecursionlimit() < 10**6:
        sys.setrecursionlimit(10**6)
    per_node_tl = float(getattr(settings, 'input_split_node_timeout', 8.0))
    fast_leaf = bool(getattr(settings, 'input_split_fast_leaf', True))
    # Per-leaf PGD on unverifiable leaves — catches narrow SAT regions
    # that root PGD missed but become attackable once the sub-box is
    # localized (cersyve pretrain_inv cases live here). Off by default
    # to avoid extra cost on benchmarks where root PGD is sufficient.
    leaf_pgd_enabled = bool(getattr(
        settings, 'input_split_leaf_pgd_enabled', False))
    leaf_pgd_time = float(getattr(
        settings, 'input_split_leaf_pgd_time', 0.1))

    # If fast_leaf is on, prepare GPU graph once at the top.
    fast_gg = None
    fast_device = fast_dtype = None
    if fast_leaf:
        from .settings import resolve_torch
        fast_device, fast_dtype = resolve_torch(settings)
        fast_gg = graph.gpu_graph(fast_device, fast_dtype)

    # Root-level PGD attack — fast_leaf path skips per-leaf PGD, so SAT
    # cases would otherwise return 'unknown'. Run ONE PGD attack on the
    # original input box before BaB to catch counter-examples cheaply
    # (~0.5s). Mirrors Phase 0 PGD in the regular pipeline.
    if (fast_leaf
            and not bool(getattr(settings, 'disable_sat_finding', False))
            and bool(getattr(settings, 'pgd_phase0_enabled', True))):
        try:
            xl_pgd = torch.tensor(
                spec.x_lo, dtype=fast_dtype, device=fast_device)
            xh_pgd = torch.tensor(
                spec.x_hi, dtype=fast_dtype, device=fast_device)
            pgd_budget = float(getattr(
                settings, 'pgd_time_budget_phase0', 5.0))
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, fast_gg, settings,
                time_budget=pgd_budget)
            if pgd_sat and pgd_witness is not None:
                return 'sat', {'phase': 'fast_leaf_pgd',
                                'witness': pgd_witness}
        except Exception:
            pass  # PGD failure is non-fatal

    def _run_node(s):
        remaining = total_budget - (time.perf_counter() - t_start)
        if remaining < 0.5:
            return 'unknown', {'phase': 'input_split_timeout'}
        if fast_leaf:
            return _input_split_fast_leaf(
                graph, s, settings, fast_gg, fast_device, fast_dtype)
        sub = copy.deepcopy(settings)
        sub.total_timeout = min(per_node_tl, max(0.5, remaining))
        # Phase 8 MILP at leaves: gives the per-leaf verifier a stronger
        # bound when LP triangle isn't enough (cifar_biasfield_28
        # required this — Phase 7 LP alone left worst-disjunct lb
        # at -1.24 even at depth 15). Use a small Phase 8 budget per
        # leaf so it doesn't blow past per_node_tl.
        sub.skip_phase8_milp = bool(getattr(
            settings, 'input_split_skip_phase8', False))
        sub.print_progress = False
        # Per-leaf stack:
        #   forward zono → per-neuron adaptive CROWN-backward (Phase 1)
        #   → Phase 2 backward CROWN
        #   → Phase 2.5 α-CROWN per-query refresh
        #   → NO MILP anywhere (Phase 7 LP, Phase 8 MILP both skipped)
        # The split itself is doing the heavy lifting; each leaf gets
        # a tighter zono enclosure than the parent, so adaptive CROWN
        # + α-CROWN often closes the spec without needing MILP. Keeping
        # adapt+α-CROWN on (vs. the older fully-lean config that turned
        # both off) costs ~1-2s extra per leaf on biasfield_28 but
        # closes leaves that plain zono+CROWN can't.
        # Override via settings.input_split_leaf_mode ∈ {'lean','medium'}:
        #   'medium' (default) — adapt+α-CROWN on, no MILP (this block)
        #   'lean' — adapt+α-CROWN off, no MILP (the original
        #            forward-zono+CROWN+LP-only config; faster per leaf
        #            but each leaf is looser, so needs more splits)
        leaf_mode = str(getattr(settings, 'input_split_leaf_mode', 'medium'))
        sub.tighten_formulation = 'skip'
        sub.phase1_method = 'legacy'
        if leaf_mode == 'lean':
            sub.phase1_adapt_enabled = False
            sub.zono_lift_enabled = False
        else:  # 'medium' (default) — keep adapt + α-CROWN on
            sub.phase1_adapt_enabled = True
            sub.zono_lift_enabled = True
        sub.input_split_enabled = False  # no nested input split
        return _run_pipeline(graph, s, sub, build_fn, impl)

    n_nodes = [0]

    def _bab(s, depth):
        n_nodes[0] += 1
        r, d = _run_node(s)
        if r in ('verified', 'sat'):
            return r, d
        # Per-leaf PGD on unknown sub-domains — sub-box is now localized,
        # so narrow SAT regions become easier to find than from the
        # original full box. Only fires when `input_split_leaf_pgd_enabled`.
        if (leaf_pgd_enabled and fast_leaf
                and not bool(getattr(settings, 'disable_sat_finding', False))):
            try:
                xl_l = torch.tensor(
                    s.x_lo, dtype=fast_dtype, device=fast_device)
                xh_l = torch.tensor(
                    s.x_hi, dtype=fast_dtype, device=fast_device)
                ok, w = _pgd_attack_general(
                    xl_l, xh_l, s, fast_gg, settings,
                    time_budget=leaf_pgd_time)
                if ok and w is not None:
                    return 'sat', {'phase': 'leaf_pgd', 'witness': w}
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                pass
        if depth >= max_depth:
            return r, d
        if time.perf_counter() - t_start > total_budget - 0.5:
            return 'unknown', d
        # Pick split axis by spec sensitivity (ew × current radius — radius
        # is folded into G column magnitude). Prefer the score the leaf
        # already computed (saves a redundant forward zono per BaB
        # node — ~250 ms each on cifar_biasfield); fall back to a fresh
        # `_score_input_axes` only if the leaf didn't stash one (e.g.
        # because it returned `verified` and we're in the rare case
        # of needing to refine after).
        leaf_scores = d.get('axis_scores') if isinstance(d, dict) else None
        if leaf_scores is not None:
            scores = leaf_scores
        else:
            try:
                scores, _ = _score_input_axes(
                    graph, s, gg=fast_gg, device=fast_device, dtype=fast_dtype)
            except Exception:
                scores = (s.x_hi - s.x_lo).astype(float).flatten()
        widths = (s.x_hi - s.x_lo).astype(float).flatten()
        # Don't split an axis that has effectively zero width (avoids
        # infinite recursion on degenerate dims).
        scores = np.where(widths > 1e-9, scores, -np.inf)
        ax = int(np.argmax(scores))
        # When score-based pick gives nothing useful, fall back to the
        # widest axis with positive width — keeps BaB making progress
        # on benchmarks where sensitivity scores under-rank some axes
        # (cersyve: after axis 1 narrows to ~1e-3, its score becomes 0
        # but axes 0/2/3 still need splitting too).
        if not np.isfinite(scores[ax]) or scores[ax] <= 0:
            widths_alive = np.where(widths > 1e-9, widths, -np.inf)
            ax = int(np.argmax(widths_alive))
            if not np.isfinite(widths_alive[ax]):
                return r, d
        mid = (float(s.x_lo.flatten()[ax]) + float(s.x_hi.flatten()[ax])) / 2.0
        sa = copy.deepcopy(s)
        sb = copy.deepcopy(s)
        sa.x_hi.flat[ax] = mid
        sb.x_lo.flat[ax] = mid
        ra, da = _bab(sa, depth + 1)
        if ra == 'sat':
            return 'sat', da
        if ra != 'verified':
            return ra, da
        rb, db = _bab(sb, depth + 1)
        if rb == 'verified':
            return 'verified', db
        return rb, db

    r, d = _bab(spec, 0)
    d['input_split_used'] = True
    d['input_split_n_nodes'] = n_nodes[0]
    d['input_split_total_time'] = time.perf_counter() - t_start
    return r, d
