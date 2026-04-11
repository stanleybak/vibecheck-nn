"""Graph verification mode: zonotope + CROWN + MILP on DAG-structured networks.

This module is a parallel implementation of the graph pipeline found in
`verify_milp._milp_verify_graph`, with two independent LP/MILP model builders
(a readable reference builder and a batched optimized builder) sharing exactly
the same op-walking, dead-neuron propagation, and variable-ref tracking.

Both builders preserve the Bug #1 fix: a conv/fc whose inputs are all dead
still outputs `bias[j]`, encoded as a fixed-bound Gurobi variable — never
returned as `None`.

All `m.optimize()` calls set `DualReductions=0` (Bug #5) and raise on
unexpected Gurobi status codes.
"""

import time
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F

from .settings import resolve_torch
from .verify_milp import (
    VerifyStats, _fire_callback,
    _compute_dead_at, _conv_sparse_matrix,
    _pgd_attack_general, _tighten_layer_parallel,
)
from .verify_zono_bnb import (
    _forward_zonotope_graph, _spec_backward_graph, _make_slopes,
    _find_shared_gens_count,
)
from .zonotope import TorchZonotope


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
                    a = m.addVar(lb=float(lo_r[j]), ub=float(hi_r[j]))
                    m.addConstr(a == z)
                    out[j] = a
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
                    expr = grb.LinExpr()
                    if va[j] is not None:
                        expr.add(va[j], 1.0)
                    if vb[j] is not None:
                        expr.add(vb[j], 1.0)
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == expr)
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
                     milp_by_layer=None, n_threads=1):
    """Batched builder using MVar + scipy.sparse for conv/fc layers.

    Produces identical constraints to _build_reference but with dramatically
    fewer Python<->C transitions. For conv layers the sparse weight matrix
    is applied as a single addConstr(out_mvar == W_live @ prev_mvar + b).

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
                # Bug #1: outputs are constants = per-row bias; skip dead outputs
                for j in range(n_out):
                    if my_dead is not None and my_dead[j]:
                        continue
                    c = float(bias_per_out[j])
                    out[j] = m.addVar(lb=c, ub=c)
                m.update()
                op_var_refs[nm] = out
                if nm == target_input_name:
                    target_vars = out
                    break
                continue

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

            # Active: a == z, bounds tightened
            for j in np.where(active_mask)[0]:
                a = m.addVar(lb=float(lo_arr[j]), ub=float(hi_arr[j]))
                m.addConstr(a == prev[j])
                out[int(j)] = a

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
                out = [None] * n
                for j in range(n):
                    if va[j] is None and vb[j] is None:
                        continue
                    expr = grb.LinExpr()
                    if va[j] is not None:
                        expr.add(va[j], 1.0)
                    if vb[j] is not None:
                        expr.add(vb[j], 1.0)
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == expr)
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
    """
    import gurobipy as grb
    j, timeout, cur_lo, cur_hi = args[:4]

    m = _graph_shared_model.copy()
    m.setParam('DualReductions', 0)
    var_idx = _graph_shared_target_indices[j]

    lb, ub = cur_lo, cur_hi
    if var_idx < 0:
        m.dispose()
        return j, lb, ub

    tv = m.getVars()[var_idx]
    m.setParam('TimeLimit', timeout)
    if abs(cur_lo) < abs(cur_hi):
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

    m.optimize()
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
        if abs(cur_lo) < abs(cur_hi):
            m.setObjective(tv, grb.GRB.MAXIMIZE)
            m.setParam('BestBdStop', -1e-6)
        else:
            m.setObjective(tv, grb.GRB.MINIMIZE)
            m.setParam('BestBdStop', 1e-6)
        m.optimize()
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


def _tighten_layer_graph(gg_ops, x_lo, x_hi, bounds_by_relu,
                          target_layer_idx, unstable, input_name,
                          build_fn, sample_timeout, n_cores, time_left_fn):
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
        cm.optimize()
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
      - feasibility: None
      - optimize:    ObjBound (float or None)
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

    m, env, op_var_refs, _ = build_fn(
        gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
        target_layer_idx=None,
        use_milp=False,
        milp_by_layer=milp_by_layer,
        n_threads=n_threads,
    )
    m.setParam('DualReductions', 0)
    m.setParam('TimeLimit', float(timeout))

    spec_expr, const = _build_spec_expression(
        m, op_var_refs, gg_ops, query_w, query_bias)

    t0 = time.perf_counter()

    ok = _ok_statuses()

    if mode == 'feasibility':
        m.addConstr(spec_expr + const <= 0)
        m.setObjective(0, grb.GRB.MINIMIZE)
        m.optimize()
        status = m.Status
        dt = time.perf_counter() - t0
        m.dispose(); env.dispose()
        assert status in ok, f'feasibility: unexpected status {status}'
        if status == grb.GRB.INFEASIBLE:
            return 'UNSAT', dt, None
        if status == grb.GRB.OPTIMAL:
            return 'SAT', dt, None
        return 'UNKNOWN', dt, None

    if mode == 'optimize':
        m.setParam('BestBdStop', 0.0)
        m.setObjective(spec_expr + const, grb.GRB.MINIMIZE)
        m.optimize()
        status = m.Status
        lb = None
        try:
            lb = float(m.ObjBound)
        except Exception:
            pass
        n_sol = m.SolCount
        dt = time.perf_counter() - t0
        m.dispose(); env.dispose()
        assert status in ok, f'optimize: unexpected status {status}'
        if status in (grb.GRB.OPTIMAL, grb.GRB.USER_OBJ_LIMIT):
            return ('UNSAT' if lb is not None and lb > 0 else 'SAT'), dt, lb
        if status == grb.GRB.TIME_LIMIT and n_sol > 0:
            return 'SAT', dt, lb
        return 'UNKNOWN', dt, lb

    # mode == 'score'
    m.setObjective(spec_expr + const, grb.GRB.MINIMIZE)
    m.optimize()
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
    m.dispose(); env.dispose()
    assert status in ok, f'score: unexpected status {status}'
    return 'SCORED', dt, (scores, lb)


def _racing_escalation_graph_correct(impl, gg_ops, x_lo, x_hi, bounds_by_relu,
                                       query_w, query_bias, scored_keys,
                                       n_cores, time_left_fn, input_name,
                                       print_progress=False):
    """Doubling-bin-schedule MILP escalation with three-way dispatch.

    Bug #4: explicit SAT / UNSAT / UNKNOWN branches, never collapses.
    Uses _solve_spec_worker_graph (Bug #1 safe).
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
        common = (impl, gg_ops, x_lo, x_hi, bounds_by_relu, query_w,
                  query_bias, scored_keys, n_bins)
        feas_args = ('feasibility',) + common + (1, tl, input_name)
        opt_args = ('optimize',) + common + (opt_threads, tl, input_name)

        pool = multiprocessing.Pool(2)
        async_feas = pool.apply_async(_solve_spec_worker_graph, (feas_args,))
        async_opt = pool.apply_async(_solve_spec_worker_graph, (opt_args,))
        while True:
            if async_feas.ready():
                feas_result, feas_dt, _ = async_feas.get()
                pool.terminate(); pool.join()
                if feas_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'feas UNSAT ({feas_dt:.1f}s) → verified')
                    return True, n_bins
                if feas_result == 'SAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'feas SAT ({feas_dt:.1f}s) → escalate')
                    break
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'feas UNKNOWN ({feas_dt:.1f}s) → escalate')
                break
            if async_opt.ready():
                opt_result, opt_dt, opt_lb = async_opt.get()
                pool.terminate(); pool.join()
                lb_s = f'{opt_lb:.4f}' if opt_lb is not None else '?'
                if opt_result == 'UNSAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'opt lb={lb_s} ({opt_dt:.1f}s) → verified')
                    return True, n_bins
                if opt_result == 'SAT':
                    if print_progress:
                        print(f'    Racing bins={n_bins}: '
                              f'opt lb={lb_s} ({opt_dt:.1f}s) → escalate')
                    break
                if print_progress:
                    print(f'    Racing bins={n_bins}: '
                          f'opt lb={lb_s} ({opt_dt:.1f}s) → escalate')
                break
            time.sleep(0.05)

    return False, bin_schedule[-1] if bin_schedule else 0


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
                                 device, dtype, neuron_subset=None):
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
    """
    j, timeout, cur_lo, cur_hi, use_milp = args
    import gurobipy as grb

    (gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, input_name) = \
        _sparse_graph_args

    m, env, tv = _build_sparse_neuron_graph(
        gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name, j,
        input_name, use_milp=use_milp, n_threads=1)
    m.setParam('TimeLimit', float(timeout))

    lb, ub = float(cur_lo), float(cur_hi)
    any_timeout = False

    if abs(cur_lo) < abs(cur_hi):
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
    m.optimize()
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
        m.optimize()
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
    m.optimize()
    dt_probe = time.perf_counter() - t_probe0
    status = m.Status
    m.dispose()
    env.dispose()
    return dt_build, dt_probe, status


def _tighten_layer_graph_sparse(gg_ops, x_lo, x_hi, bounds_by_relu,
                                  target_layer_idx, unstable, input_name,
                                  mode, sample_timeout, n_cores, time_left):
    """Per-target-neuron sparse tightening on a graph with merges.

    Each unstable neuron gets its own sparse Gurobi model covering only
    its backward dependency cone (dead neurons dropped). Models are
    solved in parallel by `_solve_sparse_neuron_graph_worker`.

    `mode` controls solver choice, mirroring
    `_tighten_sequential_with_probe`:
      - 'probe': probe MILP; on failure fall through to LP probe
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

    if mode == 'probe':
        # Try MILP first.
        dt_b, dt_p, status = _probe_sparse_neuron(
            gg_ops, x_lo, x_hi, bounds_by_relu, target_op_name,
            probe_j, input_name, use_milp=True,
            sample_timeout=sample_timeout)
        dt_build_total += dt_b
        dt_probe_total += dt_p
        if status is not None and status != grb.GRB.TIME_LIMIT:
            use_milp = True
            per_neuron_est = dt_b + dt_p
        # Else fall through to LP probe below.

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
      - 'lp': skip the MILP probe entirely (a previous layer's MILP
        already timed out). Probe only LP; if LP survives, solve all
        with LP, else skip.
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
    half = n_sample // 2
    rng = np.random.RandomState(int(seq_li))
    sample_idx = rng.choice(unstable, n_sample, replace=False)

    is_fc = layers_np_seq[seq_li]['type'] == 'fc'
    # FC layers' MILP probes tend to blow up; follow milp_verify's
    # heuristic of skipping the MILP sample on FC layers past the first.
    try_milp = (mode == 'probe') and not (is_fc and seq_li > 0)

    t0 = time.perf_counter()
    milp_any_timeout = not try_milp
    if try_milp and half > 0:
        _, _, milp_any_timeout = _tighten_layer_parallel(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
            use_milp=True, timeout=sample_timeout,
            n_cores=n_cores, neuron_subset=sample_idx[:half])

    lp_any_timeout = False
    lp_probe_start = half if try_milp else 0
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
    if not lp_any_timeout:
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
# Interleaved zonotope forward with per-layer tightening
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_zonotope_interleaved(
        xl, xh, gg, gg_ops_ser, x_lo_64, x_hi_64,
        build_fn, sample_timeout, n_cores, time_left,
        device, dtype, settings, print_progress=False, verbose_cb=None):
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

    Returns:
        sb: {layer_idx: (lo_tensor, hi_tensor)} of the *tightened*
            pre-activation bounds actually used in the relu relaxation
        bounds_by_relu: {layer_idx: (lo_np64, hi_np64)} same bounds as np
        z_final: the final zonotope after the last op
    """
    input_name = gg['input_name']
    forks = gg['fork_points']

    z_init = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    zono_state = {input_name: z_init}
    gen_count = {input_name: z_init.generators.shape[1]}
    sb = {}
    bounds_by_relu = {}

    # Sticky mode: once the LP (or MILP) probe times out at any layer,
    # downshift and never probe that solver again on later layers.
    # Transitions: 'probe' → 'lp' → 'skip'. Mirrors
    # _milp_verify._milp_verify's tighten_mode. The per-neuron timeout
    # is `milp_sample_timeout` (default 5 s); there's no separate
    # cumulative budget — we trust the per-probe cap plus sticky mode.
    tighten_mode = 'probe'
    lp_per_worker = bool(getattr(settings, 'milp_lp_per_worker', True))

    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    def _get(name):
        return zono_state[name].copy() if name in forks else zono_state[name]

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']

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
            if 'layer_idx' not in op:
                # Output relu (rare) — just apply with internal bounds.
                z.apply_relu()
                zono_state[name] = z
                gen_count[name] = z.generators.shape[1]
                for inp in op['inputs']:
                    if last_use.get(inp) == op_idx and inp in zono_state:
                        del zono_state[inp]
                continue

            li = op['layer_idx']

            # Step 1: seed bounds_by_relu from the zonotope's internal bounds.
            z_lo, z_hi = z.bounds()
            lo_np = z_lo.cpu().numpy().astype(np.float64)
            hi_np = z_hi.cpu().numpy().astype(np.float64)
            bounds_by_relu[li] = (lo_np, hi_np)
            unstable_initial = np.where((lo_np < 0) & (hi_np > 0))[0]

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
                # using already-tightened bounds_by_relu[0..li-1].
                t_a = time.perf_counter()
                adapt_lo, adapt_hi = _per_neuron_adaptive_bounds(
                    gg, xl, xh, bounds_by_relu, li, device, dtype,
                    neuron_subset=unstable_initial)
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

                if (len(after_adapt) > 0 and time_left() > 2
                        and tighten_mode != 'skip'):
                    if _has_merge_before(gg_ops_ser, li):
                        # Sparse per-target-neuron model: one small
                        # Gurobi model per unstable neuron, covering
                        # only that neuron's backward dependency cone.
                        # Same construction logic as the sequential
                        # sparse path; the only difference is that the
                        # dependency walker follows both branches of a
                        # merge-Add. MILP is tried first, then LP.
                        gs_lo, gs_hi, gs_method, dt_build, dt_probe, dt_solve = \
                            _tighten_layer_graph_sparse(
                                gg_ops_ser, x_lo_64, x_hi_64,
                                bounds_by_relu, li, after_adapt,
                                gg['input_name'], mode=tighten_mode,
                                sample_timeout=sample_timeout,
                                n_cores=n_cores, time_left=time_left)
                        new_lo = np.maximum(new_lo, gs_lo)
                        new_hi = np.minimum(new_hi, gs_hi)
                        method = f'adapt+graph-{gs_method}'
                        if gs_method == 'skip':
                            tighten_mode = 'skip'
                        elif (gs_method.startswith('lp')
                              and tighten_mode == 'probe'):
                            tighten_mode = 'lp'
                    else:
                        # Sequential subgraph (no merge upstream): probe
                        # MILP and LP on a sample, then solve all with
                        # whichever survived. Conv layers use sparse
                        # per-neuron MILP — typically faster *and*
                        # tighter than LP at L1/L2. The `tighten_mode`
                        # is sticky: once MILP times out we drop to
                        # 'lp', once LP times out we drop to 'skip'.
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
                                mode=tighten_mode)
                        new_lo = np.maximum(new_lo, seq_lo)
                        new_hi = np.minimum(new_hi, seq_hi)
                        method = f'adapt+seq-{seq_method}'
                        if seq_method == 'lp' and tighten_mode == 'probe':
                            tighten_mode = 'lp'
                        elif seq_method == 'skip':
                            tighten_mode = 'skip'
                    after_lp = np.where((new_lo < 0) & (new_hi > 0))[0]
                    lp_fixed = len(after_adapt) - len(after_lp)
                    width_lp = _mean_w(new_lo, new_hi, unstable_initial)
                else:
                    # Adaptive ran (above) but LP/MILP probing didn't —
                    # either because tighten_mode is 'skip' (a prior
                    # layer's probe failed), or adapt already closed all
                    # unstable neurons, or the time budget is used up.
                    # The adaptive pass still tightened bounds even when
                    # crown_fixed == 0, so 'adapt-only' is the right
                    # label either way.
                    method = 'adapt-only'
                    if tighten_mode == 'skip':
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

            # Step 4: apply the relu using the tightest available bounds.
            # The zonotope's internal bounds are intersected with
            # (tight_lo_t, tight_hi_t) inside apply_relu itself.
            lo_used, hi_used = z.apply_relu(
                tight_lo=tight_lo_t, tight_hi=tight_hi_t)
            sb[li] = (lo_used.clone(), hi_used.clone())
            lo_final = lo_used.cpu().numpy().astype(np.float64)
            hi_final = hi_used.cpu().numpy().astype(np.float64)
            bounds_by_relu[li] = (lo_final, hi_final)
            zono_state[name] = z
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
                    z = TorchZonotope(
                        z.center + torch.tensor(
                            bias.flatten(), dtype=dtype, device=device),
                        z.generators.clone())
                zono_state[name] = z

        elif t == 'sub':
            z = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                z = TorchZonotope(
                    z.center - torch.tensor(
                        bias.flatten(), dtype=dtype, device=device),
                    z.generators.clone())
            zono_state[name] = z

        elif t == 'reshape':
            zono_state[name] = _get(op['inputs'][0])

        gen_count[name] = zono_state[name].generators.shape[1]
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]

    z_final = zono_state[gg['ops'][-1]['name']]
    return sb, bounds_by_relu, z_final


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

    def time_left():
        return max(0.0, deadline - time.perf_counter())

    stats = VerifyStats()
    timing = {}
    details = {'timing': timing}
    if verbose:
        per_layer_timing = {}
        avg_layer_width = {}
        details['per_layer_timing'] = per_layer_timing
        details['avg_layer_width'] = avg_layer_width
        details['build_time_total'] = 0.0

    def _finalize(result_str, phase, **extra):
        details['result'] = result_str
        details['phase'] = phase
        details['time'] = time.perf_counter() - t_start
        details['n_splits'] = _compute_n_splits(gg, bounds_by_relu)
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
    # main verification is float64.
    gg_pgd = graph.gpu_graph(device, torch.float32)
    xl_pgd = xl_g.to(torch.float32)
    xh_pgd = xh_g.to(torch.float32)

    # Pre-seed bounds_by_relu as empty so _finalize never fails
    bounds_by_relu = {}

    # Prepare serialized ops + sparse conv matrices up front so phase 1
    # can do in-line LP probes at each relu.
    gg_ops_ser = _serialize_gg_ops(gg)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])
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

    # --- Phase 1: interleaved zonotope forward + per-layer tightening ---
    # At every ReLU we pause, run the per-neuron adaptive backward and
    # an optional LP probe, then apply the ReLU using the tightened
    # pre-activation bounds. This lets subsequent layers' zonotope
    # propagation benefit from the tightening (smaller new generators),
    # so deeper layers start with tighter bounds for free.
    t0 = time.perf_counter()
    try:
        sb, bounds_by_relu, _ = _forward_zonotope_interleaved(
            xl_g, xh_g, gg, gg_ops_ser, x_lo_64, x_hi_64, build_fn,
            sample_timeout, n_cores, time_left, device, dtype, settings,
            print_progress=print_progress, verbose_cb=_verbose_cb)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device.type != 'cpu':
            if print_progress:
                print(f'  GPU OOM/runtime ({e!s:.60}); falling back to CPU')
            device = torch.device('cpu')
            gg = graph.gpu_graph(device, dtype)
            gg_pgd = graph.gpu_graph(device, torch.float32)
            xl_g = xl_g.cpu(); xh_g = xh_g.cpu()
            xl_pgd = xl_pgd.cpu(); xh_pgd = xh_pgd.cpu()
            spec_ew = {qi: (w.cpu(), b) for qi, (w, b) in spec_ew.items()}
            sb, bounds_by_relu, _ = _forward_zonotope_interleaved(
                xl_g, xh_g, gg, gg_ops_ser, x_lo_64, x_hi_64, build_fn,
                sample_timeout, n_cores, time_left, device, dtype,
                print_progress=print_progress, verbose_cb=_verbose_cb)
        else:
            raise
    timing['phase1_zono_tighten'] = time.perf_counter() - t0

    # --- Phase 2: CROWN backward ---
    t0 = time.perf_counter()
    all_qids = set(spec_ew.keys())
    with torch.no_grad():
        spec_lbs, _ = _spec_backward_graph(
            sb, xl_g, xh_g, gg, spec_ew, all_qids, nh, device, dtype)
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

    # --- Phase 3: initial PGD ---
    t0 = time.perf_counter()
    try:
        pgd_sat, pgd_witness = _pgd_attack_general(xl_pgd, xh_pgd, spec, gg_pgd, settings)
    except RuntimeError:
        pgd_sat, pgd_witness = False, None
    timing['phase3_pgd'] = time.perf_counter() - t0
    if print_progress:
        print(f'Phase 3 (PGD): {timing["phase3_pgd"]:.2f}s  sat={pgd_sat}')
    if pgd_sat:
        return _finalize('sat', 'pgd', witness=pgd_witness)

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

    # --- Phase 7: score open queries via LP ---
    remaining_qids = set()
    for di in still_open_disj:
        for qi, _, _ in disj_queries[di]:
            if spec_lbs.get(qi, -1) <= 0:
                remaining_qids.add(qi)

    with torch.no_grad():
        _, _, ew_at_relu = _spec_backward_graph(
            sb, xl_g, xh_g, gg, spec_ew,
            remaining_qids, nh, device, dtype, return_ew=True)

    # --- Phase 7a: parallel LP feasibility pre-check (fast verify path) ---
    # Run one worker per open query with feasibility mode. Queries that
    # return UNSAT are verified immediately via LP; the rest fall through
    # to scoring + MILP escalation.
    t0 = time.perf_counter()
    still_needs_milp = set()
    if remaining_qids and time_left() > 2:
        n_q = len(remaining_qids)
        per_worker_threads = max(1, n_cores // max(1, n_q))
        feas_tl = time_left() * 0.5
        feas_tasks = []
        for qi in sorted(remaining_qids):
            _, q_w, q_bias = queries[qi]
            feas_tasks.append((
                'feasibility', impl, gg_ops_ser, x_lo_64, x_hi_64,
                bounds_by_relu, q_w, q_bias, [], 0, per_worker_threads,
                feas_tl, gg['input_name']))
        pool_size = min(n_q, max(1, n_cores // max(1, per_worker_threads)))
        with multiprocessing.Pool(pool_size) as pool:
            results = pool.map(_solve_spec_worker_graph, feas_tasks)
        for qi, (res, dt_feas, _) in zip(sorted(remaining_qids), results):
            if print_progress:
                print(f'  Phase 7a query {qi}: feas={res} ({dt_feas:.1f}s)')
            if res == 'UNSAT':
                spec_lbs[qi] = 1.0
            else:
                still_needs_milp.add(qi)
    else:
        still_needs_milp = set(remaining_qids)
    timing['phase7_feasibility'] = time.perf_counter() - t0

    # --- Phase 7b: seed MILP scores from CROWN ew + zono width ---
    # When the LP didn't finish in phase 7a (all 5 queries timed out), the
    # scoring LP will also time out. Use the cheap CROWN-derived estimate
    # directly; the true LP-fractional scoring can come back later once
    # we have a tighter bound store.
    per_query_scored = {}
    t0 = time.perf_counter()
    for qi in sorted(still_needs_milp):
        q_ew = ew_at_relu.get(qi, {})
        q_scores = {}
        for li in range(nh):
            lo_l, hi_l = bounds_by_relu[li]
            unstable = np.where((lo_l < 0) & (hi_l > 0))[0]
            if li in q_ew:
                ew = np.abs(q_ew[li])
                for i in unstable:
                    ew_i = float(ew[i]) if i < len(ew) else 1.0
                    frac = float(hi_l[i]) * abs(float(lo_l[i])) / float(
                        hi_l[i] - lo_l[i])
                    q_scores[(li, int(i))] = ew_i * frac
            else:
                for i in unstable:
                    q_scores[(li, int(i))] = float(hi_l[i]) * abs(float(lo_l[i])) / 2
        per_query_scored[qi] = sorted(
            q_scores.keys(), key=lambda k: q_scores[k], reverse=True)
    timing['phase7_score'] = time.perf_counter() - t0

    # --- Phase 8: MILP racing escalation (correct worker, Bug #1 safe) ---
    t_phase8 = time.perf_counter()
    for qi in sorted(still_needs_milp):
        if time_left() <= 0:
            break
        _, q_w, q_bias = queries[qi]
        scored_keys = per_query_scored.get(qi, [])
        if print_progress:
            print(f'  MILP query {qi} (disjunct {queries[qi][0]}):')
        verified, _ = _racing_escalation_graph_correct(
            impl, gg_ops_ser, x_lo_64, x_hi_64, bounds_by_relu,
            q_w, q_bias, scored_keys, n_cores, time_left,
            gg['input_name'], print_progress)
        if verified:
            spec_lbs[qi] = 1.0
    timing['phase8_milp'] = time.perf_counter() - t_phase8

    verified_disj = {di for di, qlist in disj_queries.items()
                      if all(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist)}
    still_open_disj = set(disj_queries.keys()) - verified_disj

    # --- Phase 9: final PGD ---
    t0 = time.perf_counter()
    if still_open_disj and time_left() > 0:
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

    Returns (result_str, details_dict).
    """
    impl = str(getattr(settings, 'graph_impl', 'optimized'))
    assert impl in _BUILDERS, f'unknown graph_impl: {impl!r}'
    build_fn = _BUILDERS[impl]
    return _run_pipeline(graph, spec, settings, build_fn, impl)
