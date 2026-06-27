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
from .pgd import pgd_box_expand_amount
from . import verify_gen_lp


_OK_STATUSES_GLOBAL = None

_FAST_DA_VERIFIERS = {}


def _get_fast_da_verifier(device, ls, K, compile, sweeps=1):
    """Cached fast GPU dual-ascent Verifier (one per (device, ls, K, compile, sweeps)).

    The `torch.compile` warmup (~3 s) is paid once per process, then the same
    Verifier is reused across every query/case in that process — so build it
    lazily and memoize. See `fast_dual_ascent` for the kernel.

    `sweeps > 1` routes to the K-step dual-ascent kernel
    (`fast_verify_dual.Verifier`, K = sweeps): it runs `sweeps` λ-ascent
    iterations per BaB node (warm-started), giving a much tighter per-node
    bound than the default 1-step logbucket verifier. Far fewer nodes survive
    per level → the frontier stays small enough to fit (the metaroom q8 case
    needs this: 1-step → 70M-node OOM; ~20 sweeps → ~1M frontier, closes)."""
    key = (str(device), ls, int(K), bool(compile), int(sweeps))
    v = _FAST_DA_VERIFIERS.get(key)
    if v is None:
        if int(sweeps) > 1:
            from .fast_dual_ascent.fast_verify_dual import Verifier as _KStep
            v = _KStep(device=str(device), compile=compile, K=int(sweeps))
        else:
            from .fast_dual_ascent import Verifier
            v = Verifier(device=str(device), compile=compile, ls=ls, K=K)
        _FAST_DA_VERIFIERS[key] = v
    return v


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
        elif t in ('sigmoid', 'tanh'):
            if 'layer_idx' in op:
                d['layer_idx'] = op['layer_idx']
        elif t == 'add':
            d['is_merge'] = op.get('is_merge', False)
            if not d['is_merge']:
                d['bias'] = op.get('bias')
        elif t == 'sub':
            d['bias'] = op.get('bias')
        elif t == 'mul':
            d['scale'] = op.get('scale')
            d['in_shapes_nd'] = op.get('in_shapes_nd')
            d['out_shape_nd'] = op.get('out_shape_nd')
        elif t == 'reshape':
            pass    # flat passthrough; consumers alias the input vars
        elif t in ('slice', 'gather'):
            d['flat_idx'] = op['flat_idx']
        elif t in ('matmul_bilinear', 'softmax', 'concat', 'squeeze',
                   'exp', 'reciprocal', 'reduce_sum', 'mul_bilinear',
                   'sub_bilinear'):
            # carried with full geometry; consumers that cannot handle
            # these types still raise in their own dispatch.
            # sub_bilinear (a-b, linear) appears in the maxpool_to_relu
            # decomposition's binary-max tree (max(a,b)=a+ReLU(b-a)).
            d['axis'] = op.get('axis')
            d['axes'] = op.get('axes')
            d['keepdims'] = op.get('keepdims')
            d['in_shapes_nd'] = op.get('in_shapes_nd')
            d['out_shape_nd'] = op.get('out_shape_nd')
        else:
            raise NotImplementedError(
                f'gg serializer: unsupported op {t!r} at '
                f"{op.get('name')!r} — serializing without its params "
                f'would make consumers run a different network')
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

        else:
            raise NotImplementedError(
                f'LP builder: unsupported op {t!r} at {nm!r} — skipping '
                f'it would encode a different network')

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

        else:
            raise NotImplementedError(
                f'compact-LP builder: unsupported op {t!r} at {nm!r} — '
                f'skipping it would encode a different network')

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
    except (grb.GurobiError, AttributeError):
        # ObjBound unavailable (no LP relaxation produced); keep prior bound.
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
        except (grb.GurobiError, AttributeError):
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
        except (grb.GurobiError, AttributeError):
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
        except (grb.GurobiError, AttributeError):
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
                    except (grb.GurobiError, AttributeError):
                        # .X unavailable without integer solution; skip.
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


def _resolve_high_bin_count(raw):
    """Resolve ``phase8_high_bin_count`` to an int cap, or ``None`` = "all".

    The high-bin fallback binarizes up to this many unstable neurons. The
    sentinel ``'all'`` (also ``None`` / ``float('inf')`` / any value <= 0) means
    "binarize EVERY unstable neuron" (a full MILP) and returns ``None`` so the
    caller can resolve it to the per-query neuron count — replacing the old
    magic ``10000`` ("bigger than any net") with an explicit flag.
    """
    if raw is None or raw == 'all':
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v == float('inf') or v <= 0:
        return None
    return int(v)


def _solve_high_bin_query(state_q, qw_q, qb_q, scored_keys_q, *, n_bins,
                          time_limit, n_threads, use_bbs, bbs_tol,
                          stop_event=None):
    """Build + solve the high-bin MILP for ONE query (Gurobi, ``n_threads`` cores).

    Returns ``(closed: bool, lb: float | None)``. SOUND: ``closed`` iff Gurobi
    PROVED ``min(qw·y+qb) >= bbs_tol > 0`` (BestBdStop ObjBound certificate) or
    (legacy) the spec-halfspace model is INFEASIBLE — never inferred from an
    early termination. ``stop_event`` (when set mid-solve) makes Gurobi
    ``terminate()`` so the loser of a BnB∥MILP race stops promptly; a terminated
    solve that hasn't proved the bound simply returns ``closed=False``.

    This is the same logic as the sequential high-bin fallback below, factored
    out so the parallel-race path can call it from a worker thread.
    """
    import gurobipy as _grb
    from .gurobi_util import optimize_checked, GurobiNumericTrouble
    try:
        m_fb, env_fb, _, _ = verify_gen_lp.build_gen_lp_from_state(
            state_q, qw_q, qb_q,
            milp_set=set(scored_keys_q[:n_bins]),
            n_threads=n_threads,
            unsafe_halfspace=('none' if use_bbs else 'inequality'))
        m_fb.setParam('TimeLimit', float(time_limit))
        m_fb.setParam('BestBdStop', bbs_tol if use_bbs else _grb.GRB.INFINITY)
        _cb = None
        if stop_event is not None:
            def _cb(_m, _where):     # terminate promptly once the race is decided
                if stop_event.is_set():
                    _m.terminate()
        try:
            optimize_checked(m_fb, user_callback=_cb)
        except (GurobiNumericTrouble, _grb.GurobiError):
            m_fb.dispose(); env_fb.dispose()
            return False, None
        status = m_fb.Status
        try:
            ob = float(m_fb.ObjBound)
        except (_grb.GurobiError, AttributeError):
            ob = None
        m_fb.dispose(); env_fb.dispose()
    except (_grb.GurobiError, GurobiNumericTrouble):
        return False, None
    if use_bbs:
        return (ob is not None and ob >= bbs_tol), ob
    return (status == _grb.GRB.INFEASIBLE), ob


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
            # ew passes through unchanged → only sound with no outer-broadcast
            # operand (reduce-sum adjoint lives in _spec_backward_graph).
            from .broadcast_util import assert_no_outer_broadcast
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'lb_direct')
            if op.get('is_merge'):
                for inp in op['inputs']:
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
            else:
                bias = op.get('bias')
                if bias is not None:
                    from .alpha_crown import _bias_dot_ew
                    acc += float(_bias_dot_ew(ew, bias, dtype, device))
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'sub':
            from .broadcast_util import assert_no_outer_broadcast
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'lb_direct')
            bias = op.get('bias')
            if bias is not None:
                from .alpha_crown import _bias_dot_ew
                acc -= float(_bias_dot_ew(ew, bias, dtype, device))
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'reshape':
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'mul':
            from .alpha_crown import _mul_scale_backward
            ew_back = _mul_scale_backward(op, ew, dtype, device)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t in ('sigmoid', 'tanh'):
            from .verify_zono_bnb import _sigmoid_tanh_linear_bounds
            L = op.get('layer_idx')
            lo_pre_np, hi_pre_np = bounds_by_relu[L]
            lo_pre = torch.as_tensor(lo_pre_np, dtype=dtype, device=device)
            hi_pre = torch.as_tensor(hi_pre_np, dtype=dtype, device=device)
            lo_s, lo_t_b, up_s, up_t_b = _sigmoid_tanh_linear_bounds(
                lo_pre, hi_pre, t)
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc += float((ep * lo_t_b).sum() + (en * up_t_b).sum())
            ew_back = ep * lo_s + en * up_s
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        else:
            raise NotImplementedError(
                f'_compute_lb_direct backward: unsupported op {t!r} '
                f'(name={op.get("name")!r}). Silent skip would drop ew → unsound.')

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
    if target_op_name is None:
        # The per-neuron adaptive bounds are a ReLU-triangle method; they only
        # apply to layers that have a ReLU op. A non-ReLU activation (sigmoid/
        # tanh) at this layer index means the caller invoked the ReLU-adaptive
        # path on the wrong layer (a mixed net like dist_shift's mnist_concat).
        # Fail loud rather than silently mis-tighten a non-ReLU layer.
        _types_here = sorted({op['type'] for op in ops
                              if op.get('layer_idx') == target_layer_idx})
        raise NotImplementedError(
            f'per-neuron adaptive (ReLU-triangle) bounds requested for layer '
            f'{target_layer_idx}, which has no ReLU op (op types at this layer: '
            f'{_types_here}). These bounds apply only to ReLU layers; a non-ReLU '
            f'activation (sigmoid/tanh) must be excluded from the adaptive loop.')

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
            # ew passes through unchanged → only sound with no outer-broadcast
            # operand (reduce-sum adjoint lives in _spec_backward_graph).
            from .broadcast_util import assert_no_outer_broadcast
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'per_neuron')
            if op.get('is_merge'):
                for inp in op['inputs']:
                    _accum(ew_at_lb, inp, ew_lb)
                    _accum(ew_at_ub, inp, ew_ub)
            else:
                bias = op.get('bias')
                if bias is not None:
                    from .alpha_crown import _bias_dot_ew
                    acc_lb = acc_lb + _bias_dot_ew(ew_lb, bias, dtype, device)
                    acc_ub = acc_ub + _bias_dot_ew(ew_ub, bias, dtype, device)
                inp = op['inputs'][0]
                _accum(ew_at_lb, inp, ew_lb)
                _accum(ew_at_ub, inp, ew_ub)

        elif t == 'sub':
            from .broadcast_util import assert_no_outer_broadcast
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'per_neuron')
            bias = op.get('bias')
            if bias is not None:
                from .alpha_crown import _bias_dot_ew
                acc_lb = acc_lb - _bias_dot_ew(ew_lb, bias, dtype, device)
                acc_ub = acc_ub - _bias_dot_ew(ew_ub, bias, dtype, device)
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb)
            _accum(ew_at_ub, inp, ew_ub)

        elif t == 'reshape':
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb)
            _accum(ew_at_ub, inp, ew_ub)

        elif t == 'mul':
            from .alpha_crown import _mul_scale_backward
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp,
                   _mul_scale_backward(op, ew_lb, dtype, device))
            _accum(ew_at_ub, inp,
                   _mul_scale_backward(op, ew_ub, dtype, device))

        elif t in ('sigmoid', 'tanh'):
            from .verify_zono_bnb import _sigmoid_tanh_linear_bounds
            L_idx = op.get('layer_idx')
            lo_k, hi_k = bounds_by_relu[L_idx]
            lo_k_t = torch.as_tensor(lo_k, dtype=dtype, device=device)
            hi_k_t = torch.as_tensor(hi_k, dtype=dtype, device=device)
            lo_s, lo_t_b, up_s, up_t_b = _sigmoid_tanh_linear_bounds(
                lo_k_t, hi_k_t, t)
            # Lower bound backward: use lower-bound slope for positive
            # part, upper-bound slope for negative part.
            ep_lb = ew_lb.clamp(min=0); en_lb = ew_lb.clamp(max=0)
            acc_lb = acc_lb + (ep_lb * lo_t_b).sum(dim=-1) \
                            + (en_lb * up_t_b).sum(dim=-1)
            ew_lb_back = ep_lb * lo_s + en_lb * up_s
            # Upper bound backward: swap.
            ep_ub = ew_ub.clamp(min=0); en_ub = ew_ub.clamp(max=0)
            acc_ub = acc_ub + (ep_ub * up_t_b).sum(dim=-1) \
                            + (en_ub * lo_t_b).sum(dim=-1)
            ew_ub_back = ep_ub * up_s + en_ub * lo_s
            inp = op['inputs'][0]
            _accum(ew_at_lb, inp, ew_lb_back)
            _accum(ew_at_ub, inp, ew_ub_back)

        else:
            raise NotImplementedError(
                f'_per_neuron_adaptive_bounds backward: unsupported op {t!r} '
                f'(name={op.get("name")!r}). Silent skip would drop ew.')

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

        else:
            # Unknown op: retention bookkeeping only — conservatively mark
            # ALL of its inputs fully needed (sound superset; never lets a
            # lagging op list silently release bounds a new op still uses).
            for inp in op['inputs']:
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

        else:
            raise NotImplementedError(
                f'window builder: unsupported op {t!r} at {name!r} — '
                f'skipping it would encode a different network')

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
    except (grb.GurobiError, AttributeError):
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
        except (grb.GurobiError, AttributeError):
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
    if target_op_name is None:
        # ReLU-triangle adaptive bounds — only valid on ReLU layers. A non-ReLU
        # activation (sigmoid/tanh) at this layer means the adaptive loop was
        # invoked on the wrong layer; fail loud rather than mis-tighten it.
        raise NotImplementedError(
            f'per-neuron adaptive (ReLU-triangle) bounds requested for layer '
            f'{target_layer_idx}, which has no ReLU op; these apply only to '
            f'ReLU layers (exclude sigmoid/tanh layers from the adaptive loop).')

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

        else:
            raise NotImplementedError(
                f'_sparse_neuron forward zono: unsupported op {t!r} '
                f'(name={name!r}). Silent skip would propagate stale zono.')

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
    # Use the live forward zono's center as the source of truth for
    # n_output. The op-list serializer strips shape metadata for some
    # op types (e.g. 'add' at the tail of a Conv+Add output layer),
    # which would otherwise fall back to the last ReLU's width.
    n_output = int(z_final_initial.center.numel())
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
    # Pure-linear nets (no ReLU layers) — nothing to BaB-tighten; CROWN
    # is already exact. Bail early.
    if not bounds_by_relu:
        sb = {}
        tb['phase1_bab_refine'] = time.perf_counter() - t_total
        return sb, bounds_by_relu, z_final_initial, tb
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

        On a wide net the per-target backward can OOM. Rather than SKIP the
        wide layers (a gate that leaves their bounds loose — the old
        `phase1_alpha_refresh_mem_elems` behavior), CHUNK harder: catch the
        OOM and retry with double the S-split (which cuts peak autograd
        retention ~1/N), up to a cap. The trace records each step.
        """
        _dir_mode = str(settings.alpha_crown_dir_mode
                         if 'alpha_crown_dir_mode' in settings
                         else 'auto')
        _s_split = max(1, int(settings.alpha_crown_s_split_n
                              if 'alpha_crown_s_split_n' in settings
                              else 1))
        _s_split_cap = int(getattr(
            settings, 'alpha_crown_s_split_max', 64))
        # sparse_alpha (ABC's `sparse_alpha: true`): allocate α (+Adam state)
        # only for UNSTABLE neurons per layer, not all n. The α/Adam state is
        # the dominant memory here, so this shrinks it ~n/n_unstable× and fits
        # the wide cnn7's 131072-neuron refresh with no target cap — same math
        # as dense (stable neurons' slopes are fixed regardless), measured
        # identical roots on idx9074. Default on.
        _sparse_a = bool(getattr(settings, 'alpha_refresh_sparse_alpha', True))
        while True:
            try:
                _, _, best_bounds, _ = ac.run_alpha_crown_batched(
                    gg, xl, xh, bbr_now, w_qs, b_qs, S_nodes, un_idx,
                    device, dtype, n_iters=n_iters, lr=alpha_lr,
                    lr_decay=0.98, early_stop_on_positive=False,
                    dir_mode=_dir_mode, s_split_n=_s_split,
                    sparse_alpha=_sparse_a, time_left_fn=time_left)
                return best_bounds
            except torch.cuda.OutOfMemoryError:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if _s_split >= _s_split_cap:
                    # Even fully chunked (s_split == cap) and with sparse_alpha
                    # the per-target backward won't fit. The α-refresh is a pure
                    # TIGHTENING step: returning {} leaves the caller's merge
                    # loop a no-op, so `bounds_by_relu` keeps its SOUND (looser)
                    # Phase-1 intermediate bounds. Degrade to looser-but-sound
                    # rather than erroring — a sound timeout, never a wrong
                    # verdict. NOT the banned silent-OOM-swallow (that hides
                    # wrong RESULTS); the fallback is provably sound + logged.
                    # NEXT LEVER IF THIS EVER FIRES (not implemented): cap the
                    # refresh to the top-K widest-gap (loosest) unstable neurons
                    # per layer (ABC's select_unstable_idx / max_crown_size) —
                    # fewer targets = less α state. sparse_alpha alone fit every
                    # cct2026 case so far, so it's unbuilt.
                    print(f'  [phase0.5] α-refresh OOM at s_split={_s_split} '
                          f'(cap) even with sparse_alpha -> keeping looser-but-'
                          f'sound Phase-1 bounds (sound timeout, not an error). '
                          f'To go further, cap targets to top-K widest-gap '
                          f'unstable neurons (ABC select_unstable_idx; '
                          f'not implemented).', flush=True)
                    return {}
                _s_split *= 2
                if print_progress:
                    print(f'  [phase0.5] α-refresh OOM at s_split='
                          f'{_s_split // 2} -> chunk harder, retry '
                          f's_split={_s_split}', flush=True)

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
    # Memory cap on the α-refresh start nodes: the per-target backward in
    # `_alpha_refresh_best_bounds` materializes a peak tensor of ~n_targets ×
    # ALL unstable layers are refreshed (no mem-cap skip). Wide layers that
    # would OOM the per-target backward are handled by CHUNKING harder inside
    # `_alpha_refresh_best_bounds` (OOM → double the S-split), not by skipping
    # them — skipping left their intermediate bounds loose, which gave a loose
    # root and exploded the BaB frontier (cct2026 idx9074: cap-skip → root
    # −0.69 / q6 explodes; refresh-all → root −0.44 / q2 closes). The legacy
    # `phase1_alpha_refresh_mem_elems` cap is retired.
    _unstable_layers = [
        Lk for Lk in bounds_by_relu if Lk > 0 and
        ((bounds_by_relu[Lk][0] < 0)
         & (bounds_by_relu[Lk][1] > 0)).any()]
    intermediate_start_nodes_init = list(_unstable_layers)
    unstable_indices_init = {
        Lk: np.where((bounds_by_relu[Lk][0] < 0)
                      & (bounds_by_relu[Lk][1] > 0))[0].tolist()
        for Lk in bounds_by_relu}
    if print_progress:
        _wid = {Lk: len(bounds_by_relu[Lk][0])
                for Lk in intermediate_start_nodes_init}
        print(f'  [phase0.5] α-refresh: all '
              f'{len(intermediate_start_nodes_init)} unstable layers '
              f'(widths {_wid}); wide layers chunked on OOM, not skipped',
              flush=True)
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
        # per-spec α (a separate α per open query) is much tighter than a
        # shared α when only a few queries remain open and they pull α in
        # different directions (cct2026 idx9074 q2/q6: shared −0.58 vs ABC's
        # per-spec −0.37). Gated by a setting so other benchmarks (where the
        # shared α already matched ABC) are unchanged; default False.
        _per_spec_a = bool(getattr(
            settings, 'phase05_per_spec_alpha', False))
        spec_lbs_phase05, alpha_dict_p05, _, _ = (
            ac.run_alpha_crown_fixed_intermediate_batched(
                gg, xl, xh, bounds_by_relu, w_qs, b_qs,
                device, dtype, n_iters=phase05_spec_iters, lr=alpha_lr,
                lr_decay=0.98, early_stop_on_positive=True,
                per_spec_alpha=_per_spec_a, time_left_fn=time_left))
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
        # Stash Phase 0.5's per-layer α tensors so Phase 8's per-query
        # α-CROWN can warm-start from already-optimised slopes instead
        # of restarting at min-area. Keyed by spec qi for per-query
        # passthrough (Phase 0.5 uses a shared α across queries; the
        # same tensor is reused per qi).
        tb['spec_alpha_phase05'] = spec_alpha_phase05
        if print_progress:
            n_closed = len(queries_flat) - len(open_qis)
            # Record the GAP on the open queries (their spec LB) — this is
            # the root bound the BaB starts from. A loose root here (vs ABC's
            # α-CROWN) is what makes the dual-ascent frontier explode; show it
            # per open query so the trace pinpoints it without re-deriving.
            _open_gaps = {queries_flat[qi][0]: round(float(spec_lbs_phase05[qi]), 4)
                          for qi in open_qis}
            print(f'  [phase0.5] α-CROWN closed {n_closed}/'
                  f'{len(queries_flat)} specs; open-query root LBs '
                  f'{_open_gaps}', flush=True)

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
        # Skip the per-layer gen-LP MILP block entirely when settings
        # force `tighten_formulation='skip'` (e.g. graphs with Pow,
        # ReduceSum, mul/div_bilinear ops the gen-LP state builder
        # doesn't support).
        if str(getattr(settings, 'tighten_formulation', 'gen_cone')) == 'skip':
            break
        # Skip L=0: zonotope forward is exact at the first hidden layer
        # (z_0 = W_0 x + b_0 over a box-shaped input has tight closed-form
        # interval bounds), so per-neuron MILP at L=0 has no binaries to
        # branch on (nothing upstream is unstable) and produces the same
        # bound as the box. Saves the ~0.2s no-op and matches AB-CROWN's
        # `if relu_idx >= 1` guard at lp_mip_solver.py:1849.
        # gen-LP MILP only encodes ReLU triangles. If any sigmoid/tanh
        # layer exists at index ≤ L, the dependency cone for L's MILP
        # would need to walk back through that nonlinear op which the
        # gen-LP precompute can't handle. Cap max_layer to one less
        # than the first sigmoid/tanh index.
        non_relu_layer_ids = sorted({op['layer_idx'] for op in gg_ops_ser
                                      if op['type'] in ('sigmoid', 'tanh')
                                      and 'layer_idx' in op})
        if non_relu_layer_ids:
            cap = non_relu_layer_ids[0] - 1
            if cap < max_layer:
                max_layer = cap
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
            # Propagate the just-tightened L's bounds to ALL downstream
            # layers Lk > L. Two passes:
            #   (a) per-neuron adaptive backward CROWN at each Lk > L
            #       (cheap, ~10 ms/layer; USES `bounds_by_relu[L]` as
            #       fixed triangle slopes, so the MILP gain at L feeds
            #       forward exactly).
            #   (b) joint α-CROWN refresh on ALL intermediate bounds
            #       (heavier, ~0.5-1 s; jointly optimizes α per layer).
            # The joint α-CROWN re-computes L's bound from scratch and
            # may DROP the MILP gain at L — we always max() to keep the
            # tighter of (MILP, α-CROWN) per layer.
            t_a = time.perf_counter()
            # (a) Adaptive backward for downstream layers. ALWAYS run —
            # this is ~10 ms / layer × few layers; even if MILP overran
            # the Phase 8 budget reserve, the extra is negligible and
            # propagates the L tightening to all Lk > L (the joint
            # α-CROWN refresh below re-computes L's bound from scratch
            # and discards the MILP gain, so this is the ONLY path the
            # MILP-at-L gain reaches Lk > L).
            for Lk in sorted(bounds_by_relu.keys()):
                if Lk <= L:
                    continue
                lo_k, hi_k = bounds_by_relu[Lk]
                un_k = np.where((lo_k < 0) & (hi_k > 0))[0]
                if un_k.size == 0:
                    continue
                try:
                    new_lo, new_hi = _per_neuron_adaptive_bounds_chunked(
                        gg, xl, xh, bounds_by_relu, Lk,
                        device, dtype, neuron_subset=un_k)
                    bounds_by_relu[Lk] = (
                        np.maximum(lo_k, new_lo),
                        np.minimum(hi_k, new_hi))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as _e:
                    # Cascade tightening failure (GPU OOM or runtime). Log
                    # and keep prior bounds for this layer — the cascade is
                    # an opportunistic tightener, not load-bearing for soundness.
                    if getattr(settings, 'print_progress', False):
                        print(f'  [cascade] layer {Lk} tighten failed: '
                              f'{type(_e).__name__}', flush=True)
            # (b) Joint α-CROWN refresh (only if we still have time).
            if time_left() > 0.5:
                # All unstable layers (no mem-cap pre-filter): the chunked,
                # OOM-graceful `_alpha_refresh_best_bounds` handles wide layers
                # by escalating the S-split and degrading to sound-looser at the
                # cap, so the old `_refresh_fits(Lk)` memory gate is retired
                # (it was removed with `phase1_alpha_refresh_mem_elems`).
                intermediate_start_nodes = [
                    Lk for Lk in bounds_by_relu if Lk > 0
                    and ((bounds_by_relu[Lk][0] < 0)
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
            # Now check whether to continue cascade to next layer.
            if time_left() <= 1:
                break
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
    # Re-run spec α-CROWN with the cascade-tightened intermediate bounds
    # so Phase 8 can warm-start from the post-tighten optimal α. Without
    # this, Phase 8 warm-starts from Phase 0.5's pre-tighten α, which is
    # stale w.r.t. the new bounds. Cost: one batched spec α call (~0.1-1s).
    if queries_flat and bool(getattr(
            settings, 'phase1_final_spec_alpha', True)) and time_left() > 1.0:
        t_a = time.perf_counter()
        try:
            _, alpha_dict_p1, _, _ = (
                ac.run_alpha_crown_fixed_intermediate_batched(
                    gg, xl, xh, bounds_by_relu, w_qs, b_qs,
                    device, dtype, n_iters=phase05_spec_iters, lr=alpha_lr,
                    lr_decay=0.98, early_stop_on_positive=True,
                    per_spec_alpha=False, time_left_fn=time_left))
            spec_alpha_phase1 = alpha_dict_p1.get('spec')
            if spec_alpha_phase1 is not None:
                tb['spec_alpha_phase05'] = spec_alpha_phase1
        except (RuntimeError, torch.cuda.OutOfMemoryError) as _e:
            # Spec α-CROWN refresh: GPU runtime/OOM. Falling back to the
            # warm-start α is sound but loses tightening. Log so we notice
            # when this is happening at scale.
            if getattr(settings, 'print_progress', False):
                print(f'  [phase0.5] spec α-CROWN refresh failed: '
                      f'{type(_e).__name__}', flush=True)
        tb['phase1_alpha_total'] += time.perf_counter() - t_a
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
        else:
            # NOTE: the maxpool->relu lowering (mp2relu) emits a chain of
            # specialised ops — `slice` (phase gather) then `sub_bilinear` (the
            # nonlinear max relaxation) — that this DENSE keep-pre forward does not
            # implement (full-image maxpool uses the patches forward in
            # verify_zono_bnb instead). Raising here is the sound behaviour: a
            # silent skip would propagate a stale pre-activation snapshot. This path
            # is not reached in normal operation (sat cases resolve via PGD first;
            # full-image cases use patches), so we keep the honest refusal rather
            # than a half-ported maxpool that still breaks at `sub_bilinear`.
            raise NotImplementedError(
                f'_forward_keep_pre_gpu: unsupported op {t!r} '
                f'(name={name!r}). Silent skip would propagate stale zono.')
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
            if closed_here:
                # ANY-closure: one refuted conjunct closes the whole
                # disjunct — skip its sibling queries (saves ~26s/query on
                # yolo's 5-conjunct specs).
                break
        info['per_query'][di] = q_info
        # ANY-closure: the disjunct is a conjunction — one refuted conjunct
        # (closed query) closes it. any_open_after stays as the legacy
        # all-closed signal; either condition is sound, ANY is complete
        # for multi-conjunct specs (yolo). Single-conjunct disjuncts
        # (regular track) behave identically.
        _any_closed = any(bool(v.get('closed')) for v in q_info.values())
        if not any_open_after or _any_closed:
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


def _sat_disposition(graph, spec, settings, witness, info):
    """Classify an ORT-VALIDATED sat witness (from `_validate_sat_witness`, ok=True).

    Always returns 'real'. Under the VNN-COMP 2026 output-strict rule the validation
    gate runs with out_atol=0 and only admits a genuine violation (worst margin
    <= 0 for the spec's `<=`/`>=` comparison), so any witness reaching here IS a real
    counterexample — commit 'sat' and stop. There is no within-output-tolerance
    near-miss any more (that emit-early-then-keep-searching path is removed); the
    return value is kept for the call sites that branch on it.
    """
    return 'real'


def _clamp_witness_to_box(witness, x_lo, x_hi, slack=0.0):
    """Clamp a counterexample witness into the input box `[x_lo, x_hi]` (widened
    by `slack` on each side) so it stays inside even after the float32 cast ORT /
    the VNNCOMP scorer applies.

    A box edge (e.g. `x >= 9.2`) is not generally representable in float32, so
    a witness sitting exactly on it can round to the *outside* of the box when
    cast (`float32(9.2) < 9.2`), failing the scorer's `x < lb - tol` test. Here
    we clamp into the box in float64, then pull any component whose float32 cast
    landed outside back toward the interior by one float32 ULP. The result is a
    float64 array that is provably within `[x_lo-slack, x_hi+slack]` both as
    float64 and as float32 (the coarser grid → also safe for float64 models).

    `slack` (default 0.0) widens the clamp target to `[lo-slack, hi+slack]` for
    the box-expansion path: a counterexample that the box-expanded PGD found just
    OUTSIDE the box must be kept there (clamping it strictly to `[lo, hi]` would
    re-evaluate the output at the boundary and lose the violation). With
    `slack <= sat_validate_atol` the result is still within the scorer's
    `[lo-atol, hi+atol]` acceptance region (scores CORRECT_WITH_TOLERANCE).

    Pure function: uses `np.nextafter`/`np.clip` only — it does NOT touch any FP
    rounding mode, so verification arithmetic elsewhere is unaffected (the clamp
    runs only at witness validation / output time).
    """
    w = np.asarray(witness, np.float64).flatten()
    lo = np.asarray(x_lo, np.float64).flatten() - slack
    hi = np.asarray(x_hi, np.float64).flatten() + slack
    w = np.minimum(np.maximum(w, lo), hi)            # into [lo, hi] in float64
    w32 = w.astype(np.float32)
    cast = w32.astype(np.float64)                    # value the scorer sees
    below = cast < lo                                # rounded under the floor
    above = cast > hi                                # rounded over the ceiling
    w32 = np.where(below, np.nextafter(w32, np.float32(np.inf)), w32)
    w32 = np.where(above, np.nextafter(w32, np.float32(-np.inf)), w32)
    return w32.astype(np.float64)


_ORT_VALIDATE_SESSIONS = {}


def _ort_session_for(onnx_path):
    """Cached CPU InferenceSession for `onnx_path` (handles `.onnx.gz`)."""
    sess = _ORT_VALIDATE_SESSIONS.get(onnx_path)
    if sess is None:
        import onnxruntime as ort
        if onnx_path.endswith('.gz'):
            import gzip
            with gzip.open(onnx_path, 'rb') as _f:
                sess = ort.InferenceSession(_f.read(),
                                            providers=['CPUExecutionProvider'])
        else:
            sess = ort.InferenceSession(onnx_path,
                                        providers=['CPUExecutionProvider'])
        _ORT_VALIDATE_SESSIONS[onnx_path] = sess
    return sess


def _validate_witness_ort(onnx_path, witnesses, boxes, output_violated, atol=1e-4,
                          emit_slack=0.0):
    """Unified ORT-CPU witness validator for EVERY path (graph + surrogate/attack).

    This is the single authoritative gate: load the ORIGINAL ONNX, replay the
    witness on CPU onnxruntime, and check it actually violates the spec — catching
    spurious counterexamples from PGD/MILP/graph-builder bugs. It is multi-input
    (the surrogate/attack paths produce multi-tensor witnesses) and the
    output-violation rule is supplied by the caller (`output_violated`), so each
    spec representation keeps its own correct semantics (the graph path's inclusive
    `spec.check`/disjunctive `check_witness`; the surrogate's strict `>`/`<` margin
    with `sat_strict_buffer`).

    witnesses: list of per-input flat float arrays (single-input -> 1-element list).
    boxes:     list of (lo, hi) flat arrays per input, or None to skip the box check.
    output_violated(inbox_list, y_flat) -> (violated: bool, info_updates: dict).

    Returns (proceed, info). proceed=True means "emit this sat" — the output
    genuinely violates, OR validation was skipped (no onnx_path / onnxruntime
    missing, so we don't reject what we cannot check). proceed=False means
    reject/downgrade (out-of-box, ORT failure, or output does not violate). info
    carries 'out', 'witness_inbox'/'witnesses_inbox', 'spec_check', and any
    `output_violated` updates (e.g. 'worst_margin').
    """
    info = {'ok': False, 'reason': None}
    if onnx_path is None:
        info['reason'] = 'no onnx_path stashed on graph; skipping validation'
        return True, info
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        info['reason'] = 'onnxruntime not installed; skipping validation'
        return True, info
    # Per-input box check (witness up to `atol` outside is allowed). Validation +
    # emission then run on the float32-safe CLAMPED point (a box edge can round
    # outside under the float32 cast ORT/the scorer apply; see
    # `_clamp_witness_to_box`), never the raw one. Try the STRICT in-box clamp
    # first (scores CORRECT); only if that does NOT violate and `emit_slack` > 0 do
    # we retry keeping the witness up to `emit_slack` OUTSIDE the box — the case
    # where box-expanded PGD found a CE just outside (clamping it strictly in would
    # re-evaluate the output at the boundary and lose the violation; the just-
    # outside witness scores CORRECT_WITH_TOLERANCE, no penalty).
    raw, boxes_lh = [], []
    for k, w in enumerate(witnesses):
        w = np.asarray(w, np.float64).ravel()
        box = boxes[k] if boxes is not None else None
        if box is not None:
            lo = np.asarray(box[0], np.float64).ravel()
            hi = np.asarray(box[1], np.float64).ravel()
            if w.shape != lo.shape:
                info['reason'] = (f'witness[{k}] shape {w.shape} != box shape '
                                  f'{lo.shape}')
                return False, info
            if np.any(w < lo - atol) or np.any(w > hi + atol):
                info['reason'] = f'witness[{k}] outside input box (atol={atol})'
                info['out_of_box'] = (float((lo - w).max()), float((w - hi).max()))
                return False, info
            box = (lo, hi)
        raw.append(w)
        boxes_lh.append(box)

    def _eval_at(slack):
        """Clamp each witness into [lo-slack, hi+slack], ORT-replay the ORIGINAL
        model. Returns (violated, inbox, y, updates); violated is None on ORT
        error (then updates carries the 'reason')."""
        inbox = [(_clamp_witness_to_box(w, b[0], b[1], slack=slack)
                  if b is not None else w)
                 for w, b in zip(raw, boxes_lh)]
        try:
            sess = _ort_session_for(onnx_path)
            feeds = {}
            for k, im in enumerate(sess.get_inputs()):
                shp = [d if isinstance(d, int) and d > 0 else 1 for d in im.shape]
                feeds[im.name] = inbox[k].reshape(shp).astype(np.float32)
            out = sess.run(None, feeds)[0]
            y = np.asarray(out).flatten().astype(np.float64)
        except (RuntimeError, OSError, ValueError, IndexError, KeyError) as e:
            return None, inbox, None, {
                'reason': f'ORT forward failed: {type(e).__name__}: {e}'}
        violated, updates = output_violated(inbox, y)
        return violated, inbox, y, updates

    violated, inbox, y, updates = _eval_at(0.0)
    if violated is False and float(emit_slack) > 0.0:
        b_violated, b_inbox, b_y, b_updates = _eval_at(float(emit_slack))
        if b_violated is True:                      # CE only holds just outside box
            violated, inbox, y, updates = b_violated, b_inbox, b_y, b_updates
    info['witnesses_inbox'] = inbox
    info['witness_inbox'] = inbox[0] if len(inbox) == 1 else None
    if violated is None:                            # ORT failure -> reject
        info['reason'] = updates.get('reason', 'ORT forward failed')
        return False, info
    info['out'] = y
    if updates:
        info.update(updates)
    info['spec_check'] = 'unknown' if violated else 'verified'
    if violated:
        info['ok'] = True
        return True, info
    if not info.get('reason'):
        info['reason'] = 'ORT output does not violate spec'
    return False, info


def _validate_sat_witness(onnx_path, spec, witness, atol=1e-4, out_atol=0.0,
                          emit_slack=0.0):
    """Run a SAT witness through ONNXRuntime + check it actually violates
    the spec. Catches spurious counterexamples from PGD/MILP bugs OR from
    graph-builder bugs (vibecheck's internal forward might compute a
    different value than the original ONNX). Mirrors VNNCOMP scoring's
    counterexample-validation step.

    VNN-COMP 2026 ruling (evaluation chairs): a witness is CORRECT iff its
    input satisfies the VNN-LIB input constraints AND the *replayed* ORT
    output satisfies the output constraints. The 1e-4 absolute tolerance
    applies ONLY to the INPUT box (a witness up to `atol` outside the box is
    CORRECT_WITH_TOLERANCE — no penalty, but not SAT ground truth). The
    OUTPUT must violate the spec with NO tolerance. Hence `atol` gates the
    input box and `out_atol` (default 0.0 = strict) gates the output. Output
    tolerance is NOT scorer-accepted under the 2026 rule — keep `out_atol=0`.

    Returns (ok, info_dict). `ok=True` iff witness is in the input box
    (within `atol`) AND its ORT output satisfies the unsafe condition
    (i.e., `spec.check(out, out)` returns 'unknown', i.e. worst margin <= 0
    within `out_atol` on constraint margins — inclusive of the boundary,
    matching the official checker's `<=`/`>=` comparison at zero tolerance).
    """
    # Single-input VNNSpec wrapper over the unified `_validate_witness_ort`. The
    # output-violation rule is the graph path's: a per-disjunct X-subrange spec
    # (nn4sys lindex, acasxu prop_6) uses `check_witness(x, y)` (else `spec.check`
    # ignores the X constraints and could report a false SAT); otherwise the
    # Y-only `spec.check` with the output band `out_atol` (default 0.0 = strict,
    # boundary inclusive per the official `<=`/`>=` comparison).
    w = np.asarray(witness).flatten().astype(np.float64)
    _disjunctive = any(getattr(c, 'input_lo', None) is not None
                       for c in spec.disjuncts)

    def _output_violated(inbox, y):
        if _disjunctive:
            is_ce, _ = spec.check_witness(inbox[0].astype(np.float64), y)
            return is_ce, ({} if is_ce else {
                'reason': ('ORT output does not violate any disjunct whose '
                           'X-subrange contains the witness (check_witness rejected)')})
        check_res, check_info = spec.check(y - out_atol, y + out_atol)
        upd = {'worst_margin': check_info.get('worst_margin')}
        if check_res != 'unknown':
            upd['reason'] = (f'ORT output does not violate spec '
                             f'(worst_margin={check_info.get("worst_margin"):.4g}, '
                             f'out_atol={out_atol})')
        return (check_res == 'unknown'), upd

    return _validate_witness_ort(onnx_path, [w], [(spec.x_lo, spec.x_hi)],
                                 _output_violated, atol, emit_slack=emit_slack)


def _validate_verified_with_samples(onnx_path, spec, n_samples=32,
                                     atol=1e-4, rng_seed=0):
    """Defense-in-depth for VERIFIED verdicts.

    Sample N points from the input box, forward through the original ONNX,
    and check that NONE of them counterexamples the spec. If any sample
    is a real counterexample, our 'verified' verdict is unsound — return
    (ok=False, info) so the caller downgrades to 'unknown'.

    Soundness rationale: a true UNSAT property has no counterexample
    anywhere in the input box; finite sampling is NOT a soundness proof
    of UNSAT (we miss adversarial inputs), but ANY counterexample
    discovered is a true counterexample. This catches Class-1 unsoundness
    (verifier silently certified a spec that's actually SAT) at near-zero
    cost.

    For specs with per-disjunct X-subranges, sample uniformly from each
    disjunct's subbox so every disjunct gets coverage.
    """
    info = {'ok': True, 'reason': None, 'n_samples': n_samples,
             'n_checked': 0}
    if onnx_path is None:
        info['reason'] = 'no onnx_path stashed on graph; skipping'
        return True, info
    try:
        import onnxruntime as ort
    except ImportError:
        info['reason'] = 'onnxruntime not installed; skipping'
        return True, info
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
    except (RuntimeError, OSError) as e:
        # ORT session-load: bad onnx file or unsupported op. Sampling
        # validation is best-effort; skip and report ok=True (no extra info).
        info['reason'] = f'ORT session failed: {type(e).__name__}: {e}'
        return True, info
    in_meta = sess.get_inputs()[0]
    in_shape = [d if isinstance(d, int) and d > 0 else 1
                for d in in_meta.shape]
    rng = np.random.default_rng(rng_seed)
    # Per-disjunct subboxes if present, else global x_lo/x_hi.
    subboxes = []
    for ci, conj in enumerate(spec.disjuncts):
        lo = conj.input_lo if conj.input_lo is not None else spec.x_lo
        hi = conj.input_hi if conj.input_hi is not None else spec.x_hi
        subboxes.append((np.asarray(lo, dtype=np.float64),
                          np.asarray(hi, dtype=np.float64)))
    # Distribute samples roughly evenly across subboxes; always at least 1.
    n_per = max(1, n_samples // max(1, len(subboxes)))
    for (lo, hi) in subboxes:
        # 4 corners + (n_per - 4) interior uniform; small subboxes
        # benefit from corner coverage (extremes), interior from
        # uniform coverage.
        n_corner = min(4, n_per)
        samples = []
        for k in range(n_corner):
            mask = (rng.random(lo.shape) < 0.5)
            samples.append(np.where(mask, lo, hi))
        for _ in range(n_per - n_corner):
            samples.append(lo + (hi - lo) * rng.random(lo.shape))
        for x in samples:
            x32 = x.reshape(in_shape).astype(np.float32)
            try:
                out = sess.run(None, {in_meta.name: x32})[0]
            except (RuntimeError, ValueError) as e:
                # Sample-time ORT run failure (rare; shape mismatch).
                # Treat as inconclusive ok=True (sampling is best-effort).
                info['reason'] = (f'ORT forward failed: '
                                    f'{type(e).__name__}: {e}')
                return True, info
            out_flat = np.asarray(out).flatten().astype(np.float64)
            info['n_checked'] += 1
            is_ce, _ = spec.check_witness(
                x.flatten().astype(np.float64), out_flat)
            if is_ce:
                info['ok'] = False
                info['reason'] = ('sample is a real counterexample '
                                    '(VERIFIED is unsound)')
                info['witness'] = x.flatten().astype(np.float64)
                info['witness_out'] = out_flat
                return False, info
    return True, info


def _zono_alpha_close(gg, xl, xh, w_q, b_q, device, dtype,
                      settings, time_left, n_iters=60, lr=0.1,
                      tight_bounds=None, op_clamps=None):
    """Optimize per-neuron ReLU slopes (lam in [0,1]) of the PLAIN graph
    zono forward against ONE open query's margin (alpha-zono, forward
    mode). Every lam is a sound relaxation, so the best-of over iters is
    sound; returns True iff some iterate proves lb > 0. Built for the
    vit attention nets where no CROWN backward exists."""
    w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
        np.asarray(w_q, np.float64), device=device, dtype=dtype)
    relu_ops = [op for op in gg['ops']
                if op['type'] == 'relu' and 'layer_idx' in op]
    # init at min-area-ish 0.5; sizes from a probe forward
    sb0, _ = _forward_zonotope_graph(xl, xh, gg, device, dtype,
                                     settings=settings,
                                     tight_bounds=tight_bounds,
                                     op_clamps=op_clamps)
    lams = {op['layer_idx']: torch.full(
                (sb0[op['layer_idx']][0].numel(),), 0.5,
                device=device, dtype=dtype, requires_grad=True)
            for op in relu_ops if op['layer_idx'] in sb0}
    # convex 1-D ops (softmax decomposition): slope params keyed by op
    # NAME — the forward interpolates f'(l)..f'(u), any value is sound.
    for op in gg['ops']:
        if op['type'] in ('exp', 'reciprocal'):
            _sh = op.get('out_shape_nd') or op.get('in_shapes_nd', [None])[0]
            if _sh is None:
                continue
            lams[op['name']] = torch.full(
                (int(np.prod(_sh)),), 0.5, device=device, dtype=dtype,
                requires_grad=True)
    if not lams:
        return False
    opt = torch.optim.Adam(list(lams.values()), lr=lr)
    for it in range(n_iters):
        if time_left() <= 1.0:
            return False
        opt.zero_grad()
        _, zf = _forward_zonotope_graph(
            xl, xh, gg, device, dtype, settings=settings,
            relu_lambdas=lams, tight_bounds=tight_bounds,
            op_clamps=op_clamps)
        wv = w_t.to(zf.center.dtype)
        lb = wv @ zf.center + float(b_q) - (wv @ zf.generators).abs().sum()
        if float(lb) > 0:
            return True
        (-lb).backward()
        opt.step()
        with torch.no_grad():
            for p_ in lams.values():
                p_.clamp_(0, 1)
    return False


def _zono_relu_split_close(gg, xl, xh, w_q, b_q, device, dtype,
                            settings, time_left, lams=None, max_nodes=512,
                            plane_params=None):
    """Worst-first BnB on ONE open query over the plain graph zono
    forward, with two split types:

    * **ReLU split** on unstable pre-ReLU neuron (L, j): clamps that
      neuron's bounds via the forward's `tight_bounds` intersection —
      standard ReLU-BaB soundness: each child's relaxation is valid on
      its subdomain (z_j <= 0 / z_j >= 0) and the children cover the
      parent.
    * **Bilinear-input split** on an exp-input coordinate (pre-softmax
      score) — ABC's splittable-Mul analog. Clamping the op's input
      interval at its midpoint (via the forward's `op_clamps`
      intersection) shrinks both exp parallelograms/planes AND the
      downstream product remainder. Same value-split soundness argument
      as the ReLU case (split on the TRUE score value; children cover
      the parent). Proposed only when no unstable relu remains — the
      'exhausted' leaves whose residual gap is pure bilinear slack.

    `lams` (optional alpha-zono slopes) are reused as a warm relaxation.
    Built for the vit nets (~100 unstables, 0.2 s per forward).

    Clamp keys: relu = (layer_idx:int, j) -> side 0/1; exp =
    (op_name:str, j) -> (lo, hi) accumulated interval."""
    import heapq
    w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
        np.asarray(w_q, np.float64), device=device, dtype=dtype)
    exp_names = {op['name'] for op in gg['ops'] if op['type'] == 'exp'}
    exp_numel = {}

    def _mk_oc(clamps):
        oc = {}
        for key, val in clamps.items():
            if not isinstance(key[0], str):
                continue
            nm, j = key
            if nm not in oc:
                oc[nm] = (torch.full((exp_numel[nm],), -np.inf,
                                     device=device, dtype=dtype),
                          torch.full((exp_numel[nm],), np.inf,
                                     device=device, dtype=dtype))
            oc[nm][0][j] = val[0]
            oc[nm][1][j] = val[1]
        return oc or None

    def _mk_rc(clamps):
        # relu split halfspaces for the rbeta Lagrangian: side 0 means
        # z_j <= 0 (finite hi), side 1 means z_j >= 0 (finite lo)
        rc = {}
        for key, side in clamps.items():
            if isinstance(key[0], str):
                continue
            L, j = key
            if L not in rc:
                n = base_sb[L][0].size
                rc[L] = (torch.full((n,), -np.inf, device=device,
                                    dtype=dtype),
                         torch.full((n,), np.inf, device=device,
                                    dtype=dtype))
            if side == 0:
                rc[L][1][j] = 0.0
            else:
                rc[L][0][j] = 0.0
        return rc or None

    def bound(clamps):
        ob = {}
        oc = _mk_oc(clamps)
        rc = _mk_rc(clamps)
        sb_l, zf = _forward_zonotope_graph(
            xl, xh, gg, device, dtype, settings=settings,
            relu_lambdas=lams,
            tight_bounds={L: _mk_tb(L, clamps) for L in
                          {Lj[0] for Lj in clamps
                           if not isinstance(Lj[0], str)}}
                          if clamps else None,
            op_clamps=oc, op_bounds=ob)
        wv = w_t.to(zf.center.dtype)
        lb = float(wv @ zf.center + float(b_q)
                   - (wv @ zf.generators).abs().sum())
        # best-of with the backward CROWN under the SAME clamped relu
        # bounds + this node's op_bounds — each split then re-tightens
        # the bilinear/softmax planes. When the root alpha pass supplied
        # optimized plane parameters, evaluate THOSE planes against the
        # node's own bounds (the params are interpolation coefficients,
        # sound for any bounds).
        try:
            if plane_params is not None:
                from .attn_crown import attn_crown_lb
                with torch.no_grad():
                    lb_bw = float(attn_crown_lb(
                        gg, xl, xh, sb_l, ob, wv, float(b_q),
                        plane_params))
            else:
                bw_lbs, _ = _spec_backward_graph(
                    sb_l, xl, xh, gg, {0: (wv, float(b_q))}, {0},
                    len(sb_l), device, dtype, op_bounds=ob)
                lb_bw = float(bw_lbs[0])
            lb = max(lb, lb_bw)
            if lb <= 0 and (oc is not None or rc is not None):
                # Split node still open: the split constraints enter the
                # bound only through their beta Lagrangian terms — bound
                # clamping alone keeps spurious traces from inputs
                # OUTSIDE the subdomain (a clamped relu routes true z>0
                # through y=0; a clamped exp plane still concretizes
                # over the FULL input box). Short per-node beta/plane
                # optimization, warm-started from the root plane params
                # when available — beta-CROWN's core mechanism.
                from .attn_crown import attn_crown_alpha
                _wp = None
                if plane_params is not None:
                    _wp = {k: v.detach().clone().requires_grad_(True)
                           for k, v in plane_params.items()}
                    for nm2, (cl2, _ch2) in (oc or {}).items():
                        _wp[('beta', nm2)] = torch.zeros(
                            (2, cl2.numel()), device=device, dtype=dtype,
                            requires_grad=True)
                    for L2, (cl2, _ch2) in (rc or {}).items():
                        _wp[('rbeta', L2)] = torch.zeros(
                            (2, cl2.numel()), device=device, dtype=dtype,
                            requires_grad=True)
                lb_beta, _ = attn_crown_alpha(
                    gg, xl, xh, sb_l, ob, wv, float(b_q),
                    n_iters=20, lr=0.2, time_left=time_left,
                    params=_wp, op_clamps=oc, relu_clamps=rc)
                lb = max(lb, lb_beta)
        except NotImplementedError:
            pass    # nets whose backward lacks an op: forward-only node
        return lb, sb_l, ob

    base_sb = {}

    def _mk_tb(L, clamps):
        lo0, hi0 = base_sb[L]
        lo = lo0.copy(); hi = hi0.copy()
        for (L2, j), side in clamps.items():
            if L2 != L:
                continue
            if side == 0:
                hi[j] = min(hi[j], 0.0)
            else:
                lo[j] = max(lo[j], 0.0)
        return lo, hi

    lb0, sb0, ob0 = bound({})
    for L, (lo_t, hi_t) in sb0.items():
        base_sb[L] = (lo_t.detach().cpu().numpy().astype(np.float64),
                      hi_t.detach().cpu().numpy().astype(np.float64))
    for nm in exp_names:
        if nm in ob0:
            exp_numel[nm] = int(ob0[nm][0].numel())
    if lb0 > 0:
        return True, 1, 'root'

    # Branching scores: instability mass x |root spec-ew| at the neuron
    # (BaBSR-flavoured; measured ~2x better child bounds than pure
    # instability on vit). Root ew reused for the whole tree.
    ew_w = {}
    try:
        ob_r = {}
        sb_r, _zf_r = _forward_zonotope_graph(
            xl, xh, gg, device, dtype, settings=settings, op_bounds=ob_r)
        _, _, _ew_at = _spec_backward_graph(
            sb_r, xl, xh, gg, {0: (w_t, float(b_q))}, {0}, len(sb_r),
            device, dtype, op_bounds=ob_r, return_ew=True)
        for L, _e in _ew_at.get(0, {}).items():
            ew_w[L] = torch.as_tensor(
                np.abs(np.asarray(_e, np.float64)), device=device,
                dtype=dtype)
    except NotImplementedError:
        pass

    def pick_split(sb_l, clamps, ob):
        best = None; best_score = -1.0
        for L, (lo_t, hi_t) in sb_l.items():
            lo = lo_t.detach(); hi = hi_t.detach()
            uns = (lo < 0) & (hi > 0)
            if not bool(uns.any()):
                continue
            score = torch.minimum(-lo, hi) * uns
            if L in ew_w and ew_w[L].numel() == score.numel():
                score = score * (ew_w[L] + 1e-12)
            # avoid re-picking clamped neurons: zero them
            for (L2, j2) in clamps:
                if L2 == L:
                    score[j2] = -1.0
            j = int(score.argmax())
            s = float(score[j])
            if s > best_score and (L, j) not in clamps:
                best_score = s; best = ('relu', L, int(j))
        if best is not None:
            return best
        # no unstable relu left: propose a bilinear-input (exp) split.
        # Score = the exp parallelogram's slack height (chord gap minus
        # tangency gap) — the slack a midpoint split actually removes.
        for nm in exp_names:
            if nm not in ob:
                continue
            lo, hi = ob[nm]
            w_in = hi - lo
            k = (torch.exp(hi) - torch.exp(lo)) / w_in.clamp(min=1e-12)
            xs = torch.log(k.clamp(min=1e-300))
            xs = torch.minimum(torch.maximum(xs, lo), hi)
            slack = (torch.exp(lo) - k * lo) - (torch.exp(xs) - k * xs)
            slack = torch.where(w_in > 1e-6, slack.clamp(min=0.0),
                                torch.full_like(slack, -1.0))
            j = int(slack.argmax())
            s = float(slack[j])
            if s > best_score:
                best_score = s
                best = ('exp', nm, j, float(lo[j]), float(hi[j]))
        return best

    heap = []
    cnt = 0
    sp0 = pick_split(sb0, {}, ob0)
    if sp0 is None:
        return False, 1, 'no_split'
    heapq.heappush(heap, (lb0, cnt, {}, sp0))
    nodes = 1
    while heap:
        if time_left() <= 1.0:
            return False, nodes, 'time'
        if nodes >= max_nodes:
            return False, nodes, 'cap'
        lb, _, clamps, sp = heapq.heappop(heap)
        for side in (0, 1):
            c2 = dict(clamps)
            if sp[0] == 'relu':
                c2[(sp[1], sp[2])] = side
            else:
                _, nm, j, l_, u_ = sp
                m_ = 0.5 * (l_ + u_)
                c2[(nm, j)] = (l_, m_) if side == 0 else (m_, u_)
            lb2, sb2, ob2 = bound(c2)
            nodes += 1
            if lb2 <= 0:
                sp2 = pick_split(sb2, c2, ob2)
                if sp2 is None:
                    # No splittable relu or exp coordinate left in this
                    # leaf; a short backward-alpha rescue under the
                    # leaf's clamps before giving up.
                    _tb_leaf = {L: _mk_tb(L, c2)
                                for L in {Lj[0] for Lj in c2
                                          if not isinstance(Lj[0], str)}}
                    _oc_leaf = _mk_oc(c2)
                    try:
                        from .attn_crown import attn_crown_alpha
                        ob_leaf = {}
                        sb_leaf, _zfL = _forward_zonotope_graph(
                            xl, xh, gg, device, dtype, settings=settings,
                            tight_bounds=_tb_leaf, op_bounds=ob_leaf,
                            op_clamps=_oc_leaf)
                        _best_leaf, _ = attn_crown_alpha(
                            gg, xl, xh, sb_leaf, ob_leaf, w_t,
                            float(b_q), n_iters=30, lr=0.2,
                            time_left=time_left, op_clamps=_oc_leaf,
                            relu_clamps=_mk_rc(c2))
                        if _best_leaf > 0:
                            continue
                    except NotImplementedError:
                        pass
                    if _zono_alpha_close(
                            gg, xl, xh, w_q, b_q, device, dtype,
                            settings, time_left, n_iters=40, lr=0.15,
                            tight_bounds=_tb_leaf, op_clamps=_oc_leaf):
                        continue
                    return False, nodes, 'exhausted'
                cnt += 1
                heapq.heappush(heap, (lb2, cnt, c2, sp2))
    return True, nodes, 'closed'


def _zono_input_split_close(gg, xl, xh, w_q, b_q, device, dtype,
                             settings, time_left, max_nodes=4096):
    """Worst-first input-split BnB on ONE open query using the plain
    graph zonotope forward. Each node re-propagates the sub-box and
    checks lb(w·y+b) > 0; the split dim is the input dim with the
    largest |(w@G)| contribution (input dims own the leading generator
    columns). Sound: every leaf bound is a valid zonotope lower bound
    over its sub-box; True is returned only when every leaf closed.

    Built for the vit attention nets (the only phase-1 path there is the
    plain forward — no CROWN backward), but net-agnostic."""
    import heapq
    w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
        np.asarray(w_q, np.float64), device=device, dtype=dtype)

    def bound(lo_x, hi_x):
        _, zf = _forward_zonotope_graph(lo_x, hi_x, gg, device, dtype,
                                        settings=settings)
        c = zf.center; G = zf.generators
        wv = w_t.to(c.dtype)
        wg = wv @ G
        lb = float(wv @ c + float(b_q) - wg.abs().sum())
        nzd = torch.nonzero(hi_x - lo_x).flatten()
        K_in = min(int(nzd.numel()), G.shape[1])
        if K_in == 0:
            return lb, -1
        col = int(wg[:K_in].abs().argmax())
        return lb, int(nzd[col])

    lb0, dim0 = bound(xl, xh)
    if lb0 > 0:
        return True, 1
    if dim0 < 0:
        return False, 1
    heap = []
    cnt = 0
    heapq.heappush(heap, (lb0, cnt, xl, xh, dim0))
    nodes = 1
    while heap:
        if time_left() <= 1.0 or nodes >= max_nodes:
            return False, nodes
        lb, _, lo_x, hi_x, d = heapq.heappop(heap)
        mid = float((lo_x[d] + hi_x[d]) / 2)
        for half in (0, 1):
            l2 = lo_x.clone(); h2 = hi_x.clone()
            if half == 0:
                h2[d] = mid
            else:
                l2[d] = mid
            lb2, d2 = bound(l2, h2)
            nodes += 1
            if lb2 <= 0:
                if d2 < 0:
                    return False, nodes   # box degenerate yet open
                cnt += 1
                heapq.heappush(heap, (lb2, cnt, l2, h2, d2))
    return True, nodes


def _bound_stack_unsat(graph, xl_g, xh_g, spec, device, deadline=None,
                       alpha_iters=80, tol=1e-6, print_progress=False):
    """Memory-bounded UNSAT prover for big conv-ReLU nets whose dense forward
    zonotope OOMs (soundnessbench's 98304-wide ReLUs need ~43 GB; this fits
    ~150 MB). Stack: forward-LiRPA intermediate bounds -> backward-CROWN spec
    bound + alpha-CROWN, in float64.

    A disjunct's unsafe set (AND of threshold constraints) is empty iff SOME
    constraint is provably always-violated, i.e. max_j margin_j > 0, where
    margin_j = backward-CROWN lower bound of (w_j . Y + bias_j):
      '>=' Y_i >= v  -> w=-e_i, bias=+v  (margin = v - ub(Y_i))
      '<=' Y_i <= v  -> w=+e_i, bias=-v  (margin = lb(Y_i) - v)
      pairwise Yc>=Yp -> w=e_p-e_c, bias=0
    Whole spec is UNSAT iff EVERY disjunct's max-margin > tol.

    SOUND: the margins are valid CROWN lower bounds for any alpha in [0,1] (the
    max over alpha iterates stays a lower bound), so margin_j > 0 truly implies
    constraint j is never satisfiable -> the conjunction is empty. On a SAT
    instance every constraint can hold, so every margin_j <= 0 and this never
    returns `unsat`. float64 + a positive `tol` guard the near-zero cases.

    Only handles the single-input-box case (no per-disjunct input subboxes);
    returns None (defer to the normal pipeline) if any disjunct carries one.
    Returns ('verified', tb) when proven unsat, else None.
    """
    from .forward_lirpa import batched_forward_lirpa_layer_bounds_only
    from .verify_zono_bnb import _spec_backward_graph_batched
    if any(getattr(c, 'input_lo', None) is not None for c in spec.disjuncts):
        return None
    dt = torch.float64
    gg = graph.gpu_graph(device, dt)
    xl = xl_g.to(dt).reshape(1, -1)
    xh = xh_g.to(dt).reshape(1, -1)
    sb, last = batched_forward_lirpa_layer_bounds_only(gg, xl, xh, device, dt)
    n_out = last.lo_box.flatten().numel()
    int_layers = sorted(k for k in sb if isinstance(k, int))
    worst = float('inf')
    margins = {}
    for di, conj in enumerate(spec.disjuncts):
        ew = {}
        for j, c in enumerate(conj.constraints):
            w = torch.zeros(n_out, dtype=dt, device=device)
            if hasattr(c, 'op') and c.op is not None:
                if c.op == '>=':
                    w[c.index] = -1.0
                    b = float(c.value)
                else:
                    w[c.index] = 1.0
                    b = -float(c.value)
            else:
                w[c.pred] = 1.0
                w[c.comp] = -1.0
                b = 0.0
            ew[j] = (w, b)
        Q = len(ew)
        alpha = {}
        for L in int_layers:
            lo_L, hi_L = sb[L]
            _, up_s, _, _, _, uns = _make_slopes(lo_L, hi_L)
            alpha[L] = (((up_s > 0.5).to(dt) * uns.to(dt))
                        .unsqueeze(1).expand(-1, Q, -1)
                        .contiguous().requires_grad_(True))
        opt = torch.optim.Adam([alpha[L] for L in alpha], lr=0.1)
        best = None
        for _it in range(alpha_iters):
            if deadline is not None and time.perf_counter() > deadline:
                break
            opt.zero_grad()
            sl = _spec_backward_graph_batched(
                sb, xl, xh, gg, ew, device, dt, alpha_at_layer=alpha)
            best = (sl.detach().clone() if best is None
                    else torch.maximum(best, sl.detach()))
            (-sl.sum()).backward()
            opt.step()
            with torch.no_grad():
                for L in alpha:
                    alpha[L].clamp_(0.0, 1.0)
        if best is None:
            return None
        cm = max(best[0, k].item() for k in range(Q))
        margins[di] = cm
        worst = min(worst, cm)
        if cm <= tol:
            return None  # this disjunct not proven -> spec not proven unsat
    if print_progress:
        print(f'  [bound-stack] proved unsat '
              f'(worst disjunct margin {worst:+.5f})', flush=True)
    # 'verified' is verify_graph's convention for a proven-unsat result
    # (main maps verified -> the `unsat` results-file line + exit 0).
    return 'verified', {'phase': 'bound_stack', 'worst_margin': worst,
                        'margins': margins}


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
        # `_prevalidated`: the caller (`_sat_or_fallthrough`) already ran
        # `_validate_sat_witness` and the witness PASSED — skip the
        # redundant ORT forward below. Popped so it never leaks into
        # `details`.
        _prevalidated = bool(extra.pop('_prevalidated', False))
        # Defense-in-depth: every SAT verdict must pass ONNXRuntime
        # validation against the spec (within COUNTEREXAMPLE_ATOL).
        # Catches spurious witnesses from PGD/MILP bugs AND graph-
        # builder bugs that would otherwise leak through silently.
        # Spurious witnesses get downgraded to 'unknown' with
        # `details['spurious_witness']` set and logged.
        if (result_str == 'sat' and extra.get('witness') is not None
                and not _prevalidated
                and not bool(getattr(
                    settings, 'skip_sat_validation', False))):
            _atol = float(settings.sat_validate_atol
                           if 'sat_validate_atol' in settings else 1e-4)
            # Output tolerance FIXED at 0.0 (2026 rule) — not configurable.
            _onnx_path = getattr(graph, 'onnx_path', None)
            _ok, _info = _validate_sat_witness(
                _onnx_path, spec, extra['witness'], atol=_atol,
                out_atol=0.0, emit_slack=pgd_box_expand_amount(settings))
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
        # Defense-in-depth: every VERIFIED verdict gets a sample-based
        # independent check. ORT forwards N input samples; if any
        # falsifies the spec, that's a real counterexample → the case
        # is actually SAT, and our verifier was unsound. Return 'sat'
        # with the witness (not 'unknown' — a witness is strictly more
        # informative than unknown, and matches what PGD would return).
        if (result_str == 'verified'
                and not bool(getattr(
                    settings, 'skip_verified_validation', False))):
            _n = int(getattr(settings, 'verified_validation_samples', 0)
                      or 0)
            _atol = float(settings.sat_validate_atol
                           if 'sat_validate_atol' in settings else 1e-4)
            _onnx_path = getattr(graph, 'onnx_path', None)
            if _n > 0:
                _ok_v, _info_v = _validate_verified_with_samples(
                    _onnx_path, spec, n_samples=_n, atol=_atol)
                if not _ok_v:
                    if print_progress:
                        print(f'  [validate] SPURIOUS VERIFIED from '
                              f'{phase}: counterexample found → '
                              f'flipping to SAT', flush=True)
                    extra = dict(extra)
                    extra['spurious_verified'] = _info_v
                    extra['original_phase'] = phase
                    extra['witness'] = _info_v.get('witness')
                    result_str = 'sat'
                    phase = f'spurious_verified_{phase}'
        details['result'] = result_str
        details['phase'] = phase
        details['time'] = time.perf_counter() - t_start
        details['n_splits'] = _compute_n_splits(gg, bounds_by_relu)
        # Surface the per-query spec lower bounds for diagnostics —
        # callers often want to see how close we got to verification
        # even on 'unknown' verdicts (closer-to-0 = better strategy).
        try:
            details['spec_lbs'] = dict(spec_lbs)
        except (TypeError, NameError):
            # spec_lbs may not exist on some early-exit paths; diagnostic only.
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
        except (AttributeError, ValueError, RuntimeError):
            # Diagnostic surfacing only — tensor → numpy can fail on odd
            # device/dtype combos; not worth crashing the run for diag output.
            pass
        if verbose:
            details['avg_layer_width'] = _compute_avg_layer_width(
                gg, bounds_by_relu)
        for k, v in extra.items():
            details[k] = v
        # VNNCOMP: an undecided result that ran out of the time budget is a
        # 'timeout', distinct from a give-up 'unknown'. The timeout call sites
        # pass phase='timeout'; surface it so main.py emits the right verdict.
        if result_str not in ('sat', 'verified') and phase == 'timeout':
            details['timed_out'] = True
        details['phase'] = phase
        details['neuron_stats'] = stats.neuron_stats
        return result_str, details

    def _sat_or_fallthrough(phase, witness):
        """Validate a PGD/MILP SAT witness; return the finalized 'sat'
        result if it GENUINELY violates the spec, else return None so the
        caller FALLS THROUGH to the next (often stronger) attack/bound
        stage instead of aborting to 'unknown'.

        A weak early PGD stage that produces a near-boundary spurious
        witness must not short-circuit the full-strength downstream
        attacks. Observed on tinyimagenet SAT cases: the light Phase-0 PGD
        returned points with worst margin ≈ +1e-4 (just inside the safe
        side), which `_finalize` correctly rejected as spurious — but the
        unconditional `return` then skipped the cascade's full-restart PGD
        that finds the real counterexample. Soundness is unchanged: every
        emitted 'sat' is still ORT-validated here (or in `_finalize`).
        """
        if (witness is None
                or bool(getattr(settings, 'skip_sat_validation', False))
                or not bool(getattr(
                    settings, 'pgd_fallthrough_on_spurious', True))):
            # Nothing to validate, validation disabled, or fallthrough
            # disabled — preserve the original return-on-sat behaviour.
            return _finalize('sat', phase, witness=witness)
        _atol = float(settings.sat_validate_atol
                       if 'sat_validate_atol' in settings else 1e-4)
        # Output tolerance FIXED at 0.0 (2026 rule) — not configurable.
        _onnx_path = getattr(graph, 'onnx_path', None)
        _ok, _info = _validate_sat_witness(_onnx_path, spec, witness,
                                           atol=_atol, out_atol=0.0,
                                           emit_slack=pgd_box_expand_amount(settings))
        if _ok:
            if _sat_disposition(graph, spec, settings, witness,
                                _info) == 'real':
                return _finalize('sat', phase, witness=witness,
                                 _prevalidated=True)
            # within-tolerance near-miss: the counterexample was written to
            # the results file early by _sat_disposition; keep searching for a
            # clear counterexample or an unsat proof rather than committing.
            if print_progress:
                print(f'  [validate] within-tol near-miss from {phase} — CE '
                      f'saved, continuing for a clear CE / unsat proof',
                      flush=True)
            return None
        if print_progress:
            print(f'  [validate] SPURIOUS SAT from {phase} — continuing '
                  f'(budget left, not aborting): {_info.get("reason")}',
                  flush=True)
        details.setdefault('spurious_witnesses', []).append(
            {'phase': phase, 'info': _info})
        return None

    gg = graph.gpu_graph(device, dtype)
    nh = gg['n_relu']
    relu_names = gg['relu_names']

    # Output size. The last-fc/conv heuristic is correct only when the net ENDS
    # in a linear layer. ml4acopf ends in non-linear physics (Concat of
    # Mul/Sin/Cos branches), so the last Gemm is an intermediate (= bus count,
    # e.g. 14/118) — far smaller than the true output (186/1190). When the final
    # op isn't fc/conv, propagate to get the true count — but over a DEGENERATE
    # (point) input box, not the real box. The count (center.numel()) is the same
    # either way, but a point makes every ReLU stable, so NO error generators are
    # created: the real box OOMs the dense generator matrix on wide residual nets
    # (soundnessbench model_residual: 128->12288 Gemm + 15 convs -> ~16 GB) BEFORE
    # PGD runs, whereas the point probe stays at 0 generators. (The net's inferred
    # output shape can't be used here: it under-counts ml4acopf's Concat. The
    # point probe is exact for every graph and far cheaper than the box probe.)
    if gg['ops'][-1]['type'] in ('fc', 'conv'):
        last = gg['ops'][-1]
        n_output = (last['W'].shape[0] if last['type'] == 'fc'
                    else last['n_out'])
    else:
        _xm = torch.tensor(((np.asarray(spec.x_lo) + np.asarray(spec.x_hi))
                            / 2.0).flatten(), dtype=dtype, device=device)
        _, _zf_p = _forward_zonotope_graph(_xm, _xm, gg, device, dtype,
                                           settings=settings)
        n_output = int(_zf_p.center.numel())
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

    # Softmax max-shift retarget (exact reparametrization; only the
    # gather index changes, picked at the box center)
    from .verify_zono_bnb import retarget_softmax_shifts
    retarget_softmax_shifts(gg, 0.5 * (xl_g + xh_g))

    # PGD only needs approximate gradients — run it in float32 on a
    # dedicated float32 gpu_graph so the attack stays cheap even when the
    # main verification is float64. When `bits=32`, `gg` is already fp32,
    # so reuse it (saves ~10ms + avoids duplicating weight tensors on GPU).
    if dtype == torch.float32:
        gg_pgd = gg
    else:
        gg_pgd = graph.gpu_graph(device, torch.float32)
        # the fp32 work graph must be the SAME function as gg: copy the
        # retargeted softmax-shift indices (fresh op dicts re-emit the
        # k=0 placeholder otherwise, changing node semantics under it)
        _shift_idx = {op['name']: op['flat_idx'] for op in gg['ops']
                      if op['type'] == 'slice' and 'softmax_axis' in op}
        for _op2 in gg_pgd['ops']:
            if _op2['name'] in _shift_idx:
                _op2['flat_idx'] = _shift_idx[_op2['name']]
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

    # --- Bound-stack UNSAT route (memory-bounded; for big conv-ReLU nets whose
    # dense forward zonotope OOMs, e.g. soundnessbench's 98304-wide ReLUs).
    # forward-LiRPA + backward-CROWN + α in float64, no dense zono. Runs BEFORE
    # the PGD attack: proves the genuinely-unsat instances in ~seconds; SAT/hard
    # instances fall through to PGD (the bound-stack margin stays <=0 there, so
    # it never false-verifies a SAT case — sound on the soundness benchmark). ---
    if bool(getattr(settings, 'bound_stack_phase0', False)):
        _bs_dl = time.perf_counter() + float(
            getattr(settings, 'bound_stack_time', 60.0))
        try:
            _bs = _bound_stack_unsat(
                graph, xl_g, xh_g, spec, device, deadline=_bs_dl,
                alpha_iters=int(
                    getattr(settings, 'bound_stack_alpha_iters', 80)),
                print_progress=print_progress)
        except torch.cuda.OutOfMemoryError:
            raise  # never swallow OOM (CLAUDE.md hard rule)
        except RuntimeError as _bse:
            # Best-effort UNSAT prover: on a non-OOM failure, log loudly and
            # fall through to PGD / the rest of the pipeline rather than abort.
            print(f'  [bound-stack] failed, falling through: '
                  f'{type(_bse).__name__}: {str(_bse)[:100]}', flush=True)
            _bs = None
        if _bs is not None:
            return _bs

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
        # Optional deterministic seeding (e.g. soundnessbench) — reseed the
        # torch RNG right before the attack so the random restarts are
        # reproducible across machines and don't depend on ambient RNG state.
        _pgd_seed = getattr(settings, 'pgd_seed', None)
        if _pgd_seed is not None:
            torch.manual_seed(int(_pgd_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(_pgd_seed))
        # Persist-until-budget: for SAT-heavy / OOM-prone benchmarks where the
        # bound-prop cascade cannot help (soundnessbench is all-SAT; its dense
        # forward zonotope needs ~43 GB and OOMs), keep relaunching fresh-init
        # PGD batches until the whole Phase-0 budget is spent, instead of one
        # batch + a doomed cascade. A single 500-restart batch only has a ~90%
        # hit rate on the hardest planted basins (model_6: 9/10 seeds) — ~4
        # batches in the budget drive the miss probability to ~1e-4. Each round
        # advances the (optionally seeded) RNG so inits differ, and rotates the
        # targeted disjunct when the spec has several ("target different
        # classes"). When the budget is exhausted with no witness the cascade
        # is skipped and we report `unknown` honestly.
        _persist = bool(getattr(
            settings, 'pgd_phase0_persist_until_budget', False))
        pgd_sat, pgd_witness = False, None
        _round = 0
        try:
            while True:
                if _pgd_budget_phase0 - (time.perf_counter() - t0) <= 0.5:
                    break
                _remain = _pgd_budget_phase0 - (time.perf_counter() - t0)
                # Each batch attacks ALL disjuncts (restrict_disj=None), but
                # with per-restart disjunct assignment: restart r descends only
                # disjunct (r % n_disj)'s loss, so every disjunct gets dedicated
                # restarts instead of one joint loss all restarts share (a CEX
                # in any disjunct is still accepted by the witness screen). No-op
                # for single-disjunct specs (soundnessbench). Diversity across
                # rounds comes from the advancing RNG of the fresh inits.
                try:
                    pgd_sat, pgd_witness = _pgd_attack_general(
                        xl_pgd, xh_pgd, spec, gg_pgd, settings,
                        time_budget=_remain, per_restart_disj=True)
                except RuntimeError as _pe:
                    # A round failed (incl. GPU OOM — torch's OutOfMemoryError
                    # is a RuntimeError subclass). Don't hide it: log to stdout
                    # so the failure is visible even without --verbose, free the
                    # cached memory, and keep attacking with the remaining
                    # budget (Phase-0 PGD is a best-effort SAT-finder).
                    pgd_sat, pgd_witness = False, None
                    print(f'  [phase0-pgd] round {_round} failed: '
                          f'{type(_pe).__name__}: {str(_pe)[:90]}', flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _round += 1
                if pgd_sat or not _persist:
                    break
        finally:
            settings.pgd_iter = _orig_iter
            settings.pgd_restarts = _orig_restarts
        timing['phase0_pgd'] = time.perf_counter() - t0
        if print_progress:
            print(f'Phase 0 (PGD before cascade): '
                  f'{timing["phase0_pgd"]:.2f}s  sat={pgd_sat}'
                  + (f' ({_round} round(s))' if _persist else ''), flush=True)
        if pgd_sat:
            _r = _sat_or_fallthrough('pgd', pgd_witness)
            if _r is not None:
                return _r
        if _persist:
            # Spent the full attack budget without a valid witness. The cascade
            # is useless here (all-SAT) or would OOM, so skip it and report
            # `timeout` (timed_out=True → main maps it to the VNNCOMP `timeout`
            # line, more informative than a bare `unknown`: we ran the whole
            # wall budget attacking and couldn't decide). Always log it.
            print(f'  [phase0-pgd] persist budget exhausted after {_round} '
                  f'round(s), no witness -> timeout', flush=True)
            return 'unknown', {'phase': 'pgd_persist_exhausted',
                                'rounds': _round, 'timed_out': True}

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
        # Total wall cap across ALL disjuncts this hook attacks (not per-spec).
        # Without it, n_open × per_min grows the budget unboundedly with disjunct
        # count (44 open × 0.5s = 22s on tinyimagenet 6546 — pure overhead on a
        # robust spec that then starved Phase 8). The cap stops after the N most-
        # promising (lowest spec_lb) disjuncts; the rest are left to Phase 8 BnB +
        # the restricted Phase-9 survivor attack. Default None = no cap (unchanged
        # for other benchmarks).
        _total_cap = (float(settings.phase26_pre_cascade_total_cap)
                      if 'phase26_pre_cascade_total_cap' in settings
                      and settings.phase26_pre_cascade_total_cap is not None
                      else None)
        n_attacked = 0
        for di in ordered:
            if time_left() <= 0.5: break
            if _total_cap is not None and (
                    time.perf_counter() - t_hook) >= _total_cap:
                break
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
        if _phase1_method == 'plain':
            # Plain graph-aware zono forward (verify_zono_bnb) with NO
            # tightening cascade. The only phase-1 path that supports the
            # vit attention ops (matmul_bilinear / softmax interval
            # handlers live there); also the cheapest when
            # max_tighten_layer=0 anyway.
            _t_p1 = time.perf_counter()
            _op_bounds_p1 = {}
            sb, z_final_phase1 = _forward_zonotope_graph(
                xl_g, xh_g, gg, device, dtype, settings=settings,
                rec_zono=rec_zono, op_bounds=_op_bounds_p1)
            bounds_by_relu = {
                L: (lo.detach().cpu().numpy().astype(np.float64),
                    hi.detach().cpu().numpy().astype(np.float64))
                for L, (lo, hi) in sb.items()}
            phase1_tb = {'plain_forward': time.perf_counter() - _t_p1,
                         'op_bounds': _op_bounds_p1}
        elif _phase1_method == 'bab_refine':
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

    # Env-gated bounds dump / intersect — the "combine two tightenings"
    # experiment. VC_DUMP_BOUNDS pickles the post-Phase-1 bounds_by_relu;
    # VC_LOAD_BOUNDS loads another run's bounds and INTERSECTS them with the
    # current ones (max lo, min hi per neuron). Sound: both are valid
    # over-approximations, so the tighter of each is still valid.
    import os as _osb
    _db = _osb.environ.get('VC_DUMP_BOUNDS', '')
    if _db:
        import pickle as _pklb
        with open(_db, 'wb') as _f:
            _pklb.dump({L: (np.asarray(lo).copy(), np.asarray(hi).copy())
                        for L, (lo, hi) in bounds_by_relu.items()}, _f)
        if print_progress:
            print(f'  [dump-bounds] {len(bounds_by_relu)} layers -> {_db}',
                  flush=True)
    _lbf = _osb.environ.get('VC_LOAD_BOUNDS', '')
    if _lbf:
        import pickle as _pklb
        with open(_lbf, 'rb') as _f:
            _bload = _pklb.load(_f)
        _ntab = 0
        for L, (lo, hi) in list(bounds_by_relu.items()):
            if L in _bload:
                _lo2, _hi2 = _bload[L]
                _nlo = np.maximum(np.asarray(lo), np.asarray(_lo2))
                _nhi = np.minimum(np.asarray(hi), np.asarray(_hi2))
                _ntab += int(((_nlo > np.asarray(lo))
                              | (_nhi < np.asarray(hi))).sum())
                bounds_by_relu[L] = (_nlo, _nhi)
        if print_progress:
            print(f'  [intersect-bounds] tightened {_ntab} neuron-bounds '
                  f'from {_lbf}', flush=True)

    # --- Check parallel PGD result (thread joined inside Phase 1) ---
    if isinstance(phase1_tb, dict):
        _pp_attacks = phase1_tb.get('parallel_pgd_attacks')
        if _pp_attacks is not None:
            timing['parallel_pgd_attacks'] = _pp_attacks
        _pp_witness = phase1_tb.get('parallel_pgd_sat')
        if _pp_witness is not None:
            _r = _sat_or_fallthrough('parallel_pgd', _pp_witness)
            if _r is not None:
                return _r

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
            print(f'Phase 1 pre-cascade PGD found SAT — validating',
                  flush=True)
        _r = _sat_or_fallthrough('pgd_pre_cascade', _pre_cascade_witness)
        if _r is not None:
            return _r

    # --- Phase 2: CROWN backward ---
    # Skippable when Phase 1 already produced tighter α-CROWN spec lbs
    # (Phase 0.5 step 2). Setting `phase2_crown_enabled=False` reuses those
    # lbs instead of recomputing a (looser) plain-CROWN backward.
    t0 = time.perf_counter()
    all_qids = set(spec_ew.keys())
    _phase2_crown = bool(getattr(settings, 'phase2_crown_enabled', True))
    _ph1_spec_lbs = phase1_tb.get('spec_lbs_phase05') if isinstance(phase1_tb, dict) else None
    # Phase 0.5's optimised spec α — used to warm-start Phase 8's per-query
    # α-CROWN (saves ~10 iters worth of Adam to re-find the same local
    # optimum on a frozen-intermediate-bounds problem).
    _ph1_spec_alpha = phase1_tb.get('spec_alpha_phase05') if isinstance(phase1_tb, dict) else None
    if _phase2_crown:
        _ob_p1 = (phase1_tb.get('op_bounds')
                  if isinstance(phase1_tb, dict) else None)
        with torch.no_grad():
            spec_lbs, _ = _spec_backward_graph(
                sb, xl_g, xh_g, gg, spec_ew, all_qids, nh, device, dtype,
                op_bounds=_ob_p1)
            if (_phase1_method == 'plain'
                    and z_final_phase1 is not None
                    and hasattr(z_final_phase1, 'center')):
                # Best-of: the forward-zono margin and the backward CROWN
                # bound are both sound — take the max per query (on vit
                # the forward is tighter for some queries, the McCormick
                # backward for others).
                _zc = z_final_phase1.center
                _zG = z_final_phase1.generators
                for qi in all_qids:
                    _w, _b = spec_ew[qi]
                    _w = _w.to(_zc.dtype)
                    _fwd = float(_w @ _zc + _b - (_w @ _zG).abs().sum())
                    if _fwd > spec_lbs.get(qi, -float('inf')):
                        spec_lbs[qi] = _fwd
    elif _ph1_spec_lbs:
        # Reuse Phase 0.5 α-CROWN spec lbs (already tighter than plain CROWN).
        spec_lbs = {qi: _ph1_spec_lbs.get(qi, -1.0) for qi in all_qids}
    else:
        # No backward pass available/desired (phase1_method='plain' with
        # phase2 off — e.g. vit attention nets where the CROWN backward
        # has no handlers): sound spec margins straight from the forward
        # zonotope, lb(w·y+b) = w·c + b − Σ|w·G|.
        with torch.no_grad():
            _zc = z_final_phase1.center
            _zG = z_final_phase1.generators
            spec_lbs = {}
            for qi in all_qids:
                _w, _b = spec_ew[qi]
                _w = _w.to(_zc.dtype)
                spec_lbs[qi] = float(
                    _w @ _zc + _b - (_w @ _zG).abs().sum())
    timing['phase2_crown'] = time.perf_counter() - t0
    stats.record_bounds(sb)

    # ANY-closure: a disjunct is a CONJUNCTION of constraints (the SAT set
    # is their AND); refuting ANY single conjunct (its query lb > 0) proves
    # the whole disjunct infeasible. all() here was sound but needlessly
    # strict — multi-conjunct specs (yolo_2023: 5 objectness conjuncts per
    # disjunct) could only verify by refuting EVERY conjunct, which is
    # usually impossible. For single-conjunct disjuncts (all regular-track
    # benchmarks) any() == all(), so behavior there is unchanged.
    verified_disj = {di for di, qlist in disj_queries.items()
                      if any(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist)}
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
            _r = _sat_or_fallthrough('pgd', pgd_witness)
            if _r is not None:
                return _r

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
            _r = _sat_or_fallthrough('pgd_middle', pgd_witness)
            if _r is not None:
                return _r

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
        from . import heartbeat as _hb
        _hb.set_phase('phase7/8 gen-LP state + dual-ascent')
        if _p26_sat:
            _r = _sat_or_fallthrough('pgd_per_spec', pgd_witness)
            if _r is not None:
                return _r

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
    # Skip entirely when caller already disabled Phase 8 MILP (e.g. graphs
    # with Pow / Div / ReduceSum / MulBilinear that the gen-LP state
    # builder can't represent) — there's no consumer for the state.
    _skip_gen_lp_state = bool(getattr(settings, 'skip_phase8_milp', False))
    if remaining_qids and time_left() > 2 and not _skip_gen_lp_state:
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
    if need_ew and not bool(getattr(settings, 'skip_phase8_milp', False)):
        with torch.no_grad():
            _, _, ew_at_relu = _spec_backward_graph(
                sb, xl_g, xh_g, gg, spec_ew, set(need_ew), nh,
                device, dtype, return_ew=True)
    else:
        # No Phase-8 MILP consumer for the scores (skip_phase8_milp) —
        # don't run a backward pass that some op sets (vit attention)
        # cannot support anyway.
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
    # Exact-MILP routing for hopeless-relaxation FC nets (cora_2024): when the
    # worst still-open spec LB sits far below 0 at Phase-8 entry, the α-zono
    # BnB frontier doubles every level and cannot close within budget
    # (measured on cora: solved cases enter Phase 8 at worst ≥ −0.65 and the
    # BnB closes them in ≤1s; misses enter at ≤ −3.3 and the BnB OOMs at 67M
    # nodes, while the exact per-neuron MILP closes all 7 cifar10 misses in
    # 0.9–3.6s). Complementary hardness is real (img100: BnB-easy/MILP-hard),
    # so this routes ONLY the far-below-zero cases; near-boundary ones keep
    # the BnB. Sound: milp_verify is the exact big-M encoding (safenlp's
    # engine). Off by default; configs/cora_2024.yaml sets −2.0.
    _milp_below = getattr(settings, 'phase8_exact_milp_below', None)
    if (_milp_below is not None and still_needs_milp
            and spec_impl == 'gen_lp'):
        _unsup = {'Sigmoid', 'Tanh', 'Softmax', 'Exp', 'Erf', 'Gelu', 'Elu',
                  'LeakyRelu', 'Pow', 'Div', 'MatMulBilinear', 'ReduceSum',
                  'Conv'}
        _pure_fc = not any(getattr(n, 'op_type', '') in _unsup
                           for n in graph.nodes.values())
        _open_lbs = [spec_lbs[qi] for qi in still_needs_milp
                     if qi in spec_lbs]
        if _pure_fc and _open_lbs and min(_open_lbs) < float(_milp_below):
            import copy as _copy
            from .verify_milp import milp_verify
            from .gurobi_util import GurobiNumericTrouble
            _s2 = _copy.copy(settings)
            _s2.total_timeout = max(2.0, time_left())
            if print_progress:
                print(f'  [phase8→exact-MILP] worst open LB '
                      f'{min(_open_lbs):.4f} < {float(_milp_below)} on pure '
                      f'FC/ReLU net → routing to milp_verify '
                      f'(budget {_s2.total_timeout:.1f}s)', flush=True)
            try:
                return milp_verify(graph, spec, _s2)
            except GurobiNumericTrouble as _gnt:
                # Numerically fragile exact-MILP (e.g. cora mnist-trades
                # img6): the routed engine claims nothing, so 'unknown' is
                # the sound result — don't crash the run into
                # error_no_result.
                if print_progress:
                    print(f'  [phase8→exact-MILP] Gurobi numeric trouble '
                          f'({_gnt}); returning unknown', flush=True)
                return _finalize('unknown', 'spec_lp',
                                 remaining=len(still_needs_milp))
    if (bool(getattr(settings, 'zono_input_split_enabled', False))
            and still_needs_milp and time_left() > 2.0):
        # Last-chance input-split BnB on the open queries via the plain
        # zono forward (vit path: no backward support needed). All-or-
        # nothing: only a full close changes the verdict.
        _t_zis = time.perf_counter()
        _zis_nodes = 0
        _zis_closed = []
        _zis_cap = int(getattr(settings, 'zono_input_split_max_nodes', 4096))
        # Root forward + CROWN intermediate-bound refinement are query-
        # independent: compute ONCE and share across all open queries.
        _sb_r0 = _ob_r0 = None
        try:
            _ob_r0 = {}
            _sb_r0, _ = _forward_zonotope_graph(
                xl_g, xh_g, gg, device, dtype, settings=settings,
                op_bounds=_ob_r0)
            _ref_passes = int(getattr(
                settings, 'attn_refine_passes', 0) or 0)
            if _ref_passes > 0:
                # CROWN intermediate-bound refinement (ABC's backward
                # intermediates): tightens the exp / McCormick plane
                # ranges for the root alpha and every BnB domain.
                from .attn_crown import attn_refine_op_bounds
                _n_ref = attn_refine_op_bounds(
                    gg, xl_g, xh_g, _sb_r0, _ob_r0,
                    time_left=time_left, passes=_ref_passes)
                if print_progress:
                    print(f'  [interm-refine] tightened {_n_ref} bounds '
                          f'({time.perf_counter() - _t_zis:.1f}s)',
                          flush=True)
        except NotImplementedError:
            _sb_r0 = _ob_r0 = None
        # JOINT alpha over all open queries with differentiable
        # intermediate bounds (the ABC mechanism: spec loss backprops
        # into the planes of every intermediate-bound backward; their
        # vit pgd instances close in ~2 such steps). Certified fp64
        # bounds replace the shared root state for everything after.
        _joint_closed = set()
        _joint_params = None
        if (bool(getattr(settings, 'attn_alpha_joint', False))
                and _sb_r0 is not None and still_needs_milp
                and any(op['type'] == 'exp' for op in gg['ops'])
                and time_left() > 5.0):
            try:
                from .attn_crown import attn_alpha_joint
                _jq = sorted(still_needs_milp)
                _Wm = np.stack([
                    spec_ew[qi][0].detach().cpu().numpy().astype(
                        np.float64) for qi in _jq])
                _bv = np.array([float(spec_ew[qi][1]) for qi in _jq])
                _bab_fp32 = bool(getattr(settings, 'attn_bab_fp32',
                                         True))
                # phase budget: each joint iteration re-derives every
                # capped intermediate (expensive on 3-block nets) —
                # leave the BaB its share of the wall
                _t_j0 = time.perf_counter()
                _j_budget = min(
                    time_left() * float(getattr(
                        settings, 'attn_joint_frac', 0.35)),
                    float(getattr(settings, 'attn_joint_max_s', 30.0)))
                # Budget gate by GAP DEPTH, not query count: BaB closes
                # shallow gaps (~0.05, thousands of domains) but never
                # deep ones (measured: -0.3 gaps crawl at 24k domains;
                # ABC's alpha_iter2 ablation flips deep-gap cases to
                # timeout). Shallow worst-gap -> keep alpha short and
                # let BaB spend the wall; deep -> alpha needs its ~50
                # iterations.
                _worst_open = min(
                    (spec_lbs.get(qi, -1.0) for qi in _jq),
                    default=-1.0)
                if _worst_open > -float(getattr(
                        settings, 'attn_joint_deep_gap', 0.15)):
                    # shallow gaps: BaB closes these; alpha overhead
                    # costs cases (6769 closed pre-joint at 85.6s,
                    # lost to phase overhead afterwards)
                    _j_budget = min(
                        _j_budget,
                        float(getattr(settings, 'attn_joint_s_per_q',
                                      6.0)) * len(_jq),
                        float(getattr(settings,
                                      'attn_joint_shallow_max_s', 8.0)))
                _tl_j = (lambda: min(
                    time_left(),
                    _j_budget - (time.perf_counter() - _t_j0)))
                _lb_j, _joint_params, _ob_r0, _sb_r0 = attn_alpha_joint(
                    gg, xl_g, xh_g, _sb_r0, _ob_r0, _Wm, _bv,
                    n_iters=int(getattr(
                        settings, 'attn_joint_iters', 60)),
                    lr=float(getattr(settings, 'attn_joint_lr', 0.4)),
                    lr_decay=float(getattr(
                        settings, 'attn_joint_lr_decay', 0.98)),
                    time_left=_tl_j,
                    gg_work=gg_pgd if _bab_fp32 else None,
                    work_dtype=torch.float32 if _bab_fp32 else None,
                    max_rows=int(getattr(
                        settings, 'attn_joint_max_rows', 4096)),
                    per_row_rows=int(getattr(
                        settings, 'attn_joint_per_row_rows', 1024)),
                    freeze_tol=float(getattr(
                        settings, 'attn_joint_freeze_tol', 1e-4)),
                    freeze_patience=int(getattr(
                        settings, 'attn_joint_freeze_patience', 2)),
                    refresh_every=int(getattr(
                        settings, 'attn_joint_refresh_every', 1)),
                    freeze_refresh=int(getattr(
                        settings, 'attn_joint_freeze_refresh', 8)))
                for _qi, _lb in zip(_jq, _lb_j):
                    if _lb > 0:
                        _joint_closed.add(_qi)
                if print_progress:
                    print(f'  [joint-alpha] {len(_joint_closed)}/'
                          f'{len(_jq)} closed, worst '
                          f'{float(min(_lb_j)):+.4f}', flush=True)
            except NotImplementedError:
                _joint_params = None
        for qi in sorted(still_needs_milp):
            if qi in _joint_closed:
                _zis_closed.append(qi)
                continue
            _w_zis, _b_zis = spec_ew[qi]
            ok = _zono_alpha_close(
                gg, xl_g, xh_g, _w_zis, _b_zis, device, dtype, settings,
                time_left,
                n_iters=int(getattr(settings, 'zono_alpha_iters', 60)),
                lr=float(getattr(settings, 'zono_alpha_lr', 0.1)))
            _root_params = None
            if not ok and _sb_r0 is not None:
                # root backward-alpha (ABC's alpha-CROWN over the
                # McCormick r / tangent / relu-lambda planes); its
                # optimized params are reused by every BnB node below.
                try:
                    from .attn_crown import attn_crown_alpha
                    _sb_r, _ob_r = _sb_r0, _ob_r0
                    _a_iters = int(getattr(
                        settings, 'zono_backward_alpha_iters', 60))
                    _warm = None
                    if _joint_params is not None:
                        _warm = {k: v.detach().clone().requires_grad_(
                            True) for k, v in _joint_params.items()}
                    _best_r, _root_params = attn_crown_alpha(
                        gg, xl_g, xh_g, _sb_r, _ob_r, _w_zis,
                        float(_b_zis), n_iters=_a_iters,
                        lr=0.25, time_left=time_left, params=_warm)
                    # iterate refine <-> alpha: param-aware intermediate
                    # refinement compounds (pgd_7086 q3 measured
                    # -1.16 -> +0.0009 -> +0.23 over two rounds). The
                    # shared _sb_r0/_ob_r0 tighten in place, so gains
                    # accumulate across queries (sound: monotone
                    # intersection of valid enclosures).
                    _rounds = int(getattr(
                        settings, 'attn_refine_rounds', 0) or 0)
                    for _rr in range(_rounds):
                        if _best_r > 0 or time_left() <= 5.0:
                            break
                        from .attn_crown import attn_refine_op_bounds
                        attn_refine_op_bounds(
                            gg, xl_g, xh_g, _sb_r, _ob_r,
                            params=_root_params, time_left=time_left,
                            passes=1)
                        _b2, _root_params = attn_crown_alpha(
                            gg, xl_g, xh_g, _sb_r, _ob_r, _w_zis,
                            float(_b_zis), n_iters=_a_iters, lr=0.25,
                            time_left=time_left, params=_root_params)
                        _best_r = max(_best_r, _b2)
                    if print_progress:
                        print(f'  [bw-alpha] q{qi} root lb '
                              f'{_best_r:+.4f}', flush=True)
                    ok = _best_r > 0
                except NotImplementedError:
                    _root_params = None
            _bab_batch = int(getattr(settings, 'attn_bab_batch', 0) or 0)
            if (not ok and _bab_batch > 0 and _root_params is not None
                    and any(op['type'] == 'exp' for op in gg['ops'])):
                # Batched no-reforward beta-CROWN BaB (the ABC vit
                # recipe): ~10-30x the unbatched node throughput.
                from .attn_crown import attn_beta_bab
                _ew_w = {}
                try:
                    _, _, _ew_at = _spec_backward_graph(
                        _sb_r, xl_g, xh_g, gg,
                        {0: (_w_zis, float(_b_zis))}, {0}, len(_sb_r),
                        device, dtype, op_bounds=_ob_r, return_ew=True)
                    for _L, _e in _ew_at.get(0, {}).items():
                        _ew_w[_L] = torch.as_tensor(
                            np.abs(np.asarray(_e, np.float64)),
                            device=device, dtype=dtype)
                except NotImplementedError:
                    pass
                _bab_fp32 = bool(getattr(settings, 'attn_bab_fp32',
                                         True))
                ok, n_used, _rs_reason = attn_beta_bab(
                    gg, xl_g, xh_g, _sb_r, _ob_r, _w_zis,
                    float(_b_zis), _root_params, time_left=time_left,
                    ew_w=_ew_w or None, batch=_bab_batch,
                    n_iters=int(getattr(settings, 'attn_bab_iters', 12)),
                    lr=float(getattr(settings, 'attn_bab_lr', 0.1)),
                    print_progress=print_progress,
                    gg_work=gg_pgd if _bab_fp32 else None,
                    work_dtype=torch.float32 if _bab_fp32 else None,
                    kfsb_k=int(getattr(settings, 'attn_bab_kfsb', 4)))
                _zis_nodes += n_used
                if print_progress:
                    print(f'  [beta-bab] q{qi} '
                          f'{"closed" if ok else "open"} after {n_used} '
                          f'domains ({_rs_reason})', flush=True)
            if not ok:
                ok, n_used, _rs_reason = _zono_relu_split_close(
                    gg, xl_g, xh_g, _w_zis, _b_zis, device, dtype,
                    settings, time_left,
                    max_nodes=int(getattr(
                        settings, 'zono_relu_split_max_nodes', 512)),
                    plane_params=_root_params)
                _zis_nodes += n_used
                if print_progress and not ok:
                    print(f'  [relu-split] q{qi} open after {n_used} '
                          f'boxes ({_rs_reason})', flush=True)
            if not ok and _zis_cap > 0:
                ok, n_used = _zono_input_split_close(
                    gg, xl_g, xh_g, _w_zis, _b_zis, device, dtype,
                    settings, time_left, max_nodes=_zis_cap)
                _zis_nodes += n_used
            if not ok:
                break
            _zis_closed.append(qi)
        timing['zono_input_split'] = time.perf_counter() - _t_zis
        if print_progress:
            print(f'  [zono-input-split] closed {len(_zis_closed)}/'
                  f'{len(still_needs_milp)} open queries '
                  f'({_zis_nodes} boxes, '
                  f'{timing["zono_input_split"]:.1f}s)', flush=True)
        if len(_zis_closed) == len(still_needs_milp):
            still_needs_milp.clear()
            return _finalize('verified', 'zono_input_split')

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
            # Timeout enforcement: building a per-query α-zono state (esp. the
            # box-halfspace scoring below) is expensive on big conv nets
            # (tinyimagenet resnet: ~18s/query). Without a clock check the loop
            # blows the total budget — 6 open specs × 18s on a 100s budget runs
            # to 170s+ (prop_9215 once hit 3242s). Stop building new states once
            # the budget reserve is gone (skipped queries fall back to the shared
            # gen_lp_state), and skip the scoring heuristic adaptively when there
            # is not time for another (it falls back to ew*frac).
            _pq_reserve = float(getattr(
                settings, 'phase8_per_query_state_reserve_s', 8.0))
            _score_dt = [0.0]   # measured cost of one box-halfspace scoring
            # Phase-8 component profiler (env PHASE8_PROFILE): accumulate per-stage
            # wall to find the real bottleneck. cuda-synced for accuracy.
            import os as _os_p8
            import collections as _coll_p8
            _p8prof = bool(_os_p8.environ.get('PHASE8_PROFILE'))
            _p8t = _coll_p8.defaultdict(float)

            def _ptick(_k, _t0):
                if _p8prof and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _p8t[_k] += time.perf_counter() - _t0
                return time.perf_counter()
            import os as _os_rb
            if _os_rb.environ.get('REVERSE_BATCH_BENCH'):
                import sys as _sys_rb, time as _tm_rb
                _sys_rb.path.insert(0, _os_rb.path.expanduser('~/vibecheck/scratch/reverse_g'))
                from reverse_build import build_state_reverse as _bsr
                from vibecheck.reverse_batched import build_states_reverse_batched_safe as _bsrb
                _bq = sorted(still_needs_milp)
                # collect per-direction alpha (one α-CROWN per query)
                _alphas = []
                for _qi in _bq:
                    _, _qw, _qb = queries[_qi]
                    _, _ap, _, _ = ac.run_alpha_crown_fixed_intermediate(
                        gg, xl_g, xh_g, bounds_by_relu, _qw, float(_qb), device, dtype,
                        n_iters=int(getattr(settings, 'zono_lift_alpha_iters', 10)),
                        lr=0.25, lr_decay=0.98, early_stop_on_positive=True,
                        init_alpha=_ph1_spec_alpha)
                    _, _ew = ac.capture_ew_per_relu(gg, xl_g, xh_g, _ap['spec'],
                                                    bounds_by_relu, _qw, float(_qb), device, dtype)
                    _alphas.append(ac.build_dir_adaptive_alpha(_ap['spec'], _ew, bounds_by_relu, device, dtype))
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                _t0 = _tm_rb.perf_counter()
                _seq = [_bsr(gg, xl_g, xh_g, bounds_by_relu, _a, device, dtype) for _a in _alphas]
                torch.cuda.synchronize(); _t_seq = _tm_rb.perf_counter() - _t0
                _m_seq = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                _t0 = _tm_rb.perf_counter()
                _bench_rb = {}
                _bat = _bsrb(gg, xl_g, xh_g, bounds_by_relu, _alphas, device, dtype, _bench=_bench_rb)
                torch.cuda.synchronize(); _t_bat = _tm_rb.perf_counter() - _t0
                _m_bat = torch.cuda.max_memory_allocated() / 1e9
                # correctness: batched vs sequential rows
                _w = 0.0
                for _d in range(len(_bq)):
                    _Fs = {(u['layer_idx'], u['neuron_idx']): u for u in _seq[_d]['unstable_list']}
                    _Fb = {(u['layer_idx'], u['neuron_idx']): u for u in _bat[_d]['unstable_list']}
                    for _k in _Fs:
                        _da = np.zeros(_seq[_d]['n_gens']); _db = np.zeros(_bat[_d]['n_gens'])
                        _da[_Fs[_k]['row_indices']] = _Fs[_k]['row_values']
                        _db[_Fb[_k]['row_indices']] = _Fb[_k]['row_values']
                        _w = max(_w, float(np.abs(_da - _db).max()))
                print(f'  [REVERSE_BATCH_BENCH] D={len(_bq)} | sequential {_t_seq:.2f}s '
                      f'peak {_m_seq:.2f}GB | batched {_t_bat:.2f}s peak {_m_bat:.2f}GB '
                      f'(chunk={_bench_rb.get("final_chunk")}, n_chunks={_bench_rb.get("n_chunks")}) '
                      f'| speedup {_t_seq/_t_bat:.2f}x | max|Δrow|={_w:.2e}', flush=True)
            # --- batched reverse state build over all open directions ----------
            _rev_batched = (
                bool(getattr(settings, 'phase8_reverse_g', False))
                and bool(getattr(settings, 'phase8_reverse_g_batched', False))
                and getattr(device, 'type', None) == 'cuda'
                and not details.get('phase2p5', {}).get('per_query'))
            _batched_rev_done = set()
            if _rev_batched:
                from .reverse_batched import build_states_reverse_batched_safe
                _bq = [qi for qi in sorted(still_needs_milp)
                       if time_left() > _pq_reserve]
                if _bq:
                    _alphas = []; _ews = {}
                    _tp = time.perf_counter()
                    for qi in _bq:
                        _, qw_q, qb_q = queries[qi]
                        _, _apq, _, _ = ac.run_alpha_crown_fixed_intermediate(
                            gg, xl_g, xh_g, bounds_by_relu, qw_q, float(qb_q),
                            device, dtype,
                            n_iters=int(getattr(settings, 'zono_lift_alpha_iters', 10)),
                            lr=float(getattr(settings, 'zono_lift_alpha_lr', 0.25)),
                            lr_decay=float(getattr(settings, 'alpha_crown_lr_decay', 0.98)),
                            early_stop_on_positive=bool(getattr(
                                settings, 'alpha_crown_early_stop_on_positive', True)),
                            init_alpha=_ph1_spec_alpha)
                        _, _ewq = ac.capture_ew_per_relu(
                            gg, xl_g, xh_g, _apq['spec'], bounds_by_relu,
                            qw_q, float(qb_q), device, dtype)
                        _ews[qi] = _ewq
                        _alphas.append(ac.build_dir_adaptive_alpha(
                            _apq['spec'], _ewq, bounds_by_relu, device, dtype))
                    _ptick('alpha_crown', _tp)
                    _tp = time.perf_counter()
                    _states = build_states_reverse_batched_safe(
                        gg, xl_g, xh_g, bounds_by_relu, _alphas, device, dtype)
                    _ptick('state_build', _tp)
                    for _idx, qi in enumerate(_bq):
                        state_by_qi[qi] = _states[_idx]
                        if bool(getattr(settings, 'phase8_score_box_halfspace', True)):
                            _, qw_q, qb_q = queries[qi]
                            ew_np = {L: (t.detach().cpu().numpy().astype(np.float64)
                                         if hasattr(t, 'detach')
                                         else np.asarray(t, dtype=np.float64))
                                     for L, t in _ews[qi].items()}
                            _tp = time.perf_counter()
                            bh = verify_gen_lp.score_box_halfspace_delta_lb(
                                state_by_qi[qi], qw_q, float(qb_q), ew_np)
                            per_query_scored[qi] = sorted(
                                bh.keys(), key=lambda k: bh[k], reverse=True)
                            _p8t['scoring'] += time.perf_counter() - _tp
                        _batched_rev_done.add(qi)
                    if print_progress:
                        print(f'  [batched reverse] {len(_bq)} directions built',
                              flush=True)
            from . import heartbeat as _hb
            for qi in sorted(still_needs_milp):
                if qi in _batched_rev_done:
                    continue
                _hb.set_phase(f'alpha-zono state build q{qi}')
                if time_left() <= _pq_reserve:
                    if print_progress:
                        print(f'  Per-query α-zono state: {time_left():.1f}s left '
                              f'<= reserve {_pq_reserve:g}s; remaining queries '
                              f'use shared state', flush=True)
                    break
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
                # Warm-start from Phase 0.5's optimised α when available.
                t_ac = time.perf_counter()
                _tp = t_ac
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
                            'alpha_crown_early_stop_on_positive', True)),
                        init_alpha=_ph1_spec_alpha))
                _tp = _ptick('alpha_crown', _tp)
                _, ew_at_relu_q = ac.capture_ew_per_relu(
                    gg, xl_g, xh_g, alpha_params['spec'], merged_bbr,
                    qw_q, float(qb_q), device, dtype)
                _tp = _ptick('capture_ew', _tp)
                alpha_per_layer_q = ac.build_dir_adaptive_alpha(
                    alpha_params['spec'], ew_at_relu_q, merged_bbr,
                    device, dtype)
                _tp = _ptick('build_dir_alpha', _tp)
                # Build per-layer unstable index tensors so the forward
                # only materialises the unstable rows of the pre-ReLU G.
                unstable_per_layer_q = {}
                for L, (lo_np, hi_np) in merged_bbr.items():
                    un = np.where((np.asarray(lo_np) < 0)
                                   & (np.asarray(hi_np) > 0))[0]
                    unstable_per_layer_q[L] = torch.as_tensor(
                        un, dtype=torch.long, device=device)
                _use_rev = (bool(getattr(settings, 'phase8_reverse_g', False))
                            and getattr(device, 'type', None) == 'cuda')
                _tp = _ptick('unstable_idx', _tp)
                if _use_rev:
                    # Reverse-mode build (backward from unstable+output neurons):
                    # replaces forward_zono_dir_adaptive + state_from_alpha_zono.
                    from .reverse_g import build_state_reverse
                    state_by_qi[qi] = build_state_reverse(
                        gg, xl_g, xh_g, merged_bbr, alpha_per_layer_q,
                        device, dtype)
                else:
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
                _tp = _ptick('state_build', _tp)
                # Box+halfspace per-neuron delta_LB scoring — closed-form
                # Lagrangian, ~10ms/1k neurons. Picks kfsb-quality
                # binarisation candidates by simulating off-side
                # (y_k=0) and on-side (y_k=z_k) substitutions and
                # taking BaB worst-child LB. Overrides ew*frac for
                # this query — measured 24/30 overlap with AB-CROWN's
                # actual splits on tinyimagenet prop_7575 (vs 19/30
                # for ew*frac). Gated by `phase8_score_box_halfspace`.
                # Skip the (expensive) box-halfspace scoring when there is not
                # time for another one plus the reserve — fall back to ew*frac.
                if (bool(getattr(settings, 'phase8_score_box_halfspace', True))
                        and time_left() > _score_dt[0] * 1.3 + _pq_reserve):
                    ew_np = {
                        L: (t.detach().cpu().numpy().astype(np.float64)
                            if hasattr(t, 'detach')
                            else np.asarray(t, dtype=np.float64))
                        for L, t in ew_at_relu_q.items()}
                    _t_sc = time.perf_counter()
                    bh_scores = verify_gen_lp.score_box_halfspace_delta_lb(
                        state_by_qi[qi], qw_q, float(qb_q), ew_np)
                    per_query_scored[qi] = sorted(
                        bh_scores.keys(),
                        key=lambda k: bh_scores[k], reverse=True)
                    _score_dt[0] = max(_score_dt[0],
                                       time.perf_counter() - _t_sc)
                    _p8t['scoring'] += time.perf_counter() - _t_sc
                if print_progress:
                    print(f'  Per-query α-zono state q{qi}: '
                          f'n_gens={state_by_qi[qi]["n_gens"]}, '
                          f'unstable={len(state_by_qi[qi]["unstable_list"])} '
                          f'({time.perf_counter()-t_ac:.2f}s)')
                # Free GPU tensors before next query.
                _tp = time.perf_counter()
                if not _use_rev:
                    del z_alpha, pre_relu_gpu_q
                del alpha_per_layer_q
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _ptick('empty_cache', _tp)
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
            # Fast GPU dual-ascent node bound (drop-in for the legacy per-query
            # BaB). CUDA-only; on CPU fall back to the legacy verifier. It emits
            # only 'unsat'/'unknown' and never attacks, so SAT detection stays
            # with the PGD/witness machinery that honors `disable_sat_finding`.
            _fast_da = (bool(getattr(settings, 'phase8_fast_dual_ascent', True))
                        and getattr(device, 'type', None) == 'cuda')
            _fast_verifier = None
            if _fast_da:
                _t_fv = time.perf_counter()
                _fast_verifier = _get_fast_da_verifier(
                    device,
                    str(getattr(settings, 'phase8_fast_dual_ascent_ls',
                                'logbucket')),
                    int(getattr(settings, 'phase8_fast_dual_ascent_K', 256)),
                    bool(getattr(settings,
                                 'phase8_fast_dual_ascent_compile', True)),
                    int(getattr(settings, 'phase8_fast_dual_ascent_sweeps', 1)))
                try:
                    _ptick('fast_verifier_build(compile)', _t_fv)
                except NameError:
                    pass
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
                n_dim = w_t.shape[1]
                if n_dim == center.numel():
                    # Dense witness: one coefficient per input dim.
                    x_real = center.unsqueeze(0) + w_t * halfwidth.unsqueeze(0)
                else:
                    # Sparse witness: the LP/dual-ascent primal `e` lives in the
                    # INPUT-GENERATOR subspace (one column per VARYING input dim,
                    # in radius>0 order) — for sparse specs like dist_shift
                    # mnist_concat only 8 of 792 inputs vary, so w is [N, 8] not
                    # [N, 792]. Scatter into the varying dims; fixed dims stay at
                    # center (= x_lo = x_hi). Without this the witness check
                    # crashed (8-vs-792 broadcast) and main.py's top-level
                    # handler masked it as 'unknown' — so dual-ascent never ran.
                    var_idx = (halfwidth > 0).nonzero(as_tuple=True)[0]
                    if n_dim != var_idx.numel():
                        # A witness dimension we can't map: skip the attack (the
                        # BaB bound is still sound — this only forgoes a cex).
                        return None
                    x_real = center.unsqueeze(0).expand(
                        w_t.shape[0], -1).clone()
                    x_real[:, var_idx] = (center[var_idx].unsqueeze(0)
                                          + w_t * halfwidth[var_idx].unsqueeze(0))
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
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # PGD attack on GPU: runtime/OOM. SAT-finding fallback
                    # is best-effort; return None so the LP path continues.
                    pass
                return None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            from . import heartbeat as _hb
            for (qi, qw_q, qb_q, scored_keys_q) in query_specs:
                _hb.set_phase(f'phase8 dual-ascent BnB q{qi}')
                if time_left() <= 0:
                    raw.append((qi, 'unknown', [], None))
                    continue
                # If the per-query α-zono state build was budget-truncated, a
                # skipped query has no alpha_zono state. The dual-ascent needs
                # one (reads 'lam'); the shared gen_lp_state is a phase1 state
                # without it. Leave such queries unknown rather than crash.
                if state_by_qi and qi not in state_by_qi:
                    raw.append((qi, 'unknown', [], None))
                    continue
                state_q = (state_by_qi.get(qi, gen_lp_state)
                           if state_by_qi else gen_lp_state)
                # Both dual-ascent paths (fast `verify_query` and
                # `verify_query_dual_ascent_bab`) read each unstable
                # entry's `lam`/`mu` — they require an alpha_zono/phase1
                # state. When the per-query α-zono build is FULLY
                # budget-truncated, `state_by_qi` is empty and the
                # fallback `gen_lp_state` may be an 'alpha'-form state
                # (e_new col coeff 1.0, no lam/mu — see
                # `precompute_gen_state`). Handing that to the DA
                # crashes on `u['lam']` (cifar10 cct2026 idx4226). Skip
                # the DA for this query; the high-bin MILP fallback below
                # is form-aware (`build_gen_lp_from_state`) and still
                # gets its shot.
                _ul_q = state_q.get('unstable_list', []) if state_q else []
                if _ul_q and 'lam' not in _ul_q[0]:
                    raw.append((qi, 'unknown', [], None))
                    continue
                # BUGFIX: query_specs was built with lp_ew_frac scoring at
                # phase 8 start; bh_scores reranking later updates
                # per_query_scored but query_specs holds stale reference.
                # Re-fetch the latest per_query_scored (which IS bh_scores
                # if `phase8_score_box_halfspace=True`).
                _latest_keys = per_query_scored.get(qi, scored_keys_q)
                if list(_latest_keys) != list(scored_keys_q):
                    scored_keys_q = _latest_keys
                if print_progress:
                    # Branch (split) ORDER the dual-ascent BaB will follow.
                    # The score is ew·intercept (|ew|·(-lo·hi/(hi-lo))) reranked
                    # by the box-halfspace ΔLB when `phase8_score_box_halfspace`
                    # is on (default); else the raw ew·intercept / box-area.
                    _bsrc = ('box_halfspace' if bool(getattr(
                        settings, 'phase8_score_box_halfspace', True))
                        else 'ew·intercept')
                    _top5 = ', '.join(
                        f'L{k[0]}n{k[1]}' for k in list(scored_keys_q)[:5])
                    print(f'  [fast-dual-ascent-gpu] q{qi} branch_score='
                          f'{_bsrc}, {len(scored_keys_q)} unstable; '
                          f'top 5 split neurons: {_top5}', flush=True)
                # DEBUG: dump state passed to dual-ascent BaB if the env var OR the
                # `dump_da_bab_dir` setting is set (env takes precedence).
                import os as _os
                _dump = _os.environ.get('DA_BAB_DUMP_DIR') or getattr(
                    settings, 'dump_da_bab_dir', '')
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
                # Phase-8 Problem dump (kernel-ready: dict(problem=Problem, extras))
                # for offline BnB replay / kernel A/B — gated by the env var OR the
                # `dump_bnb_dir` setting (env takes precedence), named bnb_<qi>.pkl.
                _bnb_dir = _os.environ.get('VC_DUMP_BNB_DIR', '') or getattr(
                    settings, 'dump_bnb_dir', '')
                if _bnb_dir:
                    from .fast_dual_ascent.fast_verify_dual import (
                        _dump_bnb_instance as _dbi)
                    _dbi(state_q, qw_q, qb_q, list(scored_keys_q),
                         _bnb_dir, query_id=qi)
                if _fast_da:
                    # Fast path: returns 'unsat'(robust) / 'unknown' only.
                    _t_call = time.perf_counter()
                    # Sibling output constraints (INVPROP-style): for a
                    # CONJUNCTIVE disjunct, refutation may assume the SAT
                    # set, so every sibling conjunct (w_j·y + b_j ≤ 0) is a
                    # valid halfspace to dualize into the node bound —
                    # aggressive joint pruning (yolo_2023: 5 conjuncts).
                    # Inert for single-conjunct disjuncts (regular track).
                    _sib_hs = ()
                    if bool(getattr(settings, 'phase8_sibling_halfspaces',
                                    True)):
                        _di_q = queries[qi][0]
                        _sib_hs = [(queries[qj][1], queries[qj][2])
                                   for qj in range(len(queries))
                                   if queries[qj][0] == _di_q and qj != qi]
                    _parallel_milp = bool(getattr(
                        settings, 'phase8_parallel_milp', False))
                    if (_parallel_milp
                            and bool(getattr(settings,
                                             'phase8_high_bin_fallback', True))
                            and state_q is not None and scored_keys_q
                            and time_left() > 5.0):
                        # Race the GPU BnB against the CPU/Gurobi high-bin MILP;
                        # take whichever closes the query first, terminate the
                        # loser. Sound: the MILP worker only reports `closed` on a
                        # proven ObjBound>=tol (or INFEASIBLE) certificate.
                        import threading as _thr
                        _stop_ev = _thr.Event(); _mres = {}
                        _hbc = _resolve_high_bin_count(
                            getattr(settings, 'phase8_high_bin_count', 200))
                        _nb = (len(scored_keys_q) if _hbc is None
                               else min(_hbc, len(scored_keys_q)))
                        _tlm = min(float(getattr(
                            settings, 'phase8_high_bin_time_limit', 60.0)),
                            time_left())
                        _ubbs = bool(getattr(
                            settings, 'phase8_high_bin_bestbdstop', True))
                        _btol = float(getattr(
                            settings, 'phase8_high_bin_bestbdstop_tol', 1e-6))
                        # Cap the racing MILP's Gurobi threads so it doesn't starve the
                        # GPU BnB's host-side orchestration. 0 = auto (n_cores // 2).
                        _mthr = int(getattr(settings,
                                            'phase8_parallel_milp_threads', 0))
                        if _mthr <= 0:
                            _mthr = max(1, n_cores // 2)
                        _mdelay = float(getattr(
                            settings, 'phase8_parallel_milp_delay', 0.0))

                        def _milp_worker():
                            # Head start: if the BnB closes (sets _stop_ev) within _mdelay,
                            # skip the MILP entirely — a fast BnB-closeable case never pays
                            # the model-build contention. wait() returns True iff set in time.
                            if _mdelay > 0.0 and _stop_ev.wait(_mdelay):
                                return
                            c, lb = _solve_high_bin_query(
                                state_q, qw_q, qb_q, scored_keys_q,
                                n_bins=_nb, time_limit=_tlm, n_threads=_mthr,
                                use_bbs=_ubbs, bbs_tol=_btol,
                                stop_event=_stop_ev)
                            _mres['closed'] = c; _mres['lb'] = lb
                            if c:
                                _stop_ev.set()
                        _mt = _thr.Thread(target=_milp_worker, daemon=True)
                        _mt.start()
                        vd, info = _fast_verifier.verify_query(
                            state_q, qw_q, qb_q, [k for k in scored_keys_q],
                            time_limit=time_left(), extra_hs=_sib_hs,
                            stop_event=_stop_ev)
                        if vd == 'unsat':
                            _stop_ev.set()      # BnB won → cancel the MILP
                        _mt.join()
                        if vd != 'unsat' and _mres.get('closed'):
                            # MILP won the race (BnB OOM/stopped/unknown).
                            vd = 'unsat'
                            info = dict(info or {})
                            info['milp_parallel'] = True
                            if print_progress:
                                print(f'  [phase8-parallel] q{qi}: high-bin MILP '
                                      f'closed (lb={_mres.get("lb")}) — BnB '
                                      f'{info.get("reason", "running")}',
                                      flush=True)
                        elif (vd == 'unsat' and print_progress
                                and not _mres.get('closed')):
                            print(f'  [phase8-parallel] q{qi}: BnB closed first '
                                  f'(MILP cancelled)', flush=True)
                    else:
                        vd, info = _fast_verifier.verify_query(
                            state_q, qw_q, qb_q,
                            [k for k in scored_keys_q],
                            time_limit=time_left(),
                            extra_hs=_sib_hs)
                    try:
                        if _p8prof and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _full = time.perf_counter() - _t_call
                        _p8t['bnb_node'] += float(info.get('wall', 0.0))
                        _p8t['bnb_parse+upload'] += _full - float(info.get('wall', 0.0))
                    except NameError:
                        pass
                else:
                    vd, info = verify_query_dual_ascent_bab(
                        state_q, qw_q, qb_q,
                        [k for k in scored_keys_q],
                        time_limit=time_left(),
                        max_iter=_da_K, repair_steps=_da_rep,
                        print_progress=False, time_left_fn=time_left,
                        witness_check_fn=_da_witness_check)
                if print_progress:
                    if _fast_da:
                        # `reason` attributes WHY the BnB stopped: 'time_limit',
                        # 'splits_exhausted', 'oom' (all → unknown, frontier not
                        # emptied) vs proved unsat (frontier emptied, no reason).
                        # There is NO node cap — the only stops are time/depth/OOM.
                        _rsn = info.get('reason')
                        print(f'  [fast-dual-ascent-gpu] query {qi}: {vd} '
                              f'nodes={info.get("nodes", 0)} '
                              f'wall={info.get("wall", 0.0):.3f}s '
                              f'(peak_frontier={info.get("peak_frontier", 0)}'
                              f'{f", stop={_rsn}" if _rsn else ""})')
                    else:
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
                                'time': info.get('wall', 0.0),
                                'source': 'dual_ascent',
                                'milp_parallel': bool(info.get('milp_parallel')),
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
                bin_schedule_override=_bin_sched,
                race_all_bins=bool(getattr(settings, 'phase8_race_all_bins', True)),
                bbs_tol=float(getattr(settings, 'phase8_high_bin_bestbdstop_tol', 1e-6)))
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
                _src = ('milp_parallel' if (race_levels
                        and race_levels[0].get('milp_parallel')) else 'bnb')
                details.setdefault('phase8_close_src', {})[qi] = _src
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
    # phase8_high_bin_count caps how many neurons the fallback binarizes. The
    # sentinel 'all' (also None / inf / <= 0) means "binarize EVERY unstable
    # neuron" — a full MILP. `None` here flags that; n_bins_fb resolves it to
    # len(scored_keys_q) per query (avoids a magic 'bigger than any net' int).
    _hi_bin_count = _resolve_high_bin_count(
        getattr(settings, 'phase8_high_bin_count', 200))
    _hi_bin_time = float(getattr(
        settings, 'phase8_high_bin_time_limit', 60.0))
    # Default-on (disable to fall back to the legacy halfspace+INFEASIBLE proof):
    # minimize the spec margin qw·y+qb and stop via BestBdStop as soon as Gurobi
    # PROVES a lower bound >= tol > 0 on it. That lower bound is an explicit,
    # auditable soundness certificate (relaxation min margin >= tol > 0 ⟹ unsafe
    # region empty ⟹ verified) and avoids depending on Gurobi's numerically
    # fragile infeasibility detection. A tiny positive tol guards borderline
    # numerics (a barely-positive margin is not certified rather than risked).
    _hi_bin_bbs = bool(getattr(settings, 'phase8_high_bin_bestbdstop', True))
    _hi_bin_bbs_tol = float(getattr(
        settings, 'phase8_high_bin_bestbdstop_tol', 1e-6))
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
            n_bins_fb = (len(scored_keys_q) if _hi_bin_count is None
                         else min(_hi_bin_count, len(scored_keys_q)))
            tl = min(_hi_bin_time, time_left())
            # Fallback runs sequentially — use ALL cores for one Gurobi
            # solve. Bypass solve_spec because that path disables cuts/
            # heuristics/presolve for fast small-bin tasks; at high bins
            # we WANT Gurobi's defaults (cuts ON, presolve ON) to reach
            # INFEASIBLE quickly. Empirically (img3039 q8): 200 bins +
            # default Gurobi closes in ~19s; bins=200 with cuts off
            # times out at 60s.
            import gurobipy as _grb
            from .gurobi_util import (optimize_checked, GurobiNumericTrouble)
            try:
                m_fb, env_fb, _, _ = verify_gen_lp.build_gen_lp_from_state(
                    state_q, qw_q, qb_q,
                    milp_set=set(scored_keys_q[:n_bins_fb]),
                    n_threads=n_cores,
                    unsafe_halfspace=('none' if _hi_bin_bbs else 'inequality'))
                m_fb.setParam('TimeLimit', float(tl))
                # BestBdStop: stop as soon as the proven lower bound on the
                # minimized spec margin reaches tol > 0. Legacy path disables it
                # and relies on INFEASIBLE under the qw·y+qb<=0 halfspace.
                m_fb.setParam('BestBdStop',
                              _hi_bin_bbs_tol if _hi_bin_bbs else _grb.GRB.INFINITY)
                try:
                    optimize_checked(m_fb)
                except (GurobiNumericTrouble, _grb.GurobiError):
                    m_fb.dispose(); env_fb.dispose()
                    continue
                fb_status = m_fb.Status
                try:
                    fb_obj_bound = float(m_fb.ObjBound)
                except (_grb.GurobiError, AttributeError):
                    fb_obj_bound = None
                m_fb.dispose(); env_fb.dispose()
            except (_grb.GurobiError, GurobiNumericTrouble):
                # Outer wrapper: the fallback model build/solve itself
                # raised. Skip this query and continue with the next.
                continue
            if _hi_bin_bbs:
                # Sound: ObjBound is a valid lower bound on min(qw·y+qb); if it
                # is >= tol > 0 then {qw·y+qb <= 0} ∩ relaxation = ∅ (verified).
                _closed = (fb_obj_bound is not None
                           and fb_obj_bound >= _hi_bin_bbs_tol)
            else:
                # Sound: relaxation∩{qw·y+qb≤0}=∅ ⇔ relaxation min > 0.
                _closed = (fb_status == _grb.GRB.INFEASIBLE)
            if _closed:
                spec_lbs[qi] = 1.0
                details.setdefault('phase8_close_src', {})[qi] = 'milp_fallback'
                if print_progress:
                    _how = (f'min-margin lb={fb_obj_bound:+.5f}>={_hi_bin_bbs_tol:g}'
                            if _hi_bin_bbs else '+halfspace INFEASIBLE')
                    print(f'  Fallback q{qi} bins={n_bins_fb}: {_how} '
                          f'→ q{qi} CLOSED', flush=True)
            elif print_progress:
                status_name = {_grb.GRB.OPTIMAL: 'OPTIMAL',
                                _grb.GRB.TIME_LIMIT: 'TIMEOUT',
                                _grb.GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT',
                                _grb.GRB.INFEASIBLE: 'INFEASIBLE'}.get(
                    fb_status, f'STATUS_{fb_status}')
                ob_s = (f'{fb_obj_bound:+.4f}'
                        if isinstance(fb_obj_bound, float) else 'n/a')
                print(f'  Fallback q{qi} bins={n_bins_fb}: '
                      f'{status_name} lb={ob_s}', flush=True)
    timing['phase8_milp'] = time.perf_counter() - t_phase8
    try:
        if _p8prof:
            _tot = timing['phase8_milp']
            _acc = sum(_p8t.values())
            _items = sorted(_p8t.items(), key=lambda kv: -kv[1])
            _s = '  '.join(f'{k}={v:.2f}s' for k, v in _items)
            print(f'  [PHASE8_PROFILE] total={_tot:.2f}s  accounted={_acc:.2f}s'
                  f'  unaccounted={_tot-_acc:.2f}s | {_s}', flush=True)
    except NameError:
        pass

    # If the gen_lp path found a true counterexample from a MILP integer
    # solution, short-circuit to SAT before running final PGD.
    if milp_witness is not None:
        return _finalize('sat', 'spec_milp', witness=milp_witness)

    # ANY-closure: a disjunct is a CONJUNCTION of constraints (the SAT set
    # is their AND); refuting ANY single conjunct (its query lb > 0) proves
    # the whole disjunct infeasible. all() here was sound but needlessly
    # strict — multi-conjunct specs (yolo_2023: 5 objectness conjuncts per
    # disjunct) could only verify by refuting EVERY conjunct, which is
    # usually impossible. For single-conjunct disjuncts (all regular-track
    # benchmarks) any() == all(), so behavior there is unchanged.
    verified_disj = {di for di, qlist in disj_queries.items()
                      if any(spec_lbs.get(qi, -1) > 0 for qi, _, _ in qlist)}
    still_open_disj = set(disj_queries.keys()) - verified_disj

    # --- Phase 9: final PGD, restricted to the BnB survivors ---
    # BnB-first / attack-the-survivors schedule: Phase 8 already closed every
    # robust disjunct, so `still_open_disj` is exactly the set BnB left open —
    # on a robust spec that set is empty and we never attack (0 PGD overhead),
    # on a SAT spec it is the counterexample-bearing disjunct(s). Restricting the
    # attack to those disjuncts concentrates OSI/PGD on the live margins instead
    # of diluting the min over already-verified ones. Reuses VC's own
    # pgd_attack_general (OSI init + params from configs/default.yaml).
    t0 = time.perf_counter()
    if still_open_disj and time_left() > 0 and not _disable_sat:
        try:
            pgd_sat, pgd_witness = _pgd_attack_general(
                xl_pgd, xh_pgd, spec, gg_pgd, settings,
                restrict_disj=still_open_disj)
        except RuntimeError:
            pgd_sat, pgd_witness = False, None
        if pgd_sat:
            timing['phase9_pgd'] = time.perf_counter() - t0
            _r = _sat_or_fallthrough('pgd', pgd_witness)
            if _r is not None:
                return _r
    timing['phase9_pgd'] = time.perf_counter() - t0

    # Attribute which Phase-8 phase actually closed each query, so the verbose
    # trace and the final `phase` tag reflect reality (e.g. "verified by BnB" vs
    # "by the high-bin MILP fallback") instead of a blanket 'spec_milp'.
    _csrc = details.get('phase8_close_src', {})
    _n_bnb = sum(1 for v in _csrc.values() if v == 'bnb')
    _n_fb = sum(1 for v in _csrc.values() if v == 'milp_fallback')
    if print_progress and (_n_bnb or _n_fb):
        print(f'  [phase8] closed {_n_bnb} quer{"y" if _n_bnb == 1 else "ies"} '
              f'by dual-ascent BnB, {_n_fb} by high-bin MILP fallback '
              f'(α-CROWN/earlier closed the rest)', flush=True)
    if not still_open_disj:
        # 'spec_milp' only if the MILP fallback was load-bearing; else the BnB
        # (or earlier α-CROWN) carried it — name the phase that did the work.
        _src_phase = ('spec_milp' if _n_fb
                      else 'dual_ascent_bab' if _n_bnb else 'alpha_crown')
        return _finalize('verified', _src_phase)
    return _finalize('unknown', 'timeout', remaining=len(still_open_disj))


def _verify_per_disjunct_subboxes(graph, spec, settings):
    """Decompose a spec whose conjuncts each carry an X subbox into
    per-unique-subbox sub-verifications.

    For each unique (input_lo, input_hi) tuple across disjuncts, build a
    sub-spec with:
      - x_lo/x_hi = that subbox
      - disjuncts = the Y-only Conjuncts (input_lo cleared so the
        recursive call skips this branch) whose original subbox matches
    Each sub-verification is a standard verify_graph call. Verdicts:
      - SAT iff any sub returns SAT
      - verified iff all subs return verified
      - unknown otherwise

    Time budget is split proportionally to sub count; per-sub minimum
    is 0.2 s to avoid 0-budget no-ops on huge disjunct sets.
    """
    import copy
    from .spec import VNNSpec, Conjunct, Constraint
    t_start = time.perf_counter()
    total = float(getattr(settings, 'total_timeout', 30.0))

    # Group disjuncts by their (input_lo, input_hi) tuple. Use bytes for
    # hashable dict keys.
    groups = {}
    order = []
    for conj in spec.disjuncts:
        key = (conj.input_lo.tobytes(), conj.input_hi.tobytes())
        if key not in groups:
            groups[key] = (conj.input_lo, conj.input_hi, [])
            order.append(key)
        # Clone the Conjunct without per-disjunct X bounds — the sub-spec's
        # global x_lo/x_hi IS the subbox.
        groups[key][2].append(Conjunct(conj.constraints))

    if getattr(settings, 'print_progress', False):
        print(f'[per-disjunct] {len(groups)} unique X subboxes '
              f'across {len(spec.disjuncts)} disjuncts', flush=True)

    n_sub = len(groups)

    # Fast inline CROWN path: when n_sub is large, calling verify_graph
    # per sub-box pays ~50 ms of pipeline overhead per call (gg build,
    # phase 0 PGD setup, etc.) — 10000 subs × 50 ms = 500 s, blowing
    # past any reasonable budget. Instead, batch a single forward zono
    # over ALL subboxes (one tensor of (B, n_in) lo/hi), CROWN backward
    # per-sub, and emit verdicts inline. Falls back to per-sub
    # verify_graph for any sub the cheap path can't close.
    from .verify_zono_bnb import _forward_zonotope_graph_batched
    from .forward_lirpa import forward_lirpa_compat_zono_batched
    from .settings import resolve_torch
    dev, dt = resolve_torch(settings)
    try:
        gg_fast = graph.gpu_graph(dev, dt)
    except (RuntimeError, NotImplementedError, KeyError):
        # gpu_graph can't represent this model (unsupported op or shape).
        # Fall back to per-sub verify_graph which handles more op types.
        gg_fast = None
    # Forward LiRPA gives tighter intermediate bounds than zonotope
    # (especially for sigmoid/tanh/mul_bilinear), so basic CROWN closes
    # more subs without needing α-CROWN escalation. Same caller signature.
    _use_lirpa = bool(getattr(settings, 'use_forward_lirpa_subboxes', True))
    _fwd_fn = (forward_lirpa_compat_zono_batched
               if _use_lirpa else _forward_zonotope_graph_batched)

    fast_results = {}
    fast_wall = 0.0
    # STRUCTURAL fast-pass: detect per-sub disjunctions of pairwise
    # overlapping halfspaces, where the OR-coverage is all of R
    # regardless of network output. Mirrors the mscn cardinality_X_Y
    # pattern: each sub is `(Y_i ≥ thr_lo) ∨ (Y_i ≤ thr_hi)` with
    # thr_lo ≤ thr_hi. For these, the disjunction is trivially TRUE
    # for any Y — no verification needed.
    for ki, key in enumerate(order):
        _, _, conjs = groups[key]
        if len(conjs) < 2:
            continue
        per_dim = {}  # dim_idx -> list of (sign, threshold)
        ok = True
        for d in conjs:
            if len(d.constraints) != 1:
                ok = False; break
            c = d.constraints[0]
            # This OR-coverage fast-pass is threshold-only; a PairwiseConstraint
            # (Y_i >= Y_j, e.g. acasxu prop_6/7/8 disjunctive-input) has no
            # threshold `.op`/`.value` -> bail and let the CROWN path handle it.
            if not isinstance(c, Constraint):
                ok = False; break
            sign = +1 if c.op == '>=' else -1 if c.op == '<=' else 0
            if sign == 0:
                ok = False; break
            per_dim.setdefault(c.index, []).append((sign, c.value))
        if not ok:
            continue
        # For each dim, if it has BOTH a `>= thr_lo` and `<= thr_hi`
        # with thr_lo ≤ thr_hi, that dim's disjunction covers all R.
        if any(
            any(s == +1 for s, _ in entries)
            and any(s == -1 for s, _ in entries)
            and max(t for s, t in entries if s == +1)
                <= min(t for s, t in entries if s == -1)
            for entries in per_dim.values()
        ):
            fast_results[ki] = 'verified'
    if (fast_results and getattr(settings, 'print_progress', False)):
        print(f'[per-disjunct] structural OR-coverage closed '
              f'{len(fast_results)}/{n_sub} subs',
              flush=True)
    if gg_fast is not None:
        try:
            xls = np.stack([groups[k][0] for k in order]).astype(
                np.float64)
            xhs = np.stack([groups[k][1] for k in order]).astype(
                np.float64)
            xl_t = torch.as_tensor(xls, dtype=dt, device=dev)
            xh_t = torch.as_tensor(xhs, dtype=dt, device=dev)
            t_fast = time.perf_counter()
            import os as _os_dbg_fp
            if _os_dbg_fp.environ.get('DEBUG_PER_DISJ_PATH', '') == '1':
                print(f'[per-disj-path] _fwd_fn={_fwd_fn.__name__} '
                      f'use_lirpa={_use_lirpa}', flush=True)
            # OOM-safe chunked forward: try full batch first; on OOM,
            # halve. The full-batch path is much faster than chunked
            # (no kernel-launch overhead, better cache utilization), so
            # only halve when actually needed.
            _bs_full = xl_t.shape[0]
            _chunk = _bs_full
            while _chunk >= 1:
                try:
                    if _chunk >= _bs_full:
                        sb_b, (c_b, G_b) = _fwd_fn(
                            xl_t, xh_t, gg_fast, dev, dt)
                    else:
                        sb_parts = None; c_parts = []; G_parts = []
                        for ci0 in range(0, _bs_full, _chunk):
                            ci1 = min(ci0 + _chunk, _bs_full)
                            sb_c, (cc, GG) = _fwd_fn(
                                xl_t[ci0:ci1], xh_t[ci0:ci1], gg_fast, dev, dt)
                            if sb_parts is None:
                                sb_parts = {L: ([sb_c[L][0]], [sb_c[L][1]])
                                             for L in sb_c}
                            else:
                                for L in sb_c:
                                    sb_parts[L][0].append(sb_c[L][0])
                                    sb_parts[L][1].append(sb_c[L][1])
                            c_parts.append(cc); G_parts.append(GG)
                        sb_b = {L: (torch.cat(sb_parts[L][0], dim=0),
                                     torch.cat(sb_parts[L][1], dim=0))
                                 for L in sb_parts}
                        # G shape: (B, n, K). K may differ per chunk
                        # (Sigmoid/Tanh/Pow add new gens proportional to
                        # the unstable count, which varies per batch).
                        # Pad to common K with zeros (sound — zero gens
                        # add 0 to bounds).
                        max_K = max(G.shape[2] for G in G_parts)
                        G_padded = []
                        for G in G_parts:
                            if G.shape[2] < max_K:
                                pad = G.new_zeros(G.shape[0], G.shape[1],
                                                   max_K - G.shape[2])
                                G_padded.append(torch.cat([G, pad], dim=2))
                            else:
                                G_padded.append(G)
                        c_b = torch.cat(c_parts, dim=0)
                        G_b = torch.cat(G_padded, dim=0)
                    break  # success
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if _chunk == 1:
                        raise  # truly cannot fit
                    _chunk = max(1, _chunk // 2)
            # Stash chunk size for downstream chunked CROWN calls.
            _fb_chunk = _chunk
            n_out = c_b.shape[1]
            # For each sub-box index b, check all its Y conjuncts.
            for ki, key in enumerate(order):
                _, _, conjs = groups[key]
                lo_y = (c_b[ki] - G_b[ki].abs().sum(dim=-1))
                hi_y = (c_b[ki] + G_b[ki].abs().sum(dim=-1))
                lo_np = lo_y.detach().cpu().numpy()
                hi_np = hi_y.detach().cpu().numpy()
                # Build a sub-spec just to reuse spec.check semantics.
                sub_spec_ck = VNNSpec(
                    groups[key][0], groups[key][1], conjs)
                ck_res, _ = sub_spec_ck.check(lo_np, hi_np)
                if ck_res == 'verified':
                    fast_results[ki] = 'verified'
            fast_wall = time.perf_counter() - t_fast
            # Batched CROWN backward for sub-boxes that forward bounds
            # didn't close. Build all per-sub spec queries, run a single
            # batched backward, check per-sub verdicts. For mscn
            # (1-input per disjunct, simple Y-constraints), this closes
            # most of what forward leaves open.
            from .verify_zono_bnb import _spec_backward_graph_batched
            t_crown = time.perf_counter()
            n_crown_closed = 0
            try:
                # For each unclosed sub-box, build (w, b) per query.
                # All sub-boxes share the same Y dim n_out, so we can
                # batch all queries across all unclosed subs as (Q_total, n_out).
                unclosed = [ki for ki in range(n_sub) if ki not in fast_results]
                if unclosed:
                    # Try TRULY batched CROWN backward across all
                    # unclosed subs. This works when every sub's spec
                    # has the same `w` vector (typical for mscn:
                    # "Y_0 <= per_sub_threshold" — w = e_0 shared,
                    # bias varies per sub). One batched _spec_backward
                    # call per UNIQUE w computes spec_lbs of shape
                    # (B, 1); we then compare per-sub spec_lb > -bias.
                    #
                    # When ws differ across subs, fall back to per-sub
                    # loop below.
                    sub_queries = []
                    all_w_sub = []
                    for ki in unclosed:
                        _, _, conjs = groups[order[ki]]
                        sub_sp = VNNSpec(
                            groups[order[ki]][0], groups[order[ki]][1], conjs)
                        qs = sub_sp.as_linear_queries(n_out)
                        sub_queries.append(qs)
                        if len(qs) >= 1:
                            all_w_sub.append(np.asarray(qs[0][1],
                                                          dtype=np.float64))
                    same_w = (len(all_w_sub) > 0 and all(
                        np.array_equal(w, all_w_sub[0]) for w in all_w_sub))
                    one_q_per_sub = all(len(q) == 1 for q in sub_queries
                                         if len(q) > 0)
                    if same_w and one_q_per_sub:
                        try:
                            w_shared = torch.as_tensor(
                                all_w_sub[0], dtype=dt, device=dev)
                            # Restrict the batched backward to the UNCLOSED subs
                            # so the returned lbs align with `unclosed` (hence
                            # with `biases` and the margin loop). Passing the
                            # FULL xl_t/xh_t returns lbs indexed by sub; once the
                            # forward pass has closed some subs, `unclosed` is a
                            # strict subset and `margins[idx]` would read a
                            # DIFFERENT sub's (possibly positive) margin -> a
                            # false `verified`. The per-q and per-sub paths below
                            # already slice; this one did not.
                            _unc_t = torch.as_tensor(
                                unclosed, dtype=torch.long, device=dev)
                            xl_unc = xl_t.index_select(0, _unc_t)
                            xh_unc = xh_t.index_select(0, _unc_t)
                            tight_all = {L: (sb_b[L][0].index_select(0, _unc_t),
                                              sb_b[L][1].index_select(0, _unc_t))
                                          for L in sb_b}
                            spec_ew_shared = {0: (w_shared, 0.0)}
                            import os as _osd0
                            spec_lbs_all = _spec_backward_graph_batched(
                                tight_all, xl_unc, xh_unc, gg_fast,
                                spec_ew_shared, dev, dt)
                            lbs_np_all = spec_lbs_all[:, 0].detach().cpu().numpy()
                            biases = np.array([float(sub_queries[idx][0][2])
                                                if sub_queries[idx] else -1e9
                                                for idx in range(len(unclosed))])
                            margins = lbs_np_all + biases
                            if _osd0.environ.get('VIB_DUMP_INIT_LB', '') == '1':
                                print(f"  [VIB] batched_per_q first 5: lbs={lbs_np_all[:5].tolist()}, biases={biases[:5].tolist()}, margins={margins[:5].tolist()}", flush=True)
                            for idx, ki in enumerate(unclosed):
                                if margins[idx] > 0:
                                    fast_results[ki] = 'verified'
                                    n_crown_closed += 1
                            # α-CROWN escalation: widened threshold —
                            # on mscn many-disjunct cases, basic CROWN's
                            # unclosed subs have margins in (-0.5, 0]
                            # and α-CROWN closes ~63% of those. The
                            # batched α-CROWN cost scales linearly with
                            # the candidate count and is GPU-batched
                            # (~1-2s for 500 subs × 20 iters).
                            close_idx = [idx for idx, ki in enumerate(unclosed)
                                            if ki not in fast_results
                                            and margins[idx] > -1.0]
                            if (close_idx and
                                time.perf_counter() - t_start <
                                    total - 5.0):
                                from .verify_zono_bnb import (
                                    _run_alpha_crown_inputsplit_batched)
                                idx_t_tensor = torch.as_tensor(
                                    close_idx, dtype=torch.long, device=dev)
                                # close_idx are positions WITHIN `unclosed`, so
                                # index the unclosed slice (not the full xl_t).
                                xl_open = xl_unc.index_select(0, idx_t_tensor)
                                xh_open = xh_unc.index_select(0, idx_t_tensor)
                                spec_lbs_alpha = _run_alpha_crown_inputsplit_batched(
                                    xl_open, xh_open, gg_fast,
                                    spec_ew_shared, dev, dt, n_iters=30)
                                lbs_alpha_np = spec_lbs_alpha[:, 0].detach().cpu().numpy()
                                for i_open, idx in enumerate(close_idx):
                                    ki = unclosed[idx]
                                    bias = float(sub_queries[idx][0][2])
                                    if lbs_alpha_np[i_open] + bias > 0:
                                        fast_results[ki] = 'verified'
                                        n_crown_closed += 1
                        except (RuntimeError, torch.cuda.OutOfMemoryError):
                            # α-CROWN inputsplit on GPU: RuntimeError covers
                            # CUBLAS / shape errors, OOM is the dominant
                            # failure on dense LiRPA matrices. Skip this
                            # batch and let the per-sub fallback try.
                            pass
                    else:
                        # Mixed-q-per-sub path. mscn cardinality_X_Y_DIM
                        # has 2 disjuncts per sub (OR semantics) with
                        # SAME w per query INDEX across subs (q0: w=[1],
                        # q1: w=[-1]). Sub safe iff ANY query verifies.
                        per_sub_q_count = [len(q) for q in sub_queries]
                        all_w = []; all_b = []
                        for qs in sub_queries:
                            for _, w, b in qs:
                                all_w.append(np.asarray(w, dtype=np.float64))
                                all_b.append(float(b))
                        # Detect: ALL subs have SAME q_count AND for each
                        # query-index, all subs share the same w. If yes,
                        # run BATCHED basic+α-CROWN per query index.
                        max_q = max(per_sub_q_count) if per_sub_q_count else 0
                        uniform_q = all(c == max_q for c in per_sub_q_count)
                        per_qi_same_w = uniform_q
                        if uniform_q:
                            for qi in range(max_q):
                                w_per_sub = []
                                for sub_idx, qs in enumerate(sub_queries):
                                    w_per_sub.append(np.asarray(
                                        qs[qi][1], dtype=np.float64))
                                if not all(np.array_equal(w_per_sub[0], w)
                                           for w in w_per_sub):
                                    per_qi_same_w = False
                                    break
                        if uniform_q and per_qi_same_w and max_q >= 1:
                            # Per-q-index batched CROWN + α-CROWN.
                            try:
                                if getattr(settings, 'print_progress', False):
                                    print(f'[per-disjunct] per-q batched CROWN '
                                          f'on {len(unclosed)} subs × {max_q} q',
                                          flush=True)
                                # Track which subs are verified by ANY q.
                                sub_verified_by_q = np.zeros(
                                    len(unclosed), dtype=bool)
                                # Bias per (sub, q_idx).
                                bias_per_sub_q = np.zeros((len(unclosed), max_q))
                                for sub_idx, qs in enumerate(sub_queries):
                                    for qi in range(max_q):
                                        bias_per_sub_q[sub_idx, qi] = float(
                                            qs[qi][2])
                                # Build per-unclosed-sub xl/xh + sb.
                                ki_t = torch.as_tensor(
                                    unclosed, dtype=torch.long, device=dev)
                                xl_unc = xl_t.index_select(0, ki_t)
                                xh_unc = xh_t.index_select(0, ki_t)
                                sb_unc = {L: (sb_b[L][0].index_select(0, ki_t),
                                               sb_b[L][1].index_select(0, ki_t))
                                           for L in sb_b}
                                for qi in range(max_q):
                                    w_q = torch.as_tensor(
                                        np.asarray(sub_queries[0][qi][1],
                                                    dtype=np.float64),
                                        dtype=dt, device=dev)
                                    spec_ew_q = {0: (w_q, 0.0)}
                                    lbs_basic_q = _spec_backward_graph_batched(
                                        sb_unc, xl_unc, xh_unc, gg_fast,
                                        spec_ew_q, dev, dt)
                                    lbs_np_q = (
                                        lbs_basic_q[:, 0].detach().cpu().numpy())
                                    margins_basic_q = (lbs_np_q
                                                       + bias_per_sub_q[:, qi])
                                    sub_verified_by_q |= (margins_basic_q > 0)
                                    # α-CROWN escalation on subs still
                                    # unverified by ANY q so far AND
                                    # close to verified by THIS q.
                                    close_for_q = (
                                        (~sub_verified_by_q)
                                        & (margins_basic_q > -1.0))
                                    close_idx = np.where(close_for_q)[0]
                                    if (len(close_idx) > 0
                                            and time.perf_counter() - t_start
                                                < total - 5.0):
                                        from .verify_zono_bnb import (
                                            _run_alpha_crown_inputsplit_batched)
                                        idx_t = torch.as_tensor(
                                            close_idx, dtype=torch.long,
                                            device=dev)
                                        xl_open = xl_unc.index_select(0, idx_t)
                                        xh_open = xh_unc.index_select(0, idx_t)
                                        spec_lbs_alpha = (
                                            _run_alpha_crown_inputsplit_batched(
                                                xl_open, xh_open, gg_fast,
                                                spec_ew_q, dev, dt,
                                                n_iters=100, lr=0.25,
                                                lr_decay=0.99))
                                        lbs_alpha_np = (
                                            spec_lbs_alpha[:, 0].detach()
                                            .cpu().numpy())
                                        for i_open, sub_idx in enumerate(
                                                close_idx):
                                            margin_alpha = (
                                                lbs_alpha_np[i_open]
                                                + bias_per_sub_q[sub_idx, qi])
                                            if margin_alpha > 0:
                                                sub_verified_by_q[sub_idx] = True
                                # Now mark sub as verified if ANY q verified.
                                for sub_idx, ki in enumerate(unclosed):
                                    if sub_verified_by_q[sub_idx]:
                                        fast_results[ki] = 'verified'
                                        n_crown_closed += 1
                                if getattr(settings, 'print_progress', False):
                                    print(f'[per-disjunct] per-q closed '
                                          f'{int(sub_verified_by_q.sum())}/'
                                          f'{len(unclosed)} subs',
                                          flush=True)
                            except (RuntimeError, torch.cuda.OutOfMemoryError,
                                    ValueError, KeyError) as _e:
                                # Per-q batched CROWN: GPU runtime/OOM or
                                # shape/key mismatch from sparse selectors.
                                # Already prints traceback under print_progress;
                                # fall through to per-sub fallback path.
                                if getattr(settings, 'print_progress', False):
                                    import traceback
                                    print(f'[per-disjunct] per-q EXC: {_e}',
                                          flush=True)
                                    traceback.print_exc()
                        # Per-sub fallback loop. Skip already-closed
                        # subs (from batched per-q path above).
                        # CAPPED to small count — downstream multi-sub
                        # BAB handles big-sub-count cases faster via
                        # batched LiRPA. This loop was eating 130+s on
                        # mscn_5410_dual at 50ms per call * 2500 subs.
                        _per_sub_cap = 20
                        for _pos, (ki, q_count) in enumerate(zip(unclosed, per_sub_q_count)):
                            if _pos >= _per_sub_cap:
                                break
                            if ki in fast_results:
                                continue
                            elapsed = time.perf_counter() - t_start
                            if elapsed > total: break
                            if q_count == 0:
                                continue
                            tight_sub = {L: (sb_b[L][0][ki:ki+1],
                                              sb_b[L][1][ki:ki+1])
                                          for L in sb_b}
                            offset = sum(per_sub_q_count[:unclosed.index(ki)])
                            w_q_sub = np.stack(all_w[offset:offset+q_count])
                            b_q_sub = np.array(all_b[offset:offset+q_count])
                            spec_ew_sub = {qi: (
                                torch.as_tensor(w_q_sub[qi], dtype=dt, device=dev),
                                float(b_q_sub[qi]))
                                for qi in range(q_count)}
                            xl_sub = xl_t[ki:ki+1]
                            xh_sub = xh_t[ki:ki+1]
                            try:
                                spec_lbs = _spec_backward_graph_batched(
                                    tight_sub, xl_sub, xh_sub, gg_fast,
                                    spec_ew_sub, dev, dt)
                                lbs_np = spec_lbs[0].detach().cpu().numpy()
                                import os as _os_dump
                                if _os_dump.environ.get('VIB_DUMP_INIT_LB', '') == '1':
                                    print(f"  [VIB] fast_crown ki={ki} lbs={lbs_np[:5]}", flush=True)
                                _, _, conjs_sub = groups[order[ki]]
                                sub_sp = VNNSpec(
                                    groups[order[ki]][0],
                                    groups[order[ki]][1], conjs_sub)
                                queries = sub_sp.as_linear_queries(n_out)
                                disj_qs = {}
                                for qi, (di, _, _) in enumerate(queries):
                                    disj_qs.setdefault(di, []).append(qi)
                                if all(any(lbs_np[qi] > 0 for qi in qis)
                                       for qis in disj_qs.values()):
                                    fast_results[ki] = 'verified'
                                    n_crown_closed += 1
                            except (RuntimeError, ValueError, KeyError):
                                # Per-sub spec backward: torch/shape/key
                                # failures fall back silently (this sub stays
                                # unverified and downstream BAB will retry).
                                pass
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Outer wrapper of the CROWN-closed loop; GPU runtime issues
                # fall through to the per-sub verify_graph fallback below.
                pass
            crown_wall = time.perf_counter() - t_crown
            del c_b, G_b
        except torch.cuda.OutOfMemoryError:
            # OOM is the only expected failure mode of the batched fast
            # path — let it fall through to per-sub verify_graph below.
            torch.cuda.empty_cache()

    if getattr(settings, 'print_progress', False):
        print(f'[per-disjunct] fast batched CROWN closed '
              f'{len(fast_results)}/{n_sub} subs in {fast_wall:.2f}s',
              flush=True)

    # Fallback: per-sub UNBATCHED forward zono + (if open) spec CROWN
    # backward. The batched forward doesn't yet handle nn4sys mscn ops
    # (mul_bilinear, div_bilinear, reduce_sum, ND fc); the unbatched
    # forward does. Loop over remaining subs:
    #   1. forward zono → check bounds (closes ~35% of mscn subs)
    #   2. if open, spec CROWN backward → tighter spec lb (closes more)
    # Both are 10-50ms each, much cheaper than full verify_graph.
    # Skip the per-sub unbatched-forward+CROWN fallback when many subs
    # remain — the downstream multi-sub BAB closes the same subs ~10x
    # faster via batched GPU operations. The unbatched loop here was
    # ~22s wasted on mscn_3800 (closed 0 subs in 22s).
    _unb_threshold = 30
    if (gg_fast is not None
            and (n_sub - len(fast_results)) <= _unb_threshold):
        from .verify_zono_bnb import (
            _forward_zonotope_graph, _spec_backward_graph)
        t_unbatched = time.perf_counter()
        n_fwd_closed = 0
        n_crown_closed = 0
        for ki, key in enumerate(order):
            if ki in fast_results:
                continue
            elapsed = time.perf_counter() - t_start
            if elapsed > total:
                break
            x_lo_np, x_hi_np, conjs = groups[key]
            xl_one = torch.as_tensor(x_lo_np, dtype=dt, device=dev)
            xh_one = torch.as_tensor(x_hi_np, dtype=dt, device=dev)
            try:
                sb_u, z_one = _forward_zonotope_graph(
                    xl_one, xh_one, gg_fast, dev, dt)
                lo_y, hi_y = z_one.bounds()
                lo_np = lo_y.detach().cpu().numpy()
                hi_np = hi_y.detach().cpu().numpy()
                sub_spec_ck = VNNSpec(x_lo_np, x_hi_np, conjs)
                ck_res, _ = sub_spec_ck.check(lo_np, hi_np)
                if ck_res == 'verified':
                    fast_results[ki] = 'verified'
                    n_fwd_closed += 1
                    continue
                # Spec CROWN backward (cheap, ~10ms): tighter than
                # raw forward bounds. Use as_linear_queries to get
                # per-conjunct (w, b); verified iff all queries lb > 0.
                n_out = z_one.center.numel()
                queries = sub_spec_ck.as_linear_queries(n_out)
                if not queries:
                    continue
                # Build spec_ew dict {qi: (w, b)}
                spec_ew = {qi: (torch.as_tensor(w, dtype=dt, device=dev),
                                  float(b))
                            for qi, (di, w, b) in enumerate(queries)}
                spec_lbs, _open = _spec_backward_graph(
                    sb_u, xl_one, xh_one, gg_fast, spec_ew,
                    list(range(len(queries))), len(sb_u), dev, dt)
                # AND semantics: per-disjunct safe iff ANY query lb > 0.
                disj_queries = {}
                for qi, (di, w, b) in enumerate(queries):
                    disj_queries.setdefault(di, []).append(qi)
                all_disj_safe = all(
                    any(spec_lbs.get(qi, -1.0) > 0 for qi in qis)
                    for qis in disj_queries.values())
                if all_disj_safe:
                    fast_results[ki] = 'verified'
                    n_crown_closed += 1
            except (RuntimeError, KeyError, ValueError):
                # Per-sub CROWN backward: torch runtime/shape/key issues
                # leave the sub unverified for the BAB fallback to handle.
                pass
        if getattr(settings, 'print_progress', False):
            print(f'[per-disjunct] unbatched forward closed '
                  f'{n_fwd_closed} + CROWN closed {n_crown_closed} '
                  f'more subs in {time.perf_counter() - t_unbatched:.2f}s',
                  flush=True)

    # Sub-budget for remaining (un-closed) sub-boxes — DYNAMIC so a
    # fast-closing sub frees its unused budget for subsequent harder
    # subs. Recomputed at each iter in the loop below.
    remaining_sub_idx = [ki for ki in range(n_sub) if ki not in fast_results]
    if getattr(settings, 'print_progress', False):
        print(f'[per-disjunct] before multi-sub gate: '
              f'gg_fast={gg_fast is not None} '
              f'remaining={len(remaining_sub_idx)} '
              f'time_left={total - (time.perf_counter() - t_start):.1f}s',
              flush=True)
    # MULTI-SUB INPUT-SPLIT BAB: when many subs remain (mscn 1000+),
    # batch all their leaves into a single GPU worklist (ABC-style:
    # single "domains" list, batched bounding per iter). Each domain is
    # (sub_idx, xl_leaf, xh_leaf); sub safe iff ALL its leaves verified
    # by ANY query. Each iter:
    #   1. Pop batch of B leaves.
    #   2. Batched forward zono → spec_lb per (leaf, query).
    #   3. Per leaf, mark as safe if ANY query margin > 0.
    #   4. Unsafe leaves split on widest dim → 2 child leaves, push back.
    # `serial_disjuncts` skips this batched-together path: it lacks the
    # α-CROWN boundary closing the single-box driver has, so on acasxu
    # prop_6 it burns the whole budget (401k leaves, 0/2 closed) without
    # converging. Skipping it routes each subbox through the serial per-sub
    # `verify_graph` loop below, where each gets α-CROWN + the full
    # remaining budget and closes (sub0 ~Xs, sub1 the rest, ≤ total).
    _serial_disj = bool(getattr(settings, 'input_split_serial_disjuncts', False))
    if (gg_fast is not None and len(remaining_sub_idx) > 0 and not _serial_disj
            and total - (time.perf_counter() - t_start) > 1.0):
        # Partition remaining subs by their per-q w-tuple so each
        # group has shared w (multi-sub BAB requirement). mscn dual
        # cases have 2 groups (Y ≥ thr vs Y ≤ thr).
        from .spec import VNNSpec
        groups_by_w = {}  # tuple(w_q_tuples) → list of sub_idx
        for ki in remaining_sub_idx:
            _, _, conjs = groups[order[ki]]
            sp = VNNSpec(groups[order[ki]][0], groups[order[ki]][1], conjs)
            qs = sp.as_linear_queries(n_out)
            # Key: tuple of (w_tuple) per query.
            key = tuple(tuple(np.asarray(q[1]).tolist()) for q in qs)
            groups_by_w.setdefault(key, []).append(ki)
        total_msb_closed = 0
        # Mini-group multi-sub BAB: large w-groups (>60 subs) are split
        # into mini-groups of ~60 subs each. FIFO leaf processing in a
        # 300+ sub group means stubborn subs only get ~10 leaves
        # processed before time runs out — not enough to close subs
        # that need 40+ leaves. Mini-groups give each sub the focused
        # processing it needs.
        # Mini-group only when w-group is small enough that focused
        # processing helps. For huge w-groups (1000+), process as a
        # single multi-sub BAB call with the full budget — batched
        # LiRPA is throughput-bound and mini-groups waste time.
        # mini_group_size resolution order:
        #   1. env MINI_GROUP_SIZE (for ad-hoc A/B testing)
        #   2. settings.mini_group_size (auto-routed by `_adapt_per_disjunct`
        #      in config_profiles.py — uses n_disjuncts to pick 60/120/200)
        #   3. hard default 60 (matches legacy behavior)
        # Per-disjunct routing was added after observing that a flat
        # default=120 regressed pensieve_big_parallel (small n_disjuncts)
        # by 10 cases while only winning +2 mscn_2048d_dual cases.
        import os as _os_mg
        mini_group_size = int(_os_mg.environ.get(
            'MINI_GROUP_SIZE',
            str(int(getattr(settings, 'mini_group_size', 60)))))
        if getattr(settings, 'print_progress', False):
            print(f'[per-disjunct] entering multi-sub BAB on '
                  f'{len(groups_by_w)} w-groups '
                  f'(sizes: {[len(g) for g in groups_by_w.values()][:5]})',
                  flush=True)
        for w_key, sub_group in groups_by_w.items():
            # Subdivide large-but-not-huge w-group into mini-groups.
            # Mini-grouping helps BAB throughput (smaller batches fit GPU
            # cache better, ~26× more iters/s on mscn) — it's not purely
            # a memory bound. Tried single-call per w-group: regressed
            # 915→772 closed on cardinality_1_960 because per-iter cost
            # of huge batches dwarfs any iteration-allocation gain.
            if len(sub_group) <= mini_group_size:
                mini_groups = [sub_group]
            elif len(sub_group) > 10000:
                # Memory bound: at huge sub counts, mini-grouping bookkeeping
                # itself overflows GPU memory. Old threshold was 1000 which
                # is too aggressive — it caused cardinality_1_2260's group-1
                # (1930 subs) to fall into a single 57s call that finished
                # only 11 BAB iters. Mini-grouping to 120 gives ~17 calls
                # with clip on each, which closes far more subs per second.
                mini_groups = [sub_group]
            else:
                mini_groups = [sub_group[i:i+mini_group_size]
                                for i in range(0, len(sub_group),
                                                mini_group_size)]
            for mg in mini_groups:
                rem_for_msb = total - (time.perf_counter() - t_start) - 1.0
                if rem_for_msb <= 0:
                    break
                # Per mini-group budget = (rem time) / (remaining mini-groups)
                mg_rem = (len(mini_groups)
                          - mini_groups.index(mg))
                # Cap by remaining outer budget too — without this cap, the
                # 2s floor lets multi-sub BAB run past `total_timeout` when
                # only a few seconds remain (mscn_2048d_dual >_5000 cases
                # would overshoot by 30s and get SIGKILL'd by the sweep
                # harness at 90s, producing NO_FILE rows).
                mg_budget = min(rem_for_msb,
                                 max(2.0, rem_for_msb / max(1, mg_rem)))
                msb_closed = _multi_sub_input_split_bab(
                    mg, groups, order, gg_fast, dev, dt,
                    mg_budget, n_out, settings)
                for ki in msb_closed:
                    fast_results[ki] = 'verified'
                total_msb_closed += len(msb_closed)
        if (total_msb_closed > 0
                and getattr(settings, 'print_progress', False)):
            print(f'[per-disjunct] multi-sub BAB total closed '
                  f'{total_msb_closed}/{len(remaining_sub_idx)} subs '
                  f'across {len(groups_by_w)} w-groups',
                  flush=True)
        remaining_sub_idx = [ki for ki in range(n_sub) if ki not in fast_results]
    rem_budget = total - (time.perf_counter() - t_start)
    per_sub_budget = max(0.2, rem_budget / max(1, len(remaining_sub_idx)))

    # Track verified status per sub-box.
    verified_subs = set(fast_results)
    sub_details = [{'idx': ki, 'verdict': 'verified', 'phase': 'fast_crown'}
                    for ki in fast_results]
    for sub_pos, ki in enumerate(remaining_sub_idx):
        elapsed = time.perf_counter() - t_start
        if elapsed > total:
            sub_details.append({'idx': ki, 'verdict': 'unknown',
                                'reason': 'total_timeout'})
            continue
        # Dynamic per-sub budget: recompute based on remaining time
        # and remaining subs. Lets fast-closing subs free their unused
        # budget for harder subs later in the list.
        n_left = len(remaining_sub_idx) - sub_pos
        rem_now = total - elapsed
        # min 3s/sub — recursive verify_graph has setup overhead
        # (forward zono + per-q + fast batched ~1s) before BAB starts;
        # the BAB itself needs 1-2s on mscn 128d_dual subs. 3s lets
        # most close.
        # Serial-disjunct mode: give each sub-box the FULL remaining budget
        # (greedy, one at a time) instead of an upfront rem/n_left fraction.
        # acasxu prop_6 has 2 input sub-boxes that each need >half the 116s
        # cap on a slow GPU; the fractional split (58s each) times BOTH out,
        # but serial-with-full-remaining lets sub0 use what it needs and sub1
        # take the rest — total still bounded by `total`, so the wall cap
        # holds. Off by default: the rem/n_left split is better when there
        # are MANY subs (mscn's thousands), where one slow sub must not
        # starve the rest. See `input_split_serial_disjuncts`.
        if bool(getattr(settings, 'input_split_serial_disjuncts', False)):
            per_sub_budget_dyn = max(3.0, rem_now)
        else:
            per_sub_budget_dyn = max(3.0, rem_now / max(1, n_left))
        key = order[ki]
        x_lo, x_hi, conjs = groups[key]
        sub_spec = VNNSpec(x_lo, x_hi, conjs)
        sub_settings = copy.copy(settings)
        sub_settings.total_timeout = min(per_sub_budget_dyn, max(0.05,
                                                              total - elapsed))
        sub_settings.print_progress = False
        try:
            sub_v, sub_d = verify_graph(graph, sub_spec, sub_settings)
        except (RuntimeError, NotImplementedError, KeyError,
                ValueError, torch.cuda.OutOfMemoryError) as e:
            # Per-sub recursive verify failures: torch runtime/OOM, missing
            # op support, or shape/key bugs in a single sub. Record as
            # err:<Type> verdict so the sub fails open (counted as unknown
            # in the verifier's aggregate), and continue to next sub.
            sub_v = f'err:{type(e).__name__}'
            sub_d = {'reason': str(e)[:200]}
        if getattr(settings, 'print_progress', False):
            print(f'[per-disjunct] sub {ki}: {sub_v} '
                  f'(phase={sub_d.get("phase")}, '
                  f'reason={sub_d.get("reason", "")[:80]})',
                  flush=True)
        sub_details.append({'idx': ki, 'verdict': sub_v,
                            'phase': sub_d.get('phase')})
        if sub_v == 'sat':
            return 'sat', {
                'phase': 'per_disjunct_sat',
                'sub_idx': ki,
                'witness': sub_d.get('witness'),
                'sub_details': sub_details,
            }
        if sub_v == 'verified':
            verified_subs.add(ki)

    all_verified = (len(verified_subs) == n_sub)
    verdict = 'verified' if all_verified else 'unknown'
    return verdict, {
        'phase': 'per_disjunct_split',
        'n_subboxes': n_sub,
        'n_fast_closed': len(fast_results),
        'sub_details': sub_details,
        'wall': time.perf_counter() - t_start,
    }


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
    _vg_t_start = time.perf_counter()   # for the top-level clear-CE upgrade budget
    # Pre-pass: if conjuncts carry per-disjunct X subboxes (e.g., nn4sys
    # lindex, acasxu prop_6), split into per-subbox sub-verifications.
    # The vnnlib spec encodes the unsafe region as
    #   UNION_i [x in subbox_i AND y in unsafe_Y_i].
    # Verifying the global bounding box against the OR of all Y-disjuncts
    # is unsound (a witness x outside subbox_i but in the bounding box
    # could still falsely SAT a disjunct's Y constraint). The right
    # decomposition: one sub-verification per unique X subbox, with the
    # bound input box = that subbox and disjuncts = the Y-only conjuncts
    # whose subbox matches. Overall SAT iff any sub-SAT; verified iff all
    # verified; unknown otherwise.
    _x_constrained = [c for c in spec.disjuncts if c.input_lo is not None]
    if _x_constrained and len(_x_constrained) == len(spec.disjuncts):
        return _verify_per_disjunct_subboxes(graph, spec, settings)
    # Force gen-LP / Phase 8 MILP off for graphs containing ops the
    # generator-LP state builder doesn't support (Pow, mul_bilinear,
    # div_bilinear, ReduceSum on the activation path). Otherwise
    # `precompute_gen_state` raises NotImplementedError mid-pipeline.
    # Sound: the CROWN backward + α-CROWN path still verifies these
    # cases (pensieve_*_parallel, mscn_*). Settings the user explicitly
    # set are preserved unless they collide with this routing.
    _unsupported_ops = {'pow', 'mul_bilinear', 'div_bilinear',
                         'reduce_sum'}
    _has_unsupported = any(
        getattr(n, 'op_type', '') in (
            'Pow', 'ReduceSum', 'Div') or
        (getattr(n, 'op_type', '') == 'Mul'
            and len([i for i in n.inputs
                      if i in graph.nodes
                      or i == graph.input_name]) == 2)
        for n in graph.nodes.values())
    if _has_unsupported:
        # Fast root verify: the SOUND forward zonotope box may already refute
        # every disjunct (no BaB needed). Run it in float64 with affine-band
        # sigmoid (sound + far tighter than the box-collapse; the only graphs
        # here with a sigmoid are the ACOPF physics nets, whose single sigmoid
        # layer doesn't blow up the generator count). float64 because these
        # margins can be ~1e-6 — float32 noise would lose them. A SOUND
        # over-approximation refuting all disjuncts means no counterexample
        # exists, so this can only return 'verified' on genuinely-unsat cases.
        # Gated to the ACOPF-physics signature: Sin/Cos OR a genuine both-vary
        # Mul (mul_bilinear). This catches every ml4acopf model (full has
        # Sin/Cos + bilinear Mul; the linear-residual/nonresidual variants drop
        # the trig but keep the bilinear Mul) while EXCLUDING the other non-LP
        # graphs that share this block: mscn (point-mask Mul + Sigmoid, no trig
        # and no both-vary Mul) and pensieve (sub_bilinear, no Mul-bilinear) —
        # they keep their tuned, timing-pinned CROWN/BaB path unchanged. Using
        # sigmoid presence here would wrongly grab mscn.
        _has_trig = any(getattr(n, 'op_type', '') in ('Sin', 'Cos')
                        for n in graph.nodes.values())
        _has_mul_bilinear = any(
            getattr(n, 'op_type', '') == 'Mul'
            and len([i for i in n.inputs
                     if i in graph.nodes or i == graph.input_name]) == 2
            for n in graph.nodes.values())
        # Non-LP nonlinearity (trig OR a genuine both-vary bilinear Mul) routes to
        # the dedicated nonlinear bound pipeline (root-zono -> backward-CROWN-root
        # refine -> alpha-CROWN -> nonlinear-split BaB), the path built for the
        # ml4acopf physics nets. DEFER, though, to a config that explicitly asked
        # for the batched input-split BaB (`input_split_batched_enabled`): such a
        # net has DECLARED its verification strategy, so this heuristic auto-router
        # must not hijack it. Without the gate, lsnc_relu's Lyapunov quadratic
        # forms (V=u^T P u -> both-vary Muls) were mis-routed here and the
        # throughput BaB the config requested got force-disabled -> timeouts.
        _route_nonlinear = (_has_trig or _has_mul_bilinear) and not bool(
            getattr(settings, 'input_split_batched_enabled', False))
        if (_route_nonlinear and hasattr(spec, 'x_lo')
                and getattr(spec, 'x_lo', None) is not None):
            import torch as _t
            _prev_sig = (settings.get('sigmoid_relaxation', 'box')
                         if settings is not None else 'box')
            if settings is not None:
                settings.sigmoid_relaxation = 'affine_band'
            _bcdev = _resolve_device(settings)
            _gg64 = graph.gpu_graph(device=_bcdev, dtype=_t.float64)
            _xl = _t.tensor(np.asarray(spec.x_lo).flatten(), dtype=_t.float64,
                            device=_bcdev)
            _xh = _t.tensor(np.asarray(spec.x_hi).flatten(), dtype=_t.float64,
                            device=_bcdev)
            # Cheap nominal-point SAT probe (ORT-confirmed) before the bound
            # work: closes specs violated at the operating point itself (e.g.
            # prop1, unsafe across the whole box) that the internal PGD misses.
            _np = _nonlinear_nominal_cex_probe(graph, spec, settings, _xl, _xh)
            if _np is not None:
                if settings is not None:
                    settings.sigmoid_relaxation = _prev_sig
                return _np
            _ob_fwd = {}
            _sb_fwd, _zf = _forward_zonotope_graph(
                _xl, _xh, _gg64, _bcdev, _t.float64, settings=settings,
                op_bounds=_ob_fwd)
            if settings is not None:
                settings.sigmoid_relaxation = _prev_sig
            _olo, _ohi = _zf.bounds()
            _rv, _rd = spec.check(_olo.cpu().numpy(), _ohi.cpu().numpy())
            if _rv == 'verified':
                if getattr(settings, 'print_progress', False):
                    print('[verify_graph] root zono box (affine-band sigmoid) '
                          f'verified all disjuncts; worst_margin='
                          f'{_rd.get("worst_margin", 0.0):.3e}', flush=True)
                return 'verified', {**_rd, 'method': 'root_zono_box'}
            # Root box didn't close. Try α-CROWN first: gradient-optimize the
            # nonlinear ops' relaxation slopes (Sqr/Sigmoid/Sin/Cos) to tighten
            # the root bound (ABC's init-α-CROWN analog; sound for any α). Then
            # fall back to the dedicated trig verifier (input/nonlinear split).
            import time as _time_tg
            _t0 = _time_tg.perf_counter()
            _tot = float(getattr(settings, 'total_timeout', 60.0))
            _deadline = _t0 + _tot
            # Backward-CROWN root with topo-order intermediate-bound refinement
            # (ABC's tight-intermediate init-CROWN analog). Closes the ml4acopf
            # linear nets the forward box + α-CROWN miss (e.g. 14_ieee-linear
            # prop2 -> +3.98, beating ABC). Sound; on 'unknown' we fall through.
            # Budget-capped so big nets that won't close here leave the trig BaB
            # its time. GATED on `not _has_trig`: _spec_backward_graph (the
            # refinement's engine) has no Sin/Cos backward handler, and the FULL
            # physics nets that use real Sin/Cos are both-timeout anyway (the
            # trig BaB below handles them). The linear-residual/nonresidual
            # variants bake the trig as ReLU-PWL — no Sin/Cos op — so they run.
            # Gated on `not _has_trig` AND a net-size cap: the per-node
            # refinement (2n sequential backward passes over the WHOLE graph)
            # is grossly insufficient on the big nets (118: -20k; 300: worse)
            # AND so costly that one node blows the deadline — those nets'
            # easy props are closed by the root-box / α-CROWN instead. Only the
            # small 14_ieee linear nets (n_out~186) benefit, so cap on n_out.
            _bc_nout = int(_zf.center.numel())
            _bc_max_out = int(getattr(settings, 'nonlinear_bwd_crown_max_out', 600))
            if (not _has_trig and _bc_nout <= _bc_max_out
                    and bool(getattr(settings, 'nonlinear_backward_crown', True))):
                _bc_budget = min(_tot * 0.5, float(getattr(
                    settings, 'nonlinear_bwd_crown_budget', 60.0)))
                _bv, _bd = _nonlinear_backward_crown_root(
                    _gg64, spec, _sb_fwd, _ob_fwd, _xl, _xh, _zf,
                    settings, _bcdev, _t.float64,
                    deadline=_time_tg.perf_counter() + _bc_budget)
                if getattr(settings, 'print_progress', False):
                    print('[verify_graph] nonlinear backward-CROWN root: '
                          f'{_bv} worst_margin={_bd.get("worst_margin", 0.0):.3e} '
                          f'({_bd.get("verified_disj", "?")}/'
                          f'{_bd.get("n_disj", "?")} disj)', flush=True)
                if _bv == 'verified':
                    return 'verified', _bd
            if bool(getattr(settings, 'nonlinear_alpha_crown', True)):
                # α-CROWN gets the bulk of the budget (it's the strongest
                # lever for these bilinear-dominated specs); the BaB mops up.
                _abudget = min(_tot * 0.8, 150.0)
                _av, _ad = _nonlinear_alpha_opt(
                    graph, spec, settings, _t0, _abudget)
                if _av == 'verified':
                    return 'verified', _ad
            _rem = max(1.0, _deadline - _time_tg.perf_counter())
            return _verify_nonlinear_graph(
                graph, spec, settings, _time_tg.perf_counter(), _rem)
        if getattr(settings, 'tighten_formulation', 'gen_cone') != 'skip':
            # Count VARYING input dims (these specs typically have a handful
            # varying out of 100s of total dims). If that count is small,
            # lift `input_split_max_dims` to the TOTAL input dim so the
            # input-split fast-leaf path triggers (the gate compares TOTAL
            # `n_in = np.prod(x_lo.shape)` against `input_split_max_dims`,
            # not varying count) — it closes specs where the no-split
            # α-CROWN plateaus just below 0.
            n_var = int(((spec.x_hi - spec.x_lo) > 1e-9).sum())
            n_in_total = int(spec.x_lo.size)
            # Lift the split cap when input-split BAB is tractable.
            # Two regimes by varying-dim count:
            #  - n_var ≤ 8: few varying dims, BAB splits along those few
            #    dims, exponentially small tree depth.
            #  - n_var > 8 AND n_in_total ≤ 100: many varying dims out of a
            #    small total (parallel-branch predictors). The structural
            #    tightness fixes (α-Pow + α-Div tangent in Phase 0.5,
            #    shared-gen Div for scalar denominator) make per-leaf bounds
            #    tight enough that BAB converges within the timeout.
            if n_var <= 8 or n_in_total <= 100:
                new_split_cap = max(int(getattr(
                    settings, 'input_split_max_dims', 20)), n_in_total)
            else:
                new_split_cap = int(getattr(
                    settings, 'input_split_max_dims', 20))
            if getattr(settings, 'print_progress', False):
                print('[verify_graph] graph has non-LP ops '
                      '(Pow/Div/ReduceSum/MulBilinear); forcing '
                      'tighten_formulation=skip + skip_phase8_milp=True '
                      '+ zono_lift_enabled=False + phase2_crown_enabled=False '
                      f'+ input_split_max_dims={new_split_cap} '
                      f'(was {getattr(settings, "input_split_max_dims", 20)}; '
                      f'varying dim count={n_var})', flush=True)
            settings.tighten_formulation = 'skip'
            settings.tighten_solver = 'lp'  # placeholder; skip path ignores
            settings.skip_phase8_milp = True
            settings.zono_lift_enabled = False
            # Enable batched input-split + K-dim split + branch-aware SB for
            # these non-LP graphs. Batched = GPU-parallel leaves at ~100×
            # throughput vs non-batched. For single-disjunct graphs with many
            # varying dims and a parallel-branch (bilinear) architecture, the
            # batched split with branch-boost SB closes cases the old
            # non-batched single-dim path timed out on. The branch boost
            # compensates for backward CROWN underestimating lA at dims
            # feeding pow/div_bilinear branches, so SB correctly picks the
            # most-unstable branch.
            settings.input_split_batched_enabled = True
            settings.input_split_batched_branch_sb = True
            settings.input_split_batched_branch_boost = True
            # The BATCHED zono forward has no Sin/Cos/Floor handlers and lacks
            # the col-ID merge fix; ml4acopf physics nets must use the SOUND
            # unbatched input-split (whose _forward_zonotope_graph handles trig
            # + carries the col-ID/concat soundness fix). Only trig graphs hit
            # this; pensieve/mscn/lsnc keep the fast batched path.
            _has_trig = any(
                getattr(n, 'op_type', '') in ('Sin', 'Cos', 'Floor')
                for n in graph.nodes.values())
            if _has_trig:
                settings.input_split_batched_enabled = False
            import os as _os_ksd
            # The two input-split-branching knobs below are chosen from a
            # graph/spec property, not a benchmark name: the count of VARYING
            # input dims `n_var`. This block already requires bilinear/
            # nonlinear LAYER TYPES (Pow/Div/Mul — `_has_unsupported`), which
            # route to the batched input-split BaB. With many varying dims the
            # BaB has real split-dim *choice*, so arity and dim-selection
            # matter; with few, they're inert.
            _many_varying_dims = n_var > 8
            # Split arity. K=2 (4-way) is the sweet spot when n_var is large:
            # K=4 (16-way) over-splits high-dim boxes (measured 56k leaves/39s
            # at K=4 → 9.8k/11s at K=2, 5.7× fewer leaves). When n_var ≤ 8 the
            # arity is irrelevant to the LEAF COUNT (K=1 == K=2 == K=4, measured)
            # — so use K=1, which takes the VECTORIZED GPU 2-way split. K>1 goes
            # through a per-leaf Python loop that is catastrophic on the huge
            # queues these large-batch bilinear cases build (lsnc q2: K=4 33s vs
            # K=1 6.4s, same verdict). The old "K=1 too slow" note predates the
            # chunk worklist + vectorized split that made K=1 the fast path.
            _default_K = 2 if _many_varying_dims else 1
            settings.bab_split_depth = int(
                _os_ksd.environ.get('BAB_SPLIT_DEPTH', str(_default_K)))
            # Split-dim boost EXPONENT. The boost
            # = (1 + n_unstable_in_dominant_shallow_ReLU_layer)^exp sharpens
            # dim selection toward dims feeding the most-unstable shallow
            # branch. exp=2.0 (with K=2) cuts leaves 10–198× when n_var is
            # large AND the net has heavy SHALLOW ReLU instability. It is
            # SELF-LIMITING by topology: a graph with no shallow unstable
            # ReLUs gets boost ≈ 1 regardless of exp, so a higher exp is
            # harmless where it doesn't fit. exp only changes WHICH dims split
            # (efficiency) — never the bounds or the verdict. Linear (exp=1)
            # under-weights the unstable-branch dims; kept for n_var ≤ 8 where
            # there is no dim choice to sharpen.
            settings.input_split_batched_branch_boost_exp = (
                2.0 if _many_varying_dims else 1.0)
            # Critical: without this, Phase 2's basic CROWN OVERWRITES
            # Phase 0.5's α-CROWN spec_lb with a looser value (observed on
            # small cardinality-estimation specs: α-CROWN closes the spec at
            # lb≈+0.001, Phase 2 CROWN reports lb≈-0.003, flipping the verdict
            # verified→unknown). With tighten_formulation=skip there's no MILP
            # tightening between Phase 0.5 and Phase 2, so Phase 0.5's bound
            # is strictly tighter.
            settings.phase2_crown_enabled = False
            settings.input_split_max_dims = new_split_cap
            # NOTE: α-CROWN-on-boundary (input_split_batched_alpha_iters) is
            # left OFF here — measured ineffective on these graphs (open
            # leaves plateau at spec_lb≈-0.3, far past the ~-0.02 boundary
            # where slope-only α flips a leaf; 0 closures). The leaf-count gap
            # was a branching, not a bounding, problem — closed by the arity
            # and boost-exponent settings above.
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
    # CROWN-IBP route: inputs at/above the IBP threshold go to
    # milp_verify's graph path unconditionally — their zonotope
    # generator tensors don't fit GPU memory in ANY pipeline, and the
    # IBP Phase 1 lives there (challenging_certified_training
    # tinyimagenet). Takes precedence over auto_route_milp_for_conv so
    # a benchmark can route small inputs to the graph pipeline
    # (dual-ascent phase 8) and big ones to CROWN-IBP with one config.
    _ibp_thr = int(getattr(settings, 'phase1_ibp_input_dim_threshold', 0))
    if _ibp_thr > 0 and n_in >= _ibp_thr:
        from .verify_milp import milp_verify
        if getattr(settings, 'print_progress', False):
            print('[verify_graph] routing large-input net to milp_verify '
                  f'CROWN-IBP path (n_in={n_in} >= {_ibp_thr})', flush=True)
        return milp_verify(graph, spec, settings)
    _split_max = int(getattr(settings, 'input_split_max_dims', 20))
    if (bool(getattr(settings, 'auto_route_milp_for_conv', True))
            and bool(getattr(settings, 'input_split_enabled', True))
            and n_in > _split_max):
        has_conv = any(getattr(n, 'op_type', '') == 'Conv'
                       for n in graph.nodes.values())
        if has_conv and not graph.fork_points():
            from .verify_milp import milp_verify
            # Structural difficulty gate (settings.milp_route_pert_threshold):
            # route HIGH input-uncertainty instances (large mean input box
            # width) to milp_verify's IBP + α-CROWN + ReLU-split BaB, and
            # leave LOW-uncertainty ones on the graph pipeline's zono +
            # dual-ascent BaB, which is tighter per-direction on small
            # perturbations. Larger input uncertainty loosens the forward
            # zonotope enough that the CROWN triangle relaxation + per-domain
            # β-CROWN BaB wins; smaller keeps the zono state tight. The
            # threshold is the only knob and scales to any benchmark whose
            # difficulty tracks input-box width (default None = off → all
            # conv nets to milp_verify, the historical behavior).
            _pert_thr = getattr(settings, 'milp_route_pert_threshold', None)
            _vb = getattr(settings, 'print_progress', False)
            if _pert_thr is None or not hasattr(spec, 'x_lo'):
                # gate off → all conv nets to milp_verify (historical).
                if _vb:
                    print('[verify_graph] auto-routing conv net to '
                          'milp_verify', flush=True)
                return milp_verify(graph, spec, settings)
            _pw = float(np.mean(spec.x_hi - spec.x_lo))
            if _pw > float(_pert_thr):
                # HIGH uncertainty (large eps) → milp_verify's graph path:
                # IBP + α-CROWN + no-reforward β-CROWN BaB. Closes the hard
                # tight-root eps8 queries (CROWN's triangle beats the zono).
                settings.milp_force_graph_path = True
                settings.milp_force_ibp_phase1 = True
                if _vb:
                    print(f'[verify_graph] pert-gate: {_pw:.4f} > '
                          f'{_pert_thr} -> milp_verify IBP+BaB', flush=True)
                return milp_verify(graph, spec, settings)
            # LOW uncertainty. Budget sub-gate: the benchmark assigns longer
            # budgets to harder instances, so a long-budget low-eps instance
            # is the LOOSE-ROOT cluster the per-direction-tight β-CROWN BaB
            # can't close fast — send it to the graph pipeline's zono +
            # phase-8 dual-ascent BaB (~3 µs/node brute-forces the looser
            # frontier; closes eps2 s2/s3 in 36–87 s). Short-budget (easy)
            # low-eps instances stay on the fast layers-path zono+CROWN
            # Phase 1 (verifies them in seconds; the dual-ascent's heavier
            # Phase 1+2 would blow their 30 s budget).
            _min_b = getattr(settings, 'milp_route_dualascent_min_budget', None)
            _budget = float(getattr(settings, 'total_timeout', 0.0))
            if _min_b is not None and _budget >= float(_min_b):
                if _vb:
                    print(f'[verify_graph] pert-gate: {_pw:.4f} <= '
                          f'{_pert_thr}, budget {_budget:.0f} >= {_min_b} '
                          f'-> graph pipeline (dual-ascent)', flush=True)
                # fall through to the graph pipeline below
            else:
                settings.milp_force_graph_path = False
                settings.milp_force_ibp_phase1 = False
                if _vb:
                    print(f'[verify_graph] pert-gate: {_pw:.4f} <= '
                          f'{_pert_thr}, budget {_budget:.0f} -> '
                          f'milp_verify zono layers', flush=True)
                return milp_verify(graph, spec, settings)

    # Auto-route SMALL FC nets (few ReLU layers, no conv, no bilinear) to
    # milp_verify, where the exact per-neuron MILP both attacks (PGD) and
    # verifies (= AB-CROWN's `complete_verifier: mip`). The graph pipeline's
    # zono/CROWN relaxation is far too loose on these (e.g. safenlp_2024:
    # 30->128(ReLU)->2, worst spec_lb ~-10 on a 1-ReLU net) and input-split
    # explodes on the 30-dim box; the exact MILP is sub-second. Gated on the
    # net being structurally MILP-tractable (≤2 ReLU layers) and on
    # input_dim > the input-split cap (so tiny-input FC nets that the
    # input-split fast-leaf handles well are left alone).
    _milp_unsupported_acts = {'Sigmoid', 'Tanh', 'Softmax', 'Exp', 'Erf',
                              'Gelu', 'Elu', 'LeakyRelu', 'Pow', 'Div',
                              'MatMulBilinear', 'ReduceSum', 'Conv'}
    _is_pure_relu_fc = not any(
        getattr(n, 'op_type', '') in _milp_unsupported_acts
        for n in graph.nodes.values())
    if (bool(getattr(settings, 'auto_route_milp_for_small_fc', True))
            and not _has_unsupported
            and _is_pure_relu_fc           # milp_verify encodes only ReLU
            and n_in > _split_max
            and len(graph.relu_nodes()) <= 2
            and not graph.fork_points()):
        from .verify_milp import milp_verify
        if getattr(settings, 'print_progress', False):
            print('[verify_graph] auto-routing small FC net to milp_verify '
                  '(exact MILP; graph relaxation too loose)', flush=True)
        return milp_verify(graph, spec, settings)

    impl = str(getattr(settings, 'graph_impl', 'optimized'))
    assert impl in _BUILDERS, f'unknown graph_impl: {impl!r}'
    build_fn = _BUILDERS[impl]
    n_in = int(np.prod(spec.x_lo.shape)) if hasattr(spec, 'x_lo') else 10**9
    if (getattr(settings, 'input_split_enabled', True)
            and n_in <= int(getattr(settings, 'input_split_max_dims', 20))):
        import os as _os_isd
        if _os_isd.environ.get('DEBUG_INPUT_SPLIT_PATH', '') == '1':
            print(f'[is-path] enabled, n_in={n_in} max_dims='
                  f'{getattr(settings, "input_split_max_dims", 20)} '
                  f'batched={bool(getattr(settings, "input_split_batched_enabled", False))}',
                  flush=True)
        if bool(getattr(settings, 'use_hybrid_acasxu', False)):
            # Freeze-replay α-CROWN with TIGHTENED intermediate bounds. The
            # forward-zono intermediate bounds the batched input-split BaB uses
            # are ~1000x too loose for ACAS Xu's amplifying weights (root spec
            # margin -1597 vs true >0) -> it diverges (6.8M leaves, never
            # converges). verify_hybrid tightens per-layer pre-ReLU bounds via
            # backward α-CROWN (intersected with forward-zono, so still SOUND)
            # and converges. Adapt its {verdict} dict to (result, details), and
            # onnxruntime-validate any sat witness (no false sat).
            from .verify_hybrid_acasxu import verify_hybrid
            _hres = verify_hybrid(
                graph, spec, settings,
                timeout=float(getattr(settings, 'total_timeout', 120.0)))
            _hv = _hres.get('verdict', 'unknown')
            if _hv == 'sat':
                _onnx_p = getattr(graph, 'onnx_path', None)
                _w = _hres.get('witness')
                if _onnx_p is not None and _w is not None:
                    _ok, _info = _validate_sat_witness(
                        _onnx_p, spec, np.asarray(_w).flatten(),
                        emit_slack=pgd_box_expand_amount(settings))
                    if not _ok:
                        _hres['spurious_witness'] = _info
                        return 'unknown', _hres
                return 'sat', _hres
            return ('verified' if _hv == 'unsat' else 'unknown'), _hres
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
                    except (RuntimeError, ImportError, OSError, ValueError) as _pe:
                        # PGD via raw ONNX: RuntimeError from torch / onnxruntime,
                        # ImportError if optional deps missing, OSError on bad
                        # path, ValueError on shape mismatch. All non-fatal —
                        # we report unknown rather than crash.
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
                    except (RuntimeError, ImportError, OSError, ValueError) as _pe:
                        # PGD via raw ONNX: RuntimeError from torch / onnxruntime,
                        # ImportError if optional deps missing, OSError on bad
                        # path, ValueError on shape mismatch. All non-fatal —
                        # we report unknown rather than crash.
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
            atol=_atol, emit_slack=pgd_box_expand_amount(settings))
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
        elif (float(getattr(settings, 'clear_ce_upgrade_budget', 0.0)) > 0.0
                and not bool(getattr(settings, 'disable_sat_finding', False))
                and _info.get('worst_margin') is not None
                and float(_info['worst_margin']) > -_atol):
            # The witness is a valid but NEAR-BOUNDARY closure counterexample
            # (worst output margin ~0 — e.g. a network-pair's trivial diagonal,
            # x_f == x_g so the diff is exactly 0). It scores CORRECT, but isn't a
            # strict violation. Try a bounded margin-minimizing PGD for a CLEAR CE
            # (margin < -atol) and emit that instead when found; otherwise the
            # already-valid boundary witness stands (no sat is ever lost).
            _clear = _try_clear_ce_upgrade(graph, spec, settings, _vg_t_start)
            if _clear is not None:
                details['witness'] = _clear
                details['phase'] = f'{details.get("phase")}+clear_ce_upgrade'
    return result, details


def _try_clear_ce_upgrade(graph, spec, settings, t_start):
    """A sat witness ORT-validated but is only a NEAR-BOUNDARY closure CE (worst
    output margin > -sat_validate_atol). Run a bounded margin-minimizing PGD on
    the raw ONNX for a CLEAR counterexample (worst margin < -atol) so the emitted
    sat is a genuine strict violation rather than a boundary point — the
    network-pair benchmarks (monotonic/isomorphic acasxu) otherwise settle on a
    trivial diagonal (x_f == x_g -> output diff exactly 0) even though clear
    violations exist elsewhere in the box.

    Returns a clear witness (np.ndarray, spec.x_lo shape) or None. Bounded by
    `settings.clear_ce_upgrade_budget` and the remaining total-timeout budget;
    every returned witness is ORT-confirmed to violate with margin < -atol."""
    onnx_p = getattr(graph, 'onnx_path', None)
    if onnx_p is None:
        return None
    atol = float(settings.sat_validate_atol
                 if 'sat_validate_atol' in settings else 1e-4)
    cap = float(getattr(settings, 'clear_ce_upgrade_budget', 0.0))
    if cap <= 0.0:
        return None
    tot = float(getattr(settings, 'total_timeout', 0.0)) or cap
    # Leave ~1 s of headroom against the global deadline so the upgrade can't
    # push the instance into a hard timeout.
    budget = min(cap, tot - (time.perf_counter() - t_start) - 1.0)
    if budget <= 0.1:
        return None
    from .onnx_torch_runner import pgd_via_onnx
    try:
        _sat, _w = pgd_via_onnx(
            onnx_p, spec,
            n_restarts=int(settings.pgd_phase0_restarts
                           if 'pgd_phase0_restarts' in settings else 256),
            n_iter=int(settings.pgd_phase0_iters
                       if 'pgd_phase0_iters' in settings else 100),
            accept_margin=-atol,
            deadline=time.perf_counter() + budget)
    except (RuntimeError, ImportError, OSError, ValueError):
        # PGD-via-ONNX best-effort: torch/ORT runtime errors, missing optional
        # deps, bad path, or shape mismatch. The boundary witness already stands.
        return None
    if not _sat or _w is None:
        return None
    w = np.asarray(_w).flatten()
    _ok, _vinfo = _validate_sat_witness(onnx_p, spec, w, atol=atol, out_atol=0.0)
    _m = _vinfo.get('worst_margin')
    if _ok and _m is not None and float(_m) < -atol:
        return w
    return None


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


_ALPHA_OPS = ('pow', 'sigmoid', 'tanh', 'sin', 'cos', 'exp', 'reciprocal')


def _nonlinear_nominal_cex_probe(graph, spec, settings, xl, xh):
    """Cheap SAT pre-check: ACOPF specs are sometimes violated at the nominal
    operating point itself (the whole input box is unsafe — e.g. 14_ieee
    prop1, margin ~-0.006 at every corner). PGD on the internal point-forward
    can miss this (the internal forward may disagree with the reference ONNX
    near the boundary), so directly ORT-check the box center and both corners.
    Returns ('sat', details) for a CLEAR counterexample (sound — ORT-confirmed,
    matches α,β-CROWN's clean-input attack); a within-tolerance near-miss is
    saved early via the result sink and we return None (keep searching)."""
    if bool(getattr(settings, 'disable_sat_finding', False)):
        return None
    onnx_p = getattr(graph, 'onnx_path', None)
    if onnx_p is None:
        return None
    atol = float(settings.sat_validate_atol
                 if 'sat_validate_atol' in settings else 1e-4)
    for cand in (0.5 * (xl + xh), xl, xh):
        w = cand.detach().cpu().numpy().flatten()
        ok, info = _validate_sat_witness(onnx_p, spec, w, atol=atol,
                                         emit_slack=pgd_box_expand_amount(settings))
        if ok and _sat_disposition(graph, spec, settings, w, info) == 'real':
            return 'sat', {'witness': w, 'phase': 'nonlinear_nominal_probe'}
    return None


def _nonlinear_backward_crown_root(gg, spec, sb, op_bounds, xl, xh, zf,
                               settings, device, dtype, deadline):
    """Backward-CROWN root bound with topo-order intermediate-bound refinement
    — the ml4acopf analog of α,β-CROWN's tight-intermediate initial CROWN.

    Refines every nonlinear op's input-node bounds by backward-CROWN-to-input
    (intersected with the forward zonotope), then runs the graph spec backward
    over those tightened bounds. Returns ('verified', details) iff EVERY
    disjunct has a query whose lower bound (best-of refined backward CROWN and
    the forward-zono margin — both sound) is > 0; else ('unknown', details).

    Sound by construction: every bound used is a sound over-approximation, and
    the ANY-query-closes-the-disjunct rule matches `as_linear_queries`
    semantics (a disjunct is a conjunction; refuting one conjunct refutes it).
    """
    import torch as _t
    from .verify_zono_bnb import (_crown_refine_intermediate_graph,
                                  _spec_backward_graph)
    n_out = int(zf.center.numel())
    queries = spec.as_linear_queries(n_out)
    if not queries:
        return 'unknown', {'method': 'nonlinear_bwd_crown', 'reason': 'no_queries'}
    spec_ew = {qi: (_t.as_tensor(w, dtype=dtype, device=device), float(b))
               for qi, (di, w, b) in enumerate(queries)}
    disj_queries = {}
    for qi, (di, w, b) in enumerate(queries):
        disj_queries.setdefault(di, []).append((qi, w, b))
    # Refine on COPIES — the caller may fall through to α-CROWN / the trig BaB
    # which rebuild their own state, but don't rely on these being mutated.
    sb_r = dict(sb)
    ob_r = dict(op_bounds)
    _crown_refine_intermediate_graph(
        gg, xl, xh, sb_r, ob_r, device, dtype, deadline=deadline,
        print_progress=bool(getattr(settings, 'print_progress', False)))
    spec_lbs, _ = _spec_backward_graph(
        sb_r, xl, xh, gg, spec_ew, list(range(len(queries))), len(sb_r),
        device, dtype, op_bounds=ob_r)
    # best-of: the forward-zono spec margin is also sound — take the max per
    # query (mirrors the Phase-1+2 interleaved path).
    _zc = zf.center.to(dtype)
    _zG = zf.generators.to(dtype)
    for qi in range(len(queries)):
        _w, _b = spec_ew[qi]
        _fwd = float(_w @ _zc + _b - (_w @ _zG).abs().sum())
        if _fwd > spec_lbs.get(qi, -float('inf')):
            spec_lbs[qi] = _fwd
    # per-disjunct margin = how strongly its strongest-refuted query closes
    # (a disjunct is verified iff this is > 0).
    margins = {di: max(spec_lbs.get(qi, -1.0) for qi, _, _ in ql)
               for di, ql in disj_queries.items()}
    verified_disj = {di for di, m in margins.items() if m > 0}
    worst = min(margins.values()) if margins else -1.0
    det = {'method': 'nonlinear_bwd_crown', 'worst_margin': worst,
           'margins': margins, 'verified_disj': len(verified_disj),
           'n_disj': len(disj_queries)}
    if len(verified_disj) == len(disj_queries):
        return 'verified', det
    return 'unknown', det


def _nonlinear_alpha_opt(graph, spec, settings, t_start, total_timeout,
                     n_iters=None, lr=None):
    """α-CROWN (forward / α-zono) for the ACOPF trig graphs. Gradient-optimizes
    a per-element relaxation slope α∈[0,1] for every Sqr/Sigmoid/Tanh/Sin/Cos/
    exp/reciprocal op against the WORST-disjunct spec margin, all the way
    through the differentiable forward zonotope (affine-band sigmoid + α-bands;
    NO box-collapse, so gradient flows through every op). SOUND: any α gives a
    sound bound, so a positive margin at ANY iterate proves the property.
    Returns ('verified', details) on success, else ('unknown', details)."""
    import torch
    import time as _time
    deadline = t_start + total_timeout
    device = _resolve_device(settings); dt = torch.float64
    prev_sig = (settings.get('sigmoid_relaxation', 'box')
                if settings is not None else 'box')
    if settings is not None:
        settings.sigmoid_relaxation = 'affine_band'
    gg = graph.gpu_graph(device=device, dtype=dt)
    xl = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dt, device=device)
    xh = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dt, device=device)
    print_progress = bool(getattr(settings, 'print_progress', False))

    def _restore():
        if settings is not None:
            settings.sigmoid_relaxation = prev_sig

    # probe forward to size the α params (one per element of each α-op output)
    ob = {}
    _sb0, _zf0 = _forward_zonotope_graph(xl, xh, gg, device, dt,
                                         settings=settings, op_bounds=ob)
    alpha_op_names = [op['name'] for op in gg['ops']
                      if op['type'] in _ALPHA_OPS and op['name'] in ob
                      and torch.is_tensor(ob[op['name']][0])]
    alphas = {nm: torch.full((ob[nm][0].numel(),), 0.5, dtype=dt,
                             device=device, requires_grad=True)
              for nm in alpha_op_names}
    # ALSO α-optimize the ReLU lower-slopes (keyed by layer_idx — the forward
    # zono's ReLU reads relu_lambdas[layer_idx], the SAME mechanism α-CROWN
    # uses on every other ReLU benchmark). The ACOPF nets have ReLU MLP blocks
    # (residual / sin-cos approximations), and some specs are bound by a
    # ReLU-only path: 14_ieee prop3's binding lb(Y_5) is limited by ~6 unstable
    # ReLUs that α-CROWN lifts -0.0096 -> +0.0005 (unknown -> verified, matching
    # α,β-CROWN which also closes it purely via ReLU-slope α; NOT a bilinear
    # gap). Default 0.5 for unstable neurons; stable/dead ignore α.
    for op in gg['ops']:
        if op['type'] == 'relu':
            _L = op.get('layer_idx')
            if (_L is not None and _L in _sb0
                    and torch.is_tensor(_sb0[_L][0])):
                alphas[_L] = torch.full((_sb0[_L][0].numel(),), 0.5, dtype=dt,
                                        device=device, requires_grad=True)
    if not alphas:
        _restore()
        return 'unknown', {'method': 'nonlinear_alpha', 'reason': 'no_alpha_ops'}
    # group spec queries by disjunct (refuted iff ANY query margin > 0)
    n_out = int(_zf0.center.numel())
    qs = spec.as_linear_queries(n_out)
    if not qs:
        _restore()
        return 'unknown', {'method': 'nonlinear_alpha', 'reason': 'no_queries'}
    Wq = torch.as_tensor(np.stack([w for _, w, _ in qs]), dtype=dt, device=device)
    bq = torch.as_tensor(np.array([b for _, _, b in qs]), dtype=dt, device=device)
    di = [d for d, _, _ in qs]
    disj_groups = {}
    for qi, d in enumerate(di):
        disj_groups.setdefault(d, []).append(qi)
    grp_idx = [torch.as_tensor(v, device=device) for v in disj_groups.values()]

    n_iters = int(n_iters if n_iters is not None
                  else getattr(settings, 'nonlinear_alpha_iters', 400))
    lr = float(lr if lr is not None
               else getattr(settings, 'nonlinear_alpha_lr', 0.5))
    opt = torch.optim.Adam(list(alphas.values()), lr=lr)
    # decay LR as the margin approaches 0 (fine convergence near the boundary)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15, min_lr=1e-3)
    best = -1e30
    for it in range(n_iters):
        if _time.perf_counter() >= deadline:
            break
        opt.zero_grad()
        _, zf = _forward_zonotope_graph(xl, xh, gg, device, dt,
                                        settings=settings, relu_lambdas=alphas)
        c, G = zf.center, zf.generators
        qm = Wq @ c + bq - (Wq @ G).abs().sum(1)        # (Q,) per-query margin
        # disjunct margin = max over its queries; spec margin = min over disjuncts
        dmar = torch.stack([qm[idx].max() for idx in grp_idx])
        spec_margin = dmar.min()
        _smf = float(spec_margin.detach())   # scalar for logging/checks
        best = max(best, _smf)
        if _smf > 0:
            _restore()
            if print_progress:
                print(f'[nl_alpha] verified at iter {it}, '
                      f'margin={_smf:.3e}', flush=True)
            return 'verified', {'method': 'nonlinear_alpha', 'iter': it,
                                'worst_margin': _smf, 'margins': {0: _smf}}
        (-spec_margin).backward()   # spec_margin keeps grad for the α update
        opt.step()
        sched.step(_smf)
        with torch.no_grad():
            for p_ in alphas.values():
                p_.clamp_(0.0, 1.0)
    _restore()
    if print_progress:
        print(f'[nl_alpha] did not close; best margin={best:.3e}', flush=True)
    return 'unknown', {'method': 'nonlinear_alpha', 'best_margin': best}


def _resolve_device(settings):
    """Device string ('cuda'/'cpu') from `settings.device` via the generic
    `resolve_torch` helper — follows the configured device (default 'gpu'),
    falling back to CPU without CUDA. Any benchmark can pin CPU via its config
    (`device: cpu`, e.g. ml4acopf's float64 physics + Gurobi LP); GPU otherwise."""
    import torch
    if settings is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    from .settings import resolve_torch
    return str(resolve_torch(settings)[0])


def _verify_nonlinear_graph(graph, spec, settings, t_start, total_timeout):
    """Self-contained SOUND verifier for graphs whose nonlinearity is trig /
    AC-power-flow physics (ml4acopf: Sin/Cos/Pow/Sigmoid + element-wise
    bilinear Mul). The 9-phase `_run_pipeline` and the batched input-split do
    NOT support these end-to-end (no Sin/Cos handlers in the batched forward,
    gg serializer chokes on Pow, etc.), so we run a dedicated loop built only
    on pieces that DO handle them soundly:

      * UNSAT: input-split branch-and-bound. Each leaf (input sub-box) is
        bounded by the sound forward zonotope (`_forward_zonotope_graph`,
        float64, affine-band sigmoid) and checked with `spec.check`. A leaf is
        closed when the box refutes every disjunct; we split the
        highest-spec-sensitivity varying dim of an open leaf and recurse.
        Verified iff EVERY leaf closes (sound: each leaf bound is a sound
        over-approximation of its sub-box, and the leaves cover the input box).
      * SAT: `pgd_attack_general` on the full box (returns only a CONFIRMED
        counterexample), tried first.

    Returns (verdict, details). Falls back to 'unknown' (never a false verdict)
    when the box is too high-dimensional to split within the budget.
    """
    import torch
    import time as _time
    deadline = t_start + total_timeout
    device = _resolve_device(settings)
    dt = torch.float64
    prev_sig = (settings.get('sigmoid_relaxation', 'box')
                if settings is not None else 'box')
    if settings is not None:
        settings.sigmoid_relaxation = 'affine_band'
    gg = graph.gpu_graph(device=device, dtype=dt)
    xl0 = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dt, device=device)
    xh0 = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dt, device=device)
    nz = torch.nonzero(xh0 - xl0 > 1e-12).flatten()
    n_var = int(nz.numel())
    print_progress = bool(getattr(settings, 'print_progress', False))

    def _restore():
        if settings is not None:
            settings.sigmoid_relaxation = prev_sig

    # --- SAT: PGD on the full box (confirmed witnesses only) ---
    # First validate that the PGD point forward (_forward_batch_graph)
    # faithfully reproduces the network: compare it to the zono forward
    # evaluated at the box CENTER (a zero-radius zonotope's center IS the exact
    # network output). A wrong forward could confirm a bogus witness (false
    # sat), so PGD must only run when the point-forward is trustworthy.
    #
    # The ONLY tolerated failure is `NotImplementedError` — the point-forward
    # genuinely lacks an op of this graph. Then sat-finding is skipped (SOUND:
    # never a false verdict; the UNSAT input-split below is unaffected), but
    # LOUDLY (it flags an op to add to _forward_batch_graph). EVERY other
    # exception (e.g. a shape RuntimeError in an *implemented* op) is a BUG and
    # PROPAGATES — we do NOT silently swallow it: a broad silent catch here once
    # hid a real fc-shape bug that disabled PGD on all ml4acopf-linear graphs.
    if not bool(getattr(settings, 'disable_sat_finding', False)):
        from .verify_zono_bnb import _forward_batch_graph
        gg32 = graph.gpu_graph(device=device, dtype=torch.float32)
        xc = (0.5 * (xl0 + xh0))
        _, _zc = _forward_zonotope_graph(xc, xc, gg, device, dt, settings=settings)
        try:
            _pf = _forward_batch_graph(xc.float().unsqueeze(0), gg32)
        except NotImplementedError as e:
            pgd_ok = False
            print(f'[trig_bab] WARNING: point-forward op unimplemented '
                  f'({e}); sat-finding DISABLED for this run', flush=True)
        else:
            pgd_ok = bool(torch.allclose(
                _pf.flatten().double(), _zc.center.double(),
                atol=1e-2, rtol=1e-2))
            if not pgd_ok:
                _dev = float((_pf.flatten().double()
                              - _zc.center.double()).abs().max())
                print(f'[trig_bab] WARNING: point-forward DIVERGES from the '
                      f'zono center by {_dev:.3g} (>1e-2) — a forward bug; '
                      f'sat-finding DISABLED for this run', flush=True)
        if pgd_ok:
            _onnx_p = getattr(graph, 'onnx_path', None)
            _atol = float(settings.sat_validate_atol
                          if 'sat_validate_atol' in settings else 1e-4)

            def _try_witness(_w):
                """ORT-validate + dispose a candidate witness. Returns the
                ('sat', details) tuple for a CLEAR CE (commit now); None for a
                within-tol CE (persisted by `_sat_disposition`, keep searching)
                or a witness ORT rejects."""
                _w = np.asarray(_w).flatten()
                _ok, _vinfo = _validate_sat_witness(_onnx_p, spec, _w,
                                                    atol=_atol,
                                                    emit_slack=pgd_box_expand_amount(settings))
                if _ok and _sat_disposition(
                        graph, spec, settings, _w, _vinfo) == 'real':
                    return 'sat', {'witness': _w}
                if print_progress:
                    print('[trig_bab] witness not a clear CE '
                          f'(ok={_ok}); saved if within-tol, continuing',
                          flush=True)
                return None

            sat_budget = min(15.0, max(2.0, 0.25 * total_timeout))
            # `pgd_attack_general` runs ONE restart batch per call. When
            # `pgd_sat_min_time` is set, re-run fresh batches until that many
            # seconds elapse (capped by sat_budget) or a CE is found — the extra
            # restarts (with per-disjunct targeting + multi-α from the config)
            # catch needle CEs at a curved nonlinear-input boundary. Each batch
            # gets a distinct deterministic seed (base + loop index) so more
            # iterations explore NEW randomness yet stay reproducible for
            # seed-0..9 tuning. min_time=0 → exactly one batch (default).
            _sat_min_t = float(getattr(settings, 'pgd_sat_min_time', 0.0) or 0.0)
            _base_seed = getattr(settings, 'pgd_seed', None)
            _base_seed = int(_base_seed) if isinstance(_base_seed, (int, float)) else None
            _sat_t0 = _time.time()
            _loop_i = 0
            while True:
                _rem = sat_budget - (_time.time() - _sat_t0)
                if _rem <= 0:
                    break
                _bseed = None if _base_seed is None else _base_seed + _loop_i
                is_sat, witness = _pgd_attack_general(
                    xl0.float(), xh0.float(), spec, gg32, settings,
                    time_budget=_rem, seed=_bseed)
                _loop_i += 1
                if is_sat:
                    # ORT-validate each hit. A CLEAR CE commits 'sat' now; a
                    # within-tol CE is persisted by `_try_witness` but does NOT
                    # short-circuit — keep attacking for a clear CE (a fresh
                    # batch often deepens a near-boundary hit past the tolerance,
                    # matching the seed-tuned 10/10 clear rate) until the budget.
                    _r = _try_witness(witness)
                    if _r is not None:
                        _restore()
                        return _r
                if (_time.time() - _sat_t0) >= min(_sat_min_t, sat_budget):
                    break

    # Input-split is only tractable for a modest number of varying dims; the
    # 118/300 specs vary ~189 (the split tree is astronomically large). For
    # those, branch on NONLINEAR-OP PRE-ACTIVATIONS instead (α,β-CROWN's
    # nonlinear_split): splitting a Sqr/Sigmoid/Sin/Cos input range tightens
    # that op's relaxation AND every downstream bilinear Mul that consumes it.
    max_var = int(getattr(settings, 'trig_bab_max_var', 28))
    if n_var > max_var:
        return _verify_trig_nonlinear_split(
            graph, spec, settings, gg, xl0, xh0, dt, device, deadline,
            print_progress, _restore)

    # --- UNSAT: sound input-split BaB ---
    def _leaf(xl, xh):
        _, zf = _forward_zonotope_graph(xl, xh, gg, device, dt, settings=settings)
        olo, ohi = zf.bounds()
        v, d = spec.check(olo.cpu().numpy(), ohi.cpu().numpy())
        return v, d, zf

    def _split_dim(xl, xh, zf):
        # highest spec-sensitivity varying dim: |worst-disjunct weight · G_in|
        # times the dim width (impact of halving it).
        G = zf.generators
        K_in = n_var
        G_in = G[:, :K_in] if G.shape[1] >= K_in else G
        qs = spec.as_linear_queries(int(G.shape[0]))
        score = torch.zeros(K_in, dtype=dt, device=device)
        for _, qw, _ in qs:
            qw_t = torch.as_tensor(qw, dtype=dt, device=device)
            score = score + (qw_t @ G_in).abs()
        widths = (xh - xl)[nz[:K_in]]
        impact = score * widths
        return int(nz[int(impact.argmax())])

    # Budget: input-split is only tractable for a modest number of varying
    # dims. ml4acopf 14_ieee has ~22; the 118/300 specs have ~189 (cube blows
    # up) — for those we bound the work and return unknown rather than churn.
    # No early give-up at an arbitrary leaf/depth cap — keep splitting until the
    # TIMEOUT (deadline) or OOM (handled gracefully -> unknown with the best
    # bound so far). The caps default to effectively unbounded; the wall budget
    # is the real stop. A branch that narrows to a degenerate (zero-width) box
    # bottoms out (can't split) and is abandoned, so depth stays finite without
    # a cap. An abandoned (unclosed, can't-split) leaf -> 'unknown' (can't prove
    # unsat over it), NOT a give-up of the whole search.
    max_leaves = int(getattr(settings, 'trig_bab_max_leaves', 10**9))
    max_depth = int(getattr(settings, 'trig_bab_max_depth', 10**9))
    queue = [(xl0, xh0, 0)]
    closed = 0
    opened = 0
    abandoned = 0
    try:
        while queue:
            if _time.perf_counter() >= deadline:
                _restore()
                return 'unknown', {'method': 'trig_input_split',
                                   'reason': 'timeout', 'leaves_closed': closed}
            xl, xh, depth = queue.pop()
            v, d, zf = _leaf(xl, xh)
            if v == 'verified':
                closed += 1
                continue
            opened += 1
            di = _split_dim(xl, xh, zf)
            # Can't split a (near-)degenerate box, or hit an optional cap:
            # abandon this branch (keep exploring the rest), don't give up.
            if (di is None or float(xh[di] - xl[di]) < 1e-9
                    or opened > max_leaves or depth > max_depth):
                abandoned += 1
                continue
            mid = 0.5 * (xl[di] + xh[di])
            xh_l = xh.clone(); xh_l[di] = mid
            xl_r = xl.clone(); xl_r[di] = mid
            queue.append((xl, xh_l, depth + 1))
            queue.append((xl_r, xh, depth + 1))
    except (MemoryError, torch.cuda.OutOfMemoryError) as _oom:
        # OOM-as-outcome (allowed by the no-swallow rule): record + return the
        # best-so-far rather than crash. Re-raise only if explicitly requested.
        if getattr(settings, 'raise_on_oom', False):
            raise
        _restore()
        if print_progress:
            print(f'[trig_bab] OOM after opened={opened} closed={closed} '
                  f'-> unknown ({type(_oom).__name__})', flush=True)
        return 'unknown', {'method': 'trig_input_split', 'reason': 'oom',
                           'leaves_closed': closed}
    _restore()
    if abandoned:
        if print_progress:
            print(f'[trig_bab] exhausted: closed={closed} abandoned={abandoned} '
                  f'n_var={n_var}', flush=True)
        return 'unknown', {'method': 'trig_input_split', 'reason': 'exhausted',
                           'leaves_closed': closed, 'abandoned': abandoned}
    if print_progress:
        print(f'[trig_bab] verified: {closed} leaves', flush=True)
    return 'verified', {'method': 'trig_input_split', 'leaves_closed': closed}


_SPLITTABLE_OPS = ('pow', 'sigmoid', 'tanh', 'sin', 'cos', 'exp', 'reciprocal')


def _verify_trig_nonlinear_split(graph, spec, settings, gg, xl0, xh0, dt,
                                 device, deadline, print_progress, _restore):
    """SOUND nonlinear-split BaB (α,β-CROWN's nonlinear_split analog) for
    high-input-dim ACOPF specs where input-split is intractable. A BaB node is
    a dict of per-op-element input-range CLAMPS; the leaf bound is the sound
    forward zono under those clamps (`op_clamps`). Splitting one nonlinear op's
    element pre-activation at its midpoint into [lo,m] and [m,hi] yields two
    children whose op_clamps cover the parent's range — so both verified ⟹
    parent verified (the op_clamps soundness argument). Verifies iff every leaf
    closes; never returns a false verdict (falls back to unknown on budget).
    """
    import time as _time
    from .nonlinear_split_planes import split_point
    cand_set = {op['name'] for op in gg['ops']
                if op['type'] in _SPLITTABLE_OPS}
    op_type = {op['name']: op['type'] for op in gg['ops']}
    last_name = gg['ops'][-1]['name']
    NEG = float('-inf'); POS = float('inf')
    nz = torch.nonzero(xh0 - xl0 > 1e-12).flatten()   # varying input dims
    n_var = int(nz.numel())

    def _leaf(xl, xh, clamps):
        ob = {}; cids = {}; ofi = {}
        _, zf = _forward_zonotope_graph(xl, xh, gg, device, dt,
                                        settings=settings, op_bounds=ob,
                                        op_clamps=clamps, col_ids_out=cids,
                                        op_fresh_ids=ofi)
        olo, ohi = zf.bounds()
        v, d = spec.check(olo.cpu().numpy(), ohi.cpu().numpy())
        return v, d, ob, zf, cids, ofi

    def _pick_split(xl, xh, ob, zf, cids, ofi):
        # bbps-style sensitivity: per-column output margin-slack (summed over
        # all spec queries). Score each NONLINEAR op by the slack on the δ
        # columns it allocated, and each INPUT dim by the slack on its noise
        # column × its current width. Splitting the highest-scoring candidate
        # most tightens the binding disjunct. Input-dim splits shrink the
        # bilinear-Mul radii (which nonlinear-op clamps can't), so the hybrid
        # reaches cases neither split type closes alone.
        # Returns (kind, payload, score): kind 'nl' -> (op,elem,lo,hi);
        # 'input' -> (dim, lo, hi).
        G = zf.generators
        n_out = int(G.shape[0])
        qs = spec.as_linear_queries(n_out)
        if not qs:
            return None, None, 0.0
        W = torch.as_tensor(np.stack([w for _, w, _ in qs]),
                            dtype=dt, device=device)
        colslack = (W @ G).abs().sum(0)                     # (K,)
        out_ids = cids.get(last_name, [])
        id_to_col = {gid: k for k, gid in enumerate(out_ids)}
        best_kind = None; best_payload = None; best_score = 0.0
        # nonlinear-op candidates
        for nm, (a, b) in ofi.items():
            if nm not in cand_set:
                continue
            rng = ob.get(nm)
            if rng is None or not torch.is_tensor(rng[0]):
                continue
            contrib = sum(float(colslack[id_to_col[gid]])
                          for gid in range(a, b) if gid in id_to_col)
            if contrib <= best_score:
                continue
            lo_t, hi_t = rng
            w_el = (hi_t - lo_t); wi = int(w_el.argmax())
            if float(w_el[wi]) <= 1e-12:
                continue
            best_kind, best_payload, best_score = 'nl', (
                nm, wi, float(lo_t[wi]), float(hi_t[wi])), contrib
        # input-dim candidates: input noise has IDs 0..n_var-1 (made first),
        # mapping to varying dims nz[j]. Score = slack(col) × current width.
        for j in range(n_var):
            k = id_to_col.get(j)
            if k is None:
                continue
            di = int(nz[j])
            width = float(xh[di] - xl[di])
            score = float(colslack[k]) * width
            if score > best_score and width > 1e-9:
                best_kind, best_payload, best_score = 'input', (
                    di, float(xl[di]), float(xh[di])), score
        return best_kind, best_payload, best_score

    # No depth/leaf give-up: split until the TIMEOUT (deadline) or OOM (handled
    # gracefully). Caps default to effectively unbounded. A leaf we can't split
    # (no candidate / degenerate) is abandoned (-> unknown), the search keeps
    # going on the rest rather than aborting the whole verification.
    max_leaves = int(getattr(settings, 'trig_nl_max_leaves', 10**9))
    max_depth = int(getattr(settings, 'trig_nl_max_depth', 10**9))
    queue = [(xl0, xh0, {}, 0)]
    closed = 0; opened = 0; abandoned = 0
    while queue:
        if _time.perf_counter() >= deadline:
            _restore()
            if print_progress:
                print(f'[trig_nl] timeout: closed={closed} opened={opened}',
                      flush=True)
            return 'unknown', {'method': 'trig_nl_split', 'reason': 'timeout',
                               'leaves_closed': closed}
        xl, xh, clamps, depth = queue.pop()
        try:
            v, d, ob, zf, cids, ofi = _leaf(xl, xh, clamps)
        except (MemoryError, torch.cuda.OutOfMemoryError) as _oom:
            if getattr(settings, 'raise_on_oom', False):
                raise
            _restore()
            if print_progress:
                print(f'[trig_nl] OOM (closed={closed}) -> unknown '
                      f'({type(_oom).__name__})', flush=True)
            return 'unknown', {'method': 'trig_nl_split', 'reason': 'oom',
                               'leaves_closed': closed}
        if v == 'verified':
            closed += 1
            continue
        opened += 1
        kind, payload, sw = _pick_split(xl, xh, ob, zf, cids, ofi)
        if (kind is None or sw <= 1e-12 or opened > max_leaves
                or depth > max_depth):
            abandoned += 1   # can't split / over optional cap — abandon, keep going
            continue
        if kind == 'input':
            di, lo_i, hi_i = payload
            mid = 0.5 * (lo_i + hi_i)
            xh_l = xh.clone(); xh_l[di] = mid
            xl_r = xl.clone(); xl_r[di] = mid
            queue.append((xl, xh_l, clamps, depth + 1))
            queue.append((xl_r, xh, clamps, depth + 1))
        else:
            nm, ei, lo_i, hi_i = payload
            # Option A split point: 0 if a kink/inflection/min-at-0 op straddles
            # 0 (sigmoid/tanh/pow), else midpoint (sin/cos and asymmetric ranges).
            mid = split_point(op_type.get(nm, ''), lo_i, hi_i)
            n_elem = ob[nm][0].numel()

            def _child(set_lo, set_hi):
                c2 = {k: [v2[0].clone(), v2[1].clone()]
                      for k, v2 in clamps.items()}
                if nm not in c2:
                    c2[nm] = [torch.full((n_elem,), NEG, dtype=dt, device=device),
                              torch.full((n_elem,), POS, dtype=dt, device=device)]
                if set_lo is not None:
                    c2[nm][0][ei] = set_lo
                if set_hi is not None:
                    c2[nm][1][ei] = set_hi
                return {k: (v2[0], v2[1]) for k, v2 in c2.items()}

            queue.append((xl, xh, _child(None, mid), depth + 1))
            queue.append((xl, xh, _child(mid, None), depth + 1))
    _restore()
    if abandoned:
        if print_progress:
            print(f'[trig_nl] exhausted: closed={closed} abandoned={abandoned}',
                  flush=True)
        return 'unknown', {'method': 'trig_nl_split', 'reason': 'exhausted',
                           'leaves_closed': closed, 'abandoned': abandoned}
    if print_progress:
        print(f'[trig_nl] verified: {closed} leaves', flush=True)
    return 'verified', {'method': 'trig_nl_split', 'leaves_closed': closed}


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


def _clip_input_domain(x_L, x_U, lA, lbias, num_iters=1):
    """ABC-style closed-form input-domain shrinking.

    Given a CROWN linear lower bound `dm_lb(x) = lA·x + lbias` ≥ threshold=0
    (after merging per-sub bias into lbias), shrink (x_L, x_U) to keep ONLY
    the unverified region. For each input dim k and spec j: the value of
    x[k] where the bound crosses zero (other dims at worst-case) is
        curr_x[k] = -concrete_minus_one / lA[j,k]
    where concrete_minus_one = dm_lb - lA[j,:]·xhat + |lA[j,:]|·eps. Values
    of x[k] with lA[j,k] > 0 above curr_x make the bound positive
    (verified) so we drop x_U[k] to curr_x[k]; symmetrically for lA<0.
    Taking the MIN across specs (= disjunctive verification) is correct:
    if any spec verifies a region, we can drop it.

    Sound: removes only regions where SOME spec proves bound ≥ 0.
    Cost: ~10 lines of tensor math, O(batch · num_spec · input_dim). Per
    auto_LiRPA `input_split/clip.py`:`_clip_main_fn`, observed <1% of BAB
    iter time but lets each iter shrink 80%+ of unverified domains.

    Args:
      x_L, x_U: (B, K) input box (lower/upper) per leaf.
      lA:       (B, S, K) CROWN linear coefficients (S = num specs).
      lbias:    (B, S) constant offset, ALREADY including per-sub query bias.
      num_iters: clip iterations (refining after each shrink).
    Returns: (x_L', x_U') (B, K) shrunken box.
    """
    x_L = x_L.clone()
    x_U = x_U.clone()
    for _ in range(num_iters):
        xhat = (x_U + x_L) / 2     # (B, K)
        eps  = (x_U - x_L) / 2     # (B, K)
        # concretize dm_lb_per_spec: (B, S) = lA_pos·x_L + lA_neg·x_U + lbias
        lA_pos = lA.clamp(min=0); lA_neg = lA.clamp(max=0)
        dm_lb = (lA_pos * x_L.unsqueeze(1)
                  + lA_neg * x_U.unsqueeze(1)).sum(dim=2) + lbias  # (B, S)
        # contribution to dm_lb if x[k] were arbitrary in current box,
        # other dims at worst case: dm_lb - lA[j,:]·xhat + |lA[j,:]|·eps
        # (this is dm_lb minus the actual contribution of x[k] plus the
        #  worst-case contribution of all other dims).
        # Shape: (B, S, K).
        contrib_per_dim = (lA * xhat.unsqueeze(1)
                            - lA.abs() * eps.unsqueeze(1))   # what x[k] contributes worst-case if it sweeps box
        # concrete_minus_one[j,k] = dm_lb[j] - contrib_per_dim[j,k]
        concrete_minus_one = dm_lb.unsqueeze(2) - contrib_per_dim  # (B, S, K)
        # curr_x[j,k] = (threshold=0 - concrete_minus_one) / lA[j,k]
        # Guard div-by-zero (lA[j,k] = 0 → no constraint on x[k]).
        denom = torch.where(lA.abs() > 1e-30, lA,
                             torch.ones_like(lA))
        curr_x = -concrete_minus_one / denom
        # For lA > 0: x[k] > curr_x means bound ≥ 0 (verified). Drop x_U.
        # For lA < 0: x[k] < curr_x means bound ≥ 0 (verified). Drop x_L.
        x_U_cand = torch.where(lA > 0, curr_x, torch.full_like(curr_x, float('inf')))
        x_L_cand = torch.where(lA < 0, curr_x, torch.full_like(curr_x, float('-inf')))
        # Take MIN over specs (disjunctive: if ANY spec verifies x>curr_x, drop it).
        x_U = torch.min(x_U_cand.amin(dim=1), x_U)
        x_L = torch.max(x_L_cand.amax(dim=1), x_L)
    return x_L, x_U


def _multi_sub_input_split_bab(
        sub_idx_list, groups, order, gg, device, dtype, total_budget,
        n_out, settings):
    """Multi-sub batched input-split BAB (ABC-style).

    Single shared GPU worklist of (sub_idx, xl_leaf, xh_leaf) leaves
    across ALL unclosed sub-boxes. Each iter batches up to
    `input_split_batch_size` leaves: forward zono + backward CROWN per
    query, mark leaves verified by ANY query, split unverified on
    widest dim. Sub safe iff all its leaves verified.

    Mirrors α,β-CROWN's `BABReWeighted`/`input_split` driver which
    visits ~10k batched leaves in ~6s on mscn_3240 (12881 leaves total
    for 3240 unique X-subboxes).

    Returns: set of `sub_idx` (subset of `sub_idx_list`) marked safe.
    """
    import time, copy
    from .verify_zono_bnb import (
        _forward_zonotope_graph_batched, _spec_backward_graph_batched,
        _run_alpha_crown_inputsplit_batched)
    from .forward_lirpa import forward_lirpa_compat_zono_batched
    t0 = time.perf_counter()
    closed = set()
    if not sub_idx_list:
        return closed
    # Build per-sub spec data (queries: list of (w, b)).
    from .spec import VNNSpec
    sub_specs = {}
    for ki in sub_idx_list:
        _, _, conjs = groups[order[ki]]
        sp = VNNSpec(groups[order[ki]][0], groups[order[ki]][1], conjs)
        qs = sp.as_linear_queries(n_out)
        sub_specs[ki] = qs
    # Detect uniform-q-per-sub same-w pattern (mscn cardinality typical:
    # each sub has 2 queries, w shared across subs per q-index).
    q_counts = [len(sub_specs[ki]) for ki in sub_idx_list]
    if not q_counts or max(q_counts) == 0:
        return closed
    uniform_q = all(c == q_counts[0] for c in q_counts)
    max_q = q_counts[0]
    same_w_per_qi = uniform_q and all(
        all(np.array_equal(np.asarray(sub_specs[ki][qi][1]),
                            np.asarray(sub_specs[sub_idx_list[0]][qi][1]))
            for ki in sub_idx_list)
        for qi in range(max_q))
    if not (uniform_q and same_w_per_qi):
        # Fallback: cannot batch cleanly; skip multi-sub BAB.
        # Upstream w-grouping ensures shared w within each group, so this
        # path is only hit if a future caller bypasses that.
        if getattr(settings, 'print_progress', False):
            ref_w = np.asarray(sub_specs[sub_idx_list[0]][0][1])
            mismatch_ki = None
            for ki in sub_idx_list[:50]:
                for qi in range(max_q):
                    w_k = np.asarray(sub_specs[ki][qi][1])
                    if not np.array_equal(w_k, ref_w):
                        mismatch_ki = (ki, qi, w_k.tolist(), ref_w.tolist())
                        break
                if mismatch_ki:
                    break
            print(f'[multi-sub BAB] skipped: uniform_q={uniform_q} '
                  f'same_w={same_w_per_qi} q_counts={q_counts[:5]} '
                  f'mismatch={mismatch_ki}', flush=True)
        return closed
    w_per_q = [torch.as_tensor(np.asarray(sub_specs[sub_idx_list[0]][qi][1],
                                             dtype=np.float64),
                                  dtype=dtype, device=device)
                for qi in range(max_q)]
    # Bias per (sub, q).
    bias_per_sub_q = {ki: [float(sub_specs[ki][qi][2]) for qi in range(max_q)]
                       for ki in sub_idx_list}
    # Vectorized lookup table for biases — keyed by sub_idx → (max_q,)
    # tensor. Avoids the per-leaf .item() syncs in the BAB iter loop
    # (was 70+ ms per 4096-leaf iter). Maps sub_idx → row in this table.
    _bias_table_np = np.zeros((max(sub_idx_list) + 1, max_q),
                               dtype=np.float32)
    for _ki, _bs in bias_per_sub_q.items():
        _bias_table_np[_ki] = _bs
    bias_table_t = torch.as_tensor(_bias_table_np,
                                    dtype=dtype, device=device)
    # Initial worklist: one leaf per sub.
    xls_all = np.stack([groups[order[ki]][0] for ki in sub_idx_list]).astype(np.float64)
    xhs_all = np.stack([groups[order[ki]][1] for ki in sub_idx_list]).astype(np.float64)
    xl_t = torch.as_tensor(xls_all, dtype=dtype, device=device)
    xh_t = torch.as_tensor(xhs_all, dtype=dtype, device=device)
    sub_idx_t = torch.as_tensor(sub_idx_list, dtype=torch.long, device=device)
    # Number of varying axes shared across subs (mscn: every sub has
    # same X box shape; varying dims are the same indices but values
    # differ). Use width to pick the split axis per leaf.
    # Lift initial batch_size to 1024 — batched LiRPA now chunks
    # internally on OOM, so we benefit from larger initial batches when
    # memory allows (mscn 128d fits B=1024 easily). On 2048d_dual where
    # B=128 is the cap, chunked LiRPA auto-degrades. No iter cap.
    batch_size = int(getattr(settings, 'input_split_batch_size', 1024))
    import os as _os_bsz
    if _os_bsz.environ.get('DEBUG_BAB_QUEUE', '') == '1':
        print(f'[bab-init] input_split_batch_size={batch_size} '
              f'n_subs={len(sub_idx_list)} budget={total_budget:.1f}s',
              flush=True)
    max_iters = 10**9
    # Per-sub: count of LIVE leaves (not yet verified). Sub closed iff 0.
    live_leaves_per_sub = {int(ki): 1 for ki in sub_idx_list}
    # Optional ABC-style ROOT PRESPLIT (diagnostic / experimental, default
    # OFF). ABC presplits each disjunct's root box to 2^storage_depth boxes
    # before the first bounding, so it only ever drains; VC seeds 1 box and
    # ramps. Set BAB_ROOT_PRESPLIT=K to bisect each box's widest dim K times
    # up front (→ 2^K children/sub). Changes nothing when unset.
    import os as _os_psplit
    _root_presplit = int(_os_psplit.environ.get('BAB_ROOT_PRESPLIT', '0'))
    if _root_presplit > 0:
        for _ps in range(_root_presplit):
            w_ps = xh_t - xl_t
            if float(w_ps.max()) <= 1e-12:
                break  # all boxes fully fixed; nothing to split
            n_ps = xl_t.shape[0]
            rows_ps = torch.arange(n_ps, device=device)
            didx_ps = torch.argmax(w_ps, dim=1)        # widest dim per box
            mid_ps = (xl_t[rows_ps, didx_ps] + xh_t[rows_ps, didx_ps]) * 0.5
            xl_lo = xl_t.clone(); xh_lo = xh_t.clone()
            xh_lo[rows_ps, didx_ps] = mid_ps           # lower child
            xl_hi = xl_t.clone(); xh_hi = xh_t.clone()
            xl_hi[rows_ps, didx_ps] = mid_ps           # upper child
            xl_t = torch.cat([xl_lo, xl_hi], dim=0)
            xh_t = torch.cat([xh_lo, xh_hi], dim=0)
            sub_idx_t = torch.cat([sub_idx_t, sub_idx_t], dim=0)
        from collections import Counter as _Counter_ps
        _cnt_ps = _Counter_ps(int(s) for s in sub_idx_t.tolist())
        for _ki_ps in live_leaves_per_sub:
            live_leaves_per_sub[_ki_ps] = _cnt_ps.get(_ki_ps, 1)
        if _os_psplit.environ.get('DEBUG_BAB_QUEUE', '') == '1':
            print(f'[bab-presplit] depth={_root_presplit} '
                  f'boxes={xl_t.shape[0]} (was {len(sub_idx_list)})',
                  flush=True)
    _dump_boxes = _os_psplit.environ.get('BAB_DUMP_BOXES', '')
    if _dump_boxes:
        import json as _json_db
        _w = (xh_t - xl_t)
        _vd = (_w.max(dim=0).values > 1e-9).nonzero(as_tuple=True)[0].tolist()
        _boxes = []
        for _bi in range(xl_t.shape[0]):
            _boxes.append([[round(float(xl_t[_bi, d]), 6),
                            round(float(xh_t[_bi, d]), 6)] for d in _vd])
        # canonical-sort the box set for order-independent comparison
        _boxes_sorted = sorted(_boxes)
        _json_db.dump({'varying_dims': _vd, 'n_boxes': len(_boxes),
                       'boxes_sorted': _boxes_sorted},
                      open(_dump_boxes, 'w'), indent=1)
    # --- ABC-MATCH gated mode (diagnostic) -------------------------------
    # When ABC_MATCH=1, faithfully mirror α,β-CROWN's reorder_bab input-split
    # logic for box-for-box comparison: (1) backward-CROWN margin (≈ ABC
    # forward+crown), (2) dynamic split arity via get_split_depth over
    # storage_depth SB dims keeping degenerate duplicate children, (3) clip
    # children AFTER split (num_iters=1) instead of parents before. This is
    # NOT a production path — purely to confirm we understand ABC's algorithm.
    _abc_match = _os_psplit.environ.get('ABC_MATCH', '0') != '0'
    # ABC_DEDUP=1: filter redundant (bit-identical) boxes from the worklist
    # each iter. ABC's storage_depth split bisects degenerate (zero-width)
    # dims too, producing duplicate children it wastefully re-bounds (4× on
    # card_1_1). Deduping bounds each DISTINCT box once → strictly less work
    # than ABC, same verdict. (Only used with ABC_MATCH.)
    _abc_dedup = _os_psplit.environ.get('ABC_DEDUP', '0') != '0'
    import math as _math_abc
    # ABC: min_batch_size = min_batch_size_ratio(0.1) * batch_size(256) = 25.6
    _abc_min_batch = float(_os_psplit.environ.get('ABC_MIN_BATCH', '25.6'))
    _abc_storage_depth = min(
        max(int(_math_abc.log(max(_abc_min_batch, 1.0)) // _math_abc.log(2.0)), 1),
        int(xl_t.shape[1]))

    def _abc_get_split_depth(n_dom):
        # mirrors input_split/split.py:get_split_depth
        if n_dom == 0:
            return 1
        if n_dom < _abc_min_batch:
            return max(int(_math_abc.log(_abc_min_batch // n_dom)
                           // _math_abc.log(2.0)), 1)
        return 1
    if _abc_match:
        # Pickout batch. Bigger batches → fewer BAB iters → better GPU
        # utilization (mirrors ABC's auto_enlarge_batch_size). 2048 cut
        # card_1_2260 from 65→33 iters / 20.3→17.9s with identical leaves
        # (pickout doesn't affect split_depth → leaf count unchanged). The
        # backward-CROWN path below is OOM-halving-guarded so 2048 degrades
        # gracefully on the largest instances (e.g. card_1_11080) instead of
        # crashing. Override with ABC_PICKOUT.
        batch_size = int(_os_psplit.environ.get('ABC_PICKOUT', '2048'))
    if _abc_match and _os_psplit.environ.get('DEBUG_BAB_QUEUE', '') == '1':
        print(f'[abc-match] storage_depth={_abc_storage_depth} '
              f'min_batch={_abc_min_batch}', flush=True)
    n_iters = 0
    n_leaves_visited = 0
    # Per-iter phase timings (matches ABC's "pickout/bounding/filtering/clip/split" labels).
    # Enable via VIB_BAB_PHASE_TIMING=1 for side-by-side comparison with ABC.
    import os as _os_pt
    _phase_timing = _os_pt.environ.get('VIB_BAB_PHASE_TIMING', '') == '1'
    while xl_t.shape[0] > 0 and n_iters < max_iters:
        if time.perf_counter() - t0 > total_budget:
            break
        if _phase_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _pt_iter_start = time.perf_counter()
            _pt_pickout = _pt_iter_start
        # Pick batch.
        B = min(batch_size, xl_t.shape[0])
        # Drop already-closed sub leaves first.
        keep_mask = torch.tensor(
            [live_leaves_per_sub.get(int(s.item()), 0) > 0
             for s in sub_idx_t], device=device)
        if not keep_mask.all():
            xl_t = xl_t[keep_mask]
            xh_t = xh_t[keep_mask]
            sub_idx_t = sub_idx_t[keep_mask]
            B = min(batch_size, xl_t.shape[0])
            if B == 0:
                break
        # ABC_DEDUP: collapse bit-identical (sub_idx, xl, xh) worklist rows so
        # each distinct box is bounded ONCE. Duplicates are bound- and verify-
        # identical, so this is sound; it strips the wasted work ABC does on
        # storage_depth degenerate duplicates. Recompute live counts / closed
        # from the deduped worklist (a sub with 0 unverified boxes is closed).
        if _abc_dedup and xl_t.shape[0] > 0:
            _keyt = torch.cat(
                [sub_idx_t.unsqueeze(1).to(dtype), xl_t, xh_t], dim=1)
            _uniq, _inv = torch.unique(_keyt, dim=0, return_inverse=True)
            _perm = torch.arange(_inv.numel(), device=device)
            _first = torch.full((_uniq.shape[0],), _inv.numel(),
                                 device=device, dtype=torch.long)
            _first = _first.scatter_reduce(0, _inv, _perm, reduce='amin',
                                            include_self=True)
            _first = _first.sort().values
            if _first.numel() < xl_t.shape[0]:
                xl_t = xl_t[_first]; xh_t = xh_t[_first]
                sub_idx_t = sub_idx_t[_first]
                from collections import Counter as _Cdedup
                _cnt = _Cdedup(int(s) for s in sub_idx_t.tolist())
                for _s in live_leaves_per_sub:
                    live_leaves_per_sub[_s] = _cnt.get(_s, 0)
                    if live_leaves_per_sub[_s] == 0:
                        closed.add(_s)
                B = min(batch_size, xl_t.shape[0])
        # Routing: prefer zono+backward CROWN — after the batched
        # point-centers fix in alpha_crown.py, backward CROWN is both
        # tighter than forward LiRPA AND ~3× faster per batch on mscn
        # (0.34ms/leaf vs 1.04ms/leaf at B=256). LiRPA only kicks in
        # when backward CROWN hits OOM/NotImplemented (e.g., unsupported
        # op). Cache key kept for back-compat with old routing logic.
        if not hasattr(_multi_sub_input_split_bab, '_softmax_pattern_cache'):
            _multi_sub_input_split_bab._softmax_pattern_cache = {}
        gg_id = id(gg)
        if gg_id not in _multi_sub_input_split_bab._softmax_pattern_cache:
            _multi_sub_input_split_bab._softmax_pattern_cache[gg_id] = True
        # LiRPA forward inside BaB avoids the zonotope-generator memory
        # explosion that drives the old zono path to OOM-halve batch_size
        # from 4096 → 16 on mscn (~8 OOMs per w-group). With LiRPA we
        # keep B=4096, process the entire queue in 1 iter, and finish 2×
        # faster on cardinality_1_240 (33.2s → 16.1s). Disable via
        # BAB_USE_LIRPA=0 to fall back to the zono path.
        import os as _os_bab_lirpa
        use_lirpa = (_os_bab_lirpa.environ.get('BAB_USE_LIRPA', '1') != '0')
        xl_b = xl_t[:B]; xh_b = xh_t[:B]; si_b = sub_idx_t[:B]
        bb = None
        if use_lirpa:
            try:
                from .forward_lirpa import (
                    chunked_batched_forward_linear_bounds)
                # Chunked + streaming: OOM-resilient (chunked) AND
                # per-op A matrices freed after last consumer
                # (streaming). Best throughput on mscn dual.
                import os as _os_jt
                _use_jit_trace = _os_jt.environ.get(
                    'BAB_USE_JIT_TRACE', '0') == '1'
                _use_bwd_crown_env = _os_jt.environ.get(
                    'BAB_USE_BWD_CROWN', '0') != '0'
                bb = chunked_batched_forward_linear_bounds(
                    gg, xl_b, xh_b, device, dtype, max_chunk=B,
                    free_states=True, use_jit_trace=_use_jit_trace,
                    track_interm=_use_bwd_crown_env or _abc_match)
            except (torch.cuda.OutOfMemoryError, RuntimeError, NotImplementedError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bb = None
        if _phase_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _pt_bounding = time.perf_counter()
        leaf_safe_by_any_q = torch.zeros(B, dtype=torch.bool, device=device)
        # Accumulate per-query (spec_A, lbias_with_bias) so we can apply
        # ABC-style input-domain clipping on unsafe leaves before splitting.
        # Also stash per-q margin (dm_lb after bias) for SB score.
        clip_lA_list = []        # each (B, K)
        clip_lbias_list = []     # each (B,)
        clip_margin_list = []    # each (B,) — spec lb (= dm_lb - threshold)
        _oom_iter = False        # set if backward CROWN OOMs → halve & retry
        if bb is not None:
            # LiRPA path: bb.A_lo/A_up are on the COMPRESSED input subspace
            # (varying dims only). Use the stashed mask to compress xl_b/xh_b
            # so dimensions match. Without this the spec eval silently
            # broadcasts wrong (shape K vs n_in).
            varying_mask = chunked_batched_forward_linear_bounds.last_varying_mask
            xl_b_c = xl_b[:, varying_mask]   # (B, K)
            xh_b_c = xh_b[:, varying_mask]
            # Bound-quality fix: forward LiRPA → spec via direct A·W multiply
            # gives 69% LOOSER spec lb than backward CROWN (mscn card_1_240
            # disj 0 microbench: -0.477 vs -0.148 — matches ABC exactly).
            # Looser per-leaf bound → more BAB splits needed. Backward CROWN
            # now reuses the forward pass's interm_box (no second forward),
            # so per-iter cost is only +47ms over default. Net is a win on
            # small/medium cases (card_1_240: 15.7s→14.1s, card_1_960:
            # 46s→37s) but a loss on cases with many BAB iters (card_1_5410:
            # 234s verified→timeout) where the +47ms × 100+ iters dominates.
            # Opt-in via BAB_USE_BWD_CROWN=1.
            import os as _os_bwc
            use_bwd_crown = (_os_bwc.environ.get('BAB_USE_BWD_CROWN', '0') != '0'
                             or _abc_match)
            if use_bwd_crown:
                # Build sb (per-layer pre-act bounds) from the ALREADY-RUN
                # forward pass's interm_box (no second forward pass — was
                # the dominant cost in old BWD_CROWN path: 3.8× per-iter).
                from .forward_lirpa import (
                    _build_sb_from_interm,
                    chunked_batched_forward_linear_bounds as _cbflb)
                _interm = getattr(_cbflb, 'last_interm_box', None)
                if _interm is None:
                    sb_b_full, _ = forward_lirpa_compat_zono_batched(
                        xl_b, xh_b, gg, device, dtype)
                else:
                    sb_b_full, _bilinear_bounds = _build_sb_from_interm(
                        gg, _interm)
                _fixed_mask = ~varying_mask
                for qi in range(max_q):
                    w_q = w_per_q[qi]
                    spec_ew_q = {0: (w_q, 0.0)}
                    biases_for_q = bias_table_t[si_b, qi]
                    if _abc_match:
                        # ABC's clip/SB use the SAME (tight) bound as the
                        # margin (forward+crown). Use backward CROWN's
                        # input-space linear bound A·x + acc (return_input_
                        # linear) for clip_lA, compressed to varying dims with
                        # the fixed dims' constant contribution folded into the
                        # bias. The forward A·W lA below is far looser and
                        # produced ZERO clip shrinkage (split2 mismatch).
                        try:
                            lbs_z, A_in, acc_in = _spec_backward_graph_batched(
                                sb_b_full, xl_b, xh_b, gg, spec_ew_q, device,
                                dtype, return_input_linear=True)
                        except torch.cuda.OutOfMemoryError:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            _oom_iter = True
                            break   # halve batch_size & retry this iter
                        lbs_q_flat = lbs_z[:, 0]
                        margins_q = lbs_q_flat + biases_for_q
                        leaf_safe_by_any_q |= (margins_q > 0)
                        spec_A = A_in[:, 0, :][:, varying_mask]          # (B, K)
                        fixed_const = (A_in[:, 0, :][:, _fixed_mask]
                                       * xl_b[:, _fixed_mask]).sum(dim=1)  # (B,)
                        clip_lA_list.append(spec_A)
                        clip_lbias_list.append(acc_in[:, 0] + fixed_const
                                               + biases_for_q)
                        clip_margin_list.append(margins_q)
                        continue
                    lbs_z = _spec_backward_graph_batched(
                        sb_b_full, xl_b, xh_b, gg, spec_ew_q, device, dtype)
                    lbs_q_flat = lbs_z[:, 0]
                    # Vectorized lookup (was per-leaf .item() loop, 70+ms).
                    margins_q = lbs_q_flat + biases_for_q
                    leaf_safe_by_any_q |= (margins_q > 0)
                    # For clip + SB, we still need lA on the compressed input
                    # subspace. Compute it from forward A·W as before (these
                    # are LOOSER than backward but available cheaply).
                    w_pos = w_q.clamp(min=0); w_neg = w_q.clamp(max=0)
                    spec_A = (w_pos.unsqueeze(0).unsqueeze(-1) * bb.A_lo
                              + w_neg.unsqueeze(0).unsqueeze(-1) * bb.A_up).sum(dim=1)
                    spec_b_const = ((w_pos.unsqueeze(0) * bb.b_lo
                                     + w_neg.unsqueeze(0) * bb.b_up).sum(dim=1))
                    clip_lA_list.append(spec_A)
                    clip_lbias_list.append(spec_b_const + biases_for_q)
                    clip_margin_list.append(margins_q)
            else:
                # Vectorized over all queries (was a Python for-loop over qi —
                # 27ms × max_q per BAB iter on 4096-leaf batch). Stack once,
                # compute all queries' spec_A/lbias/margins in one pass.
                # w_per_q: list of max_q tensors (n_out,). Stack: (Q, n_out).
                W_qs = torch.stack(w_per_q, dim=0)              # (Q, n_out)
                W_pos = W_qs.clamp(min=0)                       # (Q, n_out)
                W_neg = W_qs.clamp(max=0)
                # bb.A_lo/A_up: (B, n_out, K). einsum to (B, Q, K).
                spec_A_all = (torch.einsum('qn,bnk->bqk', W_pos, bb.A_lo)
                              + torch.einsum('qn,bnk->bqk', W_neg, bb.A_up))
                spec_b_const_all = (torch.einsum('qn,bn->bq', W_pos, bb.b_lo)
                                    + torch.einsum('qn,bn->bq', W_neg, bb.b_up))
                # Concretize lbs over current box: (B, Q, K) sign-split → (B, Q).
                spec_A_pos_all = spec_A_all.clamp(min=0)
                spec_A_neg_all = spec_A_all.clamp(max=0)
                lbs_qs_flat = ((spec_A_pos_all * xl_b_c.unsqueeze(1)
                                + spec_A_neg_all * xh_b_c.unsqueeze(1)).sum(dim=2)
                               + spec_b_const_all)               # (B, Q)
                # Bias lookup over all queries: (B, Q) from (n_sub, Q) table.
                biases_all = bias_table_t[si_b]                  # (B, Q)
                margins_all = lbs_qs_flat + biases_all           # (B, Q)
                leaf_safe_by_any_q = (margins_all > 0).any(dim=1)
                # Stash per-q for clip/SB (matches prior list semantics).
                for qi in range(max_q):
                    clip_lA_list.append(spec_A_all[:, qi])
                    clip_lbias_list.append(spec_b_const_all[:, qi]
                                            + biases_all[:, qi])
                    clip_margin_list.append(margins_all[:, qi])
        else:
            # Zono path (used for softmax-pattern graphs OR LiRPA OOM).
            try:
                sb_b, (c_out_b, G_out_b) = _forward_zonotope_graph_batched(
                    xl_b, xh_b, gg, device, dtype)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = max(1, batch_size // 2)
                if new_bs == batch_size:
                    break
                import os as _os_oom2
                if _os_oom2.environ.get('DEBUG_BAB_QUEUE', '') == '1':
                    print(f'[bab-oom] forward OOM, halving batch_size '
                          f'{batch_size} -> {new_bs}', flush=True)
                batch_size = new_bs
                continue
            for qi in range(max_q):
                spec_ew_q = {0: (w_per_q[qi], 0.0)}
                try:
                    lbs_z = _spec_backward_graph_batched(
                        sb_b, xl_b, xh_b, gg, spec_ew_q, device, dtype)
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_bs = max(1, batch_size // 2)
                    if new_bs == batch_size:
                        lbs_z = 'OOM_PERM'
                        break
                    import os as _os_oom
                    if _os_oom.environ.get('DEBUG_BAB_QUEUE', '') == '1':
                        print(f'[bab-oom] backward OOM, halving batch_size '
                              f'{batch_size} -> {new_bs}', flush=True)
                    batch_size = new_bs
                    lbs_z = None
                    break
                lbs_q_flat = lbs_z[:, 0]
                # Vectorized GPU lookup (was per-leaf .item() list comp —
                # B GPU syncs per qi per iter; B=300 → 300 syncs).
                biases_for_q = bias_table_t[si_b, qi]
                margins_q = lbs_q_flat + biases_for_q
                leaf_safe_by_any_q |= (margins_q > 0)
            if isinstance(lbs_z, str) and lbs_z == 'OOM_PERM':
                break
            if lbs_z is None:
                continue
        if _oom_iter:
            # Backward CROWN OOM'd at this batch size → halve pickout and
            # retry the iter (worklist xl_t is untouched until split/add at
            # end, so retry is safe). Mirrors ABC's auto_enlarge shrink.
            new_bs = max(1, batch_size // 2)
            if new_bs == batch_size:
                break    # cannot reduce further; give up this group
            batch_size = new_bs
            continue
        n_leaves_visited += B
        import os as _os_qd
        if _os_qd.environ.get('DEBUG_BAB_QUEUE', '') == '1' and n_iters < 30:
            n_open_subs = sum(1 for v in live_leaves_per_sub.values() if v > 0)
            print(f'[bab-queue] iter={n_iters} B={B} batch_size={batch_size} '
                  f'queue_size={xl_t.shape[0]} '
                  f'open_subs={n_open_subs} closed={len(closed)}',
                  flush=True)
        # Update live counts: each verified leaf decrements its sub's
        # live count. When count hits 0, sub closed.
        for i in range(B):
            if leaf_safe_by_any_q[i].item():
                sk = int(si_b[i].item())
                live_leaves_per_sub[sk] -= 1
                if live_leaves_per_sub[sk] == 0:
                    closed.add(sk)
        # Unsafe leaves: split on top-K widest dims SIMULTANEOUSLY →
        # 2^K child leaves per unsafe leaf. Matches ABC's strategy
        # (`storage_depth = floor(log2(min_batch_size_ratio * batch_size))`
        # in `input_split/batch_branch_and_bound.py`). K=4 for nn4sys at
        # batch_size=256 (= log2(25.6)). With single-dim splits the queue
        # grows 2x/iter (only ~16 leaves visited/iter on mscn_240);
        # K-dim splits grow it 2^K and amortize bound-call overhead.
        if _phase_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _pt_filtering = time.perf_counter()
        unsafe_idx = (~leaf_safe_by_any_q).nonzero(as_tuple=True)[0]
        rest_xl = xl_t[B:]; rest_xh = xh_t[B:]; rest_si = sub_idx_t[B:]
        # Build the per-iter (B,S,K) clip lA / (B,S) clip lbias once so the
        # SB picker, pre-split clip, and post-split child clip all share it
        # (the SB block references clip_lA even when pre-split clip is off,
        # e.g. under ABC_MATCH).
        clip_lA = clip_lbias = None
        if clip_lA_list:
            clip_lA = torch.stack(clip_lA_list, dim=1)        # (B, S, K)
            clip_lbias = torch.stack(clip_lbias_list, dim=1)  # (B, S)
        # ABC-style input-domain clipping on unsafe leaves: shrink the
        # input box using CROWN lA/lbias to drop regions where ANY spec
        # proves bound ≥ 0 (verified). Reduces split count on hard
        # clusters where bounds are tight but BAB can't close in budget.
        # Only on LiRPA path (clip_lA_list populated); zono path is a
        # rare OOM fallback. Disable via env CLIP_INPUT_DOMAIN=0.
        import os as _os_clip
        clip_enabled = (clip_lA_list
                         and _os_clip.environ.get('CLIP_INPUT_DOMAIN', '1') != '0'
                         and unsafe_idx.numel() > 0
                         # ABC reorder_bab clips CHILDREN after split, not
                         # parents before. Under ABC_MATCH, suppress the
                         # pre-split parent clip (the post-split child clip
                         # below is force-enabled instead).
                         and not _abc_match)
        if clip_enabled:
            xl_b_c_u = xl_b_c[unsafe_idx]
            xh_b_c_u = xh_b_c[unsafe_idx]
            # num_iters=2: each clip iter refines after the prior shrink
            # (per ABC's `_clip_main_fn`). 2 iters is the sweet spot —
            # diminishing returns past that, but the second iter often
            # closes 10-20% more on hard clusters.
            xl_clipped, xh_clipped = _clip_input_domain(
                xl_b_c_u, xh_b_c_u,
                clip_lA[unsafe_idx], clip_lbias[unsafe_idx],
                num_iters=2)
            # Scatter shrunken box back into full (B, n_in) tensors by
            # writing only at varying-mask positions. VECTORIZED — the
            # prior Python loop over unsafe_idx (one .item() per leaf, then
            # masked-assign) was 100+ ms per BAB iter for 2000-leaf batches,
            # the actual clip bottleneck (not the clip math itself).
            varying_full_idx = varying_mask.nonzero(as_tuple=True)[0]
            # Build 2D index: rows = unsafe leaf indices (n_unsafe, K),
            # cols = varying-dim positions (n_unsafe, K).
            n_unsafe_clip = unsafe_idx.numel()
            row_idx = unsafe_idx.unsqueeze(1).expand(-1, varying_full_idx.numel())
            col_idx = varying_full_idx.unsqueeze(0).expand(n_unsafe_clip, -1)
            xl_b[row_idx, col_idx] = xl_clipped
            xh_b[row_idx, col_idx] = xh_clipped
            # Drop boxes that became empty (any dim where xl > xh): the
            # spec is verified on the whole original box for those leaves.
            # We treat these as additionally verified here.
            newly_verified = ((xl_b[unsafe_idx] > xh_b[unsafe_idx]).any(dim=1))
            if newly_verified.any():
                add_safe = unsafe_idx[newly_verified]
                leaf_safe_by_any_q[add_safe] = True
                for ai in add_safe.tolist():
                    sk = int(si_b[int(ai)].item())
                    live_leaves_per_sub[sk] -= 1
                    if live_leaves_per_sub[sk] == 0:
                        closed.add(sk)
                # Refresh unsafe_idx after marking new verifieds.
                unsafe_idx = (~leaf_safe_by_any_q).nonzero(as_tuple=True)[0]
        if _phase_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _pt_clip = time.perf_counter()
        if unsafe_idx.numel() > 0:
            K_max = int(getattr(settings, 'bab_split_depth', 4))
            n_in_dim = xl_b.shape[1]
            xl_u = xl_b[unsafe_idx]   # (n_unsafe, n_in)
            xh_u = xh_b[unsafe_idx]
            si_u = si_b[unsafe_idx]
            widths_u = xh_u - xl_u
            n_unsafe = xl_u.shape[0]
            # Per-leaf effective K = min(K_max, # dims with nonzero width).
            # When K_max exceeds the number of varying dims, top-K picks
            # zero-width dims whose splits produce duplicate children — sound
            # but wasteful. To keep the batched code uniform, pick the
            # SMALLER of K_max and the median nonzero-dim count, then drop
            # children identical to siblings (degenerate splits) post-hoc.
            nz_per_leaf = (widths_u > 1e-12).sum(dim=1)  # (n_unsafe,)
            if _abc_match:
                # ABC: one split_depth for the whole batch from get_split_depth
                # over storage_depth SB dims. Do NOT cap at #nonzero dims —
                # degenerate (zero-width) dims are split too, producing the
                # duplicate children that inflate ABC's visited-domain count.
                K_eff = int(min(_abc_get_split_depth(n_unsafe),
                                 _abc_storage_depth, n_in_dim))
                K_eff = max(K_eff, 1)
            else:
                K_eff = int(min(K_max,
                                 int(nz_per_leaf.max().item()) if n_unsafe > 0 else 1,
                                 n_in_dim))
                K_eff = max(K_eff, 1)
            # Smart Branching (SB) dim picker: rank dims by max-over-spec
            # |lA[:,s,k]| · width[k]. SB picks the dim that contributes most
            # to bound uncertainty — splitting it shrinks the bound the
            # most, leading to faster verification. ABC uses this by default
            # (branching.method=sb, sb_coeff_thresh=0.1). Width-only was a
            # weaker fallback. lA is available on LiRPA path (clip_lA_list);
            # falls back to width on zono path.
            # Disable via env SB_BRANCHING=0.
            import os as _os_sb
            use_sb = (clip_lA_list and
                      _os_sb.environ.get('SB_BRANCHING', '1') != '0')
            if use_sb:
                # ABC's SB score (input_split/branching_heuristics.py:88):
                #   score[b,s,k] = max(|lA[b,s,k]|, 0.1) * width[k]/2 +
                #                  (margin[b,s] - threshold(=0)) * sb_margin_weight
                # Then amax over spec dim (s). The clamp(min=0.1) ensures
                # wide dims with tiny lA still get scored; the margin term
                # favors the spec CLOSEST to verification (largest margin).
                lA_u = clip_lA[unsafe_idx]                         # (n_unsafe, S, K)
                widths_in_K = (xh_b_c[unsafe_idx] - xl_b_c[unsafe_idx])  # (n_unsafe, K)
                clip_margin = torch.stack(clip_margin_list, dim=1) # (B, S)
                margin_u = clip_margin[unsafe_idx]                 # (n_unsafe, S)
                lA_clamping_thresh = 0.1   # ABC default
                sb_margin_weight = 1.0     # ABC default
                # Per (leaf, spec, dim) score; broadcast margin across dims.
                score_per_spec = (
                    lA_u.abs().clamp(min=lA_clamping_thresh)
                    * widths_in_K.unsqueeze(1) * 0.5
                    + margin_u.unsqueeze(-1) * sb_margin_weight)   # (n_unsafe, S, K)
                sb_score = score_per_spec.amax(dim=1)              # (n_unsafe, K)
                # Map compressed-K → full n_in_dim via varying_mask.
                varying_full_idx = varying_mask.nonzero(as_tuple=True)[0]
                sb_full = torch.full((n_unsafe, n_in_dim), float('-inf'),
                                      dtype=dtype, device=device)
                sb_full[:, varying_full_idx] = sb_score
                # Final zero-width mask (already -inf if outside varying).
                sb_full = torch.where(widths_u > 1e-12, sb_full,
                                       torch.full_like(sb_full, float('-inf')))
                if _abc_match:
                    # ABC ranks dims with topk(score, storage_depth) ONCE, then
                    # the dynamic split uses the first get_split_depth(=K_eff)
                    # of them. torch.topk's tie order depends on k, so ranking
                    # at storage_depth (not K_eff) is required to reproduce
                    # ABC's split_idx exactly (e.g. CUDA topk(.,4)=[249,235,..]
                    # but topk(.,1)=[235] on a tie).
                    _rank_k = min(_abc_storage_depth, sb_full.shape[1])
                    top_k_dims = torch.topk(
                        sb_full, _rank_k, dim=1).indices[:, :K_eff]
                else:
                    top_k_dims = torch.topk(sb_full, K_eff, dim=1).indices
            else:
                top_k_dims = torch.topk(widths_u, K_eff, dim=1).indices
            xl_top = xl_u.gather(1, top_k_dims)
            xh_top = xh_u.gather(1, top_k_dims)
            mids = (xl_top + xh_top) * 0.5  # (n_unsafe, K)
            # Mask: for each (leaf, k), True iff width is non-trivial.
            usable = (xh_top - xl_top) > 1e-12  # (n_unsafe, K)
            n_children = 1 << K_eff
            combos = torch.arange(n_children, device=device)
            bits = ((combos.unsqueeze(-1)
                     >> torch.arange(K_eff, device=device)) & 1)  # (n_children, K)
            xl_exp = xl_u.unsqueeze(1).expand(-1, n_children, -1).contiguous()
            xh_exp = xh_u.unsqueeze(1).expand(-1, n_children, -1).contiguous()
            leaf_ax = torch.arange(n_unsafe, device=device).unsqueeze(1).expand(
                -1, n_children)
            child_ax = combos.unsqueeze(0).expand(n_unsafe, -1)
            for k in range(K_eff):
                dim_idx_k = top_k_dims[:, k:k+1].expand(-1, n_children)
                mid_k = mids[:, k:k+1].expand(-1, n_children)
                usable_k = usable[:, k:k+1].expand(-1, n_children)
                bit_k_2d = bits[:, k].unsqueeze(0).expand(n_unsafe, -1).bool()
                # Only update where dim k is usable for this leaf.
                hi_mask = bit_k_2d & usable_k
                lo_mask = (~bit_k_2d) & usable_k
                if hi_mask.any():
                    xl_exp[leaf_ax[hi_mask], child_ax[hi_mask],
                           dim_idx_k[hi_mask]] = mid_k[hi_mask]
                if lo_mask.any():
                    xh_exp[leaf_ax[lo_mask], child_ax[lo_mask],
                           dim_idx_k[lo_mask]] = mid_k[lo_mask]
            xl_new = xl_exp.reshape(-1, n_in_dim)
            xh_new = xh_exp.reshape(-1, n_in_dim)
            si_new = si_u.unsqueeze(1).expand(-1, n_children).reshape(-1)
            # Post-split clip (ABC-style): apply clip to CHILDREN using
            # parent's lA/lbias. Children's smaller box means clip's
            # axis-aligned shrink can extract regions that didn't shrink
            # on the parent's larger box. Per ABC's
            # `batch_branch_and_bound.py:175` clip step happens AFTER split.
            # Disable via env CLIP_AFTER_SPLIT=0.
            import os as _os_cas
            n_dropped_per_parent = torch.zeros(n_unsafe, dtype=torch.long,
                                                device=device)
            # Default off — tested on mscn_2048d_dual misses: net wash
            # (+44 on 2890, -24 on 2260). The pre-split clip already
            # captures most shrinking, and per-child clip just doubles
            # clip work for similar bounds.
            if (clip_lA_list and
                    (_os_cas.environ.get('CLIP_AFTER_SPLIT', '0') != '0'
                     or _abc_match)):
                # Child clip uses the PARENT's lA/lbias repeated to children
                # (clip_lA/clip_lbias built once above) — mirrors ABC reorder.
                parent_idx_per_child = unsafe_idx[leaf_ax.reshape(-1)]
                child_lA = clip_lA[parent_idx_per_child]
                child_lbias = clip_lbias[parent_idx_per_child]
                xl_new_c = xl_new[:, varying_mask]
                xh_new_c = xh_new[:, varying_mask]
                # ABC clip_iterations default = 1 (VC production uses 2).
                _clip_iters = 1 if _abc_match else 2
                xl_new_clipped, xh_new_clipped = _clip_input_domain(
                    xl_new_c, xh_new_c, child_lA, child_lbias,
                    num_iters=_clip_iters)
                # Empty child = some dim has xl > xh after clipping = box
                # is empty in that dim = whole region was provably verified.
                child_empty = (xl_new_clipped > xh_new_clipped).any(dim=1)
                xl_new[:, varying_mask] = xl_new_clipped
                xh_new[:, varying_mask] = xh_new_clipped
                # Count empty children per parent (used to adjust live count).
                n_dropped_per_parent = (
                    child_empty.reshape(n_unsafe, n_children).sum(dim=1))
                # Filter out empty children from the next-iter queue.
                if child_empty.any():
                    keep_mask = ~child_empty
                    xl_new = xl_new[keep_mask]
                    xh_new = xh_new[keep_mask]
                    si_new = si_new[keep_mask]
            # Live-leaf accounting:
            # Parent leaf was 1 entry, now becomes (n_children - n_dropped).
            # Net change per parent: (n_children - n_dropped) - 1.
            n_dropped_cpu = n_dropped_per_parent.cpu().tolist()
            for pi, s_val in enumerate(si_u.tolist()):
                net_delta = n_children - n_dropped_cpu[pi] - 1
                live_leaves_per_sub[int(s_val)] += net_delta
                if live_leaves_per_sub[int(s_val)] <= 0:
                    live_leaves_per_sub[int(s_val)] = 0
                    closed.add(int(s_val))
            xl_t = torch.cat([rest_xl, xl_new], dim=0)
            xh_t = torch.cat([rest_xh, xh_new], dim=0)
            sub_idx_t = torch.cat([rest_si, si_new], dim=0)
        else:
            xl_t = rest_xl; xh_t = rest_xh; sub_idx_t = rest_si
        if _phase_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _pt_end = time.perf_counter()
            n_safe = int(leaf_safe_by_any_q.sum().item())
            n_unsafe = B - n_safe
            print(f'[bab-phase] iter={n_iters} B={B} verified={n_safe} unsafe={n_unsafe}  '
                  f'bounding={(_pt_bounding-_pt_pickout)*1000:.1f}  '
                  f'filtering={(_pt_filtering-_pt_bounding)*1000:.2f}  '
                  f'clip={(_pt_clip-_pt_filtering)*1000:.2f}  '
                  f'split={(_pt_end-_pt_clip)*1000:.2f}  '
                  f'TOTAL={(_pt_end-_pt_iter_start)*1000:.1f}ms', flush=True)
        n_iters += 1
    if getattr(settings, 'print_progress', False):
        print(f'[multi-sub BAB] {n_iters} iters, {n_leaves_visited} leaves, '
              f'{len(closed)}/{len(sub_idx_list)} closed, '
              f'{time.perf_counter()-t0:.1f}s', flush=True)
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

    # Output size = the network's TRUE final output dim, taken from the probe
    # forward just run (z_final_leaf.center.numel()). The old last-fc/conv
    # heuristic is wrong for nets ending in non-linear physics (ml4acopf
    # linear-residual ends in Concat; its last Gemm = bus count 14/118).
    n_output = int(z_final_leaf.center.numel())
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
                import gurobipy as _grb_jt
                try:
                    closed_via_joint |= _joint_and_infeasible_triangle_lp(
                        gg['ops'], spec.x_lo, spec.x_hi, bbr,
                        qlists_for_tri)
                except (RuntimeError, GurobiNumericTrouble,
                        _grb_jt.GurobiError):
                    # Joint triangle-LP failure: Gurobi numerical trouble
                    # or builder runtime issue. Skip — opt-in pass, not
                    # required for soundness.
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
    except (RuntimeError, ValueError, IndexError):
        # Sensitivity-score computation: torch runtime on degenerate
        # tensors, ValueError on bad shape, IndexError if generators are
        # empty. Caller falls back to width-based scoring (sound).
        score_per_axis = None

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
        import gurobipy as _grb_sp
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
            except (_grb_sp.GurobiError, GurobiNumericTrouble):
                # Gen-LP solve failure (numeric trouble or Gurobi internal
                # error). Try next query; if all fail, disjunct stays open.
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
    # Use backward-CROWN intermediate bounds (AB-CROWN's `bound_prop_method:
    # crown`) instead of the loose forward zonotope. ~2x tighter -> far fewer
    # leaves on amplifying-weight nets (acasxu).
    crown_intermediate = bool(getattr(
        settings, 'input_split_crown_intermediate', False))
    # Serialize ops once (for MILP escalation; cheap, no torch tensors).
    # Only the per-leaf MILP-escalation path (gated by
    # input_split_batched_milp_escalate, default OFF) consumes this list —
    # the batched zono/CROWN BaB itself runs straight off the torch `gg`.
    # The MILP builder only supports fc/conv/relu/add/sub layers, so for a
    # graph with non-LP ops (mul/div_bilinear/reduce_sum/concat/slice/...
    # — lsnc_relu's quadratic Lyapunov forms) there is no sound MILP
    # encoding anyway. Build the serialized list ONLY when escalation is
    # requested; then the NotImplementedError still fires loudly for an
    # op the MILP builder can't honor (no silent skip). With escalation
    # off the list stays empty and the consumer at the bottom short-
    # circuits on `gg_ops_ser` being falsy.
    _milp_escalate_serialize = bool(getattr(
        settings, 'input_split_batched_milp_escalate', False))
    gg_ops_ser = []
    if _milp_escalate_serialize:
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
            elif op['type'] == 'reshape':
                pass    # flat passthrough; consumers alias the input vars
            elif op['type'] == 'conv':
                d['kernel_np'] = op['kernel_np']; d['bias_np'] = op['bias_np']
                d['in_shape'] = op['in_shape']; d['out_shape'] = op['out_shape']
                d['stride'] = op['stride']; d['padding'] = op['padding']
                d['n_out'] = op['n_out']
            else:
                raise NotImplementedError(
                    f"gg serializer (dual-ascent): unsupported op "
                    f"{op['type']!r} at {op['name']!r} — serializing without "
                    f"its params would make consumers run a different network")
            gg_ops_ser.append(d)

    # True network output width, from a single point-forward through gg.
    # The "last fc op" heuristic used below for the BaB queries is WRONG
    # for concat-output nets (lsnc_relu's output is
    # Concat(dV, V, next_state)=8 wide, but the last Gemm is the 6-wide
    # dynamics layer; nn4sys pensieve concats too). A zero-width input box
    # (lo==hi==center) propagates with NO error generators, so this is a
    # pure point eval — cheapest possible — and robust to any output
    # topology (concat / slice / reduce). Both the SAT-finder PGD and the
    # BaB queries below use it.
    _xc = ((torch.as_tensor(spec.x_lo, dtype=dtype, device=device)
            + torch.as_tensor(spec.x_hi, dtype=dtype, device=device)) / 2
           ).reshape(1, -1)
    _, (_c_probe, _) = _forward_zonotope_graph_batched(
        _xc, _xc, gg, device, dtype)
    n_output_true = int(_c_probe.shape[-1])

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
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Phase-0 PGD: GPU runtime or OOM. SAT-finding is best-effort;
            # fall through to the simpler PGD path below.
            pass
        # Also try the simpler multi-disjunct-aware PGD (verify_hybrid_acasxu
        # `_simple_pgd`): straight sign-gradient over the input box, no
        # OSI/multi-α, 10K restarts × 50 iters. Catches multi-disjunct DNF
        # SAT cases that _pgd_attack_general misses on cgan because the
        # latter doesn't reduce loss across disjuncts.
        try:
            from .verify_hybrid_acasxu import _simple_pgd
            # n_output_true (point-forward width) is robust to concat/slice
            # output topologies where the "last fc op" width is wrong.
            xl_pgd2 = xl_pgd.flatten().unsqueeze(0)
            xh_pgd2 = xh_pgd.flatten().unsqueeze(0)
            sat_simple, w_simple = _simple_pgd(
                xl_pgd2, xh_pgd2, spec, gg, n_output_true, device, dtype,
                n_restarts=10000, n_iter=50)
            if sat_simple:
                return 'sat', {'phase': 'batched_simple_pgd',
                                'witness': w_simple}
        except (RuntimeError, torch.cuda.OutOfMemoryError, ImportError):
            # Simple-PGD: GPU runtime/OOM or missing hybrid_acasxu deps.
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
        except (RuntimeError, ImportError, OSError):
            # ONNX PGD: ORT failure, missing deps, or bad path. Fall
            # through to the regular BAB pipeline.
            pass

    # True output width from the point-forward probe above (robust to
    # concat/slice/reduce output topologies, unlike the "last fc" heuristic).
    n_output = n_output_true
    if n_output is None or n_output <= 0:
        return 'unknown', {'phase': 'batched_no_output'}

    queries = spec.as_linear_queries(n_output)
    if not queries:
        return 'verified', {'phase': 'batched_no_queries'}
    # DEDUPLICATE identical (w, b) halfspaces. A global conjunct ANDed into
    # every disjunct (e.g. lsnc_relu's level-set band Y_1 in [a,b]) appears
    # once per disjunct — 26 of 39 queries are the 2 band halfspaces repeated
    # across 13 disjuncts. The CROWN spec backward cost scales with the number
    # of queries Q, so computing each UNIQUE halfspace once (Q: 39 → 15) cuts
    # the backward ~2.6x with identical verdicts. Each disjunct maps to the
    # shared unique-query batch positions; closure (any query lb > 0) is
    # unchanged because duplicate queries have identical lbs by construction.
    uniq_key_to_pos = {}
    uniq_ew = []        # (w_tensor, b) in batch order = unique position
    disj_q_idx = {}     # di -> [unique batch positions]
    for di, w, b in queries:
        w_arr = np.asarray(w, dtype=np.float64).flatten()
        key = (w_arr.tobytes(), round(float(b), 12))
        pos = uniq_key_to_pos.get(key)
        if pos is None:
            pos = len(uniq_ew)
            uniq_key_to_pos[key] = pos
            uniq_ew.append(
                (torch.as_tensor(w_arr, dtype=dtype, device=device), float(b)))
        disj_q_idx.setdefault(di, []).append(pos)
    # spec_ew keys are 0..Q-1, so the backward's sorted-key batch order IS the
    # unique position order — disj_q_idx already indexes the batch directly.
    spec_ew = {pos: ew for pos, ew in enumerate(uniq_ew)}

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
    # CHUNK-based worklist: each entry is a 2D (m, n_in) GPU tensor, not a
    # single row. The old per-row list (extend(xl_u.unbind(0)) on push,
    # torch.stack(wl[-B:]) on pop) churned MILLIONS of 1-row tensors through
    # Python on lsnc's huge queues — ~40% of wall (split+stack). Pushing and
    # popping whole chunks keeps the data on-GPU and the Python list short
    # (~2 chunks per iteration). `_wl_size` is O(#chunks) (a few hundred),
    # negligible. Correctness is identical — same boxes, LIFO order.
    worklist_xl = [xl0.reshape(1, -1)]
    worklist_xh = [xh0.reshape(1, -1)]

    def _wl_size(wl):
        return sum(int(c.shape[0]) for c in wl)

    def _wl_pop(wl_xl, wl_xh, n):
        """Pop exactly `n` rows from the END (LIFO). Caller guarantees
        n <= total size. Returns (xl_batch, xh_batch) 2D tensors."""
        got = 0
        parts_xl = []
        parts_xh = []
        while got < n:
            cxl = wl_xl[-1]
            cxh = wl_xh[-1]
            m = int(cxl.shape[0])
            if got + m <= n:
                parts_xl.append(cxl)
                parts_xh.append(cxh)
                wl_xl.pop()
                wl_xh.pop()
                got += m
            else:
                need = n - got
                parts_xl.append(cxl[-need:])
                parts_xh.append(cxh[-need:])
                wl_xl[-1] = cxl[:-need]
                wl_xh[-1] = cxh[:-need]
                got = n
        xl_b = parts_xl[0] if len(parts_xl) == 1 else torch.cat(parts_xl, 0)
        xh_b = parts_xh[0] if len(parts_xh) == 1 else torch.cat(parts_xh, 0)
        return xl_b, xh_b
    # Branch-aware SB precomputation: for parallel architectures
    # (pensieve_big_parallel), our backward-CROWN lA underestimates the
    # impact of input dims feeding parallel branches with non-linear
    # downstream ops (pow, div_bilinear). ABC's SB picks ALL K splits in
    # the single most-unstable branch; ours picks across branches and
    # misses the high-impact one. Pre-compute a one-time dim → dominant-
    # shallow-ReLU mapping (by halving each varying dim once at root and
    # seeing which shallow ReLU's pre-activation width drops most). Then
    # boost SB scores by `n_unstable_in_dominant_branch` so dims feeding
    # the unstable-heavy branch outrank dims feeding stable branches.
    _branch_boost = None  # (n_in,) per-dim multiplier; None disables
    import os as _os_bb
    # Exponent on the (1+n_unstable) boost. Default from settings (2.0 for
    # the pensieve signature, 1.0 elsewhere); BRANCH_BOOST_EXP overrides.
    _bb_exp = float(_os_bb.environ.get(
        'BRANCH_BOOST_EXP',
        str(getattr(settings, 'input_split_batched_branch_boost_exp', 1.0))))
    if (bool(getattr(settings, 'input_split_batched_branch_boost', True))
            and int(getattr(settings, 'bab_split_depth', 1)) > 1):
        try:
            _root_widths_per_var = (xh0 - xl0)
            _varying_mask = _root_widths_per_var > 1e-9
            _vary_idx = _varying_mask.nonzero(as_tuple=True)[0]
            # Forward zono once at root (already cheap).
            with torch.no_grad():
                _sb_root, _ = _forward_zonotope_graph_batched(
                    xl0.unsqueeze(0), xh0.unsqueeze(0), gg, device, dtype)
                # Restrict to "shallow" relu layers (closer to input).
                # Heuristic: layers with layer_idx in first half of relus.
                # The forward also stashes string-keyed bilinear/pool boxes
                # (e.g. '<name>__mul_bilinear_box') for backward McCormick;
                # keep only int ReLU layer indices so sorted() doesn't choke
                # on mixed str/int keys (lsnc_relu's quadratic Lyapunov forms
                # add such string boxes).
                _all_Ls = sorted(L for L in _sb_root.keys()
                                 if isinstance(L, int))
                _shallow_cutoff = max(1, len(_all_Ls) // 2)
                _shallow_Ls = _all_Ls[:_shallow_cutoff]
                _root_w = {L: ((_sb_root[L][1][0] - _sb_root[L][0][0])
                                 .mean().item())
                            for L in _shallow_Ls}
                # Per varying dim, find dominant shallow L.
                _dim_to_L = {}
                for _d_t in _vary_idx:
                    _d = int(_d_t.item())
                    _xh_h = xh0.clone()
                    _xh_h[_d] = xl0[_d] + 0.5 * (xh0[_d] - xl0[_d])
                    _sb_h, _ = _forward_zonotope_graph_batched(
                        xl0.unsqueeze(0), _xh_h.unsqueeze(0), gg, device, dtype)
                    _best_L = None; _best_delta = 1e-5
                    for _L in _shallow_Ls:
                        _new_w = ((_sb_h[_L][1][0] - _sb_h[_L][0][0])
                                   .mean().item())
                        _delta = _root_w[_L] - _new_w
                        if _delta > _best_delta:
                            _best_delta = _delta; _best_L = _L
                    if _best_L is not None:
                        _dim_to_L[_d] = _best_L
                # n_unstable per shallow L at root.
                _n_unst_per_L = {
                    L: int(((_sb_root[L][0][0] < 0)
                             & (_sb_root[L][1][0] > 0)).sum().item())
                    for L in _shallow_Ls}
                # Boost factor per dim: 1 + n_unstable_in_dominant_branch.
                _branch_boost = torch.ones(n_in, dtype=dtype, device=device)
                for _d, _L in _dim_to_L.items():
                    _branch_boost[_d] = (
                        1.0 + float(_n_unst_per_L.get(_L, 0))) ** _bb_exp
                if bool(getattr(settings, 'print_progress', False)):
                    print(f'[branch-boost] n_unstable per shallow L: '
                          f'{_n_unst_per_L}', flush=True)
                del _sb_root
        except (RuntimeError, NotImplementedError):
            _branch_boost = None

    n_leaves_visited = 0
    n_iters = 0
    n_open_at_timeout = 0

    import os as _os_isb_dbg
    _dbg_isb = (_os_isb_dbg.environ.get('DEBUG_INPUT_SPLIT_BATCHED', '') == '1')
    # Per-phase profiler (bound / clip / split timing + closure counts).
    # Gated by `settings.input_split_batched_phase_timing` (env
    # VC_PHASE_TIMING forces it on for ad-hoc profiling). Every timing site
    # is wrapped in `if _phase_timing`, so when OFF the only added cost is
    # one boolean test per iteration — no perf_counter, sync, or dict work.
    _phase_timing = (
        bool(getattr(settings, 'input_split_batched_phase_timing', False))
        or _os_isb_dbg.environ.get('VC_PHASE_TIMING', '') == '1')
    _pt = ({'bound': 0.0, 'clip': 0.0, 'split': 0.0,
            'crown_closed': 0, 'clip_closed': 0,
            'clip_shrink_sum': 0.0, 'clip_shrink_n': 0}
           if _phase_timing else None)
    _pt_t0 = time.perf_counter() if _phase_timing else 0.0

    def _pt_mark(key, t_ref):
        # Only invoked from inside `if _phase_timing` guards.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _pt[key] += time.perf_counter() - t_ref
        return time.perf_counter()
    while worklist_xl:
        _wl_n = _wl_size(worklist_xl)
        if time.perf_counter() - t_start > total_budget - 0.5:
            n_open_at_timeout = _wl_n
            break
        # Pop a batch (LIFO — depth-first) of whole chunks.
        B = min(batch_size, _wl_n)
        if _dbg_isb and (n_iters < 20 or n_iters % 50 == 0):
            print(f'[is-batched] iter={n_iters} worklist={_wl_n} '
                  f'B={B} leaves_visited={n_leaves_visited} '
                  f't_elapsed={time.perf_counter()-t_start:.2f}s',
                  flush=True)
        xl_batch, xh_batch = _wl_pop(worklist_xl, worklist_xh, B)  # (B, n_in)

        # Batched bound.
        if _phase_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _pt_tb = time.perf_counter()
        try:
            if crown_intermediate:
                # AB-CROWN-style backward-CROWN intermediate bounds (~2x tighter
                # than forward zono for ACAS Xu's amplifying weights -> far fewer
                # leaves). Sound (intersection of two over-approximations).
                from .verify_zono_bnb import _crown_intermediate_batched
                sb_b = _crown_intermediate_batched(
                    gg, xl_batch, xh_batch, device, dtype)
            else:
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
                # batch=1 still fails. A genuine OOM here is a resource
                # failure, not an "undecided" verdict: per the project
                # convention (never silently swallow OOM) re-raise it so it
                # surfaces as `error`, unless the caller opted into tolerating
                # OOM (raise_on_oom=False). A non-OOM RuntimeError stays a soft
                # `unknown` (can be a benign solver/shape edge on this leaf).
                if (isinstance(_e, torch.cuda.OutOfMemoryError)
                        and getattr(settings, 'raise_on_oom', True)):
                    raise
                return 'unknown', {'phase': 'batched_oom',
                                    'batch_size': batch_size}
            # Push the popped batch back as a chunk and retry with a
            # smaller batch_size.
            worklist_xl.append(xl_batch); worklist_xh.append(xh_batch)
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
        # `input_split_batched_alpha_all_leaves` (boolean, default OFF) applies
        # the batched α-CROWN to EVERY unclosed leaf in the batch, not just the
        # eps-boundary band. For deep nets whose per-leaf backward-CROWN lb is
        # far below 0 (tllverifybench: -30..-180 — nowhere near the eps band),
        # the boundary-only α never fires and the 2-D input-split explodes
        # (45k+ leaves vs AB-CROWN's ~369). Tightening every unclosed leaf cuts
        # that ~200×. ON per-config (cleaner than tuning a magnitude eps; may
        # cost on benchmarks where the boundary heuristic was already enough,
        # hence default OFF). The batch size B bounds the per-iteration cost.
        alpha_all = bool(getattr(
            settings, 'input_split_batched_alpha_all_leaves', False))
        alpha_eps = float(getattr(
            settings, 'input_split_batched_alpha_boundary_eps', 0.0))
        alpha_max_iters = int(getattr(
            settings, 'input_split_batched_alpha_iters', 0))
        alpha_max_leaves = int(getattr(
            settings, 'input_split_batched_alpha_max_leaves', 200))
        if (alpha_max_iters > 0 and (alpha_all or alpha_eps > 0)
                and not all_disj_closed.all()):
            from .verify_zono_bnb import _run_alpha_crown_inputsplit_batched
            best_per_disj_lb = []
            for di, q_idxs in disj_q_idx.items():
                best_per_disj_lb.append(spec_lbs_b[:, q_idxs].max(dim=1).values)
            worst_disj_best = torch.stack(best_per_disj_lb, dim=1).min(dim=1).values
            if alpha_all:
                close_mask = ~all_disj_closed
            else:
                close_mask = (~all_disj_closed) & (worst_disj_best > -alpha_eps)
            close_idx = close_mask.nonzero(as_tuple=True)[0]
            if close_idx.numel() > 0:
                if not alpha_all:
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
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # α-CROWN inputsplit on boundary leaves: GPU runtime
                    # or OOM. Keep prior spec_lbs_b — α-CROWN is an
                    # opportunistic tightener, not load-bearing.
                    pass
                disj_closed_masks = []
                for di, q_idxs in disj_q_idx.items():
                    disj_closed_masks.append(
                        (spec_lbs_b[:, q_idxs] > 0).any(dim=1))
                all_disj_closed = torch.stack(
                    disj_closed_masks, dim=1).all(dim=1)

        if _phase_timing:
            _pt_ts = _pt_mark('bound', _pt_tb)
        unclosed = (~all_disj_closed).nonzero(as_tuple=True)[0]
        n_leaves_visited += B
        n_verified_by_crown_iter = int(all_disj_closed.sum().item())

        # Leaf-level SAT search. Phase-0 PGD on the WIDE root box misses a narrow
        # SAT witness (acasxu 1_5/1_9 prop_2/prop_7: the witness fills a tiny
        # fraction of the root, and even 200k random restarts can't find it). The
        # witness-containing leaf holds a real SAT point so CROWN can NEVER close
        # it -> its margin stays the most-negative and it keeps splitting until
        # narrow, where PGD inside that leaf's box finds it. Every `_every` iters,
        # batched-PGD the leaves with the WORST (most-negative worst-disjunct)
        # margin (the hybrid's proven selection — targets the witness leaf
        # directly, vs narrowest which misses multi-disjunct prop_7). Skipped
        # under disable_sat_finding (the soundness probe).
        leaf_pgd_every = int(getattr(
            settings, 'input_split_leaf_pgd_every', 0))
        if (leaf_pgd_every > 0 and unclosed.numel() > 0
                and not bool(getattr(settings, 'disable_sat_finding', False))
                and n_iters % leaf_pgd_every == 0):
            from .verify_hybrid_acasxu import _simple_pgd_batched
            lp_max = int(getattr(
                settings, 'input_split_leaf_pgd_max_leaves', 64))
            # worst-disjunct best-query lb per unclosed leaf; most negative first.
            per_disj_lb = [spec_lbs_b[unclosed][:, q_idxs].max(dim=1).values
                           for q_idxs in disj_q_idx.values()]
            worst_disj = torch.stack(per_disj_lb, dim=1).min(dim=1).values
            sel = unclosed[worst_disj.argsort()[:lp_max]]
            sat_l, w_l = _simple_pgd_batched(
                xl_batch[sel], xh_batch[sel], spec, gg, n_output, device, dtype,
                n_restarts=int(getattr(
                    settings, 'input_split_leaf_pgd_restarts', 128)),
                n_iter=int(getattr(settings, 'input_split_leaf_pgd_iters', 50)))
            if sat_l:
                return 'sat', {'phase': 'batched_leaf_pgd', 'witness': w_l}

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
            if _phase_timing:
                _wb = (xh_u - xl_u).clamp(min=0).sum()
                _wa = (xh_c - xl_c).clamp(min=0).sum()
                _pt['clip_shrink_sum'] += float(
                    (_wa / _wb.clamp(min=1e-12)).item())
                _pt['clip_shrink_n'] += 1
            # Only feasible-after-clip leaves continue to split.
            feasible_idx = feasible_c.nonzero(as_tuple=True)[0]
            if feasible_idx.numel() > 0:
                xl_split = xl_c[feasible_idx]
                xh_split = xh_c[feasible_idx]
            else:
                xl_split = xl_u[:0]
                xh_split = xh_u[:0]
            # (Removed: a clip -> re-CROWN inner-cycle path gated by
            # `input_split_batched_clip_recrown_cycles`. It was UNSOUND on
            # bilinear graphs — with cycles>0 it false-verified every SAT
            # case at the root in 1 leaf (the re-CROWN closure declared the
            # whole box safe). No config ever enabled it (default 0), so it
            # was dead AND unsound; removed rather than masked.)
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
                    # Per-disjunct best-lb query ordering. disj_q_idx maps
                    # each disjunct to its (deduplicated) unique-query batch
                    # positions; spec_ew is keyed by that same position.
                    per_disj_queries = []
                    for di, positions in disj_q_idx.items():
                        q_order = sorted(
                            positions,
                            key=lambda pos: -spec_lbs_b[full_idx, pos].item())
                        ordered = [(
                            spec_ew[pos][0].cpu().numpy().astype(np.float64),
                            float(spec_ew[pos][1])) for pos in q_order]
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

        if _phase_timing:
            _pt_ts = _pt_mark('clip', _pt_ts)
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
            # SB scoring stays ON GPU — a per-iteration `.cpu()` sync here
            # was ~30% of wall on lsnc's millions of small leaves (every
            # iteration blocked on a host transfer of widths+sens). The
            # split-dim argmax/topk run on GPU; only the final indices feed
            # the GPU gather/scatter below, no host round-trip.
            widths = (xh_split - xl_split)
            if sb_enabled and A_lin is not None:
                # Recover surviving indices in xl_split's coordinate
                # frame. `unclosed`→A_u; clip restricts to feasible_idx
                # of that; xl_split lines up with feasible_idx.
                if clip_enabled and unclosed.numel() > 0 and A_lin is not None:
                    A_split = A_u[feasible_idx]
                else:
                    A_split = A_lin[unclosed]
                # Per-leaf sensitivity: sum_q |A_q[i]| × width_i.
                # Optional branch_boost (computed at init): boosts dims
                # feeding the most-unstable shallow ReLU branch. Without
                # this, backward-CROWN through pow/div_bilinear can
                # underestimate L2-branch lA on pensieve_big_parallel and
                # pick all 4 splits in less-unstable L0/L1/L3 branches,
                # missing the bound improvement (mean -7.42 vs +0.16).
                sens = A_split.abs().sum(dim=1)  # (B, n_in)
                scores = widths * sens
                if _branch_boost is not None:
                    scores = scores * _branch_boost.to(scores.device).unsqueeze(0)
                ax = scores.argmax(dim=1)
            else:
                ax = widths.argmax(dim=1)
            import os as _os_split_dbg
            if _os_split_dbg.environ.get('DEBUG_BAB_SPLIT', '') == '1':
                for _i in range(min(xl_split.shape[0], 5)):
                    _a = int(ax[_i].item())
                    _xl = float(xl_split[_i, _a])
                    _xh = float(xh_split[_i, _a])
                    print(f'[bab-split] leaf={_i} split dim X_{_a} '
                          f'(width={_xh-_xl:.4f}) at mid={(_xl+_xh)/2:.4f}',
                          flush=True)
            # K-dim simultaneous split — ABC-style. When the queue is
            # small (< min_batch_size), grow it by splitting on top-K
            # widest×sensitive dims per leaf at once → 2^K children.
            # K=1 reproduces the historical single-widest-dim behavior.
            # Adaptive K like ABC's `get_split_depth` would be better,
            # but a fixed `bab_split_depth` is enough to test the
            # hypothesis on pensieve_big_parallel (32 free dims → K=4
            # gives 16 children per leaf, reaching depth-24 box volume
            # in 6 iters instead of 24).
            K_max = int(getattr(settings, 'bab_split_depth', 1))
            # Adaptive arity: K>1 (2^K children/leaf) exists only to GROW a
            # SMALL queue up to the batch size fast (pensieve starts with a
            # handful of leaves). Its per-leaf Python loop below is fine for
            # a small queue but catastrophic once the queue is huge — it was
            # 97% of wall on lsnc's million-leaf queues (22s split vs 0.5s
            # bound). Once the remaining worklist already exceeds the batch
            # size, the queue is full, so drop to K=1 and take the vectorized
            # GPU 2-way split. Preserves pensieve's grow-the-queue behavior
            # (queue small → K_max) while uncapping lsnc throughput.
            # Gate on the size of the batch we just popped (B), NOT the
            # post-pop remaining worklist: when B == batch_size the queue was
            # already full before the pop, so we are in steady state and K=1
            # (vectorized GPU split) suffices. Using the post-pop size was a
            # bug — a leaf set that pops the WHOLE queue each iteration leaves
            # 0 behind and would stay on the slow K>1 Python loop forever.
            if B >= batch_size:
                K_max = 1
            # Pick top-K dims per leaf via the same width×sens score.
            if K_max > 1 and sb_enabled and A_lin is not None:
                top_k_dims = torch.topk(scores, K_max, dim=1).indices  # (B,K)
            else:
                top_k_dims = ax.unsqueeze(1)  # (B, 1)
            K_eff = top_k_dims.shape[1]
            n = xl_split.shape[0]
            if K_eff == 1:
                # Vectorized on-GPU 2-way split (single split dim per leaf).
                # The per-child Python loop + `.cpu()`/`.to(device)` round-trip
                # below was ~80% of wall on acasxu's millions of leaves (every
                # child crossed the PCIe bus). Here both children of every leaf
                # are built in a handful of GPU ops, no host transfer. Children
                # are appended low-half-batch then high-half-batch; the order
                # differs from the per-leaf interleave but BaB correctness is
                # order-independent and SAT-finding is handled by the worst-margin
                # leaf-PGD (not split order) — the regression that reverted this
                # before. K>1 (pensieve) keeps the Python loop below.
                # top_k_dims/ax are built from `widths` which is on CPU — move
                # the split-dim index to xl_split's device for the GPU gather.
                ax1 = top_k_dims[:, 0].to(xl_split.device)  # (n,) split dim/leaf
                lo_ax = xl_split.gather(1, ax1.unsqueeze(1)).squeeze(1)
                hi_ax = xh_split.gather(1, ax1.unsqueeze(1)).squeeze(1)
                usable = (hi_ax - lo_ax) > 1e-12
                n_open_at_timeout += int((~usable).sum().item())
                if usable.any():
                    u = usable.nonzero(as_tuple=True)[0]
                    xl_u = xl_split[u]
                    xh_u = xh_split[u]
                    ax_u = ax1[u].unsqueeze(1)
                    mid_u = ((lo_ax[u] + hi_ax[u]) * 0.5).unsqueeze(1)
                    # low child = [xl_u, xh with ax=mid]; high = [xl with ax=mid, xh_u]
                    xh_low = xh_u.clone().scatter_(1, ax_u, mid_u)
                    xl_high = xl_u.clone().scatter_(1, ax_u, mid_u)
                    # Append both child sets as 2D chunks (no per-row unbind).
                    worklist_xl.append(xl_u); worklist_xh.append(xh_low)
                    worklist_xl.append(xl_high); worklist_xh.append(xh_u)
            else:
                top_k_dims_cpu = top_k_dims.cpu()
                xl_cpu = xl_split.cpu()
                xh_cpu = xh_split.cpu()
                n_children = 1 << K_eff
                _kids_xl = []
                _kids_xh = []
                for i in range(n):
                    xl_i = xl_cpu[i]
                    xh_i = xh_cpu[i]
                    # Which of the K dims actually have width > 0?
                    usable_k = []
                    mids_k = []
                    for k in range(K_eff):
                        dk = int(top_k_dims_cpu[i, k].item())
                        if float(xh_i[dk] - xl_i[dk]) > 1e-12:
                            usable_k.append(dk)
                            mids_k.append(float((xl_i[dk] + xh_i[dk]) / 2))
                    if not usable_k:
                        n_open_at_timeout += 1
                        continue
                    K_u = len(usable_k)
                    # Enumerate 2^K_u children.
                    for combo in range(1 << K_u):
                        xl_c = xl_i.clone()
                        xh_c = xh_i.clone()
                        for k_idx, dk in enumerate(usable_k):
                            if (combo >> k_idx) & 1:
                                xl_c[dk] = mids_k[k_idx]  # high half
                            else:
                                xh_c[dk] = mids_k[k_idx]  # low half
                        _kids_xl.append(xl_c)
                        _kids_xh.append(xh_c)
                # Stack all children of this batch into one 2D chunk (the
                # K>1 path is only used for SMALL queues — pensieve's grow
                # phase — so a single stack per iter is cheap).
                if _kids_xl:
                    worklist_xl.append(
                        torch.stack(_kids_xl, 0).to(device, dtype=dtype))
                    worklist_xh.append(
                        torch.stack(_kids_xh, 0).to(device, dtype=dtype))
            if _wl_size(worklist_xl) > max_worklist:
                return 'unknown', {
                    'phase': 'batched_worklist_overflow',
                    'batched_n_leaves': n_leaves_visited,
                    'batched_n_iters': n_iters,
                    'worklist_size': _wl_size(worklist_xl)}
        if _phase_timing:
            _pt['crown_closed'] += n_verified_by_crown_iter
            _pt['clip_closed'] += n_verified_by_clip_iter
            _pt_ts = _pt_mark('split', _pt_ts)
        n_iters += 1

    if _phase_timing:
        _tot = time.perf_counter() - _pt_t0
        _acct = _pt['bound'] + _pt['clip'] + _pt['split']
        _shrink = (_pt['clip_shrink_sum'] / _pt['clip_shrink_n']
                   if _pt['clip_shrink_n'] else float('nan'))
        print(f'[vc-phase] BAB total={_tot:.2f}s | '
              f"bound={_pt['bound']:.2f}s ({100*_pt['bound']/_tot:.0f}%) "
              f"clip={_pt['clip']:.2f}s ({100*_pt['clip']/_tot:.0f}%) "
              f"split={_pt['split']:.2f}s ({100*_pt['split']/_tot:.0f}%) "
              f"other={_tot-_acct:.2f}s ({100*(_tot-_acct)/_tot:.0f}%) | "
              f'iters={n_iters} leaves={n_leaves_visited}', flush=True)
        print(f'[vc-phase] closures: crown={_pt["crown_closed"]} '
              f'clip={_pt["clip_closed"]} | clip avg post/pre box-width '
              f'ratio={_shrink:.4f} (1.0=no shrink) over '
              f'{_pt["clip_shrink_n"]} clip-iters', flush=True)

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
        'batched_worklist_left': _wl_size(worklist_xl),
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
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Fast-leaf phase-0 PGD: GPU runtime/OOM. Non-fatal —
            # falls through to per-leaf split.
            pass

    def _run_node(s):
        remaining = total_budget - (time.perf_counter() - t_start)
        if remaining < 0.5:
            return 'unknown', {'phase': 'input_split_timeout', 'timed_out': True}
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
    import os as _os_bab_dbg
    _dbg_bab = (_os_bab_dbg.environ.get('DEBUG_INPUT_SPLIT_BAB', '') == '1')
    _node_times = []  # (depth, wall, verdict)
    _t0_bab = time.perf_counter()

    def _bab(s, depth):
        n_nodes[0] += 1
        _t_node = time.perf_counter()
        r, d = _run_node(s)
        _node_times.append((depth, time.perf_counter() - _t_node, r))
        if _dbg_bab and (n_nodes[0] <= 30 or n_nodes[0] % 10 == 0):
            print(f'[is-bab] node {n_nodes[0]} depth={depth} '
                  f'wall={_node_times[-1][1]*1000:.0f}ms '
                  f'r={r} t_elapsed={time.perf_counter()-_t0_bab:.1f}s',
                  flush=True)
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
            except (RuntimeError, KeyError, ValueError):
                # Score-input-axes failure: GPU runtime, missing key in
                # gpu_graph, or shape issue. Fall back to width-based
                # axis picking (always correct, just less informed).
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
