"""Dense generator-based LP/MILP for spec verification.

Separates the linear transformation (G matrix per op) from the domain
(generators e). Produces a much smaller LP model than per-neuron builders:
for CIFAR100 ResNet medium ~4K vars vs ~104K for _build_optimized, with
identical LP triangle bounds. Uses GPU batched conv for G propagation.

Formulation:
  input:   y_i = c_i + G_i @ e  where e_i ∈ [-1, 1]
  conv/fc: G_new = W @ G_prev; c_new = W @ c_prev + b
  relu stable-on:  passthrough (G row unchanged)
  relu dead:       zero row
  relu unstable j: introduce e_new_j ∈ [0, hi_j] with triangle constraints
                   (or big-M if in milp_set)
  merge-Add:       G_new = G_a + G_b (padded to same columns)

See the MEMORY entry on GOLD LD sign convention for related context.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import gurobipy as grb

from .gurobi_util import optimize_checked


def forward_point(gg_ops_ser, x, input_name, output_op_name):
    """Evaluate the network at a single input point x (numpy float64).

    Supports the op types that appear in our serialized graph: conv, fc,
    relu, add (with or without is_merge), reshape. Returns numpy array.
    """
    vals = {input_name: np.asarray(x, dtype=np.float64)}
    for op in gg_ops_ser:
        nm = op['name']
        t = op['type']
        if t == 'conv':
            a = vals[op['inputs'][0]]
            C_in, H_in, W_in = op['in_shape']
            at = torch.from_numpy(a.reshape(1, C_in, H_in, W_in))
            k = torch.from_numpy(op['kernel_np']).to(at.dtype)
            b = torch.from_numpy(op['bias_np']).to(at.dtype)
            sH, sW = op['stride']; pH, pW = op['padding']
            y = F.conv2d(at, k, bias=b, stride=(sH, sW),
                         padding=(pH, pW)).flatten()
            vals[nm] = y.numpy()
        elif t == 'fc':
            a = vals[op['inputs'][0]]
            W = op['W_np']; b = op['bias_np']
            vals[nm] = W @ a + b
        elif t == 'relu':
            a = vals[op['inputs'][0]]
            vals[nm] = np.maximum(a, 0.0)
        elif t == 'add':
            a = vals[op['inputs'][0]]
            if op.get('is_merge'):
                b = vals[op['inputs'][1]]
                vals[nm] = a + b
            else:
                vals[nm] = a
        elif t == 'reshape':
            vals[nm] = vals[op['inputs'][0]]
        elif t == 'sub':
            a = vals[op['inputs'][0]]
            b = op.get('bias')
            vals[nm] = (a - np.asarray(b, dtype=np.float64).flatten()
                        if b is not None else a)
        else:
            raise NotImplementedError(
                f'forward_point: unsupported op {t!r}')
    return vals[output_op_name]


def precompute_gen_state(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                         output_op_name, *,
                         device='cuda', dtype=torch.float64,
                         formulation='sparse'):
    """Compute G/center arrays on GPU, return numpy-only state.

    The returned state is query-independent (no qw/qb) and milp_set-independent.
    It can be reused for all Phase 7/8 solves on the same (network, bbr) —
    eliminating the GPU-nondeterminism that previously made each build_gen_lp
    call produce slightly different coefficients.

    formulation:
      'dense' — every stable-on ReLU neuron passes through via the raw G
          row. Subsequent conv W @ G_prev multiplies these dense rows, so
          G accumulates nonzeros through conv layers and the coefficient
          range compounds (can span 13 orders of magnitude on a 10-layer
          ResNet). Fewer Gurobi vars but numerically fragile — Gurobi
          can silently certify wrong bounds on this formulation.

      'sparse' (default) — only stable-on neurons at the LAST hidden
          ReLU layer get a new generator variable bound to
          `v == c + G_in[j,:]@e`. Earlier layers still use dense
          passthrough (they're cheap and stay in the conv-compounded G).
          This "cut once, at the spec boundary" structure keeps the
          simplex basis away from the tiny cascaded-G coefficients that
          appear only in equality rows, while adding minimal overhead
          (~tens of vars for a typical network).

    Returns state dict with:
      - formulation: 'dense' or 'sparse'
      - n_input: number of input generators
      - n_gens:  total number of generators (input + unstable new gens [+ stable new gens])
      - unstable_list: list of per-unstable-neuron dicts with:
          layer_idx, neuron_idx, c_in (float), lo, hi, e_new_col,
          row_indices (np.int32), row_values (np.float64)
      - stable_list: (sparse only) list of per-stable-on dicts with same keys
          as unstable_list, with lo >= 0; build adds `v == c + G@e` and
          var bounds [lo, hi].
      - obj_c_out: np.ndarray of output centers
      - obj_G_out_csr: scipy.sparse.csr_matrix output G
    """
    gpu = torch.device(device)
    last_use = {}
    last_relu_idx = -1
    for i, op in enumerate(gg_ops_ser):
        for inp in op['inputs']:
            last_use[inp] = i
        if op['type'] == 'relu' and 'layer_idx' in op:
            last_relu_idx = max(last_relu_idx, op['layer_idx'])

    n_in = len(x_lo)
    center = {input_name: torch.tensor((x_hi + x_lo) / 2, dtype=dtype, device=gpu)}
    half_width = torch.tensor((x_hi - x_lo) / 2, dtype=dtype, device=gpu)
    G_by_op = {input_name: torch.diag(half_width)}
    n_gens = n_in
    unstable_list = []
    stable_list = []  # used only if formulation == 'sparse'

    def pad_cols(G):
        if G.shape[1] < n_gens:
            return torch.cat([G, torch.zeros(
                G.shape[0], n_gens - G.shape[1], dtype=dtype, device=gpu)],
                dim=1)
        return G

    for op_idx, op in enumerate(gg_ops_ser):
        nm = op['name']
        t = op['type']

        if t == 'conv':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            C_in, H_in, W_in = op['in_shape']
            kernel = torch.tensor(op['kernel_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            sH, sW = op['stride']
            pH, pW = op['padding']
            c_img = prev_c.reshape(1, C_in, H_in, W_in)
            c_out = F.conv2d(c_img, kernel, bias=bias,
                             stride=(sH, sW), padding=(pH, pW)).flatten()
            g_img = prev_G.t().contiguous().reshape(n_gens, C_in, H_in, W_in)
            g_out = F.conv2d(g_img, kernel, bias=None,
                             stride=(sH, sW), padding=(pH, pW))
            g_out = g_out.reshape(n_gens, -1).t().contiguous()
            center[nm] = c_out
            G_by_op[nm] = g_out

        elif t == 'fc':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            W = torch.tensor(op['W_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            center[nm] = W @ prev_c + bias
            G_by_op[nm] = W @ prev_G

        elif t == 'relu':
            if 'layer_idx' not in op:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]
                continue
            li = op['layer_idx']
            lo_r, hi_r = bounds_by_relu[li]
            c_in = center[op['inputs'][0]]
            G_in = pad_cols(G_by_op[op['inputs'][0]])
            n = len(c_in)
            dead_mask = hi_r <= 0
            stable_mask = (lo_r >= 0) & ~dead_mask
            unstable_mask = ~dead_mask & ~stable_mask
            unstable_idx = np.where(unstable_mask)[0]
            stable_idx = np.where(stable_mask)[0]

            do_sparse = (formulation == 'sparse' and li == last_relu_idx)
            c_in_cpu = c_in.detach().cpu().numpy()
            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(unstable_idx, device=gpu, dtype=torch.long)
                G_unstable = G_in[uidx_t].detach().cpu().numpy()
            if do_sparse and len(stable_idx) > 0:
                sidx_t_cpu = torch.tensor(stable_idx, device=gpu, dtype=torch.long)
                G_stable = G_in[sidx_t_cpu].detach().cpu().numpy()

            # Unstable neurons always get a new gen
            for local_idx, j in enumerate(unstable_idx):
                row = G_unstable[local_idx]
                nz = np.nonzero(row)[0]
                unstable_list.append({
                    'layer_idx': li,
                    'neuron_idx': int(j),
                    'c_in': float(c_in_cpu[j]),
                    'lo': float(lo_r[j]),
                    'hi': float(hi_r[j]),
                    'e_new_col': n_gens + local_idx,
                    'row_indices': nz.astype(np.int32),
                    'row_values': row[nz].astype(np.float64),
                })
            n_unstable = len(unstable_idx)

            # Sparse: stable-on neurons also get a new gen + equality constraint
            n_stable_new = 0
            if do_sparse:
                for local_idx, j in enumerate(stable_idx):
                    row = G_stable[local_idx]
                    nz = np.nonzero(row)[0]
                    stable_list.append({
                        'layer_idx': li,
                        'neuron_idx': int(j),
                        'c_in': float(c_in_cpu[j]),
                        'lo': float(lo_r[j]),
                        'hi': float(hi_r[j]),
                        'e_new_col': n_gens + n_unstable + local_idx,
                        'row_indices': nz.astype(np.int32),
                        'row_values': row[nz].astype(np.float64),
                    })
                n_stable_new = len(stable_idx)

            n_new = n_unstable + n_stable_new

            # Build output G on GPU
            G_out = torch.zeros(n, n_gens + n_new, dtype=dtype, device=gpu)
            if not do_sparse and len(stable_idx) > 0:
                # Dense: stable-on rows pass through the prior G row
                sidx_t = torch.tensor(stable_idx, device=gpu, dtype=torch.long)
                G_out[sidx_t, :n_gens] = G_in[sidx_t]
            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(unstable_idx, device=gpu, dtype=torch.long)
                new_col_idx = torch.arange(n_gens, n_gens + n_unstable,
                                            device=gpu, dtype=torch.long)
                G_out[uidx_t, new_col_idx] = 1.0
            if do_sparse and len(stable_idx) > 0:
                # Sparse: stable-on row becomes a single 1.0 at its new col
                sidx_t = torch.tensor(stable_idx, device=gpu, dtype=torch.long)
                stable_cols = torch.arange(
                    n_gens + n_unstable, n_gens + n_new,
                    device=gpu, dtype=torch.long)
                G_out[sidx_t, stable_cols] = 1.0
            c_out = torch.zeros(n, dtype=dtype, device=gpu)
            if not do_sparse and len(stable_idx) > 0:
                # Dense: stable-on center passes through
                c_out[sidx_t] = c_in[sidx_t]
            # Sparse: stable-on center is absorbed into the equality constraint
            # (v == c + G@e), so c_out[j] stays 0 and the var carries the full value
            center[nm] = c_out
            G_by_op[nm] = G_out
            n_gens += n_new

        elif t == 'add':
            if op.get('is_merge'):
                ca = center[op['inputs'][0]]
                cb = center[op['inputs'][1]]
                Ga = pad_cols(G_by_op[op['inputs'][0]])
                Gb = pad_cols(G_by_op[op['inputs'][1]])
                center[nm] = ca + cb
                G_by_op[nm] = Ga + Gb
            else:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'sub':
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            b = op.get('bias')
            if b is not None:
                bt = torch.tensor(b.flatten(), dtype=dtype, device=gpu)
                center[nm] = prev_c - bt
            else:
                center[nm] = prev_c
            G_by_op[nm] = prev_G

        elif t == 'reshape':
            center[nm] = center[op['inputs'][0]]
            G_by_op[nm] = G_by_op[op['inputs'][0]]

        for inp in op['inputs']:
            if (last_use.get(inp) == op_idx and inp in G_by_op
                    and inp != nm):
                del G_by_op[inp]
                del center[inp]

    if device == 'cuda':
        torch.cuda.synchronize()

    c_out = center[output_op_name]
    G_out = G_by_op[output_op_name]
    if G_out.shape[1] < n_gens:
        G_out = torch.cat([G_out, torch.zeros(
            G_out.shape[0], n_gens - G_out.shape[1],
            dtype=dtype, device=gpu)], dim=1)
    obj_c_out = c_out.detach().cpu().numpy().astype(np.float64)
    obj_G_out_dense = G_out.detach().cpu().numpy().astype(np.float64)
    obj_G_out_csr = sp.csr_matrix(obj_G_out_dense)

    G_by_op.clear()
    center.clear()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'n_input': n_in,
        'n_gens': n_gens,
        'formulation': formulation,
        'unstable_list': unstable_list,
        'stable_list': stable_list,
        'obj_c_out': obj_c_out,
        'obj_G_out_csr': obj_G_out_csr,
        # For witness forward-pass: cheap to carry, network-level not per-query.
        'gg_ops_ser': gg_ops_ser,
        'input_name': input_name,
        'output_op_name': output_op_name,
        'x_lo': np.asarray(x_lo, dtype=np.float64),
        'x_hi': np.asarray(x_hi, dtype=np.float64),
    }


def build_gen_lp_from_state(state, qw, qb, *, milp_set=None, n_threads=1):
    """Build Gurobi LP/MILP from a precomputed state (numpy-only).

    Same return as build_gen_lp:  (model, env, unstable_info, obj_coef)
    """
    if milp_set is None:
        milp_set = set()
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    n_input = state['n_input']
    n_gens = state['n_gens']
    e_vars = [m.addVar(lb=-1.0, ub=1.0, name=f'e_in_{i}') for i in range(n_input)]
    m.update()

    unstable_info = []
    # Interleave unstable + stable (sparse only) by e_new_col order
    combined = list(state.get('unstable_list', []))
    combined.extend([dict(s, _kind='stable') for s in state.get('stable_list', [])])
    # Mark unstable entries
    for e in combined:
        if '_kind' not in e:
            e['_kind'] = 'unstable'
    combined.sort(key=lambda e: e['e_new_col'])

    for ul in combined:
        li = ul['layer_idx']
        j = ul['neuron_idx']
        lo_j = ul['lo']
        hi_j = ul['hi']
        c_j = ul['c_in']
        e_new_col = ul['e_new_col']
        row_idx = ul['row_indices']
        row_val = ul['row_values']
        kind = ul['_kind']

        while len(e_vars) < e_new_col:
            e_vars.append(m.addVar(lb=0.0, ub=0.0, name=f'_unused_{len(e_vars)}'))

        if kind == 'stable':
            # Sparse stable-on: new gen ∈ [lo, hi], equality v == c + G@e
            e_new = m.addVar(lb=lo_j, ub=hi_j, name=f'as_L{li}_{j}')
            e_vars.append(e_new)
            m.update()
            expr_vars = [e_vars[int(k)] for k in row_idx]
            expr_coefs = [float(v) for v in row_val]
            m.addLConstr(e_new - grb.LinExpr(expr_coefs, expr_vars) == c_j,
                          name=f'eq_L{li}_{j}')
            continue

        # Unstable path (same as before)
        e_new = m.addVar(lb=0.0, ub=hi_j, name=f'a_L{li}_{j}')
        e_vars.append(e_new)
        m.update()
        expr_vars = [e_vars[int(k)] for k in row_idx]
        expr_coefs = [float(v) for v in row_val]
        m.addLConstr(grb.LinExpr(expr_coefs, expr_vars) - e_new <= -c_j,
                      name=f'tri_lo_L{li}_{j}')
        key = (li, j)
        if key in milp_set:
            s = m.addVar(vtype=grb.GRB.BINARY, name=f's_L{li}_{j}')
            m.update()
            m.addLConstr(e_new - hi_j * s <= 0,
                          name=f'bigM_hi_L{li}_{j}')
            m.addLConstr(e_new - grb.LinExpr(expr_coefs, expr_vars)
                          - lo_j * s <= c_j - lo_j,
                          name=f'bigM_z_L{li}_{j}')
        else:
            slope = hi_j / (hi_j - lo_j)
            m.addLConstr(e_new - grb.LinExpr(
                [slope * w for w in expr_coefs], expr_vars)
                <= slope * (c_j - lo_j),
                name=f'tri_up_L{li}_{j}')
        unstable_info.append({
            'layer_idx': li,
            'neuron_idx': j,
            'e_new_var_name': f'a_L{li}_{j}',
            'row_coefs_names': [(float(row_val[k]), e_vars[int(row_idx[k])].VarName)
                                for k in range(len(row_idx))],
            'c_in': c_j,
            'lo': lo_j,
            'hi': hi_j,
            'e_new_col': e_new_col,
        })

    # Pad e_vars to n_gens if needed
    while len(e_vars) < n_gens:
        e_vars.append(m.addVar(lb=0.0, ub=0.0, name=f'_unused_{len(e_vars)}'))

    # Objective: qw @ y_out + qb where y_out = c + G_out @ e
    # obj_coef = qw @ G_out_csr, obj_const = qw @ c_out + qb
    obj_coef = state['obj_G_out_csr'].T @ qw  # (n_gens,) vector
    obj_const = float(state['obj_c_out'] @ qw) + qb

    obj = grb.LinExpr()
    for k in range(n_gens):
        c = float(obj_coef[k])
        if c != 0:
            obj.add(e_vars[k], c)
    m.setObjective(obj + obj_const, grb.GRB.MINIMIZE)
    if not milp_set:
        m.setParam('Method', 1)
    m.update()

    return m, env, unstable_info, np.asarray(obj_coef).flatten()


def build_gen_lp(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                 output_op_name, qw, qb, *, milp_set=None,
                 n_threads=1, device='cuda', dtype=torch.float64):
    """Convenience wrapper: precompute state then build Gurobi.

    Same signature and return as before.
    """
    state = precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op_name, device=device, dtype=dtype)
    return build_gen_lp_from_state(state, qw, qb,
                                    milp_set=milp_set, n_threads=n_threads)


def compute_scores(m, unstable_info, obj_coef, method='lp_ew_frac'):
    """Compute neuron scores from a solved LP.

    method='lp_ew_frac' : |obj_coef[col]| * h*|l|/(h-l)
    method='lp_fractional': |a_val - max(0, z_val)| at LP solution
    """
    var_by_name = {v.VarName: v for v in m.getVars()}
    scores = {}
    for info in unstable_info:
        key = (info['layer_idx'], info['neuron_idx'])
        if method == 'lp_ew_frac':
            ew = abs(float(obj_coef[info['e_new_col']]))
            lo_j = info['lo']
            hi_j = info['hi']
            frac = hi_j * abs(lo_j) / (hi_j - lo_j)
            scores[key] = ew * frac
        elif method == 'lp_fractional':
            try:
                a_val = var_by_name[info['e_new_var_name']].X
                z_val = info['c_in'] + sum(
                    c * var_by_name[vn].X
                    for c, vn in info['row_coefs_names'])
                scores[key] = abs(a_val - max(0.0, z_val))
            except grb.GurobiError:
                scores[key] = 0.0
        else:
            raise ValueError(f'unknown scoring method {method}')
    return scores


def solve_spec(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
               output_op_name, qw, qb, *,
               milp_set=None, time_limit=60.0, best_bd_stop=None,
               n_threads=1, device='cuda', dtype=torch.float64,
               score_method='lp_ew_frac', state=None):
    """Build + solve gen LP or MILP for spec minimization.

    If state is provided (precomputed from precompute_gen_state), reuse it
    to build the Gurobi model — skips the GPU rebuild and ensures bit-identical
    coefficients across calls.

    Returns dict with: result ('UNSAT'/'SAT'/'UNKNOWN'),
    lb (ObjBound), status, solve_time, build_time, scores (if LP).
    """
    t_build = time.perf_counter()
    if state is not None:
        m, env, unstable_info, obj_coef = build_gen_lp_from_state(
            state, qw, qb, milp_set=milp_set, n_threads=n_threads)
    else:
        m, env, unstable_info, obj_coef = build_gen_lp(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            output_op_name, qw, qb, milp_set=milp_set,
            n_threads=n_threads, device=device, dtype=dtype)
    dt_build = time.perf_counter() - t_build

    m.setParam('TimeLimit', float(time_limit))
    if best_bd_stop is not None:
        m.setParam('BestBdStop', float(best_bd_stop))

    import os as _os
    _dbg = _os.environ.get('GEN_LP_DEBUG_WRITE')
    if _dbg and milp_set:
        _path = f'{_dbg}_bins{len(milp_set)}.lp'
        m.write(_path)
    t_solve = time.perf_counter()
    optimize_checked(m)
    dt_solve = time.perf_counter() - t_solve

    status = m.Status
    try:
        lb = float(m.ObjBound)
    except (grb.GurobiError, AttributeError):
        lb = None
    n_sol = m.SolCount

    scores = None
    if (not milp_set) and status == grb.GRB.OPTIMAL:
        try:
            scores = compute_scores(m, unstable_info, obj_coef,
                                    method=score_method)
        except (grb.GurobiError, AttributeError):
            scores = None

    # Classify
    if milp_set:
        if status == grb.GRB.OPTIMAL:
            result = 'UNSAT' if lb is not None and lb > 0 else 'SAT'
        elif status == grb.GRB.USER_OBJ_LIMIT:
            result = 'UNSAT' if lb is not None and lb > 0 else 'SAT'
        elif status == grb.GRB.TIME_LIMIT:
            result = 'SAT' if n_sol > 0 else 'UNKNOWN'
        else:
            result = 'UNKNOWN'
    else:
        if status == grb.GRB.OPTIMAL:
            result = 'UNSAT' if lb is not None and lb > 0 else 'SAT'
        else:
            result = 'UNKNOWN'

    # Extract input witness (e_in values) if a feasible solution exists.
    e_in = None
    if n_sol > 0 and state is not None:
        try:
            n_input = state['n_input']
            e_in = np.empty(n_input, dtype=np.float64)
            for i in range(n_input):
                e_in[i] = m.getVarByName(f'e_in_{i}').X
        except (grb.GurobiError, AttributeError):
            e_in = None

    info = {
        'n_vars': m.NumVars,
        'n_constrs': m.NumConstrs,
        'n_bins': m.NumBinVars,
        'status': status,
        'lb': lb,
        'build_time': dt_build,
        'solve_time': dt_solve,
        'scores': scores,
        'e_in': e_in,
    }

    m.dispose()
    env.dispose()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return result, dt_build + dt_solve, info


def tighten_bounds(gg_ops_ser, x_lo, x_hi, initial_bounds, input_name, *,
                    sample_timeout=5.0, time_left_fn=None, n_threads=4,
                    device='cuda', dtype=torch.float64, print_progress=False):
    """Layer-by-layer per-neuron tightening via gen LP.

    Walks ops, maintaining (center, G) state. At each ReLU, solves
    min/max LPs to tighten each unstable neuron's pre-activation bounds
    before classifying stable/dead/unstable.

    Returns tightened bounds_by_relu (new dict).
    """
    gpu = torch.device(device)
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)
    m.setParam('Method', 1)  # dual simplex (fast re-solve after obj change)

    n_in = len(x_lo)
    e_vars = [m.addVar(lb=-1.0, ub=1.0) for _ in range(n_in)]
    m.update()

    center = {input_name: torch.tensor(
        (x_hi + x_lo) / 2, dtype=dtype, device=gpu)}
    half_width = torch.tensor((x_hi - x_lo) / 2, dtype=dtype, device=gpu)
    G_by_op = {input_name: torch.diag(half_width)}
    n_gens = n_in
    bounds_by_relu = {}

    last_use = {}
    for i, op in enumerate(gg_ops_ser):
        for inp in op['inputs']:
            last_use[inp] = i

    def pad_cols(G):
        if G.shape[1] < n_gens:
            return torch.cat([G, torch.zeros(
                G.shape[0], n_gens - G.shape[1], dtype=dtype,
                device=gpu)], dim=1)
        return G

    def _tighten_neuron(row, c_j, orig_lo, orig_hi):
        """Solve min/max LP for a neuron's pre-activation."""
        nz_coefs = [(float(row[k]), e_vars[k])
                    for k in range(n_gens) if row[k] != 0]
        if not nz_coefs:
            return orig_lo, orig_hi
        obj = grb.LinExpr(nz_coefs) + float(c_j)
        # Min
        m.setObjective(obj, grb.GRB.MINIMIZE)
        optimize_checked(m)
        if m.Status == grb.GRB.OPTIMAL:
            new_lo = max(orig_lo, float(m.ObjVal))
        else:
            new_lo = orig_lo
        # Max
        m.setObjective(obj, grb.GRB.MAXIMIZE)
        optimize_checked(m)
        if m.Status == grb.GRB.OPTIMAL:
            new_hi = min(orig_hi, float(m.ObjVal))
        else:
            new_hi = orig_hi
        return new_lo, new_hi

    for op_idx, op in enumerate(gg_ops_ser):
        nm = op['name']
        t = op['type']

        if t == 'conv':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            C_in, H_in, W_in = op['in_shape']
            kernel = torch.tensor(op['kernel_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            sH, sW = op['stride']
            pH, pW = op['padding']
            c_img = prev_c.reshape(1, C_in, H_in, W_in)
            c_out = F.conv2d(c_img, kernel, bias=bias,
                             stride=(sH, sW), padding=(pH, pW)).flatten()
            g_img = prev_G.t().contiguous().reshape(
                n_gens, C_in, H_in, W_in)
            g_out = F.conv2d(g_img, kernel, bias=None,
                             stride=(sH, sW), padding=(pH, pW))
            g_out = g_out.reshape(n_gens, -1).t().contiguous()
            center[nm] = c_out
            G_by_op[nm] = g_out

        elif t == 'fc':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            W = torch.tensor(op['W_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            center[nm] = W @ prev_c + bias
            G_by_op[nm] = W @ prev_G

        elif t == 'relu':
            if 'layer_idx' not in op:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]
                continue
            li = op['layer_idx']
            orig_lo_arr, orig_hi_arr = initial_bounds[li]
            new_lo = orig_lo_arr.copy()
            new_hi = orig_hi_arr.copy()
            c_in = center[op['inputs'][0]]
            G_in = pad_cols(G_by_op[op['inputs'][0]])
            n = len(c_in)

            # Tighten unstable neurons
            unstable = np.where((orig_lo_arr < 0) & (orig_hi_arr > 0))[0]
            c_in_cpu = c_in.detach().cpu().numpy()
            if len(unstable) > 0:
                uidx_t = torch.tensor(unstable, device=gpu, dtype=torch.long)
                G_unstable = G_in[uidx_t].detach().cpu().numpy()
            t_tighten_start = time.perf_counter()
            tightened_count = 0
            for local_idx, j in enumerate(unstable):
                if time_left_fn is not None and time_left_fn() <= 0:
                    break
                row = G_unstable[local_idx]
                nlo, nhi = _tighten_neuron(
                    row, float(c_in_cpu[j]),
                    float(orig_lo_arr[j]), float(orig_hi_arr[j]))
                new_lo[j] = nlo
                new_hi[j] = nhi
                if nlo >= 0 or nhi <= 0:
                    tightened_count += 1
            dt_tighten = time.perf_counter() - t_tighten_start
            bounds_by_relu[li] = (new_lo, new_hi)
            if print_progress:
                n_after_unstable = int(((new_lo < 0) & (new_hi > 0)).sum())
                print(f'  li={li}: {len(unstable)} → {n_after_unstable} '
                      f'unstable (tightened {tightened_count} in {dt_tighten:.2f}s)')

            # Apply relu using tightened bounds, add gen vars & constraints
            dead_mask = new_hi <= 0
            stable_mask = (new_lo >= 0) & ~dead_mask
            unstable_mask = ~dead_mask & ~stable_mask
            unstable_idx = np.where(unstable_mask)[0]
            stable_idx = np.where(stable_mask)[0]
            n_new = int(unstable_mask.sum())

            new_gen_vars = [m.addVar(lb=0.0, ub=float(new_hi[j]))
                            for j in unstable_idx]
            m.update()

            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(
                    unstable_idx, device=gpu, dtype=torch.long)
                G_new_unstable = G_in[uidx_t].detach().cpu().numpy()
            for local_idx, j in enumerate(unstable_idx):
                lo_j = float(new_lo[j])
                hi_j = float(new_hi[j])
                slope = hi_j / (hi_j - lo_j)
                row = G_new_unstable[local_idx]
                expr_coefs = [(float(row[k]), e_vars[k])
                              for k in range(n_gens) if row[k] != 0]
                m.addLConstr(grb.LinExpr(expr_coefs)
                              - new_gen_vars[local_idx]
                              <= -float(c_in_cpu[j]))
                m.addLConstr(new_gen_vars[local_idx]
                              - grb.LinExpr([(slope * w, v)
                                             for w, v in expr_coefs])
                              <= slope * (float(c_in_cpu[j]) - lo_j))
            m.update()

            # Build output G
            G_out = torch.zeros(n, n_gens + n_new, dtype=dtype, device=gpu)
            if len(stable_idx) > 0:
                sidx_t = torch.tensor(
                    stable_idx, device=gpu, dtype=torch.long)
                G_out[sidx_t, :n_gens] = G_in[sidx_t]
            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(
                    unstable_idx, device=gpu, dtype=torch.long)
                new_col_idx = torch.arange(
                    n_gens, n_gens + n_new, device=gpu, dtype=torch.long)
                G_out[uidx_t, new_col_idx] = 1.0
            c_out = torch.zeros(n, dtype=dtype, device=gpu)
            if len(stable_idx) > 0:
                c_out[sidx_t] = c_in[sidx_t]
            center[nm] = c_out
            G_by_op[nm] = G_out
            e_vars.extend(new_gen_vars)
            n_gens += n_new

        elif t == 'add':
            if op.get('is_merge'):
                ca = center[op['inputs'][0]]
                cb = center[op['inputs'][1]]
                Ga = pad_cols(G_by_op[op['inputs'][0]])
                Gb = pad_cols(G_by_op[op['inputs'][1]])
                center[nm] = ca + cb
                G_by_op[nm] = Ga + Gb
            else:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'sub':
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            b = op.get('bias')
            if b is not None:
                bt = torch.tensor(b.flatten(), dtype=dtype, device=gpu)
                center[nm] = prev_c - bt
            else:
                center[nm] = prev_c
            G_by_op[nm] = prev_G

        elif t == 'reshape':
            center[nm] = center[op['inputs'][0]]
            G_by_op[nm] = G_by_op[op['inputs'][0]]

        for inp in op['inputs']:
            if (last_use.get(inp) == op_idx and inp in G_by_op
                    and inp != nm):
                del G_by_op[inp]
                del center[inp]

    if device == 'cuda':
        torch.cuda.synchronize()
    G_by_op.clear()
    center.clear()
    m.dispose()
    env.dispose()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return bounds_by_relu


def gen_lp_worker(args):
    """Multiprocessing-friendly worker wrapping solve_spec."""
    (gg_ops_ser, x_lo, x_hi, bbr, input_name, output_op_name,
     qw, qb, milp_set, time_limit, best_bd_stop, n_threads,
     device, score_method) = args
    return solve_spec(
        gg_ops_ser, x_lo, x_hi, bbr, input_name, output_op_name,
        qw, qb, milp_set=milp_set, time_limit=time_limit,
        best_bd_stop=best_bd_stop, n_threads=n_threads,
        device=device, score_method=score_method)


def racing_escalation(state, qw, qb, scored_keys, *,
                      time_left_fn, n_threads=1, print_progress=False):
    """Serial MILP racing over a doubling bin schedule with early stop.

    After each MILP solve, extracts the input witness from the integer
    solution and forward-propagates it through the real network; if
    `qw @ y + qb < 0` at that x, this is a true counterexample (early
    SAT exit). Otherwise escalates bins=2,4,8,... stopping at UNSAT or
    timeout.

    Returns (verdict, levels_info, witness_x) where verdict is
    'unsat' | 'sat' | 'unknown' and witness_x is the counterexample
    input (np.ndarray) on SAT, else None.
    """
    if state is None:
        raise ValueError('racing_escalation requires precomputed state')
    bin_schedule = []
    b = 2
    while b <= len(scored_keys):
        bin_schedule.append(b)
        b *= 2
    if scored_keys and (not bin_schedule
                        or bin_schedule[-1] < len(scored_keys)):
        bin_schedule.append(len(scored_keys))

    x_lo = state['x_lo']
    x_hi = state['x_hi']
    c = (x_hi + x_lo) / 2.0
    half_w = (x_hi - x_lo) / 2.0

    levels = []
    for n_bins in bin_schedule:
        tl = time_left_fn()
        if tl <= 0:
            break
        milp_set = set(scored_keys[:n_bins])
        result, dt, info = solve_spec(
            None, None, None, None, None, None, qw, qb,
            milp_set=milp_set, time_limit=tl, best_bd_stop=0.0,
            n_threads=n_threads, state=state)
        lb = info.get('lb')
        lb_s = f'{lb:+.4f}' if isinstance(lb, float) else 'n/a'
        levels.append({
            'n_bins': n_bins, 'result': result, 'time': dt,
            'lb': lb, 'info': info,
        })

        # Witness check on the integer solution's input assignment.
        e_in = info.get('e_in')
        witness_x = None
        if e_in is not None:
            x = c + half_w * e_in
            y = forward_point(state['gg_ops_ser'], x,
                              state['input_name'], state['output_op_name'])
            spec_val = float(np.dot(qw.astype(np.float64), y) + qb)
            if spec_val < 0:
                witness_x = x
                if print_progress:
                    print(f'    Racing bins={n_bins}: SAT (witness, '
                          f'spec={spec_val:+.6f}) lb={lb_s} ({dt:.1f}s)')
                return 'sat', levels, witness_x

        if print_progress:
            print(f'    Racing bins={n_bins}: {result} '
                  f'lb={lb_s} ({dt:.1f}s)')
        if result == 'UNSAT':
            return 'unsat', levels, None
    return 'unknown', levels, None


# Module-level globals for per-query worker (fork-shared on Linux)
_Q_STATE = None
_Q_N_THREADS = None
_Q_TIME_LEFT_DEADLINE = None


def _query_race_init(state, n_threads, deadline):
    global _Q_STATE, _Q_N_THREADS, _Q_TIME_LEFT_DEADLINE
    _Q_STATE = state
    _Q_N_THREADS = n_threads
    _Q_TIME_LEFT_DEADLINE = deadline


def _query_race_solve(args):
    qi, qw, qb, scored_keys = args
    import time as _t
    def _tl():
        return _Q_TIME_LEFT_DEADLINE - _t.perf_counter()
    verdict, levels, witness = racing_escalation(
        _Q_STATE, qw, qb, scored_keys,
        time_left_fn=_tl, n_threads=_Q_N_THREADS, print_progress=False)
    return qi, verdict, levels, witness


def _query_race_one_bin(args):
    """Solve one (spec, n_bins) task. Returns level info + witness."""
    qi, qw, qb, scored_keys, n_bins = args
    import time as _t
    tl = _Q_TIME_LEFT_DEADLINE - _t.perf_counter()
    if tl <= 0:
        return (qi, n_bins, 'unknown',
                {'n_bins': n_bins, 'result': 'timeout',
                 'time': 0.0, 'lb': None}, None)
    milp_set = set(scored_keys[:n_bins])
    result, dt, info = solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=milp_set, time_limit=tl, best_bd_stop=0.0,
        n_threads=_Q_N_THREADS, state=_Q_STATE)
    level_info = {'n_bins': n_bins, 'result': result, 'time': dt,
                  'lb': info.get('lb'), 'info': info}

    # Witness check
    witness = None
    e_in = info.get('e_in')
    if e_in is not None:
        c = (_Q_STATE['x_hi'] + _Q_STATE['x_lo']) / 2.0
        half_w = (_Q_STATE['x_hi'] - _Q_STATE['x_lo']) / 2.0
        x = c + half_w * e_in
        y = forward_point(_Q_STATE['gg_ops_ser'], x,
                          _Q_STATE['input_name'],
                          _Q_STATE['output_op_name'])
        if float(np.dot(qw.astype(np.float64), y) + qb) < 0:
            witness = x

    if result == 'UNSAT':
        verdict = 'unsat'
    elif witness is not None:
        verdict = 'sat'
    else:
        verdict = 'unknown'
    return qi, n_bins, verdict, level_info, witness


def parallel_query_racing(state, query_specs, *, time_left_fn,
                           n_threads_total=4, print_progress=False):
    """Run MILP racing across open queries with idle-core-filling.

    Submits one task per (spec, bin_level). Tasks are ordered so every
    spec's bin=2 runs before any spec's bin=4, etc. — we prefer
    completing low bins before speculating higher ones. Pool size is
    `n_threads_total`, each worker uses 1 Gurobi thread. When a spec
    resolves (UNSAT or SAT witness), later tasks for that spec still
    execute but their results are ignored.

    query_specs: list of (qi, qw, qb, scored_keys) tuples.
    Returns: list of (qi, verdict, levels, witness) in submission order,
    where verdict ∈ {'unsat', 'sat', 'unknown'}.
    """
    import multiprocessing as _mp
    import time as _t
    if not query_specs:
        return []
    tl = time_left_fn()
    if tl <= 0:
        return [(qi, 'unknown', [], None) for qi, _, _, _ in query_specs]
    deadline = _t.perf_counter() + tl

    # Build ordered task list: round-robin across specs by bin level, so
    # every spec's bin=2 comes before any bin=4, and so on.
    per_spec_schedule = []
    for qi, qw, qb, scored_keys in query_specs:
        schedule = []
        b = 2
        while b <= len(scored_keys):
            schedule.append(b); b *= 2
        if scored_keys and (not schedule or schedule[-1] < len(scored_keys)):
            schedule.append(len(scored_keys))
        per_spec_schedule.append((qi, qw, qb, scored_keys, schedule))
    max_levels = max((len(s[-1]) for s in per_spec_schedule), default=0)
    tasks = []
    for lvl_idx in range(max_levels):
        for qi, qw, qb, scored_keys, schedule in per_spec_schedule:
            if lvl_idx < len(schedule):
                tasks.append((qi, qw, qb, scored_keys, schedule[lvl_idx]))

    results = {qi: {'verdict': 'unknown', 'levels': [], 'witness': None}
               for qi, _, _, _ in query_specs}
    done_specs = set()
    found_sat = False

    n_workers = max(1, n_threads_total)
    ctx = _mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=_query_race_init,
                   initargs=(state, 1, deadline)) as pool:
        for out in pool.imap_unordered(_query_race_one_bin, tasks):
            qi, n_bins, verdict, level_info, witness = out
            r = results[qi]
            if qi in done_specs:
                continue
            r['levels'].append(level_info)
            if verdict == 'unsat':
                r['verdict'] = 'unsat'
                done_specs.add(qi)
            elif verdict == 'sat':
                r['verdict'] = 'sat'
                r['witness'] = witness
                done_specs.add(qi)
                found_sat = True
            if found_sat or len(done_specs) == len(query_specs):
                break

    return [(qi, results[qi]['verdict'], results[qi]['levels'],
             results[qi]['witness'])
            for qi, _, _, _ in query_specs]
