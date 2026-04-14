"""GOLD: Guided One-shot Lagrangian Decomposition for spec lower bounds.

Solves a single query's lower bound by (i) one LP solve of the network
LP relaxation, (ii) chain-rule extraction of optimal LD multipliers from
the LP duals, (iii) one MILP solve of the spec-adjacent layer at those
multipliers. No iterative dual ascent.

See ld_lp_dual.txt in the project root for the full derivation and
empirical motivation.
"""
import gurobipy as grb
import numpy as np

from .gurobi_util import optimize_checked

from .verify_graph import _build_reference, _build_optimized, _serialize_gg_ops


def solve_joint_lp(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                   output_op_name, qw, qb, *, n_threads=1, time_limit=60.0,
                   builder='optimized'):
    """Build joint LP, set spec objective, solve.

    builder: 'reference' or 'optimized' (default). The optimized builder
    uses MVar+sparse and is much faster for large networks.

    Returns (lp_triangle_bound, op_var_refs, model, env). Caller MUST
    dispose model+env after extracting rho.
    """
    build_fn = _build_optimized if builder == 'optimized' else _build_reference
    m, env, op_var_refs, _ = build_fn(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        n_threads=n_threads)
    m.setParam('TimeLimit', time_limit)

    last_vars = op_var_refs[output_op_name]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        w = float(qw[j])
        if w == 0:
            continue
        obj.add(last_vars[j], w)
    m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    optimize_checked(m)

    assert m.Status == grb.GRB.OPTIMAL, \
        f'joint LP not optimal: status={m.Status}'
    lp_bound = float(m.ObjVal)
    return lp_bound, op_var_refs, m, env


def find_boundary_op(gg_ops_ser, n_relu):
    """Find the op feeding the last hidden ReLU (the boundary for LD).

    Returns (boundary_op_name, last_relu_op_name).
    """
    relu_count = 0
    last_relu_op = None
    for op in gg_ops_ser:
        if op['type'] == 'relu' and 'layer_idx' in op:
            relu_count += 1
            if relu_count == n_relu:
                last_relu_op = op
                break

    boundary_op_name = last_relu_op['inputs'][0]
    return boundary_op_name, last_relu_op['name']


def find_tail_op_chain(gg_ops_ser, last_relu_op_name, output_op_name):
    """Return ordered list of op records from last_relu_op through output_op."""
    chain = []
    recording = False
    for op in gg_ops_ser:
        if op['name'] == last_relu_op_name:
            recording = True
        if recording:
            chain.append(op)
        if op['name'] == output_op_name:
            break
    return chain


def extract_rho_chain_rule(op_var_refs, boundary_op_name):
    """Read -var.RC for each variable at the boundary op.

    Returns dict mapping neuron index to rho value.
    Requires the LP to have been solved to optimality (status OPTIMAL).
    """
    boundary_vars = op_var_refs[boundary_op_name]
    rho = {}
    for i, v in enumerate(boundary_vars):
        if v is None:
            continue
        rc = v.RC
        rho[i] = -float(rc)
    return rho


def detect_rho_sign(rho_raw, bounds_by_relu, last_relu_layer_idx,
                    tail_chain, qw, qb, lp_triangle, n_threads=1):
    """Try +rho and -rho, return the sign that gives a non-negative delta."""
    best_sign = 1
    best_delta = -float('inf')
    for sign in (1, -1):
        rho_test = {i: sign * v for i, v in rho_raw.items()}
        lp_bound = _eval_tail_lp(rho_test, bounds_by_relu,
                                 last_relu_layer_idx, tail_chain,
                                 qw, qb, n_threads)
        milp_bound = _eval_tail_milp(rho_test, bounds_by_relu,
                                     last_relu_layer_idx, tail_chain,
                                     qw, qb, n_threads)
        delta = milp_bound - lp_bound
        if delta > best_delta:
            best_delta = delta
            best_sign = sign
    return best_sign


def _eval_tail_lp(rho, bounds_by_relu, last_relu_layer_idx,
                  tail_chain, qw, qb, n_threads):
    """Quick LP solve of the tail sub at given rho. Returns obj value."""
    m, env = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                            tail_chain, qw, qb, rho,
                            inner='lp', n_threads=n_threads)
    optimize_checked(m)
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    return val


def _eval_tail_milp(rho, bounds_by_relu, last_relu_layer_idx,
                    tail_chain, qw, qb, n_threads):
    """Quick MILP solve of the tail sub at given rho. Returns obj bound."""
    m, env = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                            tail_chain, qw, qb, rho,
                            inner='milp', n_threads=n_threads)
    m.setParam('TimeLimit', 10.0)
    optimize_checked(m)
    val = float(m.ObjBound)
    m.dispose()
    env.dispose()
    return val


def build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                   tail_chain, qw, qb, rho, *, inner='lp',
                   milp_neurons=None, n_threads=1):
    """Build a standalone Gurobi model for the tail sub-problem.

    The tail sub covers: input-box vars for pre_relu_N (the boundary),
    ReLU_N encoding (LP triangle or big-M), then the remaining ops
    up to the spec output. Objective includes -rho coupling on inputs
    plus the spec qw on outputs.

    milp_neurons: if set, only these neuron indices get big-M encoding
    when inner='milp'; others use LP triangle. If None, all unstable
    neurons get big-M.

    Returns (model, env).
    """
    inf = grb.GRB.INFINITY
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
    n_boundary = len(lo_r)

    boundary_vars = [None] * n_boundary
    for i in range(n_boundary):
        lo_i = float(lo_r[i])
        hi_i = float(hi_r[i])
        if hi_i <= 0:
            continue
        boundary_vars[i] = m.addVar(lb=lo_i, ub=hi_i)
    m.update()

    current_vars = [None] * n_boundary
    relu_op = tail_chain[0]
    assert relu_op['type'] == 'relu', \
        f'expected relu as first tail op, got {relu_op["type"]}'

    for i in range(n_boundary):
        if boundary_vars[i] is None:
            continue
        lo_i = float(lo_r[i])
        hi_i = float(hi_r[i])
        z = boundary_vars[i]
        use_milp_here = (inner == 'milp'
                         and (milp_neurons is None or i in milp_neurons))
        if lo_i >= 0:
            a = m.addVar(lb=lo_i, ub=hi_i)
            m.addConstr(a == z)
            current_vars[i] = a
        elif use_milp_here:
            a = m.addVar(lb=0.0, ub=hi_i)
            s = m.addVar(vtype=grb.GRB.BINARY)
            m.addConstr(a >= 0)
            m.addConstr(a >= z)
            m.addConstr(a <= hi_i * s)
            m.addConstr(a <= z - lo_i * (1 - s))
            current_vars[i] = a
        else:
            a = m.addVar(lb=0.0, ub=hi_i)
            m.addConstr(a >= z)
            slope = hi_i / (hi_i - lo_i) if hi_i != lo_i else 0.0
            m.addConstr(a <= slope * z - slope * lo_i)
            current_vars[i] = a
    m.update()

    for op in tail_chain[1:]:
        t = op['type']
        if t in ('reshape', 'flatten'):
            pass
        elif t == 'fc':
            W = op['W_np']
            bias = op['bias_np']
            n_out = W.shape[0]
            new_vars = [None] * n_out
            for j in range(n_out):
                expr = grb.LinExpr()
                has_terms = False
                for k in range(min(W.shape[1], len(current_vars))):
                    wjk = W[j, k]
                    if wjk != 0 and current_vars[k] is not None:
                        expr.add(current_vars[k], float(wjk))
                        has_terms = True
                b_j = float(bias[j])
                v = m.addVar(lb=-inf, ub=inf)
                if has_terms:
                    m.addConstr(v == expr + b_j)
                else:
                    m.addConstr(v == b_j)
                new_vars[j] = v
            current_vars = new_vars
            m.update()
        elif t == 'conv':
            from .verify_milp import _conv_connections
            kernel = op['kernel_np']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            n_out = op['n_out']
            bias = op['bias_np']
            new_vars = [None] * n_out
            for j in range(n_out):
                conns = _conv_connections(j, kernel, in_shape, stride, padding)
                expr = grb.LinExpr()
                has_terms = False
                for fi, w in conns:
                    if fi < len(current_vars) and current_vars[fi] is not None:
                        expr.add(current_vars[fi], float(w))
                        has_terms = True
                c_out = j // ((in_shape[1] + 2 * padding[0] - kernel.shape[2]) // stride[0] + 1) // ((in_shape[2] + 2 * padding[1] - kernel.shape[3]) // stride[1] + 1)
                b_j = float(bias[min(c_out, len(bias) - 1)])
                v = m.addVar(lb=-inf, ub=inf)
                if has_terms:
                    m.addConstr(v == expr + b_j)
                else:
                    m.addConstr(v == b_j)
                new_vars[j] = v
            current_vars = new_vars
            m.update()
        elif t == 'add':
            if op.get('is_merge'):
                raise NotImplementedError('tail sub: merge-Add in tail chain')
            bias = op.get('bias')
            if bias is not None:
                bias_flat = np.asarray(bias).flatten().astype(np.float64)
                new_vars = [None] * len(current_vars)
                for i in range(len(current_vars)):
                    if current_vars[i] is None:
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == current_vars[i] + float(bias_flat[i]))
                    new_vars[i] = v
                current_vars = new_vars
                m.update()
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bias_flat = np.asarray(bias).flatten().astype(np.float64)
                new_vars = [None] * len(current_vars)
                for i in range(len(current_vars)):
                    if current_vars[i] is None:
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == current_vars[i] - float(bias_flat[i]))
                    new_vars[i] = v
                current_vars = new_vars
                m.update()
        else:
            raise NotImplementedError(f'tail sub: unsupported op type {t}')

    obj = grb.LinExpr()
    for i, v in enumerate(boundary_vars):
        if v is not None and i in rho:
            obj.add(v, -float(rho[i]))
    for j in range(min(len(qw), len(current_vars))):
        w = float(qw[j])
        if w != 0 and current_vars[j] is not None:
            obj.add(current_vars[j], w)
    m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    m.update()

    return m, env


def solve_model(model, time_limit, best_bd_stop=None):
    """Set TimeLimit (and BestBdStop if given), optimize.

    Returns (obj_bound, obj_val_or_None, runtime).
    """
    model.setParam('TimeLimit', time_limit)
    if best_bd_stop is not None:
        model.setParam('BestBdStop', best_bd_stop)
    optimize_checked(model)
    runtime = model.Runtime
    try:
        obj_val = float(model.ObjVal)
    except Exception:
        obj_val = None
    try:
        obj_bound = float(model.ObjBound)
    except Exception:
        obj_bound = float('inf')
    return obj_bound, obj_val, runtime


def gold_joint_mixed_milp(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                          output_op_name, qw, qb, milp_layer_idx,
                          ambiguous_set, *, time_limit, best_bd_stop=None,
                          n_threads=4, builder='optimized'):
    """VARIANT A: build model with MILP at one layer, solve."""
    build_fn = _build_optimized if builder == 'optimized' else _build_reference
    m, env, op_var_refs, _ = build_fn(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        milp_by_layer={milp_layer_idx: ambiguous_set},
        n_threads=n_threads)

    last_vars = op_var_refs[output_op_name]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        w = float(qw[j])
        if w == 0:
            continue
        obj.add(last_vars[j], w)
    m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)

    obj_bound, obj_val, runtime = solve_model(m, time_limit, best_bd_stop)
    m.dispose()
    env.dispose()
    return obj_bound, obj_val, runtime


def gold_tail_decomposition(gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                            input_name, output_op_name, qw, qb, *,
                            time_limit, best_bd_stop=None, n_threads=1,
                            lp_threads=4, lp_time_limit=300.0,
                            builder='optimized', milp_neurons=None):
    """VARIANT B: full GOLD pipeline.

    lp_threads/lp_time_limit control the joint LP solve (which can be
    large). n_threads controls the tail sub MILP solve.
    milp_neurons: if set, only these neuron indices in the last ReLU
    get big-M encoding; others stay LP triangle.

    Returns dict with 'lp_triangle', 'rho_sign', 'v_tail_LP',
    'v_tail_MILP_bound', 'gold_bound', 'tail_runtime', 'tail_obj_val'.
    """
    n_relu = _count_relu(gg_ops_ser)
    lp_triangle, op_var_refs, m_lp, env_lp = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op_name, qw, qb, n_threads=lp_threads,
        time_limit=lp_time_limit, builder=builder)

    boundary_op_name, last_relu_op_name = find_boundary_op(gg_ops_ser, n_relu)
    tail_chain = find_tail_op_chain(gg_ops_ser, last_relu_op_name,
                                    output_op_name)

    rho_raw = extract_rho_chain_rule(op_var_refs, boundary_op_name)
    m_lp.dispose()
    env_lp.dispose()

    last_relu_layer_idx = n_relu - 1
    rho_sign = detect_rho_sign(rho_raw, bounds_by_relu, last_relu_layer_idx,
                               tail_chain, qw, qb, lp_triangle, n_threads)
    rho = {i: rho_sign * v for i, v in rho_raw.items()}

    m_tail_lp, env_tail_lp = build_tail_sub(
        bounds_by_relu, last_relu_layer_idx, tail_chain,
        qw, qb, rho, inner='lp', n_threads=n_threads)
    optimize_checked(m_tail_lp)
    v_tail_LP = float(m_tail_lp.ObjVal)
    m_tail_lp.dispose()
    env_tail_lp.dispose()

    tail_bd_stop = None
    if best_bd_stop is not None:
        tail_bd_stop = v_tail_LP - lp_triangle + best_bd_stop

    m_tail_milp, env_tail_milp = build_tail_sub(
        bounds_by_relu, last_relu_layer_idx, tail_chain,
        qw, qb, rho, inner='milp', milp_neurons=milp_neurons,
        n_threads=n_threads)
    milp_bound, milp_val, tail_runtime = solve_model(
        m_tail_milp, time_limit, tail_bd_stop)
    m_tail_milp.dispose()
    env_tail_milp.dispose()

    gold_bound = lp_triangle + (milp_bound - v_tail_LP)

    return {
        'lp_triangle': lp_triangle,
        'rho_sign': rho_sign,
        'v_tail_LP': v_tail_LP,
        'v_tail_MILP_bound': milp_bound,
        'gold_bound': gold_bound,
        'tail_runtime': tail_runtime,
        'tail_obj_val': milp_val,
    }


def _count_relu(gg_ops_ser):
    """Count hidden ReLU layers in serialized ops."""
    return sum(1 for op in gg_ops_ser
               if op['type'] == 'relu' and 'layer_idx' in op)


def gold_solve_query(gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu, qw, qb,
                     *, variant='B', time_limit=60.0, best_bd_stop=None,
                     n_threads=4, builder='optimized', milp_neurons=None):
    """High-level entry point.

    variant='A' calls gold_joint_mixed_milp.
    variant='B' calls gold_tail_decomposition.
    builder: 'reference' or 'optimized' (default).
    milp_neurons: for variant B, only these neuron indices get big-M
    in the tail sub; for variant A, used as the ambiguous_set.
    Returns a result dict with bound, runtime, and a 'method' tag.
    """
    output_op_name = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    if variant == 'A':
        n_relu = _count_relu(gg_ops_ser)
        last_relu_layer_idx = n_relu - 1
        lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
        amb = milp_neurons if milp_neurons is not None else {
            j for j in range(len(lo_r)) if lo_r[j] < 0 and hi_r[j] > 0}
        obj_bound, obj_val, runtime = gold_joint_mixed_milp(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            output_op_name, qw, qb, last_relu_layer_idx, amb,
            time_limit=time_limit, best_bd_stop=best_bd_stop,
            n_threads=n_threads, builder=builder)
        return {
            'method': 'joint_mixed_milp',
            'bound': obj_bound,
            'obj_val': obj_val,
            'runtime': runtime,
        }
    else:
        result = gold_tail_decomposition(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            output_op_name, qw, qb,
            time_limit=time_limit, best_bd_stop=best_bd_stop,
            n_threads=n_threads, lp_threads=n_threads,
            builder=builder, milp_neurons=milp_neurons)
        return {
            'method': 'gold_tail_decomposition',
            'bound': result['gold_bound'],
            'runtime': result['tail_runtime'],
            'lp_triangle': result['lp_triangle'],
            'rho_sign': result['rho_sign'],
            'v_tail_LP': result['v_tail_LP'],
            'v_tail_MILP_bound': result['v_tail_MILP_bound'],
            'tail_obj_val': result['tail_obj_val'],
        }
