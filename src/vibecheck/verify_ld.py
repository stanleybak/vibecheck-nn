"""Lagrangian-decomposition verification solver.

Additive to verify_graph.py's existing pipeline — gated behind
`settings.ld_enabled` (default False). When enabled, runs after the
phase-2 CROWN backward and may close open disjuncts before the
existing phase 7a/7b/8 MILP path. When disabled, this module is
never invoked and the existing pipeline is unchanged.

The core idea: for every hidden pre-activation neuron `z_L[j]` (and
every spec output component), build a tiny MILP that takes the
previous layer's pre-activations as box-bounded leaves, exact-encodes
the previous layer's ReLU on ambiguous neurons in the receptive field
of `j`, and has a scalar objective that is linear in the coupling
multipliers rho. Supergradient ascent on rho drives up a valid lower
bound on the spec query.
"""

import numpy as np

from .verify_milp import _conv_connections
from .gurobi_util import optimize_checked


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def verify_ld_queries(gg, gg_ops_ser, bounds_by_relu, x_lo, x_hi,
                     disj_queries, spec_lbs, still_open_disj,
                     queries, settings, time_left):
    """Run the Lagrangian-decomposition solver on still-open queries.

    Updates `spec_lbs` in place and returns telemetry. For each
    still-open query, builds the per-neuron subproblem MILPs, runs the
    supergradient iteration loop on the scalar-rho (Option-A) dual,
    and marks `spec_lbs[qi] = 1.0` if the aggregated dual bound is
    driven above zero.
    """
    input_name = gg['input_name']
    info = {
        'ld_ran': True,
        'ld_queries_run': 0,
        'ld_verified': 0,
        'per_query': {},
    }

    for di in sorted(still_open_disj):
        if time_left() <= 0:
            break
        for qi, q_w, q_bias in disj_queries[di]:
            if time_left() <= 0:
                break
            if spec_lbs.get(qi, -1) > 0:
                continue
            info['ld_queries_run'] += 1
            best_bound, n_iter = _ld_iterate_one_query(
                gg_ops_ser, bounds_by_relu, x_lo, x_hi,
                q_w, q_bias, input_name, settings, time_left)
            info['per_query'][int(qi)] = {
                'best_bound': float(best_bound),
                'iterations': int(n_iter),
            }
            if best_bound > 0:
                spec_lbs[qi] = 1.0
                info['ld_verified'] += 1

    return info


# ---------------------------------------------------------------------------
# Neuron classification
# ---------------------------------------------------------------------------

def _classify_neurons(l_np, u_np):
    """Classify pre-activation neurons by ReLU stability.

    Returns `(stable_on, stable_off, ambiguous)` boolean numpy arrays.
    The three masks form a partition:
      - `stable_on`:  l >= 0 and u > 0  (ReLU output = z, pass-through)
      - `stable_off`: u <= 0            (ReLU output = 0, dead)
      - `ambiguous`:  l < 0 and u > 0   (ReLU needs binary encoding)

    The edge case `l == u == 0` is classified as stable_off (output is
    identically zero, no binary variable needed).
    """
    l = np.asarray(l_np, dtype=np.float64)
    u = np.asarray(u_np, dtype=np.float64)
    ambiguous = (l < 0) & (u > 0)
    stable_off = (~ambiguous) & (u <= 0)
    stable_on = (~ambiguous) & (~stable_off)
    return stable_on, stable_off, ambiguous


# ---------------------------------------------------------------------------
# Look-back-1 block extraction
# ---------------------------------------------------------------------------

def _compute_linear_bias_chains(gg_ops, input_name):
    """For every linear op (conv/fc), pre-compute the pre_bias and
    post_bias arrays produced by surrounding sub/add(non-merge) ops.

    pre_bias: walking backward from the linear op's input through any
        pass-through ops (sub, add-bias, reshape) until hitting a relu,
        another linear op, a merge-Add, or the network input. The
        accumulated bias is an INPUT shift: input → input + pre_bias
        (sub negates, add adds).

    post_bias: walking forward from the linear op's output through
        pass-through ops until hitting a fork, a relu, another linear
        op, a merge-Add, or the end of the graph. Accumulated bias is
        added directly to the linear op's output.

    Returns dict: linear_op_name -> (pre_bias, post_bias). Either may
    be None when there are no contributions.
    """
    op_by_name = {op['name']: op for op in gg_ops}
    successors = {}
    for op in gg_ops:
        for inp in op['inputs']:
            successors.setdefault(inp, []).append(op['name'])

    def is_passthrough(op):
        t = op['type']
        if t == 'reshape':
            return True
        if t == 'sub':
            return True
        if t == 'add' and not op.get('is_merge', False):
            return True
        return False

    def fold(acc, raw_bias, sign):
        if raw_bias is None:
            return acc
        b_arr = sign * np.asarray(raw_bias.flatten(), dtype=np.float64)
        return b_arr if acc is None else acc + b_arr

    result = {}
    for op in gg_ops:
        if op['type'] not in ('conv', 'fc'):
            continue

        pre_bias = None
        cur_name = op['inputs'][0]
        while cur_name != input_name:
            cur = op_by_name[cur_name]
            if not is_passthrough(cur):
                break
            if cur['type'] == 'sub':
                pre_bias = fold(pre_bias, cur.get('bias'), -1.0)
            elif cur['type'] == 'add':
                pre_bias = fold(pre_bias, cur.get('bias'), 1.0)
            cur_name = cur['inputs'][0]

        post_bias = None
        cur_name = op['name']
        while True:
            succs = successors.get(cur_name, [])
            if len(succs) != 1:
                break
            nxt = op_by_name[succs[0]]
            if not is_passthrough(nxt):
                break
            if nxt['type'] == 'sub':
                post_bias = fold(post_bias, nxt.get('bias'), -1.0)
            elif nxt['type'] == 'add':
                post_bias = fold(post_bias, nxt.get('bias'), 1.0)
            cur_name = succs[0]

        result[op['name']] = (pre_bias, post_bias)

    return result


def _walk_back_to_source(op_by_name, start_name, input_name):
    """Walk backward through pass-through ops from start_name until we
    hit one of: the network input, a linear op (conv/fc), a ReLU with
    layer_idx, or a merge-Add. Returns (kind, obj) where kind is
    'input', 'linear', 'relu', or 'merge', and obj is the terminal op
    dict (or None for 'input').
    """
    cur_name = start_name
    while cur_name != input_name:
        cur = op_by_name[cur_name]
        if cur['type'] in ('conv', 'fc'):
            return 'linear', cur
        if cur['type'] == 'relu' and 'layer_idx' in cur:
            return 'relu', cur
        if cur['type'] == 'add' and cur.get('is_merge', False):
            return 'merge', cur
        cur_name = cur['inputs'][0]
    return 'input', None


def _compute_effective_bias(linear_op, pre_bias, post_bias):
    """Combine the linear op's own bias with surrounding sub/add bias
    contributions into a single per-output-neuron `effective_bias`
    array of length `n_out(linear_op)`.

    For an FC: `eff[j] = bias_orig[j] + post[j] + (W @ pre)[j]`.
    For a conv: same in spirit; the per-channel `bias_orig` is
    broadcast to flat (C*H*W), `post` is broadcast similarly when
    given per-channel, and `pre` is propagated through the conv
    sparse matrix.
    """
    if linear_op['type'] == 'fc':
        W = np.asarray(linear_op['W_np'], dtype=np.float64)
        eff = np.asarray(linear_op['bias_np'], dtype=np.float64).copy()
        if post_bias is not None:
            eff = eff + np.asarray(post_bias, dtype=np.float64)
        if pre_bias is not None:
            eff = eff + W @ np.asarray(pre_bias, dtype=np.float64)
        return eff

    out_shape = linear_op['out_shape']
    C, H, W_ = out_shape
    spatial = H * W_
    n_out = C * spatial
    bias_orig = np.asarray(linear_op['bias_np'], dtype=np.float64)
    eff = np.repeat(bias_orig, spatial)  # broadcast per-channel to flat
    if post_bias is not None:
        pb = np.asarray(post_bias, dtype=np.float64).flatten()
        if pb.shape == (C,):
            eff = eff + np.repeat(pb, spatial)
        else:
            assert pb.shape == (n_out,), (
                f'post_bias shape {pb.shape} incompatible with conv out '
                f'(C={C}, n_out={n_out})')
            eff = eff + pb
    if pre_bias is not None:
        from .verify_milp import _conv_sparse_matrix
        W_sp = _conv_sparse_matrix(
            linear_op['kernel_np'], linear_op['in_shape'],
            linear_op['stride'], linear_op['padding'])
        eff = eff + (W_sp @ np.asarray(pre_bias, dtype=np.float64))
    return eff


def _resolve_branch(op_by_name, start_name, input_name, bounds_by_relu,
                   bias_chains=None):
    """Trace one look-back-1 branch from `start_name` to a linear op
    (or an identity source at the network input).

    Supported terminals:
      - A single linear op (conv/fc) whose own input is either the
        network input or a preceding ReLU with layer_idx
      - The network input itself (identity skip from the input)

    Unsupported (assertion errors):
      - Nested merge-Add (branch crosses another merge)
      - A linear op whose input is itself a linear op (no relu in
        between) — violates look-back-1

    Branch shape:
      {
        'linear_op':      op dict or None,
        'prev_layer_idx': int or None,
        'prev_bounds':    (l_np, u_np) or (None, None),
      }
    """
    kind, terminal = _walk_back_to_source(
        op_by_name, start_name, input_name)
    assert kind in ('linear', 'input'), (
        f'unsupported branch terminal {kind!r} for look-back-1')

    if kind == 'input':
        return {
            'linear_op': None,
            'prev_layer_idx': None,
            'prev_bounds': (None, None),
        }

    linear_op = terminal
    prev_kind, prev_terminal = _walk_back_to_source(
        op_by_name, linear_op['inputs'][0], input_name)
    assert prev_kind in ('input', 'relu'), (
        f'look-back-1 violated: prev is {prev_kind!r}')
    if prev_kind == 'relu':
        prev_layer_idx = int(prev_terminal['layer_idx'])
        lo, hi = bounds_by_relu[prev_layer_idx]
        prev_bounds = (np.asarray(lo, dtype=np.float64),
                       np.asarray(hi, dtype=np.float64))
    else:
        prev_layer_idx = None
        prev_bounds = (None, None)

    effective_bias = None
    if bias_chains is not None and linear_op['name'] in bias_chains:
        pre_bias, post_bias = bias_chains[linear_op['name']]
        if pre_bias is not None or post_bias is not None:
            effective_bias = _compute_effective_bias(
                linear_op, pre_bias, post_bias)

    return {
        'linear_op': linear_op,
        'prev_layer_idx': prev_layer_idx,
        'prev_bounds': prev_bounds,
        'effective_bias': effective_bias,
    }


def _extract_look_back_1_layers(gg_ops, bounds_by_relu, input_name):
    """Walk `gg_ops` in topological order and group ops into
    look-back-1 blocks — one block per linear op (conv or fc) or per
    merge-Add preceding a ReLU.

    Each block summarizes everything a per-neuron LD subproblem needs
    to build a Gurobi model for one target output neuron:

      - `layer_idx`:   int layer index produced by this block (the
                       layer_idx of the following ReLU); None for the
                       final output block
      - `is_merge`:    True iff a merge-Add sits between the block's
                       linear op(s) and the consuming ReLU
      - `branches`:    list of branch dicts. Sequential blocks have 1
                       branch (the single linear op). Merge blocks
                       have 2+ branches, one per merge input, each
                       independently traced look-back-1 to its source
      - `linear_op`:   legacy alias for `branches[0]['linear_op']`
                       (sequential only)
      - `prev_layer_idx`, `prev_bounds`, `input_source`: legacy fields
                       for the sequential single-branch case; for
                       merge blocks these reflect the first branch and
                       `is_merge` flags to use the `branches` list

    Blocks are returned in topological order.
    """
    op_by_name = {op['name']: op for op in gg_ops}
    bias_chains = _compute_linear_bias_chains(gg_ops, input_name)

    # For each relu with a layer_idx, find the block that produces its
    # input. A block is either a sequential linear op, or a merge-Add
    # (with multiple branches). Record layer_idx -> block_spec.
    relu_to_block = {}  # layer_idx -> {'is_merge', 'branches'}
    produced_by_relu = set()  # op names that are wrapped inside a relu block
    for op in gg_ops:
        if op['type'] != 'relu' or 'layer_idx' not in op:
            continue
        li = int(op['layer_idx'])
        kind, terminal = _walk_back_to_source(
            op_by_name, op['inputs'][0], input_name)
        if kind == 'linear':
            branch = _resolve_branch(
                op_by_name, op['inputs'][0], input_name, bounds_by_relu,
                bias_chains)
            relu_to_block[li] = {
                'is_merge': False,
                'branches': [branch],
            }
            produced_by_relu.add(terminal['name'])
        else:
            assert kind == 'merge', (
                f'relu {op["name"]} has unexpected producer kind {kind!r}')
            merge_op = terminal
            branches = [
                _resolve_branch(op_by_name, inp, input_name, bounds_by_relu,
                                bias_chains)
                for inp in merge_op['inputs']
            ]
            relu_to_block[li] = {
                'is_merge': True,
                'branches': branches,
            }
            for b in branches:
                if b['linear_op'] is not None:
                    produced_by_relu.add(b['linear_op']['name'])

    blocks = []
    # Walk gg_ops in topological order; for each linear op that hasn't
    # been wrapped into a merge block and that we haven't already
    # emitted, build its block record.
    emitted = set()
    for op in gg_ops:
        # When we hit a relu with layer_idx and there's an associated
        # merge or sequential block record, emit it (at the relu's
        # position in topo order).
        if (op['type'] == 'relu' and 'layer_idx' in op
                and int(op['layer_idx']) in relu_to_block
                and int(op['layer_idx']) not in emitted):
            li = int(op['layer_idx'])
            rec = relu_to_block[li]
            first = rec['branches'][0]
            blocks.append({
                'layer_idx': li,
                'is_merge': rec['is_merge'],
                'branches': rec['branches'],
                # Legacy sequential-case aliases for the first branch:
                'linear_op': first['linear_op'],
                'prev_layer_idx': first['prev_layer_idx'],
                'prev_bounds': first['prev_bounds'],
                'input_source': 'relu' if first['prev_layer_idx'] is not None
                else 'x',
            })
            emitted.add(li)
            continue
        # Also emit the final-output block (a conv/fc that no relu
        # consumes, e.g. the spec FC layer). Sequential only for now.
        if op['type'] not in ('conv', 'fc'):
            continue
        if op['name'] in produced_by_relu:
            continue
        branch = _resolve_branch(
            op_by_name, op['name'], input_name, bounds_by_relu, bias_chains)
        blocks.append({
            'layer_idx': None,
            'is_merge': False,
            'branches': [branch],
            'linear_op': branch['linear_op'],
            'prev_layer_idx': branch['prev_layer_idx'],
            'prev_bounds': branch['prev_bounds'],
            'input_source': 'relu' if branch['prev_layer_idx'] is not None
            else 'x',
        })

    return blocks


def _rf_connections(op, target_j):
    """Return list of (input_idx, weight) for this linear op's output
    neuron target_j. Handles both conv (via `_conv_connections`) and
    fc (dense row non-zeros)."""
    if op['type'] == 'conv':
        return _conv_connections(
            int(target_j), op['kernel_np'],
            op['in_shape'], op['stride'], op['padding'])
    row = op['W_np'][int(target_j)]
    nz = np.nonzero(row)[0]
    return [(int(i), float(row[i])) for i in nz]


def _rf_bias(op, target_j):
    """Per-output bias for this linear op's output neuron target_j."""
    if op['type'] == 'conv':
        spatial = op['out_shape'][1] * op['out_shape'][2]
        return float(op['bias_np'][int(target_j) // spatial])
    return float(op['bias_np'][int(target_j)])


# ---------------------------------------------------------------------------
# Per-neuron MILP builder / objective setter / solver
# ---------------------------------------------------------------------------

def _branch_connections(branch, target_j):
    """Return `(conns, bias)` for one branch's contribution to output
    neuron `target_j`.

    For a linear branch (conv/fc): `conns` is the rf connection list
    and `bias` is the per-output effective bias (folding any
    surrounding sub/add bias ops if `branch['effective_bias']` is set,
    else falling back to the linear op's `bias_np`). For an identity
    branch (linear_op=None): `conns = [(target_j, 1.0)]` and bias=0.
    """
    op = branch['linear_op']
    if op is None:
        return [(int(target_j), 1.0)], 0.0
    eff = branch.get('effective_bias') if isinstance(branch, dict) else None
    if eff is not None:
        return _rf_connections(op, target_j), float(eff[int(target_j)])
    return _rf_connections(op, target_j), _rf_bias(op, target_j)


def _build_per_neuron_milp(block, target_j, x_lo, x_hi, z_out_bounds=None,
                          inner_mode='milp'):
    """Build a Gurobi MILP for target pre-activation `z_L[target_j]`.

    For a sequential single-branch block the MILP looks exactly as it
    did in step 3: z_in[i] for each rf neuron i (box bounds from the
    previous layer or the network input), the big-M ReLU encoding for
    ambiguous neurons, and `z_out = W[target_j, :] @ a + b`.

    For a merge-Add block the MILP has ONE set of variables per branch,
    using each branch's own prev_bounds (or input box for identity
    skips from input), and `z_out` is the sum of every branch's
    activation contribution plus the total bias. Variables from
    non-coupled branches (identity skips rooted at the network input)
    are stored in `aux_vars` instead of `z_in_vars` so the LD rho
    update and gradient computation skip them.

    The objective is NOT set here; use `_set_objective_per_neuron`.
    The caller owns the model and must dispose it via
    `refs['model'].dispose()` and `refs['env'].dispose()`.
    """
    import gurobipy as grb

    target_j = int(target_j)
    branches = block.get('branches') or [{
        'linear_op': block['linear_op'],
        'prev_layer_idx': block['prev_layer_idx'],
        'prev_bounds': block['prev_bounds'],
    }]

    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', 1)
    m.setParam('DualReductions', 0)

    # For a merged block we keep one `z_in_vars` per branch, keyed by
    # the branch's prev_layer_idx (an int) so the LD rho coupling can
    # distinguish them. For sequential (single-branch) blocks we keep
    # a flat {i: Var} so step-3 callers keep working; set the
    # `z_in_vars_by_branch` key for merge consumers.
    z_in_vars = {}             # sequential-style flat dict (for main branch)
    aux_vars = {}              # uncoupled input-box variables (identity@input)
    stable_on = set()
    ambiguous = set()
    expr_terms = []            # (coef, Var) contributions to z_out
    total_bias = 0.0
    branch_inputs = []         # per branch: {'prev_layer_idx', 'z_in_vars'}

    for bi, branch in enumerate(branches):
        is_first_layer = branch['prev_layer_idx'] is None
        if is_first_layer:
            in_lo = np.asarray(x_lo, dtype=np.float64)
            in_hi = np.asarray(x_hi, dtype=np.float64)
        else:
            in_lo, in_hi = branch['prev_bounds']

        conns, b_val = _branch_connections(branch, target_j)
        total_bias += float(b_val)
        branch_z_in = {}

        for i, w in conns:
            lo_i = float(in_lo[i])
            hi_i = float(in_hi[i])
            # Phase 1 tightening can produce numerically inconsistent
            # bounds (lo > hi) for neurons whose adapt-LP probes
            # converged from opposite sides. Clip to a feasible
            # interval before building the variable.
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            if is_first_layer:
                v = m.addVar(lb=lo_i, ub=hi_i)
                branch_z_in[int(i)] = v
                expr_terms.append((float(w), v))
                # First-layer branches are uncoupled: an identity-from
                # input skip, or a sequential block's first linear op
                # (no upstream relu → no rho). The existing sequential
                # builder exposed these as z_in_vars, so mirror that
                # behaviour when there is only one branch.
                if len(branches) == 1:
                    z_in_vars[int(i)] = v
                else:
                    aux_vars[int(i)] = v
                continue
            if hi_i <= 0:
                continue
            if lo_i >= 0:
                v = m.addVar(lb=lo_i, ub=hi_i)
                branch_z_in[int(i)] = v
                expr_terms.append((float(w), v))
                z_in_vars[int(i)] = v
                stable_on.add(int(i))
            else:
                z = m.addVar(lb=lo_i, ub=hi_i)
                a = m.addVar(lb=0.0, ub=hi_i)
                if inner_mode == 'milp':
                    s = m.addVar(vtype=grb.GRB.BINARY)
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= hi_i * s)
                    m.addConstr(a <= z - lo_i * (1 - s))
                else:  # 'lp' triangle relaxation
                    slope = hi_i / (hi_i - lo_i)
                    m.addConstr(a >= 0)
                    m.addConstr(a >= z)
                    m.addConstr(a <= slope * z - slope * lo_i)
                branch_z_in[int(i)] = z
                expr_terms.append((float(w), a))
                z_in_vars[int(i)] = z
                ambiguous.add(int(i))
        branch_inputs.append({
            'prev_layer_idx': branch['prev_layer_idx'],
            'z_in_vars': branch_z_in,
            'is_identity': branch['linear_op'] is None,
        })
    m.update()

    # When the caller knows tighter bounds on z_out (e.g., the
    # producer's pre-activation has its own `bounds_by_relu` entry),
    # apply them here. Without these, the only thing constraining
    # z_out is the linear equation derived from the (already tight)
    # input box, which is strictly looser than the zono/CROWN/LP
    # bound that the rest of the pipeline already computed.
    if z_out_bounds is not None:
        zo_lo, zo_hi = z_out_bounds
        zo_lo = float(zo_lo)
        zo_hi = float(zo_hi)
        if zo_lo > zo_hi:  # mirror the inverted-bounds clip
            zo_lo, zo_hi = zo_hi, zo_lo
    else:
        zo_lo, zo_hi = -grb.GRB.INFINITY, grb.GRB.INFINITY

    if expr_terms:
        expr = grb.LinExpr()
        for coef, var in expr_terms:
            expr.add(var, coef)
        z_out_var = m.addVar(lb=zo_lo, ub=zo_hi)
        m.addConstr(z_out_var == expr + total_bias)
    else:
        # Constant output: clamp the constant into the requested bounds.
        c = total_bias
        if z_out_bounds is not None:
            c = min(max(c, zo_lo), zo_hi)
        z_out_var = m.addVar(lb=c, ub=c)
    m.update()

    # For merge blocks, the rho coupling is per-branch, so we expose
    # the per-branch z_in_vars structure too. The legacy flat
    # `z_in_vars` is kept empty for merge blocks; callers check
    # `is_merge` in the refs to decide which to iterate.
    is_merge_refs = len(branches) > 1
    return {
        'model': m,
        'env': env,
        'z_in_vars': z_in_vars,
        'aux_vars': aux_vars,
        'z_out_var': z_out_var,
        'target_j': target_j,
        'b_val': total_bias,
        'is_first_layer': all(b['prev_layer_idx'] is None for b in branches),
        'is_merge': is_merge_refs,
        'stable_on': stable_on,
        'ambiguous': ambiguous,
        'rf_indices': sorted(z_in_vars.keys()),
        'branch_inputs': branch_inputs,
    }


def _set_objective_per_neuron(refs, producer_coef, rho_prev):
    """Set the scalar LD objective on a cached per-neuron MILP.

    obj = producer_coef * z_out - sum_{i in rf} rho_prev[i] * z_in[i]

    `producer_coef` is:
      - for non-spec subproblems: the (possibly-scaled) multiplier on
        the target neuron's producer copy, e.g. `rho_this_layer[j]` or
        `N[j] * rho_this_layer[j]` under Option-A scaling.
      - for spec subproblems: the query weight `query_w[j]`.

    `rho_prev` is either None (first-layer subproblem has no upstream
    rho) or a numpy array indexed by the previous layer's neuron
    index.
    """
    import gurobipy as grb

    m = refs['model']
    expr = grb.LinExpr()
    expr.add(refs['z_out_var'], float(producer_coef))
    if rho_prev is not None:
        for i, v in refs['z_in_vars'].items():
            rho_i = float(rho_prev[i])
            if rho_i != 0.0:
                expr.add(v, -rho_i)
    m.setObjective(expr, grb.GRB.MINIMIZE)


def _solve_subproblem(refs, timeout):
    """Solve a cached per-neuron MILP and report dual contribution.

    Returns dict with:
      - `obj_val`:  the LD-valid lower bound of this subproblem's obj,
                    taken as Gurobi's `ObjBound`.
      - `z_out_val`: primal value of z_out (None if no solution found).
      - `z_in_vals`: {i: primal value} for vars in z_in_vars (empty if
                     no solution found).
    """
    import gurobipy as grb

    m = refs['model']
    m.setParam('TimeLimit', float(timeout))
    optimize_checked(m)
    status = m.Status
    assert status in (
        grb.GRB.OPTIMAL, grb.GRB.SUBOPTIMAL, grb.GRB.TIME_LIMIT,
        grb.GRB.USER_OBJ_LIMIT,
    ), f'per-neuron MILP status={status}'

    obj_val = float(m.ObjBound)
    z_out_val = None
    z_in_vals = {}
    if m.SolCount > 0:
        z_out_val = float(refs['z_out_var'].X)
        z_in_vals = {i: float(v.X) for i, v in refs['z_in_vars'].items()}

    return {
        'obj_val': obj_val,
        'z_out_val': z_out_val,
        'z_in_vals': z_in_vals,
    }


# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------

def _compute_lr(t, n_iter, settings):
    """Compute the step size for iteration `t` given the schedule.

    Supports 'linear_decay' and 'constant'. Adam is added in step 7.
    """
    schedule = str(settings.ld_step_schedule)
    lr0 = float(settings.ld_initial_step)
    lrT = float(settings.ld_final_step)
    if schedule == 'constant' or n_iter <= 1:
        return lr0
    frac = min(1.0, float(t) / max(1, n_iter - 1))
    return lr0 + (lrT - lr0) * frac


def _build_all_subproblems(blocks, bounds_by_relu, x_lo, x_hi, query_w,
                          inner_mode='milp'):
    """Build per-neuron subproblems for a single query.

    Non-spec subproblems: one per live neuron (not stable-off) at each
    block `k in 0..L-2`, targeting that block's output neuron `j`.
    Spec subproblems: one per spec output component `j` where
    `query_w[j] != 0`, built on the last block.

    Returns:
      - `subs`: dict keyed by ('hidden', k, j) or ('spec', j), values
        are refs dicts from `_build_per_neuron_milp`.
      - `producer_consumers`: dict mapping a producer neuron `(k, j)`
        with `k<L-1` to the list of consumer subproblem keys that
        reference `j` in their rf. Used by `_set_all_objectives` to
        compute per-pair-rho producer coefficients.
    """
    L = len(blocks)
    subs = {}

    for k in range(L - 1):
        lo_k, hi_k = bounds_by_relu[k]
        for j in range(len(lo_k)):
            if hi_k[j] <= 0:
                continue  # stable-off: no producer subproblem
            refs = _build_per_neuron_milp(
                blocks[k], j, x_lo, x_hi,
                z_out_bounds=(float(lo_k[j]), float(hi_k[j])),
                inner_mode=inner_mode)
            subs[('hidden', k, int(j))] = refs

    spec_block = blocks[L - 1]
    spec_targets = np.nonzero(np.asarray(query_w))[0]
    for j in spec_targets:
        refs = _build_per_neuron_milp(
            spec_block, int(j), x_lo, x_hi, inner_mode=inner_mode)
        subs[('spec', int(j))] = refs

    producer_consumers = {}
    for ckey, refs in subs.items():
        if ckey[0] == 'hidden':
            _, sub_k, _ = ckey
            prev_k = sub_k - 1
        else:
            prev_k = L - 2
        if prev_k < 0:
            continue
        for i in refs['rf_indices']:
            producer_consumers.setdefault((prev_k, int(i)), []).append(ckey)

    return subs, producer_consumers


def _init_rho(subs, bounds_by_relu):
    """Initialize per-consumer-pair rho dictionaries.

    `rho[consumer_key]` is a numpy array indexed by the input neuron
    index in that consumer's rf. Only consumer subs (non-first hidden
    and all spec subs) have rho entries.
    """
    rho = {}
    for ckey, refs in subs.items():
        if ckey[0] == 'hidden' and ckey[1] == 0:
            continue  # first-layer sub has no upstream
        n = max(refs['rf_indices'], default=-1) + 1
        rho[ckey] = np.zeros(n, dtype=np.float64)
    return rho


def _set_all_objectives(subs, rho, producer_consumers, query_w, L):
    """Refresh every subproblem's objective from the current rho.

    Producer coefficient on sub (hidden, k, j): sum over consumers c
    of `rho[c][j]`. For spec subs, the producer coefficient is
    `query_w[j]`. Consumer terms use `rho[sub]` as rho_prev.
    """
    for skey, srefs in subs.items():
        if skey[0] == 'hidden':
            _, sub_k, sub_j = skey
            producer_coef = 0.0
            for ckey in producer_consumers.get((sub_k, sub_j), []):
                producer_coef += float(rho[ckey][sub_j])
        else:  # 'spec'
            _, sub_j = skey
            producer_coef = float(query_w[sub_j])
        rho_prev = rho.get(skey)
        _set_objective_per_neuron(srefs, producer_coef, rho_prev)


def _solve_all_subproblems(subs, timeout):
    """Serially solve every subproblem and return a results dict."""
    results = {}
    for key, refs in subs.items():
        results[key] = _solve_subproblem(refs, timeout)
    return results


def _aggregate_bound(results, query_bias):
    """Sum each subproblem's ObjBound plus the query bias."""
    total = float(query_bias)
    for r in results.values():
        total += r['obj_val']
    return total


def _compute_disagreement(results, rho, L):
    """Compute per-pair supergradients on rho.

    For each consumer subproblem `c` and each local input neuron `i`,
    the supergradient of q(rho) wrt `rho[c][i]` is
    `z_k[i]_producer - z_in_c[i]` where `k = prev_layer(c)` and the
    producer value comes from the `(hidden, k, i)` subproblem.

    Returns `grad[consumer_key][i]` in the same shape as `rho`.
    """
    grad = {ckey: np.zeros_like(arr) for ckey, arr in rho.items()}
    for ckey in rho:
        if ckey[0] == 'hidden':
            _, sub_k, _ = ckey
            prev_k = sub_k - 1
        else:
            prev_k = L - 2
        z_in_vals = results[ckey]['z_in_vals']
        for i, val in z_in_vals.items():
            pkey = ('hidden', prev_k, int(i))
            z_prod = results[pkey]['z_out_val']
            grad[ckey][i] = float(z_prod) - float(val)
    return grad


def _step_rho_linear_decay(rho, grad, t, n_iter, settings):
    """Supergradient ASCENT step on rho: rho += lr * grad.

    Grad is the per-pair supergradient of q(rho). The step direction
    is positive because q is concave in rho and we are maximizing.
    """
    lr = _compute_lr(t, n_iter, settings)
    for ckey in rho:
        rho[ckey] = rho[ckey] + lr * grad[ckey]


def _init_adam_state(rho):
    """Initialize Adam first/second moment buffers to match rho shapes."""
    return {
        'm': {ckey: np.zeros_like(arr) for ckey, arr in rho.items()},
        'v': {ckey: np.zeros_like(arr) for ckey, arr in rho.items()},
    }


def _step_rho_adam(rho, grad, adam_state, t, settings):
    """Adam ASCENT step on rho (t is the current 0-indexed iteration).

    Follows Kingma & Ba's update rule adapted for maximization:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g
      v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
      m_hat = m_t / (1 - beta1^{t+1})
      v_hat = v_t / (1 - beta2^{t+1})
      rho  += lr * m_hat / (sqrt(v_hat) + 1e-8)
    """
    lr = float(settings.ld_adam_lr)
    b1 = float(settings.ld_adam_beta1)
    b2 = float(settings.ld_adam_beta2)
    step = t + 1  # 1-indexed
    bc1 = 1.0 - (b1 ** step)
    bc2 = 1.0 - (b2 ** step)
    for ckey in rho:
        g = grad[ckey]
        m = adam_state['m'][ckey] = b1 * adam_state['m'][ckey] + (1 - b1) * g
        v = adam_state['v'][ckey] = b2 * adam_state['v'][ckey] + (1 - b2) * g * g
        m_hat = m / bc1
        v_hat = v / bc2
        rho[ckey] = rho[ckey] + lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def _dispose_all(subs):
    for refs in subs.values():
        refs['model'].dispose()
        refs['env'].dispose()


def _ld_iterate_one_query(gg_ops, bounds_by_relu, x_lo, x_hi,
                          query_w, query_bias, input_name,
                          settings, time_left):
    """Run the supergradient iteration loop for a single query.

    Returns `(best_bound, n_iter_executed)`. `best_bound` is the
    maximum aggregated dual bound over all iterations — a valid lower
    bound on `min query_w @ y + query_bias`. The caller compares it
    against 0 to decide verification.
    """
    blocks = _extract_look_back_1_layers(
        gg_ops, bounds_by_relu, input_name)
    L = len(blocks)

    inner_mode = str(getattr(settings, 'ld_inner_mode', 'milp'))
    subs, producer_consumers = _build_all_subproblems(
        blocks, bounds_by_relu, x_lo, x_hi, query_w, inner_mode=inner_mode)
    rho = _init_rho(subs, bounds_by_relu)

    schedule = str(settings.ld_step_schedule)
    adam_state = _init_adam_state(rho) if schedule == 'adam' else None

    best_bound = -np.inf
    n_iter_target = int(settings.ld_num_iterations)
    sub_timeout = float(settings.ld_subproblem_timeout)
    early_stop = bool(settings.ld_early_stop)
    log_interval = int(settings.ld_log_interval)
    t = 0
    try:
        while True:
            _set_all_objectives(subs, rho, producer_consumers, query_w, L)
            results = _solve_all_subproblems(subs, sub_timeout)
            q_rho = _aggregate_bound(results, query_bias)
            if q_rho > best_bound:
                best_bound = q_rho
            if log_interval > 0 and t % log_interval == 0 and getattr(
                    settings, 'print_progress', False):
                print(f'  [LD] iter={t} q(rho)={q_rho:.6f} '
                      f'best={best_bound:.6f}')
            if early_stop and best_bound > 0:
                t += 1
                break
            if t >= n_iter_target or time_left() <= 0:
                t += 1
                break
            grad = _compute_disagreement(results, rho, L)
            if schedule == 'adam':
                _step_rho_adam(rho, grad, adam_state, t, settings)
            else:
                _step_rho_linear_decay(
                    rho, grad, t, n_iter_target, settings)
            t += 1
    finally:
        _dispose_all(subs)

    return best_bound, t
