"""Generic multi-network VNNLIB support (isomorphic_acasxu / monotonic_acasxu, and
any 2-network v2 spec of the same shape).

A v2 multi-network spec declares networks f, g (whose `instances.csv` onnx field is a
list `[('f', a.onnx), ('g', b.onnx)]`) plus constraints relating their inputs and
outputs — a shape vibecheck's single-onnx pipeline can't ingest. This module converts
ONE such instance into a single MERGED ONNX + an equivalent v1 VNNLIB, so the rest of
vibecheck (graph mode + a normal config, e.g. acasxu) verifies it unchanged.

ONE generic code path (no per-benchmark special cases):

  1. parse_multinet: read the spec into a canonical IR — per-coord input roles
     (box / pinned-const / shared `== X_f[i] X_g[i]` / relational `>= X_f[k] X_g[k]`)
     and the output property as a DNF of LINEAR atoms over Y_f/Y_g.
  2. merge: build a net whose free input is [base(n), delta?]; derive each subnet's
     input — x_g = base, x_f = base except a relational coord k = clamp(base[k]+delta,
     lo, hi) (clamp via Min/Max => EXACT: every valid (X_g,X_f) reachable, none escape
     the box). Run f on x_f, g on x_g; concat outputs; emit each distinct atom LHS
     (a linear combo of Y_f,Y_g) as a network output via a constant MatMul.
  3. emit v1: box on the free inputs; the output DNF as simple `(>= Y_a c)`/`(<= Y_a c)`
     atoms over those baked outputs.

Isomorphic = all coords shared, no relational coord, output is an OR of |Yg-Yf|>=eps
atoms. Monotonic = one relational coord + pinned coords, single atom Yf[j]<Yg[j]. Both
fall out of the same builder. Correctness is gated by an onnxruntime oracle (merged net
vs separately-run f/g) — a bad merge raises, never silently verifies a wrong property.

Self-contained (no vibecheck imports) so it can run in prepare_instance before the
graph is built. The emitted Min/Max are rewritten to ReLU+affine at load time by
onnx_optimizer.min_max_to_relu.
"""
import ast
import os
import re
import gzip
import hashlib
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto, version_converter
import onnx.compose as compose
import onnxruntime as ort

TARGET_OPSET = 13
ORACLE_TOL = 1e-3      # max |merged - reference| allowed before we refuse the merge


# --------------------------------------------------------------------------- io

def _load_onnx(path):
    """Load an onnx model from the EXACT path given; the `.gz` sibling is only a
    fallback when the exact path doesn't exist. (The caller is handed a concrete
    path by the harness/instances.csv — that file is authoritative; reaching for a
    `.gz` first read a stale sibling and merged the wrong network.)"""
    if os.path.exists(path):
        return onnx.load(path)
    if os.path.exists(path + '.gz'):
        with gzip.open(path + '.gz') as fh:
            return onnx.load_model_from_string(fh.read())
    raise FileNotFoundError(path)


def _read_vnnlib_text(path):
    """Read the spec from the EXACT path given; `.gz` is only a fallback when the
    exact path doesn't exist (same rationale as `_load_onnx`)."""
    if os.path.exists(path):
        return open(path).read()
    if os.path.exists(path + '.gz'):
        with gzip.open(path + '.gz', 'rt') as fh:
            return fh.read()
    raise FileNotFoundError(path)


def _serialize(path):
    return _load_onnx(path).SerializeToString()


def is_network_pair_net_field(net_field):
    """True if `--net` is a pair list-string rather than a single path."""
    return isinstance(net_field, str) and net_field.lstrip().startswith('[')


def parse_network_field(net_field):
    """Parse the `--net` argument. Returns `[(label, path), ...]` for a network-PAIR
    list-string `[('f', a.onnx), ('g', b.onnx)]` (the form the competition harness
    passes for isomorphic/monotonic_acasxu), or `None` for a single onnx path.

    Uses `ast.literal_eval` (robust to the `/`, `.`, `_` etc. in real paths that a
    regex split mishandled) and rejects a malformed list loudly rather than silently
    dropping entries."""
    if not is_network_pair_net_field(net_field):
        return None
    try:
        nets = ast.literal_eval(net_field)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"--net is not a valid network list: {net_field!r} ({e})")
    if not (isinstance(nets, list) and nets and all(
            isinstance(t, (tuple, list)) and len(t) == 2
            and isinstance(t[0], str) and isinstance(t[1], str) for t in nets)):
        raise ValueError(
            f"--net network list must be a non-empty list of (label, path) "
            f"pairs, got: {nets!r}")
    return [(str(lbl), str(path)) for lbl, path in nets]


def _onnx_paths_from_field(field):
    """Ordered onnx paths from a pair field (back-compat; prefer parse_network_field)."""
    parsed = parse_network_field(field)
    return [p for _, p in parsed] if parsed else []


def declared_networks(vnnlib_text):
    """Labels of every `(declare-network LABEL ...)` in the spec, in declaration
    order (the first is the base the others relate to, e.g. f then g)."""
    return re.findall(r'\(declare-network\s+(\w+)', vnnlib_text)


def _resolve_onnx_path(p, base_dir):
    """Resolve an onnx path that may be absolute, relative to the CWD (the harness
    passes `./benchmarks/<cat>/2.0/onnx/x.onnx`), or relative to the benchmark
    version dir (instances.csv passes `onnx/x.onnx`). Prefer the form that exists."""
    if os.path.isabs(p):
        return p
    if os.path.exists(p) or os.path.exists(p + '.gz'):
        return p                                   # CWD-relative (harness form)
    return os.path.join(base_dir, p)               # version-dir-relative (csv form)


def resolve_pair_paths(net_field, vnnlib_path, base_dir=None):
    """`{label: resolved_onnx_path}` for a pair `--net`, or None for a single net.
    Same base-dir convention as `build_merged_instance`."""
    parsed = parse_network_field(net_field)
    if parsed is None:
        return None
    base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(vnnlib_path)))
    return {lbl: _resolve_onnx_path(p, base_dir) for lbl, p in parsed}


def detect_kind(vnnlib_text):
    """'iso' | 'mono' | None from the spec's relation keyword (informational)."""
    if 'isomorphic-to' in vnnlib_text:
        return 'iso'
    if 'equal-to' in vnnlib_text:
        return 'mono'
    return None


# ------------------------------------------------------------------ onnx helpers

def _free_input(model):
    init = {i.name for i in model.graph.initializer}
    free = [i for i in model.graph.input if i.name not in init]
    assert len(free) == 1, f"expected 1 free input, got {[i.name for i in free]}"
    return free[0]


def _strip_initializer_inputs(model):
    init = {i.name for i in model.graph.initializer}
    keep = [i for i in model.graph.input if i.name not in init]
    del model.graph.input[:]
    model.graph.input.extend(keep)
    return model


def _prep(model, prefix):
    """Upgrade to TARGET_OPSET, drop shadow initializer-inputs, prefix all names."""
    model = version_converter.convert_version(model, TARGET_OPSET)
    model = _strip_initializer_inputs(model)
    model = compose.add_prefix(model, prefix=prefix)
    return model


def _const(name, arr, dtype=TensorProto.FLOAT):
    arr = np.asarray(arr)
    return helper.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())


# ------------------------------------------------------------------ parse -> IR

def parse_multinet(text):
    """Parse a 2-network v2 spec into the canonical IR (see module docstring).

    Returns dict:
      n          input dim (per network)
      base_box   [(lo,hi)]*n   box for the BASE input x_g
      xf_box     [(lo,hi)]*n   clamp bounds for x_f (= f's input box)
      rel        None or {'k':int, 'dmax':float}   x_f[k]=clamp(base[k]+delta, xf_box[k]),
                 delta in [0, dmax]
      atoms      [{'lhs':{('f',i):coef,('g',i):coef}, 'op':'>='|'<=', 'rhs':float}]
      dnf        'or' | 'and'   how atoms combine
    """
    # Normalize multi-dim batch-indexed refs to the flat `X_f[i]` form every regex
    # below expects. The 2026 benchmarks were updated to VNN-LIB 2.0 shaped refs —
    # 4-D inputs `X_f[0,0,0,i]` (shape [1,1,1,5]) and 2-D outputs `Y_g[0,i]` (shape
    # [1,5]). Every leading dim is size 1 (index 0), so the flat index is the LAST
    # comma-separated component; take it. Plain 1-D refs `X_f[i]` are left untouched,
    # so both spec versions parse. (Does not touch `declare-input X_f real [..]` —
    # that has a space before `[`, so the `X_f[` pattern doesn't match it.)
    text = re.sub(r'([XY]_[fg])\[([\d\s,]+)\]',
                  lambda m: f"{m.group(1)}[{m.group(2).split(',')[-1].strip()}]", text)
    n = max(int(i) for i in re.findall(r'X_[fg]\[(\d+)\]', text)) + 1
    f_lo = [None] * n; f_hi = [None] * n
    # f input box: (<= X_f[i] HI) ... (>= X_f[i] LO)
    for m in re.finditer(r'\(<=\s*X_f\[(\d+)\]\s*([-\d.eE]+)\).*?\(>=\s*X_f\[\1\]\s*([-\d.eE]+)\)', text):
        i = int(m.group(1)); f_hi[i] = float(m.group(2)); f_lo[i] = float(m.group(3))
    # pins: (== X_f[i] CONST)
    for m in re.finditer(r'==\s*X_f\[(\d+)\]\s*([-\d.][-\d.eE]*)\s*\)', text):
        i = int(m.group(1)); c = float(m.group(2)); f_lo[i] = c; f_hi[i] = c
    assert all(v is not None for v in f_lo), f"unbound X_f coord: lo={f_lo}"

    base_lo = list(f_lo); base_hi = list(f_hi)
    rel = None
    krel = re.findall(r'>=\s*X_f\[(\d+)\]\s*X_g\[\d+\]', text)
    if krel:
        assert len(krel) == 1, f"only one relational coord supported, got {krel}"
        k = int(krel[0])
        logm = re.search(rf'>=\s*X_g\[{k}\]\s*([-\d.eE]+)', text)
        base_lo[k] = float(logm.group(1)) if logm else f_lo[k]   # x_g[k] lower bound
        rel = {'k': k, 'dmax': f_hi[k] - base_lo[k]}

    atoms, dnf = _parse_output_atoms(text)
    return dict(n=n, base_box=list(zip(base_lo, base_hi)),
                xf_box=list(zip(f_lo, f_hi)), rel=rel, atoms=atoms, dnf=dnf)


def _parse_output_atoms(text):
    """Parse the output property into (atoms, dnf). Supports the linear atom forms the
    network-pair benchmarks use: `OP Y_a[i] (+/- Y_b[i] eps)`, `OP Y_a[i] Y_b[j]`,
    `OP Y_a[i] const`. dnf='or' if the atoms sit under an `(or ...)`, else 'and'."""
    atoms = []

    def add(lhs, raw_op, raw_rhs):
        # NON-STRICT bound (sound): a strict `< 0` is encoded as its closure
        # `<= 0` (a SUPERSET of the true unsafe set), NOT `<= -margin` (a strict
        # SUBSET that would false-unsat a shallow real CE — e.g. monotonic_acasxu
        # diff = -5e-5). The strict `<` semantics are enforced DOWNSTREAM in the
        # SAT-detection: PGD + ORT confirmation only commit 'sat' on a CLEAR CE
        # (margin <= -atol); the measure-zero boundary (diff = 0, e.g. an iso
        # pair's trivial point or a monotone net's x1==x2 diagonal) validates only
        # within tolerance, so it is persisted as a within-tol sat while the
        # search keeps looking for a clear CE or an unsat proof. See
        # `verify_graph._sat_disposition`.
        op = '<=' if raw_op in ('<', '<=') else '>='
        atoms.append({'lhs': lhs, 'op': op, 'rhs': raw_rhs})

    # form 1: (OP Y_a[i] (SIGN Y_b[j] eps))  ->  Y_a[i] - Y_b[j] (OP) (±eps)
    seen = set()
    for m in re.finditer(
            r'\(([<>]=?)\s*Y_([fg])\[(\d+)\]\s*\(([-+])\s*Y_([fg])\[(\d+)\]\s*([-\d.eE]+)\s*\)\s*\)', text):
        op, a, ai, sign, b, bj, eps = m.groups()
        seen.add(m.span())
        e = float(eps) * (1.0 if sign == '+' else -1.0)
        add({(a, int(ai)): 1.0, (b, int(bj)): -1.0}, op, e)
    # form 2: (OP Y_a[i] Y_b[j])  ->  Y_a[i] - Y_b[j] (OP) 0
    for m in re.finditer(r'\(([<>]=?)\s*Y_([fg])\[(\d+)\]\s*Y_([fg])\[(\d+)\]\s*\)', text):
        op, a, ai, b, bj = m.groups()
        add({(a, int(ai)): 1.0, (b, int(bj)): -1.0}, op, 0.0)
    # form 3: (OP Y_a[i] const)  ->  Y_a[i] (OP) const
    for m in re.finditer(r'\(([<>]=?)\s*Y_([fg])\[(\d+)\]\s*([-\d.][-\d.eE]*)\s*\)', text):
        op, a, ai, c = m.groups()
        add({(a, int(ai)): 1.0}, op, float(c))

    assert atoms, "no output atoms parsed"
    # output section structure: an `(or` anywhere after the first Y_ constraint => OR
    y0 = text.find('Y_')
    dnf = 'or' if '(or' in text[y0:] else 'and'
    return atoms, dnf


def _atom_layout(ir, out_dim):
    """Map each distinct atom LHS to an output index; return (A, atom_z) where A is the
    [num_distinct, 2*out_dim] matrix (cols [Yf_0..Yf_{D-1}, Yg_0..Yg_{D-1}]) and atom_z[i]
    is the output index for atoms[i]."""
    keys = []; index = {}
    for at in ir['atoms']:
        key = tuple(sorted((t, i, c) for (t, i), c in at['lhs'].items()))
        if key not in index:
            index[key] = len(keys); keys.append(at['lhs'])
    A = np.zeros((len(keys), 2 * out_dim), np.float32)
    for r, lhs in enumerate(keys):
        for (t, i), c in lhs.items():
            A[r, (0 if t == 'f' else out_dim) + i] = c
    atom_z = [index[tuple(sorted((t, i, c) for (t, i), c in at['lhs'].items()))]
              for at in ir['atoms']]
    return A, atom_z


# --------------------------------------------------------------------- merge

def merge(nf_path, ng_path, ir, out_path):
    """Build the merged ONNX. Returns out_dim (per-net output width)."""
    mf = _prep(_load_onnx(nf_path), 'f_')
    mg = _prep(_load_onnx(ng_path), 'g_')
    in_f = _free_input(mf).name; in_g = _free_input(mg).name
    out_f = mf.graph.output[0].name; out_g = mg.graph.output[0].name
    net_shape = [d.dim_value for d in _free_input(mf).type.tensor_type.shape.dim]
    net_shape_dyn = [-1] + net_shape[1:]
    out_dim = int(np.prod([d.dim_value for d in mf.graph.output[0].type.tensor_type.shape.dim]))
    n = ir['n']; rel = ir['rel']
    A, _ = _atom_layout(ir, out_dim)

    n_in = n + (1 if rel else 0)
    Z = helper.make_tensor_value_info('X', TensorProto.FLOAT, ['batch', n_in])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['batch', A.shape[0]])
    inits = [_const('base_idx', list(range(n)), TensorProto.INT64),
             _const('net_shape', net_shape_dyn, TensorProto.INT64),
             _const('Amat', A.T)]                      # [2*out_dim, num_atoms]
    pre = [helper.make_node('Gather', ['X', 'base_idx'], ['xg_flat'], axis=1)]
    if rel:
        k = rel['k']
        e_k = np.zeros(n, np.float32); e_k[k] = 1.0
        hi = np.array([b for _, b in ir['xf_box']], np.float32)
        lo = np.array([a for a, _ in ir['xf_box']], np.float32)
        inits += [_const('delta_idx', [n], TensorProto.INT64), _const('ek', e_k),
                  _const('HI', hi), _const('LO', lo)]
        pre += [
            helper.make_node('Gather', ['X', 'delta_idx'], ['delta'], axis=1),
            helper.make_node('Mul', ['delta', 'ek'], ['dvec']),
            helper.make_node('Add', ['xg_flat', 'dvec'], ['pre_f']),
            helper.make_node('Min', ['pre_f', 'HI'], ['cl']),
            helper.make_node('Max', ['cl', 'LO'], ['xf_flat']),     # clamp(base+delta*e_k, LO, HI)
        ]
        xf_src = 'xf_flat'
    else:
        xf_src = 'xg_flat'                              # iso: x_f = x_g = base
    pre += [helper.make_node('Reshape', ['xg_flat', 'net_shape'], [in_g]),
            helper.make_node('Reshape', [xf_src, 'net_shape'], [in_f])]
    tail = [helper.make_node('Concat', [out_f, out_g], ['YY'], axis=1),  # [batch, 2*out_dim]
            helper.make_node('MatMul', ['YY', 'Amat'], ['Y'])]          # [batch, num_atoms]
    graph = helper.make_graph(
        nodes=pre + list(mf.graph.node) + list(mg.graph.node) + tail,
        name='pair_merged', inputs=[Z], outputs=[Y],
        initializer=list(mf.graph.initializer) + list(mg.graph.initializer) + inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', TARGET_OPSET)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_dim


def emit_v1(ir, out_dim, out_path):
    """Write the v1 spec over the merged net: box on free inputs + output DNF."""
    A, atom_z = _atom_layout(ir, out_dim)
    n = ir['n']; rel = ir['rel']
    n_in = n + (1 if rel else 0)
    L = ['; auto-generated v1 spec for the merged network-pair net', '']
    L += [f'(declare-const X_{i} Real)' for i in range(n_in)]
    L += [f'(declare-const Y_{a} Real)' for a in range(A.shape[0])]
    L.append('')
    for i, (lo, hi) in enumerate(ir['base_box']):
        L.append(f'(assert (<= X_{i} {hi!r}))')
        L.append(f'(assert (>= X_{i} {lo!r}))')
    if rel:
        L.append(f'(assert (<= X_{n} {ir["rel"]["dmax"]!r}))')
        L.append(f'(assert (>= X_{n} 0.0))')
    L.append('')
    clauses = []
    for at, z in zip(ir['atoms'], atom_z):
        sym = '<=' if at['op'] == '<=' else '>='
        clauses.append(f'(and ({sym} Y_{z} {at["rhs"]!r}))')
    if ir['dnf'] == 'or':
        L.append('(assert (or')
        L += ['  ' + c for c in clauses]
        L.append('))')
    else:
        # AND: every atom must hold
        for at, z in zip(ir['atoms'], atom_z):
            sym = '<=' if at['op'] == '<=' else '>='
            L.append(f'(assert ({sym} Y_{z} {at["rhs"]!r}))')
    open(out_path, 'w').write('\n'.join(L) + '\n')


def oracle(nf_path, ng_path, ir, merged_path, out_dim, n_samples=120, seed=0):
    """Check merged(Z) == atom-LHS(f(x_f), g(x_g)) on random points in the box."""
    sf = ort.InferenceSession(_serialize(nf_path))
    sg = ort.InferenceSession(_serialize(ng_path))
    sm = ort.InferenceSession(open(merged_path, 'rb').read())
    fin = _free_input(_load_onnx(nf_path)).name
    gin = _free_input(_load_onnx(ng_path)).name
    net_shape = [d.dim_value for d in _free_input(_load_onnx(nf_path)).type.tensor_type.shape.dim]
    A, _ = _atom_layout(ir, out_dim)
    n = ir['n']; rel = ir['rel']
    lo = np.array([a for a, _ in ir['base_box']]); hi = np.array([b for _, b in ir['base_box']])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_samples):
        base = lo + (hi - lo) * rng.random(n)
        z = list(base)
        xf = base.copy()
        if rel:
            k = rel['k']; delta = rng.random() * rel['dmax']; z.append(delta)
            xf[k] = np.clip(base[k] + delta, ir['xf_box'][k][0], ir['xf_box'][k][1])
        yf = sf.run(None, {fin: xf.astype(np.float32).reshape(net_shape)})[0].flatten()
        yg = sg.run(None, {gin: base.astype(np.float32).reshape(net_shape)})[0].flatten()
        ref = A @ np.concatenate([yf, yg])
        ym = sm.run(None, {'X': np.array([z], np.float32)})[0].flatten()
        worst = max(worst, float(np.abs(ym - ref).max()))
    return worst


# ----------------------------------------------------------------- entry point

def _cache_paths(net_field, vnnlib_path):
    """Deterministic temp paths keyed on the instance (onnx field + spec)."""
    h = hashlib.md5((net_field + '|' + os.path.abspath(vnnlib_path)).encode()).hexdigest()[:12]
    d = tempfile.gettempdir()
    return (os.path.join(d, f'vibecheck_pair_{h}.onnx'),
            os.path.join(d, f'vibecheck_pair_{h}.vnnlib'))


def build_merged_instance(net_field, vnnlib_path, base_dir=None, run_oracle=True):
    """Convert a network-pair instance to (merged_onnx_path, merged_vnnlib_path).

    net_field: the instances.csv onnx field `[('f', a), ('g', b)]` (paths absolute or
    relative to base_dir). vnnlib_path: the v2 pair spec. Raises if the spec isn't a
    recognized pair or if the onnx-merge oracle exceeds ORACLE_TOL.
    """
    text = _read_vnnlib_text(vnnlib_path)
    assert detect_kind(text), f"spec is not a network-pair (no isomorphic-to/equal-to): {vnnlib_path}"
    label_paths = resolve_pair_paths(net_field, vnnlib_path, base_dir)
    assert label_paths, f"--net is not a network-pair list: {net_field!r}"
    # MATCH BY NAME: the spec declares networks (f, g); pick each subnet's onnx by
    # its label, not by list position. f = the base (first declared); g = the second
    # (or f itself for an `equal-to` spec that reuses one net).
    decl = declared_networks(text)
    assert decl, f"spec declares no networks: {vnnlib_path}"
    missing = [lbl for lbl in decl if lbl not in label_paths]
    assert not missing, (f"--net provides networks {sorted(label_paths)} but the spec "
                         f"declares {decl} (missing {missing}): {net_field!r}")
    nf = label_paths[decl[0]]
    ng = label_paths[decl[1]] if len(decl) > 1 else nf
    out_onnx, out_vnnlib = _cache_paths(net_field, vnnlib_path)
    ir = parse_multinet(text)
    out_dim = merge(nf, ng, ir, out_onnx)
    emit_v1(ir, out_dim, out_vnnlib)
    if run_oracle:
        w = oracle(nf, ng, ir, out_onnx, out_dim)
        assert w < ORACLE_TOL, f"network-pair merge oracle FAIL ({w:.2e} >= {ORACLE_TOL})"
    return out_onnx, out_vnnlib
