"""Nonlinear v2 benchmark support via network augmentation (adaptive_cruise_2026).

Some 2026 v2 specs are NONLINEAR: the DNF's atoms are polynomials in X and the
network output Y, up to degree 2 (incl. input*output coupling and a nonlinear
input constraint like 200*X0 >= X1^2). vibecheck's pipeline verifies a net against
a LINEAR spec, so we transpile: build an AUGMENTED ONNX that runs the original net
f to get Y, computes the feature vector v = [all distinct monomials of X and Y],
and emits each constraint value p_c(X,Y) = W_c . v + b_c as an extra output via one
Gemm. The spec then becomes a LINEAR DNF over those outputs:
    clause -> (and (<= Y_c 0) ...).
The threshold is 0 for both strict and non-strict atoms: {Y_c <= 0} = {p_c <= 0}
is the closure of the strict unsafe set {p_c < 0} and exact for {p_c <= 0}, hence a
SUPERSET of the true unsafe set in both cases -> proving it empty soundly proves
`unsat`. (A strict threshold of -MARGIN would encode a strict SUBSET and could
false-unsat a shallow CE; the boundary over-inclusion at p_c == 0 is caught by the
ORT witness re-validation, so threshold 0 cannot false-sat.)
A nonlinear INPUT constraint g(X) {<,<=} 0 is just another atom -> another output,
folded into the DNF (so "g(X) violated OR property holds" over the full box, which
is sound: points failing g are outside the real input region).

The polynomial DNF is produced by vibecheck's own v2 parser (parse_vnnlib_v2 ->
PolynomialConstraint terms/bias/strict); only the linear-only VNNSpec adapter
rejects degree>=2, which this module bypasses. Correctness is gated by an
onnxruntime oracle (augmented output == polynomial(X, f(X))). Self-contained
(reuses network_pair's onnx helpers) so it can run in prepare_instance.
"""
import os
import hashlib
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from .network_pair import (_load_onnx, _read_vnnlib_text, _prep, _free_input,
                           _const, TARGET_OPSET)
from .vnnlib_loader import parse_vnnlib_v2, detect_version, VnnlibParseError


def _var_idx(v):
    """'X_0' -> ('X', 0), 'Y_3' -> ('Y', 3)."""
    return v[0], int(v[2:])


def is_nonlinear_v2_spec(vnnlib_text):
    """True if the spec is v2 AND has a degree>=2 monomial or an X*Y coupling
    (i.e. the linear-only adapter can't handle it -> needs augmentation)."""
    if detect_version(vnnlib_text) != '2.0':
        return False
    try:
        prop = parse_vnnlib_v2(vnnlib_text)
    except VnnlibParseError:
        # malformed/unsupported v2 spec — not something we augment; let the
        # normal loader surface the proper error.
        return False
    for cl in prop.spec.clauses:
        for c in cl.constraints:
            for mono, _ in c.terms:
                if len(mono) >= 2:        # degree>=2 (incl. X*Y, X^2, Y^2)
                    return True
    return False


def analyze(prop):
    """Return (feats, cons, clauses, xbox).
    feats: ordered distinct monomials (tuples of var names).
    cons: distinct (coef_row {feat_idx:coef}, bias, strict).
    clauses: list of constraint-index lists (the DNF).
    xbox: dict X-index -> [lo, hi] from the single-variable linear constraints."""
    feats, fidx = [], {}

    def feat_id(m):
        if m not in fidx:
            fidx[m] = len(feats); feats.append(m)
        return fidx[m]

    cons, cidx, clauses = [], {}, []
    xlo, xhi = {}, {}
    for cl in prop.spec.clauses:
        idxs = []
        for c in cl.constraints:
            # single-var linear X constraint -> also fold into the input box
            if (len(c.terms) == 1 and len(c.terms[0][0]) == 1
                    and c.terms[0][0][0].startswith('X')):
                (var,), coef = c.terms[0][0], c.terms[0][1]
                _, xi = _var_idx(var)
                if coef < 0:                       # -|c|X + b <= 0 -> X >= b/|c|
                    lo = c.bias / (-coef); xlo[xi] = min(xlo.get(xi, lo), lo)
                else:                              # cX + b <= 0 -> X <= -b/c
                    hi = -c.bias / coef; xhi[xi] = max(xhi.get(xi, hi), hi)
            row = {}
            for mono, coef in c.terms:
                row[feat_id(mono)] = row.get(feat_id(mono), 0.0) + coef
            key = (tuple(sorted(row.items())), c.bias, c.strict)
            if key not in cidx:
                cidx[key] = len(cons); cons.append((row, c.bias, c.strict))
            idxs.append(cidx[key])
        clauses.append(idxs)
    xbox = {i: [xlo.get(i, -1e6), xhi.get(i, 1e6)]
            for i in sorted(set(xlo) | set(xhi))}
    return feats, cons, clauses, xbox


def _feat_value_nodes(feats, x_in, y_out, in_dim, out_dim):
    """Nodes computing each monomial as a [batch,1] tensor; return
    (nodes, inits, feature_tensor_names)."""
    nodes, inits = [], []
    inits.append(_const('xshape', [-1, in_dim], TensorProto.INT64))
    inits.append(_const('yshape', [-1, out_dim], TensorProto.INT64))
    nodes.append(helper.make_node('Reshape', [x_in, 'xshape'], ['Xflat']))
    nodes.append(helper.make_node('Reshape', [y_out, 'yshape'], ['Yflat']))
    val = {}

    def lane(var):
        if var in val:
            return val[var]
        kind, i = _var_idx(var)
        src = 'Xflat' if kind == 'X' else 'Yflat'
        nm = f'{kind.lower()}_{i}'
        inits.append(_const(f'idx_{nm}', [i], TensorProto.INT64))
        nodes.append(helper.make_node('Gather', [src, f'idx_{nm}'], [nm], axis=1))
        val[var] = nm
        return nm

    fnames = []
    for k, mono in enumerate(feats):
        if len(mono) == 1:
            fnames.append(lane(mono[0]))
        else:
            a, b = lane(mono[0]), lane(mono[1])
            nm = f'mono_{k}'
            nodes.append(helper.make_node('Mul', [a, b], [nm]))
            fnames.append(nm)
    return nodes, inits, fnames


def augment(f_path, prop, out_path):
    feats, cons, clauses, xbox = analyze(prop)
    mf = _prep(_load_onnx(f_path), 'f_')
    fin = _free_input(mf).name
    fout = mf.graph.output[0].name
    in_shape = [d.dim_value for d in _free_input(mf).type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in mf.graph.output[0].type.tensor_type.shape.dim]
    in_dim = int(np.prod([d for d in in_shape if d > 0])) or in_shape[-1]
    out_dim = int(np.prod([d for d in out_shape if d > 0])) or out_shape[-1]
    n_in = in_shape[-1]
    M, F = len(cons), len(feats)

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, ['batch'] + in_shape[1:])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['batch', M])
    wire = [helper.make_node('Identity', ['X'], [fin], name='wire')]
    fnodes, finits, fnames = _feat_value_nodes(feats, fin, fout, in_dim, out_dim)
    catv = [helper.make_node('Concat', fnames, ['feat_v'], axis=1)]
    W = np.zeros((M, F), np.float32); b = np.zeros(M, np.float32)
    for c, (row, bias, _strict) in enumerate(cons):
        for fi, coef in row.items():
            W[c, fi] = coef
        b[c] = bias
    finits += [_const('Wc', W), _const('bc', b)]
    gemm = [helper.make_node('Gemm', ['feat_v', 'Wc', 'bc'], ['Y'], transB=1)]
    graph = helper.make_graph(
        nodes=wire + list(mf.graph.node) + fnodes + catv + gemm,
        name='nl_augmented', inputs=[X], outputs=[Y],
        initializer=list(mf.graph.initializer) + finits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', TARGET_OPSET)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return feats, cons, clauses, xbox, n_in, in_shape


def emit_v1(cons, clauses, xbox, n_in, out_path):
    L = ['; auto-generated v1 spec for the nonlinear-augmented net',
         '; Y_c = constraint polynomial c;  clause = AND of (p_c {<,<=} 0)', '']
    L += [f'(declare-const X_{i} Real)' for i in range(n_in)]
    L += [f'(declare-const Y_{c} Real)' for c in range(len(cons))]
    L.append('')
    for i in range(n_in):
        lo, hi = xbox.get(i, [-1e6, 1e6])
        L.append(f'(assert (<= X_{i} {hi!r}))')
        L.append(f'(assert (>= X_{i} {lo!r}))')
    L.append('')
    # rectangular clauses (equal length); pad by repeating the last constraint
    # (a no-op inside the AND) — keeps the disjunctive-spec handling uniform.
    #
    # Threshold is 0 for BOTH strict and non-strict atoms. The original atom is
    # p_c {<,<=} 0; its unsafe set is {p_c < 0} (strict) or {p_c <= 0}. Encoding
    # the unsafe set as {Y_c <= 0} = {p_c <= 0} is the closure of the strict set
    # and EXACT for the non-strict one — i.e. a SUPERSET of the true unsafe set in
    # both cases. Proving that superset empty soundly proves the true unsafe set
    # empty (sound `unsat`). A strict-threshold of -MARGIN would instead encode
    # {p_c <= -MARGIN}, a strict SUBSET missing the band (-MARGIN, 0): a real
    # shallow CE with p_c = -5e-5 would be declared safe -> FALSE UNSAT (unsound).
    # The only over-inclusion at the measure-zero boundary p_c == 0 (a strict
    # non-CE) is caught downstream by the ORT witness re-validation against the
    # original spec, so it cannot cause a false-sat.
    maxlen = max(len(idxs) for idxs in clauses)
    L.append('(assert (or')
    for idxs in clauses:
        padded = list(idxs) + [idxs[-1]] * (maxlen - len(idxs))
        parts = [f'(<= Y_{ci} 0.0)' for ci in padded]
        L.append('  (and ' + ' '.join(parts) + ')')
    L.append('))')
    open(out_path, 'w').write('\n'.join(L) + '\n')


def _poly_eval(cons, X, Y, feats):
    fv = np.empty(len(feats))
    val = {f'X_{i}': X[i] for i in range(len(X))}
    val.update({f'Y_{i}': Y[i] for i in range(len(Y))})
    for k, mono in enumerate(feats):
        p = 1.0
        for v in mono:
            p *= val[v]
        fv[k] = p
    out = np.empty(len(cons))
    for c, (row, bias, _strict) in enumerate(cons):
        out[c] = sum(coef * fv[fi] for fi, coef in row.items()) + bias
    return out


def oracle(f_path, aug_path, feats, cons, xbox, n_in, in_shape, n=120, seed=0):
    sf = ort.InferenceSession(_load_onnx(f_path).SerializeToString())
    sm = ort.InferenceSession(open(aug_path, 'rb').read())
    fin = _free_input(_load_onnx(f_path)).name
    rng = np.random.default_rng(seed)
    lo = np.array([xbox.get(i, [-1, 1])[0] for i in range(n_in)])
    hi = np.array([xbox.get(i, [-1, 1])[1] for i in range(n_in)])
    worst = 0.0
    for _ in range(n):
        x = (lo + (hi - lo) * rng.random(n_in)).astype(np.float32)
        y = sf.run(None, {fin: x.reshape(in_shape)})[0].flatten()
        am = sm.run(None, {'X': x.reshape([1] + in_shape[1:])})[0].flatten()
        ref = _poly_eval(cons, x, y, feats)
        worst = max(worst, float(np.abs(am - ref).max()))
    return worst


def _cache_paths(net_path, vnnlib_path):
    h = hashlib.md5((os.path.abspath(net_path) + '|'
                     + os.path.abspath(vnnlib_path)).encode()).hexdigest()[:12]
    d = tempfile.gettempdir()
    return (os.path.join(d, f'vibecheck_nlaug_{h}.onnx'),
            os.path.join(d, f'vibecheck_nlaug_{h}.vnnlib'))


def _tighten_xbox(prop, xbox, n_in):
    """Contract the emitted input box with the nonlinear input constraints
    (input_feasibility.tighten_input_box) so the verifier propagates over a
    smaller box. SOUND: the contracted box still contains the true input region
    (the nonlinear atoms remain enforced exactly in the spec, so the verdict is
    unchanged — only the bounds tighten). A tiny inflation, clamped to the
    declared box, absorbs any float rounding in the root solve so a near-boundary
    counterexample is never cut. No-op if nothing tightens / region is empty (the
    empty case is handled upstream by main._maybe_empty_input)."""
    from .input_feasibility import tighten_input_box
    init = [list(xbox.get(i, [-1e6, 1e6])) for i in range(n_in)]
    tb = tighten_input_box(prop, n_in=n_in, init_box=init)
    if tb is None or len(tb) < n_in:    # empty (handled upstream) or indeterminate
        return xbox
    INFL = 1e-6
    out = dict(xbox)
    for i in range(n_in):
        dlo, dhi = init[i]
        out[i] = [max(dlo, tb[i][0] - INFL), min(dhi, tb[i][1] + INFL)]
    return out


def build_augmented_instance(net_path, vnnlib_path, run_oracle=True):
    """Convert a nonlinear-v2 instance to (aug_onnx_path, aug_v1_spec_path).
    Raises if the oracle (augmented output vs the true polynomial) exceeds 5e-3."""
    text = _read_vnnlib_text(vnnlib_path)
    assert is_nonlinear_v2_spec(text), f"not a nonlinear v2 spec: {vnnlib_path}"
    prop = parse_vnnlib_v2(text)
    out_onnx, out_vnnlib = _cache_paths(net_path, vnnlib_path)
    feats, cons, clauses, xbox, n_in, in_shape = augment(net_path, prop, out_onnx)
    xbox = _tighten_xbox(prop, xbox, n_in)
    emit_v1(cons, clauses, xbox, n_in, out_vnnlib)
    if run_oracle:
        w = oracle(net_path, out_onnx, feats, cons, xbox, n_in, in_shape)
        assert w < 5e-3, f"nonlinear-augment oracle FAIL ({w:.2e} >= 5e-3)"
    return out_onnx, out_vnnlib
