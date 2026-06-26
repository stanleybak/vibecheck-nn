"""Surrogate-PGD attack mode for models vibecheck cannot build a sound symbolic
graph for — specifically INT8-quantized ONNX (DequantizeLinear/QuantizeLinear), whose
ops vibecheck's graph and ABC's onnx2pytorch both reject.

Idea (incomplete / attack-only — never returns unsat):
  1. detect quantized ops; fold Q/DQ to a continuous FLOAT surrogate ONNX
     (weight DequantizeLinear -> float const incl. per-axis; activation Q/DQ -> Identity).
  2. load the surrogate with onnx2torch (robust for transformers) -> torch autograd on GPU.
  3. PGD for the whole timeout, maximizing the output-spec violation over the L-inf box,
     using the SURROGATE only for the gradient direction.
  4. validate every candidate by replaying it on the ORIGINAL (quantized) ONNX with CPU
     onnxruntime (the VNNCOMP scoring engine): witness must be in-box AND violate the
     output spec within atol. The verdict is decided ONLY by the original model.

The surrogate is the gradient oracle; soundness of a `sat` rests entirely on the ORT-CPU
replay of the original model (so a mismatched surrogate can never produce a false sat).
"""
import gzip
import os
import re
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# --------------------------------------------------------------------------- detect

def has_quantized_ops(onnx_path):
    """True if the ONNX uses DequantizeLinear/QuantizeLinear (INT8 quantization)."""
    model = _load_onnx_model(onnx_path)
    return any(n.op_type in ('DequantizeLinear', 'QuantizeLinear') for n in model.graph.node)


def _load_onnx_model(path):
    if path.endswith('.gz'):
        with gzip.open(path) as fh:
            return onnx.load_model_from_string(fh.read())
    return onnx.load(path)


def _model_input_shapes(onnx_path):
    """Free-input (non-initializer) shapes of the ONNX, in graph order — the authoritative
    tensor shapes for feeding the model (the spec only carries a flat per-index box)."""
    m = _load_onnx_model(onnx_path)
    init = {i.name for i in m.graph.initializer}
    return [[d.dim_value if d.dim_value > 0 else 1 for d in i.type.tensor_type.shape.dim]
            for i in m.graph.input if i.name not in init]


# ---------------------------------------------------------------------- fold surrogate

def build_float_surrogate(onnx_path, out_path):
    """Fold Q/DQ into a continuous float ONNX (the STE surrogate). Returns out_path.

    weight DequantizeLinear (data is an initializer) -> baked float constant;
    activation Quantize/Dequantize -> Identity (drops the rounding => differentiable)."""
    m = _load_onnx_model(onnx_path)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    new_nodes, add_init = [], []
    for n in g.node:
        if n.op_type == 'QuantizeLinear':
            new_nodes.append(helper.make_node('Identity', [n.input[0]], [n.output[0]]))
            continue
        if n.op_type == 'DequantizeLinear':
            x = n.input[0]
            if x in init:                                       # weight/const -> float
                w = init[x].astype(np.float64)
                s = init[n.input[1]].astype(np.float64)
                z = init[n.input[2]].astype(np.float64) if len(n.input) > 2 and n.input[2] in init else 0.0
                axis = next((a.i for a in n.attribute if a.name == 'axis'), 1)
                if np.ndim(s) > 0:                              # per-axis scale
                    shp = [1] * w.ndim
                    shp[axis % w.ndim] = s.shape[0]
                    s = s.reshape(shp)
                    z = np.reshape(z, shp) if np.ndim(z) > 0 else z
                add_init.append(numpy_helper.from_array(((w - z) * s).astype(np.float32), n.output[0]))
            else:                                               # activation -> identity
                new_nodes.append(helper.make_node('Identity', [x], [n.output[0]]))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(add_init)
    keep = [o for o in m.opset_import if (o.domain or 'ai.onnx') in ('ai.onnx', '')]  # drop unused custom domains
    del m.opset_import[:]
    m.opset_import.extend(keep)
    m.ir_version = 8
    onnx.save(m, out_path)
    return out_path


def build_fakequant_surrogate(onnx_path, out_path):
    """Fold Q/DQ into a continuous float ONNX that REPRODUCES the INT8 rounding (Path B).

    Like build_float_surrogate, but each ACTIVATION QuantizeLinear/DequantizeLinear becomes
    an explicit FAKE-QUANT in float ops instead of Identity:
      QuantizeLinear(x)    -> Clip(Round(x/scale) + zp, qmin, qmax)   (kept in float)
      DequantizeLinear(q)  -> (q - zp) * scale
    so the model carries the activation rounding the float (STE) surrogate drops. Round is
    ONNX round-half-to-even. NON-differentiable (Round grad is 0 a.e.); a fast GPU eval oracle
    to rank PGD candidates before the authoritative ORT-CPU confirm.

    Fidelity (MEASURED on smart_turn, 61 points spanning 4 quant cells): the fold matches the
    original quantized model under ORT on ~59/61, diverging only at EXACT cell boundaries — a
    rounding TIE where ONNX `Round` and ORT's fused `QuantizeLinear` pick different int codes
    (localized to the first conv's quant; one flipped code -> a ~0.06 output swing on this
    boundary-sensitive model). A reciprocal-multiply variant `round(x*(1/scale))` was tried and
    made NO difference (same 2/61), so it is the rounding tie, not the division. Executed on a
    different float backend (onnx2torch GPU), the conv/matmul accumulation adds a couple more
    boundary flips (3/61). This is the SAME boundary float-sensitivity as a CPU/arch change
    (the box-vs-local platform effect), not a fixable bug; the fq eval is therefore an
    APPROXIMATE ranking oracle and ORT-CPU remains the deciding oracle (the attack's ORT gate
    is conservative — it only skips the ORT confirm when fq is CLEARLY safe). Weight/bias
    DequantizeLinear (initializer input) is baked to a float constant as build_float_surrogate."""
    m = _load_onnx_model(onnx_path)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    init_dtype = {i.name: i.data_type for i in g.initializer}
    new_nodes, add_init = [], []
    uid = [0]

    def _const(arr, base):
        uid[0] += 1
        nm = f'_fq_{base}_{uid[0]}'
        add_init.append(numpy_helper.from_array(np.asarray(arr, np.float32), nm))
        return nm

    def _tmp(base):
        uid[0] += 1
        return f'_fq_{base}_{uid[0]}'

    for n in g.node:
        if n.op_type == 'QuantizeLinear':
            x = n.input[0]
            if x in init:
                raise NotImplementedError('fake-quant: QuantizeLinear with initializer input '
                                          '(weight quant) not supported — expected activation only')
            s = init[n.input[1]].astype(np.float64)
            z = init[n.input[2]].astype(np.float64) if len(n.input) > 2 and n.input[2] in init else 0.0
            if np.size(s) > 1:
                raise NotImplementedError('fake-quant: per-axis activation QuantizeLinear '
                                          '(non-scalar scale) needs axis-aware broadcast')
            zdt = init_dtype.get(n.input[2], TensorProto.UINT8) if len(n.input) > 2 else TensorProto.UINT8
            qmin, qmax = (0.0, 255.0) if zdt == TensorProto.UINT8 else (-128.0, 127.0)
            s_nm, z_nm = _const(s, 'qs'), _const(z, 'qz')
            lo_nm, hi_nm = _const(qmin, 'qlo'), _const(qmax, 'qhi')
            t_div, t_rnd, t_add = _tmp('qdiv'), _tmp('qrnd'), _tmp('qadd')
            new_nodes.append(helper.make_node('Div', [x, s_nm], [t_div]))
            new_nodes.append(helper.make_node('Round', [t_div], [t_rnd]))
            new_nodes.append(helper.make_node('Add', [t_rnd, z_nm], [t_add]))
            new_nodes.append(helper.make_node('Clip', [t_add, lo_nm, hi_nm], [n.output[0]]))
            continue
        if n.op_type == 'DequantizeLinear':
            x = n.input[0]
            s = init[n.input[1]].astype(np.float64)
            z = init[n.input[2]].astype(np.float64) if len(n.input) > 2 and n.input[2] in init else 0.0
            axis = next((a.i for a in n.attribute if a.name == 'axis'), 1)
            if x in init:                                       # weight/bias const -> baked float
                w = init[x].astype(np.float64)
                if np.ndim(s) > 0:
                    shp = [1] * w.ndim
                    shp[axis % w.ndim] = s.shape[0]
                    s = s.reshape(shp)
                    z = np.reshape(z, shp) if np.ndim(z) > 0 else z
                add_init.append(numpy_helper.from_array(((w - z) * s).astype(np.float32), n.output[0]))
            else:                                               # activation DQ -> (q - z) * s
                if np.size(s) > 1:
                    raise NotImplementedError('fake-quant: per-axis activation DequantizeLinear '
                                              '(non-scalar scale) needs axis-aware broadcast')
                s_nm, z_nm = _const(s, 'ds'), _const(z, 'dz')
                t_sub = _tmp('dsub')
                new_nodes.append(helper.make_node('Sub', [x, z_nm], [t_sub]))
                new_nodes.append(helper.make_node('Mul', [t_sub, s_nm], [n.output[0]]))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(add_init)
    keep = [o for o in m.opset_import if (o.domain or 'ai.onnx') in ('ai.onnx', '')]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    m.ir_version = 8
    onnx.save(m, out_path)
    return out_path


# ----------------------------------------------------------------------------- spec

class SurrogateSpec:
    """Minimal box+output spec for the attack: per-input L-inf box and an output DNF.

    inputs:  list of (name, shape, lo_flat, hi_flat) in ONNX input order.
    out_dnf: list of clauses; each clause is a list of (out_index, op, rhs) where op in
             {'gt','lt'} meaning the VIOLATION region is out[i] > rhs (gt) / < rhs (lt).
             A witness violates the spec iff SOME clause has ALL its constraints hold.
    """

    def __init__(self, inputs, out_dnf):
        self.inputs = inputs
        self.out_dnf = out_dnf

    def violated(self, y, atol=0.0):
        for clause in self.out_dnf:
            if all((y[i] > rhs - atol) if op == 'gt' else (y[i] < rhs + atol) for i, op, rhs in clause):
                return True
        return False


def parse_box_and_output(vnnlib_path):
    """Parse a v1 OR v2 box-robustness spec into a SurrogateSpec. Supports per-input
    boxes (multi-input v2) and an output DNF of single-output threshold constraints
    (the L-inf-robustness / classification case the surrogate mode targets)."""
    # Resolve via ensure_decompressed: instances.csv references the PLAIN name while the
    # benchmark ships only `.gz` (raw open() then FileNotFoundError'd the smart_turn sweep),
    # and a stale decompressed sibling older than the `.gz` (the smart_turn vnnlib was
    # regenerated; the local unzip predates it) gets re-inflated to the current spec.
    from .io_util import ensure_decompressed
    vnnlib_path = ensure_decompressed(vnnlib_path)
    if vnnlib_path.endswith('.gz'):
        with gzip.open(vnnlib_path, 'rt') as fh:
            txt = fh.read()
    else:
        txt = open(vnnlib_path).read()
    is_v2 = 'vnnlib-version' in txt or 'declare-network' in txt or 'declare-input' in txt
    if is_v2:
        return _parse_v2(txt)
    return _parse_v1(txt)


def _parse_v2(txt):
    # input tensors: (declare-input NAME real [d0, d1, ...])
    inputs = []
    for m in re.finditer(r'\(declare-input\s+(\w+)\s+\w+\s*\[([\d,\s]+)\]\)', txt):
        name = m.group(1)
        shape = tuple(int(x) for x in m.group(2).split(','))
        n = int(np.prod(shape))
        lo = np.full(n, -np.inf, np.float64)
        hi = np.full(n, np.inf, np.float64)
        # box asserts: (>= NAME[i,j,..] LO) and (<= NAME[i,j,..] HI).
        # VECTORIZED scatter: a high-dim spec (smart_turn ~1.2M bounds, 124 MB)
        # parsed per-match with a Python _flat() call took ~7 s/case; batching the
        # index/value extraction + numpy stride-dot + scatter cuts it to <1 s.
        strides = np.asarray(_c_strides(shape), dtype=np.int64)
        for op, arr in (('>=', lo), ('<=', hi)):
            pairs = re.findall(rf'\({op}\s*{name}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt)
            if not pairs:
                continue
            idx_strs, val_strs = zip(*pairs)
            flat = np.array([s.split(',') for s in idx_strs],
                            dtype=np.int64) @ strides
            arr[flat] = np.asarray(val_strs, dtype=np.float64)
        inputs.append((name, shape, lo, hi))
    # output name (single output tensor assumed)
    om = re.search(r'\(declare-output\s+(\w+)\s', txt)
    yname = om.group(1) if om else 'Y'
    # output threshold constraints: (> Y[..] c) / (< Y[..] c). Treat top-level asserts as
    # an AND of single-clause constraints (smart_turn is one constraint); each becomes a
    # one-element clause (a sufficient violation condition).
    out_dnf = []
    for mm in re.finditer(rf'\(>\s*{yname}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1).split(',')[-1]), 'gt', float(mm.group(2)))])
    for mm in re.finditer(rf'\(<\s*{yname}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1).split(',')[-1]), 'lt', float(mm.group(2)))])
    if not inputs or not out_dnf:
        raise NotImplementedError('surrogate spec parse: unsupported v2 structure '
                                  f'(inputs={len(inputs)}, out_dnf={len(out_dnf)})')
    return SurrogateSpec(inputs, out_dnf)


def _parse_v1(txt):
    n = len(re.findall(r'\(declare-const\s+X_\d+\s+Real\)', txt))
    lo = np.full(n, -np.inf, np.float64)
    hi = np.full(n, np.inf, np.float64)
    # VECTORIZED scatter (same rationale as _parse_v2): batch index/value extract.
    for op, arr in (('>=', lo), ('<=', hi)):
        pairs = re.findall(rf'\({op}\s*X_(\d+)\s*([-\d.eE]+)\)', txt)
        if pairs:
            idx, val = zip(*pairs)
            arr[np.asarray(idx, dtype=np.int64)] = np.asarray(val, dtype=np.float64)
    out_dnf = []
    for mm in re.finditer(r'\(>\s*Y_(\d+)\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1)), 'gt', float(mm.group(2)))])
    for mm in re.finditer(r'\(<\s*Y_(\d+)\s*([-\d.eE]+)\)', txt):
        out_dnf.append([(int(mm.group(1)), 'lt', float(mm.group(2)))])
    if n == 0 or not out_dnf:
        raise NotImplementedError('surrogate spec parse: unsupported v1 structure')
    return SurrogateSpec([('X', (n,), lo, hi)], out_dnf)


def _c_strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def _flat(idx_str, strides):
    return int(sum(int(a) * s for a, s in zip(idx_str.split(','), strides)))


# ------------------------------------------------------------------------ ORT validate

_ORT_SESSION_CACHE = {}


def _ort_eval(onnx_path, feed):
    import onnxruntime as ort
    # Cache the InferenceSession per model: building it LOADS + optimizes the ONNX
    # (~0.53s for the 1.1GB smart_turn model), and surrogate-attack calls this once
    # PER PGD STEP to confirm a witness — re-creating it each call made the replay,
    # not inference, dominate wall (78s/100s on smart_turn). One session per model.
    sess = _ORT_SESSION_CACHE.get(onnx_path)
    if sess is None:
        sess = ort.InferenceSession(_decompressed(onnx_path),
                                    providers=['CPUExecutionProvider'])
        _ORT_SESSION_CACHE[onnx_path] = sess
    names = [i.name for i in sess.get_inputs()]
    return np.asarray(sess.run(None, {names[k]: feed[k].astype(np.float32) for k in range(len(names))})[0]).ravel()


def _decompressed(path):
    if path.endswith('.gz'):
        return _load_onnx_model(path).SerializeToString()
    return path


# ------------------------------------------------------------------------------- PGD

def surrogate_attack(onnx_path, vnnlib_path, settings, timeout, surrogate_path=None, log=print):
    """Run surrogate-PGD. Returns (verdict, witness) where verdict in {'sat','timeout',
    'unknown'} and witness is a list of per-input np.ndarrays (None unless sat).

    Candidates considered (all ORT-CPU-confirmed on the ORIGINAL quantized model, the
    authoritative oracle): the box CENTER, the box CORNERS, and each PGD restart's best
    point. PGD gradients come from the float (STE) surrogate; the FAKE-QUANT surrogate
    (Path B), which reproduces the INT8 rounding and so tracks ORT, RANKS candidates so the
    most promising hits ORT first. Disposition (VNN-COMP 2026 output-strict rule): only a
    CLEAR CE (the output STRICTLY crosses the threshold) returns 'sat'. A boundary point
    (output == threshold, e.g. a quantization-pinned Y==rhs) is NOT a counterexample, so if
    no strict CE is found this incomplete mode returns 'timeout' (not a within-tol sat)."""
    import torch
    from onnx2torch import convert

    t0 = time.time()
    spec = parse_box_and_output(vnnlib_path)
    # Use the MODEL's input shapes (spec only carries a flat per-index box); reconciles
    # v1 (flat X_i) and v2 (declared shape) with the real model. Fail fast on a mismatch.
    mshapes = _model_input_shapes(onnx_path)
    if len(mshapes) != len(spec.inputs):
        raise NotImplementedError(
            f'surrogate spec inputs ({len(spec.inputs)}) != model inputs ({len(mshapes)})')
    restarts = int(getattr(settings, 'surrogate_attack_restarts', 1))
    steps = int(getattr(settings, 'surrogate_attack_steps', 50))
    # `sat_validate_atol` (1e-4) is the INPUT-box tolerance only (used for the
    # in-box assertion). The replayed OUTPUT must violate with NO tolerance
    # (VNN-COMP 2026), so a CE is accepted iff a candidate STRICTLY crosses the
    # output threshold; there is no within-output-tolerance fallback.
    atol = float(getattr(settings, 'sat_validate_atol', 1e-4))
    # Strict output constraints (`>`/`<`): require the output to cross the threshold
    # by at least this buffer (in float64) so a point sitting exactly on the
    # threshold (e.g. a quantization-pinned Y == c) is NOT a counterexample and an
    # emitted CE robustly satisfies the strict, zero-tolerance competition check.
    # (A bare next-float shift is invisible in float32 and gave false sats.)
    strict_buffer = float(getattr(settings, 'sat_strict_buffer', 1e-9))
    use_quant_eval = bool(getattr(settings, 'surrogate_quant_eval', True))

    if surrogate_path is None or not os.path.exists(surrogate_path):
        surrogate_path = (surrogate_path or '/tmp/_vibecheck_surrogate.onnx')
        build_float_surrogate(onnx_path, surrogate_path)
        log(f'[surrogate] built float surrogate -> {surrogate_path}')
    # Device follows settings.device (GPU default). The float surrogate's forward is
    # GPU-architecture-dependent (cuBLAS/cuDNN reduction order), so PGD's sign-steps can
    # follow a different trajectory on a different GPU — but that no longer loses CEs: the
    # FAKE-QUANT eval model (Path B) reproduces the INT8 rounding and RANKS whatever points
    # the trajectory visits, the box CENTER/CORNERS are checked regardless of trajectory,
    # and the authoritative ORT-CPU replay of the ORIGINAL model decides 'sat'.
    _want_gpu = (getattr(settings, 'device', 'gpu') == 'gpu')
    dev = 'cuda' if (_want_gpu and torch.cuda.is_available()) else 'cpu'
    model = convert(surrogate_path).eval().to(dev)
    eval_model = None
    if use_quant_eval:
        fq_path = (surrogate_path[:-5] if surrogate_path.endswith('.onnx') else surrogate_path) + '_fq.onnx'
        if not os.path.exists(fq_path):
            build_fakequant_surrogate(onnx_path, fq_path)
        eval_model = convert(fq_path).eval().to(dev)
    log(f'[surrogate] loaded on {dev} in {time.time()-t0:.1f}s; '
        f'inputs={[(n, s) for n, s, _, _ in spec.inputs]} restarts={restarts} steps={steps} '
        f'quant_eval={"on" if eval_model is not None else "off"}')

    def to_t(a, shp):
        return torch.tensor(a.astype(np.float32).reshape(tuple(shp)), device=dev)
    los = [to_t(lo, mshapes[k]) for k, (_, _, lo, _) in enumerate(spec.inputs)]
    his = [to_t(hi, mshapes[k]) for k, (_, _, _, hi) in enumerate(spec.inputs)]
    cens = [(l + h) / 2 for l, h in zip(los, his)]

    def viol_loss(y):
        # smooth surrogate loss for the GRADIENT: max over clauses of min over constraints.
        clause_vals = []
        for clause in spec.out_dnf:
            margins = [(y[i] - rhs) if op == 'gt' else (rhs - y[i]) for i, op, rhs in clause]
            clause_vals.append(torch.stack(margins).min())
        return torch.stack(clause_vals).max()

    def margin_np(y):
        # Violation margin on a numpy output, computed in FLOAT64. (A float32 y minus
        # a python-float rhs collapses to float32 under numpy-2 NEP-50 promotion,
        # which would hide a sub-float32 strict buffer — so cast each element to
        # float64 first.) margin >= strict_buffer means the output crossed the
        # threshold by a robust amount.
        best = -np.inf
        for clause in spec.out_dnf:
            m = min((float(y[i]) - rhs) if op == 'gt' else (rhs - float(y[i]))
                    for i, op, rhs in clause)
            best = max(best, m)
        return float(best)

    def fq_margin(pts):
        # fake-quant GPU eval margin for RANKING (≈ORT, no ORT cost); None if quant_eval off.
        if eval_model is None:
            return None
        with torch.no_grad():
            y = eval_model(*pts)
            y = (y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
        return margin_np(y.detach().cpu().numpy())

    _t_val = _t_fwd = 0.0
    _n_steps = _n_val = 0

    def ort_consider(pts, tag):
        """ORT-CPU replay of the ORIGINAL quantized model. CLEAR CE (strict output
        crossing, margin > 0) -> return ('sat', (feed,y)). Otherwise -> None (a
        boundary/non-violating point is not a counterexample)."""
        nonlocal _t_val, _n_val
        feed = [p.detach().cpu().numpy().reshape(mshapes[k]) for k, p in enumerate(pts)]
        # in-box invariant: every candidate is built inside [lo,hi] (center/corner/projected
        # PGD); assert it loudly rather than silently shipping an out-of-box witness.
        for f, (_, _, lo, hi) in zip(feed, spec.inputs):
            ff = f.ravel()
            assert (ff >= lo - atol).all() and (ff <= hi + atol).all(), \
                'surrogate produced an out-of-box witness'
        _v0 = time.time()
        y = _ort_eval(onnx_path, feed)
        _t_val += time.time() - _v0
        _n_val += 1
        m = margin_np(y)
        # Strict `>`/`<`: accept only if the output crosses the threshold by at
        # least strict_buffer (float64). A point exactly on the threshold (e.g.
        # quantization-pinned Y == 0.5 for `Y > 0.5`) has m == 0 < buffer and is
        # skipped; m >= buffer means Y robustly satisfies the strict, zero-tolerance
        # competition check.
        if m >= strict_buffer:
            log(f'[surrogate] CLEAR SAT at {tag} (ORT margin={m:.3e})')
            return ('sat', (feed, y))
        # m <= 0: the output does NOT strictly violate -> not a counterexample
        # (VNN-COMP 2026 output-strict rule). There is no within-tolerance fallback
        # any more; keep searching for a strict CE.
        return None

    rng = torch.Generator(device='cpu')
    _bs = getattr(settings, 'pgd_seed', None)
    base_seed = int(_bs) if isinstance(_bs, (int, float)) else None
    alphas = list(getattr(settings, 'surrogate_alphas', None) or [0.05, 0.1, 0.2, 0.02])

    # 1) CENTER. For a quantized model whose output is constant over the box (the box sits
    #    inside one quantization cell, so PGD can't move the output), the center value IS
    #    the verdict — clear SAT (e.g. Y=0.918 > 0.5) or within-tol (e.g. Y pinned at 0.5).
    res = ort_consider(cens, 'center')
    if res is not None:
        return 'sat', res[1][0]

    # 2) PGD restarts on the float surrogate (gradient). The first restart is seeded from
    #    the CENTER, the rest from seeded RANDOM in-box points (`pgd_seed + r`). Each step is
    #    a GRADUAL L-inf step `alpha*(h-l)*sign(grad)` (alpha < 1, several steps — NOT a
    #    single jump to a box vertex) CLAMPED back into [lo,hi]. No box-corner enumeration
    #    (1.27 M dims). When the center is only within-tol, this is exactly the "keep
    #    searching for a clear CE" pass; the within-tol witness stays stashed as the fallback.
    for r in range(restarts):
        if time.time() - t0 > timeout:
            break
        alpha = alphas[r % len(alphas)]
        if r == 0:
            pts = [c.clone() for c in cens]
        else:
            rng.manual_seed((base_seed + r) if base_seed is not None else r)
            pts = [l + (h - l) * torch.rand(l.shape, generator=rng).to(dev) for l, h in zip(los, his)]
        best_loss = float('-inf')
        best_pts = [p.detach().clone() for p in pts]
        best_fq = float('-inf')         # best fake-quant margin seen (the ≈ORT proxy)
        best_fq_pts = None
        for it in range(steps):
            if time.time() - t0 > timeout:
                break
            _f0 = time.time()
            for p in pts:
                p.requires_grad_(True)
            y = model(*pts)
            y = (y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
            loss = viol_loss(y)
            _lv = float(loss.detach())
            if _lv > best_loss:        # most-violating SURROGATE point (pre-step)
                best_loss = _lv
                snap = [p.detach().clone() for p in pts]
                best_pts = snap
                # EARLY-CONFIRM: this is a newly-promising point — score it with the
                # accurate fake-quant eval (≈ORT). If fake-quant says it CLEARLY violates,
                # ORT-confirm it RIGHT NOW (return at the step the CE appears, not after all
                # `steps`). fq is checked only on surrogate-improving steps to bound its cost.
                fqm_s = fq_margin(snap)
                if fqm_s is not None and fqm_s > best_fq:
                    best_fq, best_fq_pts = fqm_s, snap
                if fqm_s is not None and fqm_s >= strict_buffer:
                    res = ort_consider(snap, f'restart{r} step{it} (a={alpha},fq={fqm_s:.3e})')
                    if res is not None:
                        _t_fwd += time.time() - _f0
                        return 'sat', res[1][0]
            grads = torch.autograd.grad(loss, pts)
            with torch.no_grad():
                pts = [torch.minimum(torch.maximum(p + alpha * (h - l) * g.sign(), l), h)
                       for p, g, l, h in zip(pts, grads, los, his)]
            _t_fwd += time.time() - _f0
            _n_steps += 1
        # End of restart: confirm the best candidate. Prefer the best fake-quant point (the
        # ≈ORT proxy) over the best surrogate point; gate the (slower) ORT replay so we only
        # confirm when fake-quant says it at least reaches within-tol (skip when clearly safe).
        # No eval oracle -> always confirm the best surrogate point.
        cand = best_fq_pts if best_fq_pts is not None else best_pts
        fqm = best_fq if best_fq_pts is not None else fq_margin(best_pts)
        if fqm is None or fqm >= -atol:
            res = ort_consider(cand, f'restart{r}(a={alpha}'
                               + (f',fq={fqm:.3e})' if fqm is not None else ')'))
            if res is not None:
                return 'sat', res[1][0]

    # 3) No strict CE found. This is an incomplete (attack-only) mode that cannot
    #    prove unsat, so "didn't find one in the budget" is a timeout (more
    #    time/restarts might find a CE), not a definitive unknown.
    log(f'[surrogate] no CE (t={time.time()-t0:.1f}s; steps={_n_steps} '
        f'fwd={_t_fwd:.1f}s validate={_t_val:.1f}s/{_n_val})')
    return 'timeout', None
