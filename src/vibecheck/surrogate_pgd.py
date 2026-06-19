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
from onnx import helper, numpy_helper


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
        # box asserts: (>= NAME[i,j,..] LO) and (<= NAME[i,j,..] HI)
        strides = _c_strides(shape)
        for mm in re.finditer(rf'\(>=\s*{name}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
            lo[_flat(mm.group(1), strides)] = float(mm.group(2))
        for mm in re.finditer(rf'\(<=\s*{name}\[([\d,]+)\]\s*([-\d.eE]+)\)', txt):
            hi[_flat(mm.group(1), strides)] = float(mm.group(2))
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
    for mm in re.finditer(r'\(>=\s*X_(\d+)\s*([-\d.eE]+)\)', txt):
        lo[int(mm.group(1))] = float(mm.group(2))
    for mm in re.finditer(r'\(<=\s*X_(\d+)\s*([-\d.eE]+)\)', txt):
        hi[int(mm.group(1))] = float(mm.group(2))
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

def _ort_eval(onnx_path, feed):
    import onnxruntime as ort
    sess = ort.InferenceSession(_decompressed(onnx_path), providers=['CPUExecutionProvider'])
    names = [i.name for i in sess.get_inputs()]
    return np.asarray(sess.run(None, {names[k]: feed[k].astype(np.float32) for k in range(len(names))})[0]).ravel()


def _decompressed(path):
    if path.endswith('.gz'):
        return _load_onnx_model(path).SerializeToString()
    return path


# ------------------------------------------------------------------------------- PGD

def surrogate_attack(onnx_path, vnnlib_path, settings, timeout, surrogate_path=None, log=print):
    """Run surrogate-PGD. Returns (verdict, witness) where verdict in {'sat','timeout',
    'unknown'} and witness is a list of per-input np.ndarrays (None unless sat)."""
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
    atol = float(getattr(settings, 'sat_validate_atol', 1e-4))

    if surrogate_path is None or not os.path.exists(surrogate_path):
        surrogate_path = (surrogate_path or '/tmp/_vibecheck_surrogate.onnx')
        build_float_surrogate(onnx_path, surrogate_path)
        log(f'[surrogate] built float surrogate -> {surrogate_path}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = convert(surrogate_path).eval().to(dev)
    log(f'[surrogate] loaded on {dev} in {time.time()-t0:.1f}s; '
        f'inputs={[(n, s) for n, s, _, _ in spec.inputs]} restarts={restarts} steps={steps}')

    def to_t(a, shp):
        return torch.tensor(a.astype(np.float32).reshape(tuple(shp)), device=dev)
    los = [to_t(lo, mshapes[k]) for k, (_, _, lo, _) in enumerate(spec.inputs)]
    his = [to_t(hi, mshapes[k]) for k, (_, _, _, hi) in enumerate(spec.inputs)]
    cens = [(l + h) / 2 for l, h in zip(los, his)]

    def viol_loss(y):
        # maximize the best-clause violation margin (smooth: max over clauses of min over constraints)
        clause_vals = []
        for clause in spec.out_dnf:
            margins = [(y[i] - rhs) if op == 'gt' else (rhs - y[i]) for i, op, rhs in clause]
            clause_vals.append(torch.stack(margins).min())
        return torch.stack(clause_vals).max()

    def validate(pts):
        feed = [p.detach().cpu().numpy().reshape(mshapes[k])
                for k, p in enumerate(pts)]
        # in-box invariant: PGD projects every step to [lo,hi], so a witness is in-box by
        # construction. Assert it (loud on any future projection bug) rather than silently
        # dropping — an out-of-box witness must never reach the original-model replay.
        for f, (_, _, lo, hi) in zip(feed, spec.inputs):
            ff = f.ravel()
            assert (ff >= lo - atol).all() and (ff <= hi + atol).all(), \
                'surrogate PGD produced an out-of-box witness'
        y = _ort_eval(onnx_path, feed)
        # STRICT output violation (no atol slack toward the boundary) so the on-threshold
        # center (e.g. Y==rhs) is NOT a sat; PGD must find a clear crossing. atol is only
        # for the in-box check above.
        return spec.violated(y, atol=0.0), (feed, y)

    rng = torch.Generator(device='cpu')
    best_verdict = 'unknown'
    for r in range(restarts):
        if r == 0:
            pts = [c.clone() for c in cens]
        else:
            rng.manual_seed(r)
            pts = [l + (h - l) * torch.rand(l.shape, generator=rng).to(dev) for l, h in zip(los, his)]
        # quick center/restart check first (covers trivially-SAT)
        ok, wy = validate(pts)
        if ok:
            log(f'[surrogate] SAT at restart {r} init (validated on original ORT-CPU)')
            return 'sat', wy[0]
        for it in range(steps):
            if time.time() - t0 > timeout:
                log(f'[surrogate] timeout after {time.time()-t0:.1f}s (restart {r}, step {it})')
                return ('timeout' if best_verdict != 'sat' else 'sat'), None
            for p in pts:
                p.requires_grad_(True)
            y = model(*pts)
            y = (y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
            loss = viol_loss(y)
            grads = torch.autograd.grad(loss, pts)
            with torch.no_grad():
                pts = [torch.minimum(torch.maximum(p + (h - l) / 2 * g.sign(), l), h)
                       for p, g, l, h in zip(pts, grads, los, his)]
            ok, wy = validate(pts)
            if ok:
                log(f'[surrogate] SAT at restart {r} step {it+1} '
                    f'(validated on original ORT-CPU; t={time.time()-t0:.1f}s)')
                return 'sat', wy[0]
    log(f'[surrogate] no counterexample after {restarts}x{steps} (t={time.time()-t0:.1f}s)')
    return 'unknown', None
