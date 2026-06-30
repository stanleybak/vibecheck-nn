"""Monotone-output inversion verifier route (SETTINGS-GATED, default OFF).

CONVERTED-NETWORK/SPEC approach. When a spec output factors as

    Y_i = scale_i * g(z_i) + bias_i

with ``g`` a MONOTONE 1-D op (merged ``pwl`` / ``sigmoid`` / ``tanh``) and ``z``
produced by a BILINEAR-FREE linear/ReLU cone, we:

  1. INVERT each output threshold through ``g`` to an EXACT, SOUND linear threshold
     on ``z`` (``Y_i >= rhs  <=>  z_i >= t`` for scale>0, etc.). The threshold ``t``
     is forward-validated in fp32 and nudged to the conservative (sound) side.
  2. TRUNCATE the network at ``z`` (drop the head + everything downstream, incl.
     all bilinears) -> a pure ReLU/linear net that outputs ``z``.
  3. Hand the converted (truncated-net, inverted-z-spec) to the STANDARD prover
     (``verify_graph``: zono -> alpha-CROWN -> BaB), which tightens ``z`` with the
     full arsenal. Short-circuit to ``verified`` only when it proves the converted
     spec; otherwise return None -> caller falls through UNCHANGED.

Soundness: the truncation is exact (same computation up to z); the inverted
thresholds are conservative (converted-unsafe region CONTAINS the original-unsafe
region, validated by an fp32 forward of ``g(t)``); so converted-unreachable =>
original-safe. INERT unless the setting is on AND the structure matches.
"""
import copy
import numpy as np
import torch

from .nl_pwl import PWLRelax
from .spec import Constraint, Conjunct, VNNSpec
from .network import ComputeGraph

_MONO_OPS = ('pwl', 'sigmoid', 'tanh')
_BILINEAR_OPS = ('mul_bilinear', 'sub_bilinear', 'div_bilinear', 'pow')


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _cone_is_bilinear_free(byn, start, input_name):
    seen = set(); stack = [start]
    while stack:
        nm = stack.pop()
        if nm in seen or nm == input_name or nm not in byn:
            continue
        seen.add(nm)
        if byn[nm]['type'] in _BILINEAR_OPS:
            return False
        stack.extend(byn[nm]['inputs'])
    return True


def detect(gg):
    """Detect the `scale * g(z) + bias` monotone head, or None."""
    byn = {o['name']: o for o in gg['ops']}
    inm = gg['input_name']
    for o in gg['ops']:
        if o['type'] != 'mul' or o.get('scale') is None:
            continue
        gnm = o['inputs'][0]; gop = byn.get(gnm)
        if gop is None or gop['type'] not in _MONO_OPS:
            continue
        znm = gop['inputs'][0]
        if not _cone_is_bilinear_free(byn, znm, inm):
            continue
        add = next((a for a in gg['ops']
                    if a['type'] == 'add' and o['name'] in a['inputs']
                    and len(a['inputs']) == 1), None)
        scale = _np(o['scale']).ravel()
        if add is not None and 'bias' in add:
            bias = _np(add['bias']).ravel(); head_name = add['name']
        else:
            bias = np.zeros_like(scale); head_name = o['name']
        relax = (PWLRelax(gop['offsets'], gop['weights'], gop.get('bias', 0.0))
                 if gop['type'] == 'pwl' else None)
        return dict(head_name=head_name, g_type=gop['type'], z_name=znm,
                    scale=scale, bias=bias, relax=relax)
    return None


def _g_scalar(g_type, relax, z, dt=np.float64):
    z = dt(z)
    if g_type == 'pwl':
        off = _np(relax.offsets).astype(dt); w = _np(relax.weights).astype(dt)
        return float(dt(relax.bias) + (np.clip(z - off, 0, None) * w).sum())
    if g_type == 'sigmoid':
        return float(1.0 / (1.0 + np.exp(-float(z))))
    if g_type == 'tanh':
        return float(np.tanh(float(z)))
    raise NotImplementedError(g_type)


def _flat_output_ranges(gg, sizes):
    byn = {o['name']: o for o in gg['ops']}
    pos = [0]; ranges = {}

    def walk(nm):
        o = byn.get(nm)
        if o is not None and o['type'] == 'concat':
            for inp in o['inputs']:
                walk(inp)
        else:
            sz = int(sizes[nm]); ranges[nm] = (pos[0], pos[0] + sz); pos[0] += sz
    walk(gg['ops'][-1]['name'])
    return ranges


_NEG, _POS = -1e30, 1e30


def _invert_threshold(g_type, relax, c, unsafe_ge, lo, hi):
    """Return (inv_op, t) for an inverted z-constraint that is SOUND (converted-
    unsafe contains original-unsafe), forward-validated + nudged in fp32.

    unsafe_ge: original unsafe condition is g(z) >= c (True) or g(z) <= c (False).
    lo,hi: a bracket on z for the bisection (g monotone increasing).
    """
    glo = _g_scalar(g_type, relax, lo); ghi = _g_scalar(g_type, relax, hi)
    # edge: c outside the achievable g-range over the (wide) bracket
    if unsafe_ge:
        inv_op = '>='                       # unsafe if z >= t ; need t <= z*
        if c <= glo:                        # g >= c everywhere -> unsafe always
            return inv_op, _NEG
        if c > ghi:                         # g >= c never -> safe always
            return inv_op, _POS
    else:
        inv_op = '<='                       # unsafe if z <= t ; need t >= z*
        if c >= ghi:                        # g <= c everywhere -> unsafe always
            return inv_op, _POS
        if c < glo:                         # g <= c never -> safe always
            return inv_op, _NEG
    # bisect for z* with g(z*) = c
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if _g_scalar(g_type, relax, m) < c:
            a = m
        else:
            b = m
    zstar = 0.5 * (a + b)
    # fp32 forward-validate + conservative nudge (your mechanism): keep small.
    step = max(abs(zstar), 1.0) * 1e-6 + 1e-9
    if unsafe_ge:                           # need g(t) <= c (fp32) -> nudge DOWN
        t = zstar
        for _ in range(64):
            if _g_scalar(g_type, relax, t, np.float32) <= c:
                break
            t -= step; step *= 2
        else:
            return inv_op, _NEG             # couldn't validate -> conservative
    else:                                   # need g(t) >= c (fp32) -> nudge UP
        t = zstar
        for _ in range(64):
            if _g_scalar(g_type, relax, t, np.float32) >= c:
                break
            t += step; step *= 2
        else:
            return inv_op, _POS
    return inv_op, float(t)


def _invert_spec(spec, det, head_range, zlo, zhi):
    """Build the inverted VNNSpec on z, or None if any constraint is not a
    single-output head threshold (-> caller falls through)."""
    head_lo, head_hi = head_range
    scale, bias = det['scale'], det['bias']
    g_type, relax = det['g_type'], det['relax']
    blo = float(min(zlo.min(), -50.0)) - 50.0   # wide bisection bracket
    bhi = float(max(zhi.max(), 50.0)) + 50.0
    new_disj = []
    for conj in spec.disjuncts:
        if getattr(conj, 'input_lo', None) is not None:
            return None
        new_cons = []
        for con in conj.constraints:
            if not isinstance(con, Constraint):
                return None                 # pairwise etc. -> not invertible here
            i = con.index
            if not (head_lo <= i < head_hi):
                return None                 # non-head query -> fall through
            j = i - head_lo
            sj, bj, rhs = float(scale[j]), float(bias[j]), float(con.value)
            if sj == 0.0:                   # constant output: trivially (un)safe
                unsafe = (bj >= rhs) if con.op == '>=' else (bj <= rhs)
                # represent: always-unsafe -> z>= -inf ; always-safe -> z>= +inf
                new_cons.append(Constraint(j, '>=', _NEG if unsafe else _POS))
                continue
            c = (rhs - bj) / sj
            # unsafe g-direction: op '>=' unsafe Y>=rhs; '<=' unsafe Y<=rhs.
            if con.op == '>=':
                unsafe_ge = sj > 0          # Y>=rhs <=> g>=c (sj>0) or g<=c (sj<0)
            else:
                unsafe_ge = sj < 0          # Y<=rhs <=> g<=c (sj>0) or g>=c (sj<0)
            inv_op, t = _invert_threshold(g_type, relax, c, unsafe_ge, blo, bhi)
            new_cons.append(Constraint(j, inv_op, t, strict=con.strict))
        new_disj.append(Conjunct(new_cons))
    return VNNSpec(np.asarray(spec.x_lo), np.asarray(spec.x_hi), new_disj)


def _fp32_gate(spec, det, head_range, zlo, zhi):
    """Final NUMERICAL-soundness gate: re-validate the ORIGINAL spec on the head
    Y_i = scale_i*g(z_i)+bias_i over the z box, EVALUATED IN FP32 (the net's
    precision). Returns True iff every disjunct is provably unreachable in fp32
    (with a small ulp bump). The converted/inner proof is fp64; for the
    tolerance-boundary cases (margins ~1e-6) the saturated head trivialises the
    z-spec, so without this gate a fp64-positive-but-fp32-negative margin would
    false-verify. Monotone head => worst Y is at a z-box corner."""
    head_lo, head_hi = head_range
    scale, bias = det['scale'], det['bias']
    g_type, relax = det['g_type'], det['relax']
    for conj in spec.disjuncts:
        conj_safe = False
        for con in conj.constraints:
            if not isinstance(con, Constraint):
                return False
            i = con.index
            if not (head_lo <= i < head_hi):
                return False
            j = i - head_lo
            sj, bj, rhs = float(scale[j]), float(bias[j]), float(con.value)
            glo = _g_scalar(g_type, relax, zlo[j], np.float32)
            ghi = _g_scalar(g_type, relax, zhi[j], np.float32)
            lo_h = sj * glo + bj; hi_h = sj * ghi + bj
            if sj < 0:
                lo_h, hi_h = hi_h, lo_h
            bump = max(abs(lo_h), abs(hi_h)) * 8.0 * 2 ** -24 + 1e-12
            # constraint's unsafe region provably empty over the z box (fp32)?
            ok = (hi_h < rhs - bump) if con.op == '>=' else (lo_h > rhs + bump)
            if ok:
                conj_safe = True; break
        if not conj_safe:
            return False
    return True


def _truncate_at_z(graph, z_name):
    """Shallow-copy `graph` truncated to output `z_name` (prune downstream)."""
    if z_name not in graph.nodes:
        return None
    keep = set(); stack = [z_name]
    while stack:
        nm = stack.pop()
        if nm in keep:
            continue
        keep.add(nm)
        if nm in graph.nodes:
            stack.extend(graph.nodes[nm].inputs)
    gz = copy.copy(graph)
    gz.nodes = {nm: graph.nodes[nm] for nm in keep if nm in graph.nodes}
    gz.output_name = z_name
    # the cached topo order lists ALL original nodes; filter to the kept cone
    # (still a valid topo order, ending at z).
    if getattr(graph, 'topo_order', None) is not None:
        gz.topo_order = [n for n in graph.topo_order if n in gz.nodes]
    return gz


def try_verify(graph, spec, settings, device='cpu', log=None):
    """SETTINGS-GATED entry. Returns ('verified', info) only when the converted
    (truncated-net, inverted-z-spec) is proven by the standard prover; else None
    (gate off / structure absent / non-head query / not proven) -> fall through."""
    if not getattr(settings, 'monotone_output_inversion', False):
        return None
    if getattr(spec, 'input_lo', None) is not None:
        return None
    gg = graph.gpu_graph(device=device, dtype=torch.float64)
    det = detect(gg)
    if det is None:
        return None
    from .verify_zono_bnb import _forward_zonotope_graph
    xl = np.asarray(spec.x_lo, np.float64); xh = np.asarray(spec.x_hi, np.float64)
    ab = {}
    _, _zf = _forward_zonotope_graph(
        torch.tensor(xl), torch.tensor(xh), gg, torch.device(device),
        torch.float64, settings=settings, all_bounds=ab)
    sizes = {nm: ab[nm][0].numel() for nm in ab}; sizes[gg['input_name']] = xl.size
    ranges = _flat_output_ranges(gg, sizes)
    head_range = ranges.get(det['head_name'])
    if head_range is None or det['z_name'] not in ab:
        return None
    zlo, zhi = (b.detach().cpu().numpy() for b in ab[det['z_name']])
    if zlo.size != head_range[1] - head_range[0]:
        return None
    z_spec = _invert_spec(spec, det, head_range, zlo, zhi)
    if z_spec is None:
        return None
    gz = _truncate_at_z(graph, det['z_name'])
    if gz is None:
        return None
    # run the standard prover on the converted problem; disable this route inside
    # to avoid recursion.
    _prev = getattr(settings, 'monotone_output_inversion', False)
    try:
        settings.monotone_output_inversion = False
        from .verify_graph import verify_graph as _vg
        v, info = _vg(gz, z_spec, settings)
    finally:
        settings.monotone_output_inversion = _prev
    # NUMERICAL-soundness gate: the converted/inner proof is fp64; require the
    # ORIGINAL spec to also hold under an fp32 forward of the head over the z box
    # (catches tolerance-boundary cases where the saturated head trivialises the
    # z-spec and the fp64 proof would otherwise not see the tiny fp32 Y-margin).
    fp32_ok = _fp32_gate(spec, det, head_range, zlo, zhi) if v == 'verified' else False
    if log is not None:
        log(f"[mono_invert] converted z-spec ({z_spec.n_constraints} cons) "
            f"-> prover={v}  fp32_gate={fp32_ok}")
    if v == 'verified' and fp32_ok:
        return 'verified', {'method': 'monotone_invert_converted',
                            'converted_verdict': v, 'fp32_gate': True}
    return None
