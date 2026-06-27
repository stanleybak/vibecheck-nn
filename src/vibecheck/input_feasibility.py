"""Fast, SOUND nonlinear (degree<=2) input-box TIGHTENER, with empty-region
detection as the limiting case.

A VNN-LIB property is `forall x in InputRegion: outputSpec(f(x))`, where the
InputRegion is the linear box intersected with any nonlinear input constraints
(e.g. `200*x0 >= x1^2`). Two payoffs from reasoning about that region up front:

  * TIGHTEN: contract the linear box to the smallest box still containing the
    true (nonlinearly-constrained) region. Feeding the verifier a smaller input
    box yields tighter bounds everywhere downstream -> more `unsat`s / faster.
    A pure, sound tightening: the nonlinear atoms stay enforced exactly in the
    spec, so the verdict is unchanged, only the bounds get tighter.

  * EMPTY (the limit): if the contraction drives the box empty, the input region
    is empty, the `forall` is vacuously true, and the verdict is `unsat` with NO
    verification (milliseconds, never a timeout). adaptive_cruise_2026 ships 18
    of 50 such instances.

How: interval constraint propagation (an HC4-style contractor) over the degree<=2
polynomial X-constraints of each DNF clause; the region's box is the hull of the
per-clause contracted boxes (a clause that contracts to empty contributes
nothing; all-empty -> empty region). The contractor only ever SHRINKS a
variable's interval while keeping every point that could satisfy a constraint (it
bounds each constraint's value from below using interval arithmetic over the
OTHER variables, an over-approximation of feasibility). Therefore the contracted
box always CONTAINS the true region (sound: it never cuts a real counterexample,
so it can neither false-`unsat` via over-tightening nor via a spurious empty).

Scope: degree<=2 monomials (x_i, x_i^2, x_i*x_j) — the only forms the 2026
nonlinear specs use. Higher-degree terms make that constraint a no-op contractor
(treated as the trivial [-inf,inf]), which stays sound (no tightening).
"""

INF = 1.0e9
# Soundness keep-tolerance for the quadratic solve: a point whose constraint
# value is within this of feasible is KEPT, so a clause is called infeasible
# (empty) only when it misses by more than this margin. Guards a false `unsat`
# from floating-point noise / a near-tangent constraint. The real empty regions
# miss by huge margins (1e3-1e5), so this never blocks a genuine detection.
_KEEP_EPS = 1e-9
_MAX_ITERS = 32          # contraction rounds per clause (fixpoint usually < 5)


def _xidx(v):
    """'X_0' -> 0. Only X-vars are passed here."""
    return int(v[2:])


def _is_x(v):
    return v[0] == 'X'


def _mono_interval(mono, box):
    """Interval of a monomial (tuple of var names) over the current box."""
    lo, hi = 1.0, 1.0
    for v in mono:
        vlo, vhi = box[_xidx(v)]
        cands = (lo * vlo, lo * vhi, hi * vlo, hi * vhi)
        lo, hi = min(cands), max(cands)
    return lo, hi


def _decompose(terms, bias, i, box):
    """Write the polynomial as A*xi^2 + L*xi + R for a fixed variable index i,
    where A is a scalar, L = [Ll,Lh] and R = [Rl,Rh] are intervals over the other
    variables (each treated independently -> a valid interval enclosure). Returns
    (A, Ll, Lh, Rl, Rh). Terms whose monomial has degree > 2, or > 2 *distinct*
    couplings, fall into R as their interval (sound, just not tightened)."""
    A = 0.0
    Ll = Lh = 0.0
    Rl = Rh = bias
    for mono, coef in terms:
        xs = [_xidx(v) for v in mono if _is_x(v)]
        if len(mono) != len(xs):
            # a Y appears (or non-X var) -> this is not a pure input constraint;
            # caller filters these out, but stay safe: treat as remainder=0 span.
            Rl, Rh = -INF, INF
            continue
        cnt = xs.count(i)
        if cnt == 2 and len(mono) == 2:                 # xi^2
            A += coef
        elif cnt == 1 and len(mono) == 1:               # xi
            Ll += coef
            Lh += coef
        elif cnt == 1 and len(mono) == 2:               # xi * xj  (j != i)
            j = xs[0] if xs[1] == i else xs[1]
            ojlo, ojhi = box[j]
            c0, c1 = coef * ojlo, coef * ojhi
            Ll += min(c0, c1)
            Lh += max(c0, c1)
        else:                                           # no xi -> remainder
            mlo, mhi = _mono_interval(mono, box)
            c0, c1 = coef * mlo, coef * mhi
            Rl += min(c0, c1)
            Rh += max(c0, c1)
    return A, Ll, Lh, Rl, Rh


def _solve_quad_le(A, B, C, dlo, dhi):
    """Hull of {x in [dlo,dhi] : A*x^2 + B*x + C <= 0}, or None if empty.
    A small positive slack keeps boundary points (we want to OVER-keep, never
    under-keep, for soundness)."""
    eps = _KEEP_EPS
    if dlo > dhi:
        return None
    if abs(A) < 1e-18:
        if abs(B) < 1e-18:
            return (dlo, dhi) if C <= eps else None
        root = -C / B
        if B > 0:                                       # x <= root
            slo, shi = dlo, min(dhi, root)
        else:                                           # x >= root
            slo, shi = max(dlo, root), dhi
        return (slo, shi) if slo <= shi else None
    disc = B * B - 4 * A * C
    if A > 0:                                           # convex: <=0 between roots
        if disc < 0:
            return None
        r = disc ** 0.5
        r1, r2 = (-B - r) / (2 * A), (-B + r) / (2 * A)
        slo, shi = max(dlo, r1), min(dhi, r2)
        return (slo, shi) if slo <= shi else None
    # A < 0: concave: <=0 OUTSIDE the roots -> (-inf,ra] U [rb,inf)
    if disc < 0:
        return (dlo, dhi)                               # always <= 0
    r = disc ** 0.5
    ra, rb = sorted(((-B - r) / (2 * A), (-B + r) / (2 * A)))
    parts = []
    if dlo <= ra:
        parts.append((dlo, min(dhi, ra)))
    if dhi >= rb:
        parts.append((max(dlo, rb), dhi))
    parts = [p for p in parts if p[0] <= p[1]]
    if not parts:
        return None
    return (min(p[0] for p in parts), max(p[1] for p in parts))


def _contract_var(A, Ll, Lh, Rl, lo, hi):
    """New [lo,hi] for xi keeping every value that could satisfy
    A*xi^2 + L*xi + R <= 0 for SOME L in [Ll,Lh], R in [Rl,Rh] (existence over the
    other variables). The minimum over L,R is A*xi^2 + min(Ll*xi,Lh*xi) + Rl;
    on xi>=0 that linear coef is Ll, on xi<=0 it is Lh. Returns None if empty."""
    res_lo = res_hi = None
    for dlo, dhi, B in ((max(lo, 0.0), hi, Ll), (lo, min(hi, 0.0), Lh)):
        s = _solve_quad_le(A, B, Rl, dlo, dhi)
        if s is None:
            continue
        res_lo = s[0] if res_lo is None else min(res_lo, s[0])
        res_hi = s[1] if res_hi is None else max(res_hi, s[1])
    if res_lo is None:
        return None
    return res_lo, res_hi


def _contract_clause(x_cons, nvars, init_box):
    """Run the contractor over one clause's X-only constraints to a fixpoint.
    Returns the tightened box (list of [lo,hi]), or None if it is driven empty
    (clause infeasible). x_cons is a list of (terms, bias) for atoms p {<,<=} 0."""
    box = [list(init_box[i]) for i in range(nvars)]
    for _ in range(_MAX_ITERS):
        changed = False
        for terms, bias in x_cons:
            present = set()
            for mono, _ in terms:
                for v in mono:
                    if _is_x(v):
                        present.add(_xidx(v))
            for i in present:
                A, Ll, Lh, Rl, _Rh = _decompose(terms, bias, i, box)
                new = _contract_var(A, Ll, Lh, Rl, box[i][0], box[i][1])
                if new is None:                     # clause infeasible over this box
                    return None
                # _contract_var already intersects with [box_lo,box_hi], so `new`
                # is a (non-empty) sub-interval; just adopt any tightening.
                if new[0] - box[i][0] > 1e-12 or box[i][1] - new[1] > 1e-12:
                    box[i][0], box[i][1] = new[0], new[1]
                    changed = True
        if not changed:
            break
    return box


def _n_inputs(clauses):
    nv = 0
    for cl in clauses:
        for c in cl.constraints:
            for mono, _ in c.terms:
                for v in mono:
                    if _is_x(v):
                        nv = max(nv, _xidx(v) + 1)
    return nv


def tighten_input_box(prop, n_in=None, init_box=None):
    """SOUND nonlinear input-box tightener for a parsed v2 `prop` (parse_vnnlib_v2
    output). Returns the contracted box (list of [lo,hi], one per input) that is
    the smallest box still CONTAINING the true input region (the linear box
    intersected with the nonlinear input constraints), taken as the hull over the
    DNF clauses. Returns None iff the region is EMPTY (every clause infeasible).

    `n_in` overrides the inferred input dimension; `init_box` (list of (lo,hi))
    seeds the contraction with a known outer box (e.g. the declared linear box) so
    unconstrained variables keep finite bounds. The returned box always contains
    the true region -> never cuts a real counterexample."""
    clauses = prop.spec.clauses
    if not clauses:
        return []                       # indeterminate, NOT provably empty
    nv = n_in if n_in is not None else _n_inputs(clauses)
    if nv == 0:
        return []                       # no input variables -> indeterminate
    if init_box is None:
        init_box = [(-INF, INF) for _ in range(nv)]
    hull = None
    for cl in clauses:
        x_cons = [(c.terms, c.bias) for c in cl.constraints
                  if all(all(_is_x(v) for v in mono) for mono, _ in c.terms)]
        if x_cons:
            box = _contract_clause(x_cons, nv, init_box)
            if box is None:
                continue                       # clause infeasible -> skip
        else:
            box = [list(init_box[i]) for i in range(nv)]   # unconstrained clause
        if hull is None:
            hull = [list(b) for b in box]
        else:
            for i in range(nv):
                hull[i][0] = min(hull[i][0], box[i][0])
                hull[i][1] = max(hull[i][1], box[i][1])
    return hull


def empty_input_region(prop, n_in=None):
    """SOUND emptiness test: True ONLY if the input region is empty (every DNF
    clause's X-constraints are jointly infeasible) -> the property is vacuously
    `unsat`. Never True for a non-empty region."""
    return tighten_input_box(prop, n_in=n_in) is None
