"""Verification specification types.

A VNNSpec defines input bounds and output constraints (the unsafe region).
The unsafe region is a disjunction of conjuncts (DNF). Verification succeeds
if ALL disjuncts are provably unreachable.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Constraint:
    """Threshold constraint: Y[index] op value."""
    index: int
    op: str       # '>=' or '<='
    value: float

    def margin(self, output_lo, output_hi):
        """Positive margin = verified safe."""
        if self.op == '>=':
            # Unsafe if Y[idx] >= val. Safe if hi < val.
            return self.value - output_hi[self.index]
        else:
            # Unsafe if Y[idx] <= val. Safe if lo > val.
            return output_lo[self.index] - self.value

    def __str__(self):
        return f'Y_{self.index} {self.op} {self.value}'


@dataclass
class PairwiseConstraint:
    """Pairwise constraint: unsafe if Y[comp] >= Y[pred]."""
    pred: int
    comp: int

    def margin(self, output_lo, output_hi):
        """Positive margin = pred provably beats comp."""
        return output_lo[self.pred] - output_hi[self.comp]

    def __str__(self):
        return f'Y_{self.comp} >= Y_{self.pred}'


@dataclass
class Conjunct:
    """Conjunction of constraints. All must hold for the unsafe region.

    Optionally carries per-disjunct input bounds (`input_lo`, `input_hi`)
    when the source vnnlib placed X constraints inside (and ...) blocks
    (e.g., nn4sys lindex, acasxu prop_6). When set, witness validation
    must additionally check `x in [input_lo, input_hi]`; a witness with
    x outside the subrange is NOT a counterexample for this conjunct.
    """
    constraints: list
    input_lo: 'np.ndarray | None' = None
    input_hi: 'np.ndarray | None' = None

    def margin(self, output_lo, output_hi):
        """Best margin across constraints — conjunction is safe iff ANY
        constraint is provably violated for all outputs in [lo, hi].

        The conjunct's unsafe region is `c1 AND c2 AND ...`. To prove
        the conjunct's unsafe region unreachable, we only need to prove
        ONE of the constraints provably-not-satisfied (since AND fails
        if any operand fails). So conjunct safe iff max(c.margin) > 0.

        Earlier this used `min`, which only declared safe when ALL
        constraints were individually safe — wrong semantics for AND.
        Silent on benchmarks where every conjunct has one constraint
        (cifar100, tinyimagenet — `min(single) == max(single)`);
        triggered on cersyve where each conjunct has two constraints.
        The ORT-based `_validate_sat_witness` in verify_graph is the
        pipeline-level defense-in-depth that catches the SAT side of
        this kind of bug; this fix corrects the verification side.

        NOTE: this method considers only output constraints. Input
        constraints (when present in `input_lo`/`input_hi`) are handled
        separately by `VNNSpec.check_witness`.
        """
        return max(c.margin(output_lo, output_hi) for c in self.constraints)

    def x_satisfied(self, x):
        """True iff the witness x is inside this conjunct's X subrange
        (or the conjunct has no per-disjunct X constraints)."""
        if self.input_lo is None:
            return True
        return bool(np.all(x >= self.input_lo - 1e-9)
                    and np.all(x <= self.input_hi + 1e-9))

    def __str__(self):
        return ' AND '.join(str(c) for c in self.constraints)


@dataclass
class VNNSpec:
    """VNNLIB specification: input bounds + disjunction of conjuncts.

    The unsafe region is OR of conjuncts. Verified if ALL disjuncts have
    positive margin (every unsafe region is provably unreachable).
    """
    x_lo: np.ndarray
    x_hi: np.ndarray
    disjuncts: list  # list of Conjunct

    def check(self, output_lo, output_hi):
        """Check spec against output bounds.

        Returns:
            result: 'verified' or 'unknown'
            details: dict with margins per disjunct and worst_margin
        """
        margins = {}
        for i, conj in enumerate(self.disjuncts):
            margins[i] = conj.margin(output_lo, output_hi)

        worst = min(margins.values()) if margins else 0.0
        return ('verified' if worst > 0 else 'unknown'), {
            'margins': margins,
            'worst_margin': float(worst),
        }

    def check_witness(self, x, y):
        """Check if a witness (x, y) is a real counterexample.

        Evaluates each disjunct AT the witness point — needed when
        conjuncts carry per-disjunct input subranges
        (`Conjunct.input_lo`/`input_hi`, e.g., nn4sys lindex). A point
        violates the full spec iff there is at least one disjunct whose
        X-subrange contains `x` AND whose Y-constraints are violated by
        `y`.

        Returns:
            (is_counterexample, details) — `is_counterexample` is True
            iff some disjunct's full conjunction holds at (x, y).
            details: list of (disjunct_idx, x_in_subrange, margin) for
            each disjunct.
        """
        details = []
        is_ce = False
        for i, conj in enumerate(self.disjuncts):
            x_ok = conj.x_satisfied(x)
            m = conj.margin(y, y) if x_ok else None
            details.append((i, x_ok, m))
            if x_ok and m is not None and m <= 0:
                is_ce = True
        return is_ce, details

    def as_pairwise(self):
        """Extract (pred, comps_set) if all constraints are pairwise with same pred.

        Returns (pred, {comp1, comp2, ...}) or None if not applicable.
        """
        preds = set()
        comps = set()
        for conj in self.disjuncts:
            for c in conj.constraints:
                if not isinstance(c, PairwiseConstraint):
                    return None
                preds.add(c.pred)
                comps.add(c.comp)
        if len(preds) != 1:
            return None
        return preds.pop(), comps

    def as_linear_queries(self, n_output):
        """Convert spec to linear queries for MILP/CROWN verification.

        Each disjunct produces one or more linear queries. A disjunct is
        verified if ALL its queries have positive minimum.

        Returns list of (disjunct_idx, w, bias) where:
        - w: numpy array of shape (n_output,) — linear weights on output
        - bias: float — constant term
        - Verified safe when min(w @ output + bias) > 0

        For pairwise: w = e_pred - e_comp, bias = 0
        For threshold Y[i] >= val: w = -e_i, bias = val
        For threshold Y[i] <= val: w = e_i, bias = -val
        """
        queries = []
        for di, conj in enumerate(self.disjuncts):
            for c in conj.constraints:
                w = np.zeros(n_output, dtype=np.float64)
                if isinstance(c, PairwiseConstraint):
                    w[c.pred] = 1.0
                    w[c.comp] = -1.0
                    queries.append((di, w, 0.0))
                elif isinstance(c, Constraint):
                    if c.op == '>=':
                        w[c.index] = -1.0
                        queries.append((di, w, c.value))
                    else:  # '<='
                        w[c.index] = 1.0
                        queries.append((di, w, -c.value))
        return queries

    @property
    def n_constraints(self):
        return sum(len(d.constraints) for d in self.disjuncts)

    def __str__(self):
        parts = [f'input: {len(self.x_lo)}D  '
                 f'[{self.x_lo.min():.4f}, {self.x_hi.max():.4f}]']
        if len(self.disjuncts) == 1:
            parts.append(f'unsafe if: {self.disjuncts[0]}')
        else:
            parts.append(f'unsafe if any of {len(self.disjuncts)} disjuncts:')
            for i, d in enumerate(self.disjuncts):
                parts.append(f'  [{i}] {d}')
        return '\n'.join(parts)
