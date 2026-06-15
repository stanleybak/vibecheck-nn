"""VNNLIB file parsing into VNNSpec objects."""

import numpy as np
import re
import gzip

from .spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint


def load_vnnlib(vnnlib_path, dtype=np.float32):
    """Parse a VNNLIB file into a VNNSpec object."""
    # Accept gzipped inputs, and resolve a `.vnnlib` reference whose only
    # on-disk copy is `.vnnlib.gz` (common in VNNCOMP benchmark dirs).
    from .io_util import ensure_decompressed
    vnnlib_path = ensure_decompressed(vnnlib_path)
    if vnnlib_path.endswith('.gz'):
        with gzip.open(vnnlib_path, 'rt') as f:
            text = f.read()
    else:
        with open(vnnlib_path, 'r') as f:
            text = f.read()
    return parse_vnnlib_text(text, dtype=dtype)


def parse_vnnlib_text(text, dtype=np.float32):
    """Parse VNNLIB text into a VNNSpec object.

    Supports:
    - Pairwise constraints: (>= Y_i Y_j) or (<= Y_i Y_j)
    - Threshold constraints: (>= Y_i val) or (<= Y_i val)
    - (or (and ...)) disjunctive normal form with mixed X/Y constraints
    """
    # Check for (or (and ...)) blocks first
    or_match = re.search(r'\(assert\s+\(or\s(.+)\)\s*\)', text, re.DOTALL)
    if or_match:
        return _parse_or_and(text, or_match.group(1), dtype=dtype)

    # Parse top-level input bounds
    x_lo, x_hi = _parse_input_bounds(text, dtype=dtype)

    # Parse output constraints
    constraints = _parse_output_constraints(text)

    return VNNSpec(x_lo, x_hi, [Conjunct(constraints)])


# ---------------------------------------------------------------------------
# Input bounds parsing
# ---------------------------------------------------------------------------

def _parse_input_bounds(text, dtype=np.float32):
    """Extract x_lo, x_hi arrays from top-level assertions."""
    x_bounds = {}
    for m in re.finditer(r'\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', text):
        x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', text):
        x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))

    if not x_bounds:
        # Fallback: X_i lo hi format
        for m in re.finditer(r'X_(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)', text):
            x_bounds[int(m.group(1))] = [float(m.group(2)), float(m.group(3))]

    if not x_bounds:
        raise ValueError("No input bounds found in VNNLIB")

    n_input = max(x_bounds.keys()) + 1
    x_lo = np.array([x_bounds.get(i, [0, 0])[0] or 0 for i in range(n_input)], dtype=dtype)
    x_hi = np.array([x_bounds.get(i, [0, 0])[1] or 0 for i in range(n_input)], dtype=dtype)
    return x_lo, x_hi


# ---------------------------------------------------------------------------
# Output constraint parsing
# ---------------------------------------------------------------------------

def _parse_output_constraints(text):
    """Parse output constraints from top-level assertions."""
    constraints = []

    # Pairwise: (>= Y_comp Y_pred)
    for m in re.finditer(r'>=\s+Y_(\d+)\s+Y_(\d+)', text):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(2)), comp=int(m.group(1))))

    # Pairwise: (<= Y_pred Y_comp)
    for m in re.finditer(r'<=\s+Y_(\d+)\s+Y_(\d+)', text):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(1)), comp=int(m.group(2))))

    if not constraints:
        # Threshold: (>= Y_i constant) or (<= Y_i constant)
        for m in re.finditer(r'\(>=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', text):
            constraints.append(Constraint(
                int(m.group(1)), '>=', float(m.group(2))))
        for m in re.finditer(r'\(<=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', text):
            constraints.append(Constraint(
                int(m.group(1)), '<=', float(m.group(2))))

    if not constraints:
        raise ValueError("Cannot parse output constraints from VNNLIB")

    return constraints


def _parse_block_constraints(block):
    """Parse constraints from an (and ...) block."""
    constraints = []

    # Y threshold
    for m in re.finditer(r'\(>=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        constraints.append(Constraint(
            int(m.group(1)), '>=', float(m.group(2))))
    for m in re.finditer(r'\(<=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        constraints.append(Constraint(
            int(m.group(1)), '<=', float(m.group(2))))

    # Y pairwise
    for m in re.finditer(r'>=\s+Y_(\d+)\s+Y_(\d+)', block):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(2)), comp=int(m.group(1))))
    for m in re.finditer(r'<=\s+Y_(\d+)\s+Y_(\d+)', block):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(1)), comp=int(m.group(2))))

    return constraints


def _parse_block_x_bounds(block):
    """Parse a single (and ...) block's X constraints.

    Returns a dict `{i: [lo_or_None, hi_or_None]}` LOCAL to this block
    (caller is responsible for merging across blocks). Pre-fix, this
    function mutated a shared dict — successive blocks overwrote each
    other's bounds, so the parsed input box was the LAST block's
    range rather than the UNION. acasxu prop_6 sampled the second
    X-or block's bounds and missed the first; nn4sys lindex_* picked
    a single subrange of the 10000 listed.
    """
    block_bounds = {}
    for m in re.finditer(r'\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        block_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        block_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))
    return block_bounds


# ---------------------------------------------------------------------------
# (or (and ...)) parsing
# ---------------------------------------------------------------------------

def _parse_top_level_y_constraints(text, or_body):
    """Output constraints asserted at TOP LEVEL, outside the (or ...) block.

    By VNNLIB semantics every top-level `(assert ...)` is a conjunct of the
    property, so a `(assert (<= Y_i v))` sitting beside an `(assert (or ...))`
    output disjunction must be ANDed into EVERY disjunct:
        P = (OR_k disj_k) AND (global_Y conjuncts)
          = OR_k (disj_k AND global_Y).
    Dropping these ENLARGES the unsafe region -> latent false-SAT. lsnc_relu
    quadrotor2d encodes a Lyapunov level-set band `Y_1 in [0.3554, 0.4055]`
    as two trailing top-level asserts; without them a witness with Y_1 far
    outside the band (e.g. 28.3) falsely satisfied the OR and we returned a
    spurious `sat`. The in-OR constraints are wrapped in `(and ...)`, the
    global ones directly in `(assert ...)`, so anchoring the regex on
    `(assert (op ...` matches only the latter; an in-(and ...) constraint
    never matches. (or_body is accepted for signature symmetry with the
    X-bound parser and to document the in/out-of-OR distinction.)
    """
    cons = []
    # Threshold: (assert (>= Y_i const)) / (assert (<= Y_i const)).
    for m in re.finditer(
            r'\(assert\s+\(>=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)\s*\)', text):
        cons.append(Constraint(int(m.group(1)), '>=', float(m.group(2))))
    for m in re.finditer(
            r'\(assert\s+\(<=\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)\s*\)', text):
        cons.append(Constraint(int(m.group(1)), '<=', float(m.group(2))))
    # Pairwise: (assert (>= Y_i Y_j)) / (assert (<= Y_i Y_j)).
    for m in re.finditer(
            r'\(assert\s+\(>=\s+Y_(\d+)\s+Y_(\d+)\s*\)\s*\)', text):
        cons.append(PairwiseConstraint(
            pred=int(m.group(2)), comp=int(m.group(1))))
    for m in re.finditer(
            r'\(assert\s+\(<=\s+Y_(\d+)\s+Y_(\d+)\s*\)\s*\)', text):
        cons.append(PairwiseConstraint(
            pred=int(m.group(1)), comp=int(m.group(2))))
    return cons


def _parse_or_and(text, or_body, dtype=np.float32):
    """Parse (or (and ...) (and ...) ...) blocks."""
    # Find all (and ...) blocks
    and_blocks = []
    depth = 0
    start = None
    for i, ch in enumerate(or_body):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and start is not None:
                block = or_body[start:i + 1]
                if block.strip().startswith('(and'):
                    and_blocks.append(block)
                start = None

    assert and_blocks, "No (and ...) blocks found in (or ...)"

    # Classify each (and ...) block. A block with output (Y) constraints is an
    # OUTPUT disjunct; a block with ONLY X constraints is one box of an INPUT
    # disjunction (acasxu prop_6/7/8 split the heading psi into two ranges, in
    # a SEPARATE `(assert (or ...))` from the output disjunction).
    all_block_x = []           # every block's X box (for the global hull)
    x_only_boxes = []          # boxes of the input disjunction
    y_disjuncts = []           # (constraints, own_x_box) output disjuncts
    for block in and_blocks:
        block_x = _parse_block_x_bounds(block)
        all_block_x.append(block_x)
        constraints = _parse_block_constraints(block)
        if constraints:
            y_disjuncts.append((constraints, block_x))
        elif block_x:
            x_only_boxes.append(block_x)

    # Top-level X bounds (outside the or block) apply to EVERY disjunct.
    top_x_bounds = {}
    for m in re.finditer(r'\(assert\s+\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        top_x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(assert\s+\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        top_x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))

    # UNION across ALL blocks + top-level for the global bounding box. Take
    # min of lo, max of hi. (Each block contributes a sub-box of the input;
    # the verification region is the UNION; the bounding box over-approximates
    # that union and is what the rest of the pipeline works against. Note: the
    # hull is SOUND for the verification/`check` side, but NOT for witness
    # validation, which is why per-disjunct boxes are attached below.)
    union = dict(top_x_bounds)
    for bx in all_block_x:
        for i, (lo, hi) in bx.items():
            if i not in union:
                union[i] = [lo, hi]
            else:
                ulo, uhi = union[i]
                # min of los
                if lo is not None:
                    union[i][0] = lo if ulo is None else min(ulo, lo)
                # max of his
                if hi is not None:
                    union[i][1] = hi if uhi is None else max(uhi, hi)

    assert union, "No input bounds found in VNNLIB (or/and format)"

    n_input = max(union.keys()) + 1
    x_lo = np.array([union.get(i, [0, 0])[0] or 0 for i in range(n_input)], dtype=dtype)
    x_hi = np.array([union.get(i, [0, 0])[1] or 0 for i in range(n_input)], dtype=dtype)

    # CROSS-PRODUCT a separate INPUT-OR with the OUTPUT-OR. The full unsafe
    # region is (union of input boxes) AND (union of output conditions);
    # collapsing the input disjunction to its hull is UNSOUND — a point in the
    # GAP between input boxes that violates the output would be accepted as a
    # false counterexample (acasxu prop_6 net 1_1 returned a spurious `sat`).
    # Expand to DNF: pair each output disjunct with each input box, so every
    # resulting conjunct carries its real input sub-box (which
    # `VNNSpec.check_witness` enforces, rejecting gap points).
    if x_only_boxes:
        expanded = []
        for constraints, own_x in y_disjuncts:
            for in_box in x_only_boxes:
                merged = {i: list(v) for i, v in own_x.items()}
                for i, (lo, hi) in in_box.items():
                    if i not in merged:
                        merged[i] = [lo, hi]
                    else:
                        if lo is not None:
                            merged[i][0] = lo if merged[i][0] is None else max(merged[i][0], lo)
                        if hi is not None:
                            merged[i][1] = hi if merged[i][1] is None else min(merged[i][1], hi)
                expanded.append((constraints, merged))
        disjuncts = expanded
    else:
        disjuncts = y_disjuncts

    assert disjuncts, "No output constraints found in (or (and ...)) blocks"

    # GLOBAL output conjuncts asserted at top level beside the (or ...)
    # block (e.g. lsnc_relu's level-set band `Y_1 in [0.3554, 0.4055]`).
    # By VNNLIB semantics these AND into EVERY disjunct; dropping them
    # enlarges the unsafe region and yields false-SAT. Prepend so they
    # apply to every conjunct below.
    global_y = _parse_top_level_y_constraints(text, or_body)

    # Build Conjuncts with their per-disjunct X subbox (merged with
    # top-level X bounds). The subbox is stored on the Conjunct so
    # witness validation can check `x in subbox AND y violates`.
    conj_list = []
    for constraints, block_x in disjuncts:
        constraints = list(constraints) + list(global_y)
        # Combined per-disjunct bounds: intersect block_x with top_x_bounds
        # (both apply simultaneously per disjunct). Where unconstrained,
        # fall back to the UNION bounding box so x_satisfied uses
        # a sound over-approximation.
        per_lo = np.empty(n_input, dtype=dtype); per_hi = np.empty(n_input, dtype=dtype)
        for i in range(n_input):
            top_lo, top_hi = top_x_bounds.get(i, [None, None])
            blk_lo, blk_hi = block_x.get(i, [None, None])
            cand_lo = [v for v in (top_lo, blk_lo) if v is not None]
            cand_hi = [v for v in (top_hi, blk_hi) if v is not None]
            per_lo[i] = max(cand_lo) if cand_lo else x_lo[i]
            per_hi[i] = min(cand_hi) if cand_hi else x_hi[i]
        # Only attach per-disjunct bounds if the conjunct actually has
        # X constraints; otherwise leave as None (the conjunct accepts
        # any x in the global box — preserves backward compat for
        # Y-only or-and specs like malbeware, cifar100, etc.).
        if block_x:
            conj_list.append(Conjunct(constraints, input_lo=per_lo, input_hi=per_hi))
        else:
            conj_list.append(Conjunct(constraints))

    return VNNSpec(x_lo, x_hi, conj_list)
