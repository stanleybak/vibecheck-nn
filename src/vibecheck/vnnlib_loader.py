"""VNNLIB file parsing into VNNSpec objects."""

import numpy as np
import re
import gzip

from .spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint


def load_vnnlib(vnnlib_path, dtype=np.float32):
    """Parse a VNNLIB file into a VNNSpec object."""
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

    # Per-block X bounds, then merge as UNION for the global x_lo/x_hi.
    block_x_bounds_list = []  # one dict per disjunct
    disjuncts = []
    for block in and_blocks:
        block_x = _parse_block_x_bounds(block)
        block_x_bounds_list.append(block_x)
        constraints = _parse_block_constraints(block)
        if constraints:
            disjuncts.append((constraints, block_x))

    # Top-level X bounds (outside the or block) apply to EVERY disjunct.
    top_x_bounds = {}
    for m in re.finditer(r'\(assert\s+\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        top_x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(assert\s+\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        top_x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))

    # UNION across disjuncts + top-level for global bounding box. Take
    # min of lo, max of hi. (Each disjunct contributes a sub-box of the
    # input; the verification region is the UNION; the bounding box
    # over-approximates that union and is what the rest of the pipeline
    # works against.)
    union = dict(top_x_bounds)
    for bx in block_x_bounds_list:
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

    assert disjuncts, "No output constraints found in (or (and ...)) blocks"

    # Build Conjuncts with their per-disjunct X subbox (merged with
    # top-level X bounds). The subbox is stored on the Conjunct so
    # witness validation can check `x in subbox AND y violates`.
    conj_list = []
    for constraints, block_x in disjuncts:
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
