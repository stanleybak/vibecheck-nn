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


def _parse_block_x_bounds(block, x_bounds):
    """Extract X bounds from an (and ...) block into x_bounds dict."""
    for m in re.finditer(r'\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))


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

    all_x_bounds = {}
    disjuncts = []

    for block in and_blocks:
        _parse_block_x_bounds(block, all_x_bounds)
        constraints = _parse_block_constraints(block)
        if constraints:
            disjuncts.append(Conjunct(constraints))

    # Also check for top-level X bounds (outside the or block)
    for m in re.finditer(r'\(assert\s+\(>=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        all_x_bounds.setdefault(int(m.group(1)), [None, None])[0] = float(m.group(2))
    for m in re.finditer(r'\(assert\s+\(<=\s+X_(\d+)\s+([-\d.eE+]+)\s*\)\)', text):
        all_x_bounds.setdefault(int(m.group(1)), [None, None])[1] = float(m.group(2))

    assert all_x_bounds, "No input bounds found in VNNLIB (or/and format)"

    n_input = max(all_x_bounds.keys()) + 1
    x_lo = np.array([all_x_bounds.get(i, [0, 0])[0] or 0 for i in range(n_input)], dtype=dtype)
    x_hi = np.array([all_x_bounds.get(i, [0, 0])[1] or 0 for i in range(n_input)], dtype=dtype)

    assert disjuncts, "No output constraints found in (or (and ...)) blocks"

    return VNNSpec(x_lo, x_hi, disjuncts)
