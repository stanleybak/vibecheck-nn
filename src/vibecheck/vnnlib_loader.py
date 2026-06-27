"""VNNLIB file parsing into VNNSpec objects.

Two spec formats are supported, auto-detected per file by `detect_version`:

- **v1** — flat `(declare-const X_0 Real)` … `(assert (<= X_780 -0.41))` …
  `(assert (or (and (>= Y_0 Y_1)) ...))`. Parsed by the original regex path
  (`_parse_or_and` / `_parse_input_bounds` / `_parse_output_constraints`),
  which carries VC-specific soundness fixes (per-disjunct X subboxes,
  top-level-Y conjuncts ANDed in, input-OR cross-product).

- **v2** — `(vnnlib-version <2.0>)` + `(declare-network N (declare-input X
  float32 [1,784]) (declare-output Y float32 [1,10]))`, then tensor-indexed
  `(assert (<= X[0,780] -0.41))` … `(and (>= Y[0,0] Y[0,1]))`. Parsed by a
  self-contained recursive-descent s-expr parser (`parse_vnnlib_v2`, ported
  from the VNNCOMP-2026 reference `scripts/vnnlib_parser.py`) into a
  version-agnostic `VnnlibProperty` (canonical flat `X_<i>`/`Y_<i>` names via
  C-order flattening), then mapped to `VNNSpec` by `_vnnlib_v2_to_spec`.

v1 and v2 are semantically equivalent (tensor indices flatten C-order to the
v1 flat indices). Because the v2 path is INDEPENDENT code (not a transpile),
the load-bearing correctness gate is the v1<->v2 equivalence check: the
inline-string cases in `tests/test_spec.py`, plus a one-time full-corpus run
over every 2026 benchmark's `1.0/`+`2.0/` vnnlib pair.
"""

import numpy as np
import re
import gzip
from dataclasses import dataclass, field

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
    # VNNLIB v2 (network-header + tensor indexing) routes to the ported
    # s-expr parser + adapter; v1 (flat declare-const) keeps the regex path
    # below unchanged (preserving its VC-specific soundness fixes).
    if detect_version(text) == '2.0':
        return _vnnlib_v2_to_spec(parse_vnnlib_v2(text), dtype=dtype)

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

    # Pairwise: (>= Y_comp Y_pred) / strict (> Y_comp Y_pred) — `(=?)` captures
    # strictness so the CE-check rejects the boundary on a strict `>`/`<`.
    for m in re.finditer(r'>(=?)\s+Y_(\d+)\s+Y_(\d+)', text):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(3)), comp=int(m.group(2)), strict=(m.group(1) == '')))

    # Pairwise: (<= Y_pred Y_comp) / strict (< Y_pred Y_comp)
    for m in re.finditer(r'<(=?)\s+Y_(\d+)\s+Y_(\d+)', text):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(2)), comp=int(m.group(3)), strict=(m.group(1) == '')))

    if not constraints:
        # Threshold: (>= Y_i const) / (<= Y_i const), strict (> / <).
        for m in re.finditer(r'\(>(=?)\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', text):
            constraints.append(Constraint(
                int(m.group(2)), '>=', float(m.group(3)), strict=(m.group(1) == '')))
        for m in re.finditer(r'\(<(=?)\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', text):
            constraints.append(Constraint(
                int(m.group(2)), '<=', float(m.group(3)), strict=(m.group(1) == '')))

    if not constraints:
        raise ValueError("Cannot parse output constraints from VNNLIB")

    return constraints


def _parse_block_constraints(block):
    """Parse constraints from an (and ...) block."""
    constraints = []

    # Y threshold (strict `>`/`<` captured via `(=?)`)
    for m in re.finditer(r'\(>(=?)\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        constraints.append(Constraint(
            int(m.group(2)), '>=', float(m.group(3)), strict=(m.group(1) == '')))
    for m in re.finditer(r'\(<(=?)\s+Y_(\d+)\s+([-\d.eE+]+)\s*\)', block):
        constraints.append(Constraint(
            int(m.group(2)), '<=', float(m.group(3)), strict=(m.group(1) == '')))

    # Y pairwise
    for m in re.finditer(r'>(=?)\s+Y_(\d+)\s+Y_(\d+)', block):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(3)), comp=int(m.group(2)), strict=(m.group(1) == '')))
    for m in re.finditer(r'<(=?)\s+Y_(\d+)\s+Y_(\d+)', block):
        constraints.append(PairwiseConstraint(
            pred=int(m.group(2)), comp=int(m.group(3)), strict=(m.group(1) == '')))

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


# ===========================================================================
# VNNLIB v2 parser (ported from VNNCOMP-2026 reference scripts/vnnlib_parser.py)
#
# A self-contained recursive-descent s-expr parser for the v2 format. It
# canonicalizes every variable to a flat "X_<i>" / "Y_<i>" name (C-order
# flatten across declared tensors, in declaration order), so a v2 spec and its
# v1 translation are directly comparable. The parser is INDEPENDENT of VC's
# regex v1 path — the equivalence oracle is what proves they agree. No onnx
# dependency.
#
# A PolynomialConstraint means: sum(coeff * product(monomial)) + bias <= 0
# (or < 0 if strict); each monomial is a sorted tuple of variable names, so
# ("X_0",) is a linear term and ("X_1","X_1") is X_1 squared (needed for the
# nonlinear 2026 benchmarks; VC's VNNSpec adapter rejects degree>=2 loudly).
# ===========================================================================

CMP_OPS = ("<=", ">=", "<", ">", "==", "=", "!=")

_NEGATED_OP = {"<=": ">", "<": ">=", ">=": "<", ">": "<=",
               "==": "!=", "=": "!=", "!=": "=="}

# nn4sys legitimately has a 120k-clause disjunction (lindex_60000.vnnlib);
# the cap only guards against accidental cross-products of multiple or-asserts.
MAX_DNF_CLAUSES = 2_000_000


class VnnlibParseError(Exception):
    pass


@dataclass(frozen=True)
class TensorDecl:
    name: str
    dtype: str
    shape: tuple  # () for scalar

    @property
    def size(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class NetworkDecl:
    name: str
    inputs: tuple   # tuple[TensorDecl]
    outputs: tuple  # tuple[TensorDecl]
    relations: tuple = ()  # e.g. (("isomorphic-to", "f"),) or (("equal-to", "f"),)


@dataclass(frozen=True)
class PolynomialConstraint:
    """sum(coeff * product(monomial vars)) + bias <= 0 (or < 0 if strict).

    Each monomial is a sorted tuple of variable names; linear constraints
    have only length-1 monomials.
    """
    terms: tuple    # tuple[(monomial, coefficient)], sorted by monomial
    bias: float
    strict: bool = False

    @property
    def is_linear(self):
        return all(len(m) == 1 for m, _ in self.terms)

    def __str__(self):
        rendered = " + ".join(f"{c}*{'*'.join(m)}" for m, c in self.terms)
        op = "<" if self.strict else "<="
        return f"{rendered} + {self.bias} {op} 0"


@dataclass
class ConjunctiveSpec:
    """AND of polynomial constraints."""
    constraints: list = field(default_factory=list)


@dataclass
class DisjunctiveSpec:
    """OR of conjunctive clauses."""
    clauses: list = field(default_factory=list)


@dataclass
class VnnlibProperty:
    version: str
    networks: list
    spec: DisjunctiveSpec

    @property
    def num_inputs(self):
        return sum(t.size for n in self.networks for t in n.inputs)

    @property
    def num_outputs(self):
        return sum(t.size for n in self.networks for t in n.outputs)


# ---------------------------------------------------------------- tokenizing

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_COMMENT_RE = re.compile(r";[^\n]*")


def _tokenize(text):
    text = _COMMENT_RE.sub("", text)
    raw = _TOKEN_RE.findall(text)

    # re-join bracket groups that were split by whitespace: "X[0," "1]" or
    # "[1," "1," "5]" (shape literals)
    tokens = []
    i = 0
    n = len(raw)
    while i < n:
        tok = raw[i]
        if tok.count("[") > tok.count("]"):
            parts = [tok]
            while parts[-1].count("]") < tok.count("["):
                i += 1
                if i >= n:
                    raise VnnlibParseError("unterminated '[' bracket")
                parts.append(raw[i])
            tok = "".join(parts)
        tokens.append(tok)
        i += 1
    return tokens


def _parse_sexprs(tokens):
    """Token list -> list of nested python lists/str."""
    out = []
    stack = [out]
    for tok in tokens:
        if tok == "(":
            new = []
            stack[-1].append(new)
            stack.append(new)
        elif tok == ")":
            stack.pop()
            if not stack:
                raise VnnlibParseError("unbalanced ')'")
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise VnnlibParseError("unbalanced '('")
    return out


def _as_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- var map

class _VarMap:
    """Resolves source variable tokens to canonical 'X_<i>' / 'Y_<i>' names."""

    def __init__(self):
        self.input_tensors = {}   # name -> (offset, TensorDecl)
        self.output_tensors = {}
        self.n_inputs = 0
        self.n_outputs = 0

    def add(self, decl, is_input):
        table = self.input_tensors if is_input else self.output_tensors
        if decl.name in table:
            raise VnnlibParseError(f"duplicate tensor declaration: {decl.name}")
        if is_input:
            table[decl.name] = (self.n_inputs, decl)
            self.n_inputs += decl.size
        else:
            table[decl.name] = (self.n_outputs, decl)
            self.n_outputs += decl.size

    def resolve(self, token):
        """'name[i,j,...]' or bare 'name' -> canonical flat var name."""
        if "[" in token:
            name, idx_str = token.split("[", 1)
            idx = tuple(int(p) for p in idx_str.rstrip("]").split(",") if p.strip())
        else:
            name, idx = token, ()

        for table, prefix in ((self.input_tensors, "X"), (self.output_tensors, "Y")):
            if name in table:
                offset, decl = table[name]
                if len(idx) != len(decl.shape):
                    raise VnnlibParseError(
                        f"index {token} does not match shape {decl.shape}")
                flat = 0
                for i, d in zip(idx, decl.shape):
                    if not 0 <= i < d:
                        raise VnnlibParseError(f"index out of bounds: {token}")
                    flat = flat * d + i
                return f"{prefix}_{offset + flat}"
        return None


# ---------------------------------------------------------------- poly exprs

def _poly_expr(expr, resolve):
    """expr -> dict {monomial tuple: coefficient}; () is the constant term."""
    if isinstance(expr, str):
        val = _as_float(expr)
        if val is not None:
            return {(): val}
        var = resolve(expr)
        if var is None:
            raise VnnlibParseError(f"unknown variable: {expr!r}")
        return {(var,): 1.0}

    if not expr:
        raise VnnlibParseError("empty expression")
    op = expr[0]

    if op == "+":
        poly = {}
        for sub in expr[1:]:
            for m, c in _poly_expr(sub, resolve).items():
                poly[m] = poly.get(m, 0.0) + c
        return poly

    if op == "-":
        if len(expr) == 2:  # unary
            return {m: -c for m, c in _poly_expr(expr[1], resolve).items()}
        poly = dict(_poly_expr(expr[1], resolve))
        for sub in expr[2:]:
            for m, c in _poly_expr(sub, resolve).items():
                poly[m] = poly.get(m, 0.0) - c
        return poly

    if op == "*":
        poly = {(): 1.0}
        for sub in expr[1:]:
            factor = _poly_expr(sub, resolve)
            product = {}
            for m1, c1 in poly.items():
                for m2, c2 in factor.items():
                    m = tuple(sorted(m1 + m2))
                    product[m] = product.get(m, 0.0) + c1 * c2
            poly = product
        return poly

    if op == "/":
        poly = _poly_expr(expr[1], resolve)
        for sub in expr[2:]:
            divisor = _poly_expr(sub, resolve)
            if set(divisor) - {()}:
                raise VnnlibParseError("division by variable")
            d = divisor.get((), 0.0)
            poly = {m: c / d for m, c in poly.items()}
        return poly

    raise VnnlibParseError(f"unsupported arithmetic operator: {op!r}")


def _atom(op, lhs, rhs, resolve):
    """Comparison -> DNF (list of clauses of PolynomialConstraint)."""
    diff = dict(_poly_expr(lhs, resolve))
    for m, c in _poly_expr(rhs, resolve).items():
        diff[m] = diff.get(m, 0.0) - c
    bias = diff.pop((), 0.0)
    diff = {m: c for m, c in diff.items() if c != 0.0}

    def make(sign, strict):
        return PolynomialConstraint(
            tuple(sorted((m, sign * c) for m, c in diff.items())),
            sign * bias, strict)

    if op in ("<=", "<"):
        return [[make(1, op == "<")]]
    if op in (">=", ">"):
        return [[make(-1, op == ">")]]
    if op in ("==", "="):
        return [[make(1, False), make(-1, False)]]
    if op == "!=":
        return [[make(1, True)], [make(-1, True)]]
    raise VnnlibParseError(f"unsupported comparison: {op!r}")


# ---------------------------------------------------------------- DNF

def _negate(expr):
    """Boolean expr -> its negation (still an expr tree)."""
    if not isinstance(expr, list) or not expr:
        raise VnnlibParseError(f"cannot negate: {expr!r}")
    # normalize infix atom first
    if len(expr) == 3 and isinstance(expr[1], str) and expr[1] in CMP_OPS:
        expr = [expr[1], expr[0], expr[2]]
    op = expr[0]
    if op == "not":
        return expr[1]
    if op == "and":
        return ["or"] + [_negate(s) for s in expr[1:]]
    if op == "or":
        return ["and"] + [_negate(s) for s in expr[1:]]
    if op in _NEGATED_OP:
        return [_NEGATED_OP[op]] + expr[1:]
    raise VnnlibParseError(f"cannot negate operator: {op!r}")


def _to_dnf(expr, resolve):
    """Boolean expr -> list of clauses, each a list of PolynomialConstraint."""
    if not isinstance(expr, list) or not expr:
        raise VnnlibParseError(f"expected boolean expression, got {expr!r}")

    # quirk: infix atom (a < b)
    if len(expr) == 3 and isinstance(expr[1], str) and expr[1] in CMP_OPS:
        return _atom(expr[1], expr[0], expr[2], resolve)

    op = expr[0]

    if op == "and":
        clauses = [[]]
        for sub in expr[1:]:
            sub_dnf = _to_dnf(sub, resolve)
            clauses = [c + s for c in clauses for s in sub_dnf]
            if len(clauses) > MAX_DNF_CLAUSES:
                raise VnnlibParseError("DNF explosion in 'and'")
        return clauses

    if op == "or":
        clauses = []
        for sub in expr[1:]:
            clauses.extend(_to_dnf(sub, resolve))
        if len(clauses) > MAX_DNF_CLAUSES:
            raise VnnlibParseError("DNF explosion in 'or'")
        return clauses

    if op == "not":
        return _to_dnf(_negate(expr[1]), resolve)

    if isinstance(op, str) and op in CMP_OPS:
        if len(expr) == 3:
            return _atom(op, expr[1], expr[2], resolve)
        # quirk: (<= (0.744 (- Y_0 ...))) -- doubled parens around both args
        if (len(expr) == 2 and isinstance(expr[1], list) and len(expr[1]) == 2
                and isinstance(expr[1][0], str) and _as_float(expr[1][0]) is not None):
            return _atom(op, expr[1][0], expr[1][1], resolve)
        raise VnnlibParseError(f"malformed comparison: {expr!r}")

    raise VnnlibParseError(f"unsupported boolean operator: {op!r}")


def _conjoin_asserts(asserts, resolve):
    clauses = [[]]
    for a in asserts:
        a_dnf = _to_dnf(a, resolve)
        if len(a_dnf) == 1:  # common case, avoid quadratic copying
            for c in clauses:
                c.extend(a_dnf[0])
        else:
            clauses = [c + s for c in clauses for s in a_dnf]
            if len(clauses) > MAX_DNF_CLAUSES:
                raise VnnlibParseError("DNF explosion combining asserts")
    return DisjunctiveSpec([ConjunctiveSpec(c) for c in clauses])


# ---------------------------------------------------------------- v2 parser

def _parse_shape(tok):
    body = tok.strip()
    if not (body.startswith("[") and body.endswith("]")):
        raise VnnlibParseError(f"expected shape literal, got {tok!r}")
    body = body[1:-1].strip()
    if not body:
        return ()
    return tuple(int(p) for p in body.split(","))


def detect_version(text):
    """'2.0' if the file uses v2 syntax markers, else '1.0'."""
    stripped = _COMMENT_RE.sub("", text)
    if "vnnlib-version" in stripped or "declare-network" in stripped:
        return "2.0"
    return "1.0"


def parse_vnnlib_v2(text):
    """Parse v2 vnnlib text into a version-agnostic VnnlibProperty."""
    sexprs = _parse_sexprs(_tokenize(text))

    version = "2.0"
    varmap = _VarMap()
    networks = []
    asserts = []

    for s in sexprs:
        if not isinstance(s, list) or not s:
            raise VnnlibParseError(f"unexpected top-level token: {s!r}")
        kind = s[0]

        if kind == "vnnlib-version":
            version = str(s[1]).strip("<>")

        elif kind == "declare-network":
            name = s[1]
            inputs, outputs, relations = [], [], []
            for item in s[2:]:
                if not isinstance(item, list) or not item:
                    raise VnnlibParseError(f"bad declare-network item: {item!r}")
                if item[0] == "declare-input":
                    decl = TensorDecl(item[1], item[2], _parse_shape(item[3]))
                    inputs.append(decl)
                    varmap.add(decl, is_input=True)
                elif item[0] == "declare-output":
                    decl = TensorDecl(item[1], item[2], _parse_shape(item[3]))
                    outputs.append(decl)
                    varmap.add(decl, is_input=False)
                elif item[0] in ("isomorphic-to", "equal-to"):
                    relations.append((item[0], item[1]))
                else:
                    raise VnnlibParseError(
                        f"unsupported declare-network item: {item[0]!r}")
            networks.append(NetworkDecl(name, tuple(inputs), tuple(outputs),
                                        tuple(relations)))

        elif kind == "assert":
            if len(s) != 2:
                raise VnnlibParseError(f"malformed assert: {s!r}")
            asserts.append(s[1])

        else:
            raise VnnlibParseError(f"unsupported v2 statement: {kind!r}")

    spec = _conjoin_asserts(asserts, varmap.resolve)
    return VnnlibProperty(version=version, networks=networks, spec=spec)


# ===========================================================================
# Adapter: VnnlibProperty (canonical DNF) -> VC's VNNSpec
#
# Splits each DNF clause into an X box (per-disjunct) and a list of Y
# constraints, mirroring the v1 regex path's VNNSpec shape exactly:
#   - global x_lo/x_hi = UNION (min lo, max hi) across all clause X boxes,
#     missing bounds filled with 0 (matching the v1 `or 0` fallback);
#   - per-disjunct input_lo/input_hi attached ONLY when the X box genuinely
#     varies across disjuncts (input-OR specs like acasxu prop_6 / nn4sys
#     lindex). When all clauses share one X box (simple + output-OR specs),
#     input_lo is None — identical to v1, and keeps the verifier off the
#     `_verify_per_disjunct_subboxes` path.
# Linear-only: a degree>=2 monomial, a mixed X/Y constraint, or a Y
# comparison VC can't represent as threshold/pairwise raises NotImplementedError
# (loud, never silently dropped).
# ===========================================================================

def _classify_var(name):
    """'X_3' -> ('X', 3); 'Y_2' -> ('Y', 2)."""
    kind, _, idx = name.partition("_")
    return kind, int(idx)


def _norm_zero(v):
    """Map IEEE -0.0 to +0.0 so adapter values match the v1 regex path's.

    `make(-1)` in `_atom` computes `-1 * 0.0 == -0.0`, which only differs from
    the v1 path's `+0.0` in sign-of-zero (numerically identical everywhere it
    is used). Normalizing keeps constraint values clean.
    """
    return 0.0 if v == 0.0 else v


def _adapt_y_constraint(var_coeffs, bias):
    """A pure-Y linear PolynomialConstraint -> VC Constraint / PairwiseConstraint.

    The constraint is `sum(c_i * Y_i) + bias <= 0` (the asserted/unsafe
    condition). Strictness is dropped: closing a strict `<` to `<=` enlarges
    the unsafe region, which is sound for verification (and matches the v1
    path, which also ignores strictness).
    """
    items = sorted(var_coeffs.items(), key=lambda kv: _classify_var(kv[0])[1])

    if len(items) == 1:
        v, c = items[0]
        _, i = _classify_var(v)
        thr = _norm_zero(-bias / c)
        # c>0: Y_i <= thr ; c<0: Y_i >= thr
        return Constraint(i, '<=', thr) if c > 0 else Constraint(i, '>=', thr)

    if len(items) == 2 and bias == 0.0:
        (va, ca), (vb, cb) = items
        # Pairwise needs opposite-sign, equal-magnitude unit-style coeffs:
        # c*(Y_pred - Y_comp) <= 0  <=>  Y_comp >= Y_pred (the unsafe region).
        # comp = the negative-coeff var, pred = the positive-coeff var.
        if ca * cb < 0 and abs(abs(ca) - abs(cb)) <= 1e-9 * max(abs(ca), abs(cb)):
            _, ia = _classify_var(va)
            _, ib = _classify_var(vb)
            if ca < 0:   # va has -coeff -> comp=ia, pred=ib
                return PairwiseConstraint(pred=ib, comp=ia)
            return PairwiseConstraint(pred=ia, comp=ib)   # vb has -coeff

    raise NotImplementedError(
        f'unsupported Y output constraint (VNNSpec cannot represent): '
        f'coeffs={var_coeffs} bias={bias}')


def _adapt_clause(clause, xbox, ycons):
    """Split one DNF clause's PolynomialConstraints into X box + Y constraints.

    `xbox` (dict i->[lo,hi]) and `ycons` (list) are accumulated in place.
    Multiple X bounds on the same var within a clause are INTERSECTED
    (conjunction): max of los, min of his.
    """
    for pc in clause.constraints:
        if not pc.is_linear:
            raise NotImplementedError(
                f'unsupported nonlinear vnnlib constraint (degree>=2 monomial): '
                f'{pc}')
        var_coeffs = {}
        for mono, c in pc.terms:
            var_coeffs[mono[0]] = var_coeffs.get(mono[0], 0.0) + c
        kinds = {_classify_var(v)[0] for v in var_coeffs}

        if kinds == {'X'}:
            if len(var_coeffs) != 1:
                raise NotImplementedError(
                    f'unsupported multi-variable X constraint (not a box): {pc}')
            v, c = next(iter(var_coeffs.items()))
            _, i = _classify_var(v)
            bound = _norm_zero(-pc.bias / c)
            slot = xbox.setdefault(i, [None, None])
            if c > 0:   # X_i <= bound (upper)
                slot[1] = bound if slot[1] is None else min(slot[1], bound)
            else:       # X_i >= bound (lower)
                slot[0] = bound if slot[0] is None else max(slot[0], bound)
        elif kinds == {'Y'}:
            ycons.append(_adapt_y_constraint(var_coeffs, pc.bias))
        else:
            raise NotImplementedError(
                f'unsupported vnnlib constraint shape (empty or mixed X/Y): {pc}')


def _vnnlib_v2_to_spec(prop, dtype=np.float32):
    """Map a parsed VnnlibProperty (v2) to VC's VNNSpec."""
    n_in = prop.num_inputs

    # Split every clause into (X box dict, Y constraint list).
    parsed = []
    for clause in prop.spec.clauses:
        xbox, ycons = {}, []
        _adapt_clause(clause, xbox, ycons)
        parsed.append((xbox, ycons))

    if not parsed:
        raise ValueError("No disjuncts found in v2 vnnlib spec")

    # Global bounding box = UNION across all clause X boxes (min lo, max hi).
    union = {}
    for xbox, _ in parsed:
        for i, (lo, hi) in xbox.items():
            slot = union.setdefault(i, [None, None])
            if lo is not None:
                slot[0] = lo if slot[0] is None else min(slot[0], lo)
            if hi is not None:
                slot[1] = hi if slot[1] is None else max(slot[1], hi)

    if not union:
        raise ValueError("No input bounds found in v2 vnnlib spec")

    # Match the v1 `or 0` fallback for missing/None bounds.
    x_lo = np.array([(union.get(i, [0, 0])[0] or 0) for i in range(n_in)],
                    dtype=dtype)
    x_hi = np.array([(union.get(i, [0, 0])[1] or 0) for i in range(n_in)],
                    dtype=dtype)

    # Attach per-disjunct boxes only when the X box genuinely varies across
    # disjuncts (input-OR). When all clauses share one box (simple +
    # output-OR), leave input_lo=None — identical to the v1 path.
    box_keys = {tuple(sorted((i, tuple(v)) for i, v in xbox.items()))
                for xbox, _ in parsed}
    attach = len(parsed) > 1 and len(box_keys) > 1

    disjuncts = []
    for xbox, ycons in parsed:
        if not ycons:
            raise ValueError(
                "v2 vnnlib disjunct has no output (Y) constraints")
        if attach:
            per_lo = np.empty(n_in, dtype=dtype)
            per_hi = np.empty(n_in, dtype=dtype)
            for i in range(n_in):
                blo, bhi = xbox.get(i, [None, None])
                per_lo[i] = x_lo[i] if blo is None else blo
                per_hi[i] = x_hi[i] if bhi is None else bhi
            disjuncts.append(Conjunct(ycons, input_lo=per_lo, input_hi=per_hi))
        else:
            disjuncts.append(Conjunct(ycons))

    return VNNSpec(x_lo, x_hi, disjuncts)
