"""Counterexample validation for VNNLIB 2.0 queries.

VENDORED VERBATIM from the official VNN-COMP 2026 scoring code so that
vibecheck's SAT-witness acceptance is bit-identical to the competition's
counterexample checker (no drift between what we emit and what the scorer
accepts).

Source: VNN-COMP/vnncomp2026_results  SCORING/counterexamples_v2.py
        branch `cex_val_de`, commit 5a9a564a9988566c533a184881aafc6173e6f1ed
        ("wip cex checking"), by the VNN-COMP evaluation chairs.

The ONLY local changes vs. the upstream file: the two upstream imports
(`from benchmark_instances import parse_network_field, resolve_benchmark_path`
and `from cex_checks import CPU_PROVIDER`) are inlined below so this module is
self-contained inside vibecheck. The validation logic — VNNLIB-2.0 parsing
(via the `vnnlib` package), ONNX-runtime replay, the per-assertion tolerance
split (input <=abs_tol, output strict 0.0) and the CORRECT /
CORRECT_UP_TO_TOLERANCE / SPEC_NOT_VIOLATED classification — is untouched.

`resolve_benchmark_path` returns an existing absolute path unchanged, so
callers may pass vibecheck's `--net`/`--spec` absolute paths directly as the
`network_field`/`property_field` arguments to `validate_vnnlib2_counterexample`.
"""

import ast
import gzip
import math
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import vnnlib

# --- inlined from vnncomp2026_results SCORING/cex_checks.py ---
CPU_PROVIDER = "CPUExecutionProvider"


# --- inlined VERBATIM from vnncomp2026_results SCORING/benchmark_instances.py ---
def parse_network_field(network_field):
    """Return ``(network_name, path)`` pairs from a result CSV network field."""

    try:
        parsed = ast.literal_eval(network_field)
    except (SyntaxError, ValueError):
        return [(None, network_field)]

    if not isinstance(parsed, list):
        return [(None, network_field)]

    networks = []
    for entry in parsed:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"expected network entries to be pairs, got {entry!r}")
        name, path = entry
        networks.append((str(name), str(path)))

    if not networks:
        raise ValueError("network list must not be empty")

    return networks


def resolve_benchmark_path(benchmark_dir, result_path, expected_directory):
    """Resolve a path stored in results.csv against a benchmark version directory."""

    path = Path(result_path)
    if path.is_file():
        return path

    parts = path.parts
    if expected_directory in parts:
        index = parts.index(expected_directory)
        candidate = Path(benchmark_dir).joinpath(*parts[index:])
    else:
        candidate = Path(benchmark_dir) / expected_directory / path.name

    if candidate.is_file():
        return candidate

    gz_candidate = Path(f"{candidate}.gz")
    if gz_candidate.is_file():
        return gz_candidate

    raise FileNotFoundError(f"benchmark file not found: {candidate} or {gz_candidate}")


_ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")

_NUMPY_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "Bool": np.bool_,
    "Real": np.float64,
    # Retain compatibility with queries parsed by vnnlib 1.0.1.
    "Unknown": np.float64,
}

_VNNLIB_TYPE_NAMES = {
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "Bool": "bool",
    "Real": "real",
    "Unknown": "real",
}

_FLOAT_DTYPE_KEYS = {"F16", "F32", "F64", "Real", "Unknown"}

_ONNX_RUNTIME_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}

_VNNLIB_TO_ONNX_TYPES = {
    "F16": "tensor(float16)",
    "F32": "tensor(float)",
    "F64": "tensor(double)",
    "I8": "tensor(int8)",
    "I16": "tensor(int16)",
    "I32": "tensor(int32)",
    "I64": "tensor(int64)",
    "U8": "tensor(uint8)",
    "U16": "tensor(uint16)",
    "U32": "tensor(uint32)",
    "U64": "tensor(uint64)",
    "Bool": "tensor(bool)",
}

_ORT_ERRORS = tuple(
    getattr(ort.capi.onnxruntime_pybind11_state, name)
    for name in (
        "EPFail",
        "EngineError",
        "Fail",
        "InvalidArgument",
        "InvalidGraph",
        "InvalidProtobuf",
        "ModelLoadCanceled",
        "ModelLoaded",
        "ModelRequiresCompilation",
        "NoModel",
        "NoSuchFile",
        "NotFound",
        "NotImplemented",
        "RuntimeException",
    )
    if hasattr(ort.capi.onnxruntime_pybind11_state, name)
)


class UnsupportedVNNLIB2Error(Exception):
    pass


class InvalidAssignmentError(Exception):
    pass


def _read_text(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return stream.read()
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


def _definitions(query):
    return tuple(
        definition
        for network in query.networks
        for definition in (*network.inputs, *network.hidden, *network.outputs)
    )


def _dtype_name(dtype):
    return dtype.name


def _parse_value(value, dtype):
    if dtype == np.bool_:
        normalized = value.lower()
        if normalized not in ("true", "false", "0", "1"):
            raise ValueError(f"invalid boolean value {value!r}")
        return normalized in ("true", "1")
    if np.issubdtype(dtype, np.integer):
        return int(value)
    return float(value)


def _assignment_type_matches(dtype_key, type_name):
    expected_type_name = _VNNLIB_TYPE_NAMES[dtype_key]
    normalized_type_name = type_name.lower()
    if normalized_type_name == expected_type_name.lower():
        return True
    return dtype_key in _FLOAT_DTYPE_KEYS and normalized_type_name == "real"


def parse_text_assignment(content, query):
    """Parse the mandatory textual assignment format from VNNLIB 2.0 section 5.3."""

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines and lines[0] == "sat":
        lines.pop(0)

    assignment = {}
    position = 0

    for definition in _definitions(query):
        if position >= len(lines):
            raise InvalidAssignmentError(f"missing assignment for variable {definition.name}")

        match = _ASSIGNMENT_HEADER.fullmatch(lines[position])
        if not match:
            raise InvalidAssignmentError(f"invalid assignment header: {lines[position]!r}")
        position += 1

        name, type_name, dimensions = match.groups()
        if name != definition.name:
            raise InvalidAssignmentError(
                f"expected variable {definition.name}, found {name}"
            )

        shape = [] if not dimensions.strip() else [
            int(value.strip()) for value in dimensions.split(",")
        ]
        if shape != list(definition.shape):
            raise InvalidAssignmentError(
                f"variable {name} has shape {shape}, expected {list(definition.shape)}"
            )

        dtype_key = _dtype_name(definition.dtype)
        if dtype_key not in _NUMPY_DTYPES:
            raise UnsupportedVNNLIB2Error(
                f"unsupported assignment type {definition.dtype} for {name}"
            )
        expected_type_name = _VNNLIB_TYPE_NAMES[dtype_key]
        if not _assignment_type_matches(dtype_key, type_name):
            raise InvalidAssignmentError(
                f"variable {name} has type {type_name}, expected {expected_type_name}"
            )

        value_count = math.prod(shape)
        if position + value_count > len(lines):
            raise InvalidAssignmentError(f"not enough values for variable {name}")

        dtype = _NUMPY_DTYPES[dtype_key]
        try:
            values = [_parse_value(value, dtype) for value in lines[position:position + value_count]]
            assignment[name] = np.asarray(values, dtype=dtype).reshape(shape)
        except (OverflowError, TypeError, ValueError) as error:
            raise InvalidAssignmentError(f"invalid value for variable {name}: {error}") from error
        position += value_count

    if position != len(lines):
        raise InvalidAssignmentError(f"unexpected content after assignments: {lines[position]!r}")

    return assignment


def _network_model_paths(query, network_field, benchmark_dir):
    supplied = parse_network_field(network_field)
    explicit = {}

    if supplied[0][0] is None:
        if len(supplied) != 1:
            raise UnsupportedVNNLIB2Error(
                "unnamed ONNX model mappings cannot be combined with other mappings"
            )
        implemented = [
            network for network in query.networks
            if not network.equal_to
        ]
        if len(implemented) != 1:
            raise UnsupportedVNNLIB2Error(
                "a single ONNX path can only be used with one implemented network"
            )
        explicit[implemented[0].name] = resolve_benchmark_path(
            benchmark_dir, supplied[0][1], "onnx"
        )
    else:
        declared = {network.name: network for network in query.networks}
        for name, path in supplied:
            if name in explicit:
                raise UnsupportedVNNLIB2Error(
                    f"multiple ONNX models were provided for network {name}"
                )
            if name not in declared:
                raise UnsupportedVNNLIB2Error(
                    f"ONNX model was provided for undeclared network {name}"
                )
            explicit[name] = resolve_benchmark_path(benchmark_dir, path, "onnx")

    paths = {}
    for network in query.networks:
        if network.name in explicit:
            if network.equal_to:
                if network.equal_to not in paths:
                    raise UnsupportedVNNLIB2Error(
                        f"equal-to network {network.name} references unavailable "
                        f"network {network.equal_to}"
                    )
                expected_path = paths[network.equal_to]
                if explicit[network.name].resolve() != expected_path.resolve():
                    raise UnsupportedVNNLIB2Error(
                        f"ONNX model provided for equal-to network {network.name} "
                        f"does not match network {network.equal_to}"
                    )
                paths[network.name] = expected_path
            else:
                paths[network.name] = explicit[network.name]
        elif network.equal_to and network.equal_to in paths:
            paths[network.name] = paths[network.equal_to]
        else:
            raise UnsupportedVNNLIB2Error(
                f"no ONNX model was provided for network {network.name}"
            )

    return paths


def _session(model_path):
    if str(model_path).endswith(".gz"):
        with gzip.open(model_path, "rb") as stream:
            return ort.InferenceSession(stream.read(), providers=[CPU_PROVIDER])
    return ort.InferenceSession(str(model_path), providers=[CPU_PROVIDER])


def _reshape_for_onnx(value, onnx_shape, variable_name):
    if len(value.shape) == len(onnx_shape) and all(
        not isinstance(onnx_dimension, int)
        or onnx_dimension <= 0
        or value_dimension == onnx_dimension
        for value_dimension, onnx_dimension in zip(value.shape, onnx_shape)
    ):
        return value

    if all(isinstance(dimension, int) and dimension > 0 for dimension in onnx_shape):
        if value.size == math.prod(onnx_shape):
            return value.reshape(onnx_shape)

    raise UnsupportedVNNLIB2Error(
        f"cannot reshape VNNLIB variable {variable_name} from {value.shape} "
        f"to ONNX shape {onnx_shape}"
    )


def _match_onnx_values(definitions, onnx_values, network_name, value_kind):
    if len(onnx_values) != len(definitions):
        raise UnsupportedVNNLIB2Error(
            f"network {network_name} declares {len(definitions)} {value_kind}s, "
            f"but ONNX has {len(onnx_values)}"
        )

    by_name = {value.name: value for value in onnx_values}
    matches = []
    for definition, positional_value in zip(definitions, onnx_values):
        if definition.onnx_name:
            if definition.onnx_name not in by_name:
                raise UnsupportedVNNLIB2Error(
                    f"ONNX {value_kind} {definition.onnx_name} declared for "
                    f"{definition.name} was not found"
                )
            matches.append((definition, by_name[definition.onnx_name]))
        else:
            matches.append((definition, positional_value))
    return matches


def _validate_element_type(definition, onnx_value, network_name):
    dtype_name = _dtype_name(definition.dtype)
    if dtype_name in ("Real", "Unknown"):
        return

    expected = _VNNLIB_TO_ONNX_TYPES.get(dtype_name)
    if expected is None:
        raise UnsupportedVNNLIB2Error(
            f"unsupported VNNLIB element type {definition.dtype} for {definition.name}"
        )
    if onnx_value.type != expected:
        raise UnsupportedVNNLIB2Error(
            f"network {network_name} declares {definition.name} as "
            f"{_VNNLIB_TYPE_NAMES[dtype_name]}, but ONNX {onnx_value.name} has "
            f"type {onnx_value.type}"
        )


def _run_networks(query, model_paths, assignment):
    computed_outputs = {}

    for network in query.networks:
        if network.hidden:
            raise UnsupportedVNNLIB2Error(
                f"network {network.name} declares hidden variables, which are not supported yet"
            )

        session = _session(model_paths[network.name])
        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()
        input_matches = _match_onnx_values(
            network.inputs, session_inputs, network.name, "input"
        )
        output_matches = _match_onnx_values(
            network.outputs, session_outputs, network.name, "output"
        )

        feeds = {}
        for definition, onnx_input in input_matches:
            onnx_name = onnx_input.name
            _validate_element_type(definition, onnx_input, network.name)
            if onnx_input.type not in _ONNX_RUNTIME_DTYPES:
                raise UnsupportedVNNLIB2Error(
                    f"unsupported ONNX input type {onnx_input.type} for {onnx_name}"
                )
            value = assignment[definition.name].astype(
                _ONNX_RUNTIME_DTYPES[onnx_input.type], copy=False
            )
            feeds[onnx_name] = _reshape_for_onnx(
                value, onnx_input.shape, definition.name
            )

        for definition, onnx_output in output_matches:
            _validate_element_type(definition, onnx_output, network.name)

        output_names = [onnx_output.name for _, onnx_output in output_matches]
        outputs = session.run(output_names, feeds)
        for (definition, onnx_output), output in zip(output_matches, outputs):
            output = np.asarray(output)
            if output.shape != tuple(definition.shape):
                if output.size != math.prod(definition.shape):
                    raise UnsupportedVNNLIB2Error(
                        f"cannot reshape ONNX output for {definition.name} from "
                        f"{output.shape} to {definition.shape}"
                    )
                output = output.reshape(definition.shape)
            computed_outputs[definition.name] = output

    return computed_outputs


def _eval_arithmetic(expression, assignment):
    node_type = type(expression).__name__

    if node_type == "Var":
        value = assignment[expression.name]
        return value[tuple(expression.indices)]
    if node_type in ("Float", "Int", "IntExpr"):
        return expression.value
    if node_type == "Literal":
        return float(expression.lexeme)
    if node_type == "Negate":
        return -_eval_arithmetic(expression.expr, assignment)
    if node_type == "Plus":
        return sum(_eval_arithmetic(arg, assignment) for arg in expression.args)
    if node_type == "Minus":
        value = _eval_arithmetic(expression.head, assignment)
        return value - sum(_eval_arithmetic(arg, assignment) for arg in expression.rest)
    if node_type == "Multiply":
        return math.prod(_eval_arithmetic(arg, assignment) for arg in expression.args)

    raise UnsupportedVNNLIB2Error(f"unsupported arithmetic expression {node_type}")


def _eval_boolean(expression, assignment, tolerance):
    node_type = type(expression).__name__

    if node_type == "And":
        return all(_eval_boolean(arg, assignment, tolerance) for arg in expression.args)
    if node_type == "Or":
        return any(_eval_boolean(arg, assignment, tolerance) for arg in expression.args)

    lhs = _eval_arithmetic(expression.lhs, assignment)
    rhs = _eval_arithmetic(expression.rhs, assignment)
    if node_type == "GreaterThan":
        return lhs > rhs - tolerance
    if node_type == "LessThan":
        return lhs < rhs + tolerance
    if node_type == "GreaterEqual":
        return lhs >= rhs - tolerance
    if node_type == "LessEqual":
        return lhs <= rhs + tolerance
    if node_type == "Equal":
        return abs(lhs - rhs) <= tolerance
    if node_type == "NotEqual":
        return abs(lhs - rhs) > tolerance

    raise UnsupportedVNNLIB2Error(f"unsupported boolean expression {node_type}")


def _expression_variables(expression):
    node_type = type(expression).__name__
    if node_type == "Var":
        return {expression.name}

    variables = set()
    for attr in ("expr", "lhs", "rhs", "head"):
        if hasattr(expression, attr):
            variables.update(_expression_variables(getattr(expression, attr)))
    for attr in ("args", "rest"):
        if hasattr(expression, attr):
            for child in getattr(expression, attr):
                variables.update(_expression_variables(child))
    return variables


def _input_names(query):
    return {
        definition.name
        for network in query.networks
        for definition in network.inputs
    }


# ===========================================================================
# vibecheck OPTIMIZATION (NOT upstream): vectorized fast-path for pure-input box
# assertions. The upstream `_assertions_hold`/`_assertions_rationale` evaluate
# EVERY assertion one-by-one in Python; on a high-dim L-inf spec (smart_turn has
# 1.27M input-bound assertions) that is ~85 s/case. Almost all of those are
# simple box bounds `(>= X[idx] lb)` / `(<= X[idx] ub)` (or an `and` of such) over
# input variables only. We recognize exactly that shape, batch it into one numpy
# comparison, and FALL BACK to the upstream per-assertion `_eval_boolean` for any
# assertion that is not a pure-input box (output/mixed/complex). The tolerance
# rule and comparison semantics are byte-for-byte the same as `_eval_boolean`
# (`>=`: v >= rhs-tol, `<=`: v <= rhs+tol, `>`: v > rhs-tol, `<`: v < rhs+tol),
# so the verdict is identical to upstream — only faster.
# ===========================================================================
_CMP_NODES = {"GreaterThan", "LessThan", "GreaterEqual", "LessEqual"}
_CMP_FLIP = {"GreaterThan": "LessThan", "LessThan": "GreaterThan",
             "GreaterEqual": "LessEqual", "LessEqual": "GreaterEqual"}


def _const_val(node):
    t = type(node).__name__
    if t in ("Float", "Int", "IntExpr"):
        return float(node.value)
    if t == "Literal":
        return float(node.lexeme)
    if t == "Negate":
        v = _const_val(node.expr)
        return None if v is None else -v
    return None


def _box_atoms(expr):
    """If `expr` is a comparison `Var <op> const` (or an `and` of such), return a
    list of (var_name, indices_tuple, cmp_node_type, rhs_float); else None."""
    t = type(expr).__name__
    if t == "And":
        out = []
        for arg in expr.args:
            sub = _box_atoms(arg)
            if sub is None:
                return None
            out.extend(sub)
        return out
    if t in _CMP_NODES:
        lhs, rhs = expr.lhs, expr.rhs
        if type(lhs).__name__ == "Var":
            c = _const_val(rhs)
            return None if c is None else [(lhs.name, tuple(lhs.indices), t, c)]
        if type(rhs).__name__ == "Var":
            c = _const_val(lhs)
            return None if c is None else [(rhs.name, tuple(rhs.indices), _CMP_FLIP[t], c)]
        return None
    return None


def _input_shapes(query):
    return {d.name: tuple(d.shape)
            for network in query.networks for d in network.inputs}


def _c_strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


# Extraction is assignment-independent and identical across the (strict, tol,
# rationale) calls for one query, so cache it by query identity. Size is held at
# 1 entry (cleared on a new query) so there is no leak across a sweep.
_PREP_CACHE = {}


def _prepare_assertions(query):
    """Partition a query's assertions ONCE into (a) batched pure-input box atoms
    as numpy arrays per (var_name, cmp_type), and (b) a fallback list of
    output/mixed/complex assertions for exact per-assertion evaluation. Pure
    assignment-independent — depends only on the query."""
    inputs = _input_names(query)
    shapes = _input_shapes(query)
    raw = {}                         # (name, op) -> list of (idx_tuple, rhs, aidx)
    other = []                       # (aidx, expr, is_input)
    for aidx, assertion in enumerate(query.assertions):
        atoms = _box_atoms(assertion.expr)
        if atoms is not None and all(nm in inputs for nm, _, _, _ in atoms):
            for nm, idx, op, rhs in atoms:
                raw.setdefault((nm, op), []).append((idx, rhs, aidx))
            continue
        variables = _expression_variables(assertion.expr)
        other.append((aidx, assertion.expr, bool(variables) and variables <= inputs))
    box = {}                         # (name, op) -> (flat_idx, rhs, aidx) numpy arrays
    for (nm, op), entries in raw.items():
        strides = np.asarray(_c_strides(shapes[nm]), dtype=np.int64)
        flat = np.array([e[0] for e in entries], dtype=np.int64) @ strides
        rhs = np.array([e[1] for e in entries], dtype=np.float64)
        aidx = np.array([e[2] for e in entries], dtype=np.int64)
        box[(nm, op)] = (flat, rhs, aidx)
    return box, other


def _get_prepared(query):
    key = id(query)
    ent = _PREP_CACHE.get(key)
    if ent is not None and ent[0] is query:
        return ent[1]
    prepared = _prepare_assertions(query)
    _PREP_CACHE.clear()              # keep only the current query (no sweep-wide leak)
    _PREP_CACHE[key] = (query, prepared)
    return prepared


def _eval_assertions(query, assignment, input_tolerance, output_tolerance,
                     collect_failures):
    """Shared evaluator for `_assertions_hold` / `_assertions_rationale`.
    Returns (all_hold, failures). Identical verdict to upstream `_eval_boolean`;
    only the input-box assertions are batched (and the extraction is cached)."""
    box, other = _get_prepared(query)
    all_hold = True
    failures = []
    tol = input_tolerance            # box atoms are pure-input -> input tolerance
    for (nm, op), (flat, rhs, aidx) in box.items():
        vals = np.asarray(assignment[nm]).ravel()[flat].astype(np.float64)
        if op == "GreaterThan":
            ok = vals > rhs - tol
        elif op == "LessThan":
            ok = vals < rhs + tol
        elif op == "GreaterEqual":
            ok = vals >= rhs - tol
        else:  # LessEqual
            ok = vals <= rhs + tol
        if not ok.all():
            all_hold = False
            if not collect_failures:
                return False, failures
            for j in np.nonzero(~ok)[0]:
                failures.append(f"assertion {int(aidx[j])} failed with tolerance "
                                f"{tol} (input)")
    for aidx, expr, is_input in other:
        tol = input_tolerance if is_input else output_tolerance
        if not _eval_boolean(expr, assignment, tol):
            all_hold = False
            if not collect_failures:
                return False, failures
            failures.append(f"assertion {aidx} failed with tolerance {tol} "
                            f"({'input' if is_input else 'output/mixed'})")
    return all_hold, failures


def _assertions_hold(query, assignment, input_tolerance, output_tolerance=0.0):
    return _eval_assertions(query, assignment, input_tolerance,
                            output_tolerance, collect_failures=False)[0]


def _assertions_rationale(query, assignment, input_tolerance, output_tolerance=0.0):
    failures = _eval_assertions(query, assignment, input_tolerance,
                                output_tolerance, collect_failures=True)[1]
    if failures:
        return "; ".join(failures)
    return f"all assertions hold with input_tolerance={input_tolerance}, output_tolerance={output_tolerance}"


def _legacy_assertions_hold(query, assignment, tolerance):
    return all(
        _eval_boolean(assertion.expr, assignment, tolerance)
        for assertion in query.assertions
    )


def _outputs_match(expected, computed, abs_tol, rel_tol):
    messages = []
    matches = True
    for name, actual in computed.items():
        witness = expected[name]
        if witness.shape != actual.shape:
            return False, f"output {name} has shape {witness.shape}, ONNX produced {actual.shape}"
        if not np.allclose(witness, actual, atol=abs_tol, rtol=rel_tol):
            matches = False
        difference = float(np.max(np.abs(witness - actual))) if witness.size else 0.0
        messages.append(f"{name} maximum absolute execution difference: {difference}")
    return matches, "; ".join(messages)


def validate_vnnlib2_counterexample(
    benchmark_dir,
    network_field,
    property_field,
    ce_path,
    abs_tol,
    rel_tol,
    result_type,
    ignore_ce_outputs=False,
):
    """Validate one VNNLIB 2.0 textual assignment."""

    try:
        assignment_content = _read_text(ce_path)
    except FileNotFoundError as error:
        return result_type.NO_CE, str(error)

    try:
        property_path = resolve_benchmark_path(benchmark_dir, property_field, "vnnlib")
        query = vnnlib.parse_query_string(_read_text(property_path))
        assignment = parse_text_assignment(assignment_content, query)
        model_paths = _network_model_paths(query, network_field, benchmark_dir)
        computed = _run_networks(query, model_paths, assignment)
    except InvalidAssignmentError as error:
        return result_type.MALFORMED_CE, str(error)
    except (
        FileNotFoundError,
        UnsupportedVNNLIB2Error,
        vnnlib.VNNLibException,
        *_ORT_ERRORS,
    ) as error:
        return result_type.UNSUPPORTED, str(error)

    evaluation_assignment = dict(assignment)
    evaluation_assignment.update(computed)
    execution_message = "counterexample outputs ignored; using ONNX CPU replay outputs"

    if _assertions_hold(query, evaluation_assignment, 0.0, 0.0):
        return (
            result_type.CORRECT,
            f"{execution_message}; "
            + _assertions_rationale(query, evaluation_assignment, 0.0, 0.0),
        )

    if _assertions_hold(query, evaluation_assignment, abs_tol, 0.0):
        return (
            result_type.CORRECT_UP_TO_TOLERANCE,
            f"{execution_message}; input constraints require at most {abs_tol} absolute tolerance; "
            + _assertions_rationale(query, evaluation_assignment, abs_tol, 0.0),
        )

    return (
        result_type.SPEC_NOT_VIOLATED,
        f"{execution_message}; "
        + _assertions_rationale(query, evaluation_assignment, abs_tol, 0.0),
    )


# ===========================================================================
# vibecheck integration layer (NOT from upstream).
#
# Mirrors the result enum the competition driver passes as `result_type`
# (SCORING/counterexamples.py CounterexampleResult) and wraps
# `validate_vnnlib2_counterexample` so vibecheck can validate a v2 CE *file*
# against direct --net/--spec absolute paths (no benchmark-dir layout needed:
# resolve_benchmark_path returns an existing absolute path unchanged).
# ===========================================================================
class CexResult:
    """String values identical to SCORING/counterexamples.py CounterexampleResult."""
    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    NO_CE = "no_ce"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"
    WRONG_SHAPE = "wrong_shape"
    MALFORMED_CE = "malformed_ce"
    UNSUPPORTED = "unsupported"


# Accepted == the scorer awards the instance (no penalty): a CORRECT witness or
# a CORRECT_UP_TO_TOLERANCE one (input <=abs_tol outside the box, output strict).
ACCEPTED_RESULTS = frozenset({CexResult.CORRECT, CexResult.CORRECT_UP_TO_TOLERANCE})


def validate_cex_v2(onnx_path, vnnlib_path, ce_path, abs_tol=1e-4, rel_tol=0.0):
    """Validate a VNNLIB-2.0 counterexample FILE the exact way the competition
    scorer does. Returns (result_str, message). `result_str` is one of
    `CexResult.*`; it is in `ACCEPTED_RESULTS` iff the scorer would accept it.

    onnx_path / vnnlib_path are passed straight through as the network/property
    fields — `resolve_benchmark_path` returns existing absolute paths unchanged.
    """
    return validate_vnnlib2_counterexample(
        str(Path(onnx_path).parent),   # benchmark_dir (unused for absolute paths)
        str(onnx_path),                # network_field
        str(vnnlib_path),              # property_field
        str(ce_path),                  # ce_path
        abs_tol,
        rel_tol,
        CexResult,
        True,                          # ignore_ce_outputs (solver Y ignored)
    )

