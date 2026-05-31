"""Parser coverage across VNNLIB spec shapes, with an emphasis on the
SOUNDNESS of disjunctive INPUT regions.

The headline regression: a spec whose input region is a DISJUNCTION of boxes
with a GAP between them (acasxu prop_6/7/8: heading psi split into two ranges)
was flattened to its convex-hull bounding box, losing the gap. A point in the
gap that violates the output condition was then accepted as a counterexample
(`check_witness` -> True) even though it is not in the actual input region ->
false `sat` (unsound).

These tests use an IDENTITY net (Y == X) so the ground truth is trivial: a point
x is a counterexample iff x is in the (true, possibly disjunctive) input region
AND y == x violates the output condition. Anything in the excluded gap must
NEVER be a counterexample.
"""
import numpy as np
import onnx
import onnx.helper as oh
import pytest

from vibecheck.vnnlib_loader import parse_vnnlib_text


# ---------------------------------------------------------------------------
# Spec-level tests (pure parser + spec.check_witness; no net, CI-safe)
# ---------------------------------------------------------------------------

def test_disjunctive_input_gap_rejects_gap_witness():
    """X_0 in [0,1] OR [2,3] (gap (1,2) excluded); unsafe Y_0 in [1.2,1.8]
    (reachable, under identity, ONLY in the gap). A gap point must NOT be a
    counterexample. Before the fix the input disjunction collapsed to the hull
    [0,3] and the gap point was wrongly accepted."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0))
        (and (>= X_0 2.0) (<= X_0 3.0))))
    (assert (or
        (and (>= Y_0 1.2) (<= Y_0 1.8))))
    """
    spec = parse_vnnlib_text(txt)
    # Every disjunct must carry a real input sub-box (not the hull) so the gap
    # is enforced.
    assert all(d.input_lo is not None for d in spec.disjuncts), \
        'disjunctive input collapsed to hull (no per-disjunct input box)'
    # Gap point x=1.5 with identity output y=1.5: output unsafe, but x is in
    # NEITHER input box -> not a counterexample.
    is_ce, _ = spec.check_witness(np.array([1.5]), np.array([1.5]))
    assert not is_ce, 'gap point wrongly accepted as CEX (input region flattened to hull)'


def test_disjunctive_input_real_witness_in_box_accepted():
    """The fix must not over-restrict: a genuine CEX inside one of the input
    boxes is still accepted."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0))
        (and (>= X_0 2.0) (<= X_0 3.0))))
    (assert (or
        (and (>= Y_0 2.5))))
    """
    spec = parse_vnnlib_text(txt)
    # x=2.7 is in box [2,3] and y=2.7 >= 2.5 -> real CEX.
    is_ce, _ = spec.check_witness(np.array([2.7]), np.array([2.7]))
    assert is_ce, 'real CEX inside an input box wrongly rejected'
    # x=0.5 is in box [0,1] but y=0.5 < 2.5 -> not a CEX.
    is_ce2, _ = spec.check_witness(np.array([0.5]), np.array([0.5]))
    assert not is_ce2


def test_cross_product_disjunctive_input_and_output():
    """prop_6 shape: input-OR (2 boxes) AND output-OR (2 conditions) in SEPARATE
    asserts. Must cross-product to 4 disjuncts, each with a real input box, and
    reject a gap point regardless of which output condition it would satisfy."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0))
        (and (>= X_0 2.0) (<= X_0 3.0))))
    (assert (or
        (and (>= Y_0 1.3))
        (and (>= Y_1 1.3))))
    """
    spec = parse_vnnlib_text(txt)
    assert len(spec.disjuncts) == 4, \
        f'expected 2x2 cross-product, got {len(spec.disjuncts)} disjuncts'
    assert all(d.input_lo is not None for d in spec.disjuncts)
    # Gap point x=1.5 with output (1.5, 1.5): both Y_0>=1.3 and Y_1>=1.3 hold,
    # but x is in the gap -> not a CEX.
    is_ce, _ = spec.check_witness(np.array([1.5]), np.array([1.5, 1.5]))
    assert not is_ce, 'gap point accepted in cross-product spec'
    # Real CEX: x=2.4 in box [2,3], Y_0=2.4 >= 1.3.
    is_ce2, _ = spec.check_witness(np.array([2.4]), np.array([2.4, 2.4]))
    assert is_ce2


def test_cross_product_intersects_overlapping_x_constraint():
    """Mixed case: an input-OR AND an output block that ALSO carries its own X
    constraint on an overlapping dim. The cross-product must INTERSECT the two X
    constraints per disjunct (dropping the block's own X would be unsound)."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 5.0))
        (and (>= X_0 10.0) (<= X_0 15.0))))
    (assert (or
        (and (>= X_0 2.0) (>= Y_0 1.0))))
    """
    spec = parse_vnnlib_text(txt)
    assert len(spec.disjuncts) == 2
    boxes = sorted((float(d.input_lo[0]), float(d.input_hi[0]))
                   for d in spec.disjuncts)
    # box [0,5] ∩ {X_0>=2} = [2,5]; box [10,15] ∩ {X_0>=2} = [10,15].
    assert boxes == [(2.0, 5.0), (10.0, 15.0)], boxes
    # x=1.0 in [0,5] but NOT in the intersection [2,5] -> not a CEX.
    assert not spec.check_witness(np.array([1.0]), np.array([5.0]))[0]
    # x=3.0 in [2,5] and y=5 >= 1 -> CEX.
    assert spec.check_witness(np.array([3.0]), np.array([5.0]))[0]


def test_single_box_threshold_unchanged():
    """Plain single-box threshold spec keeps a single disjunct, no input box
    attached (global box governs), witness logic unaffected."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (>= X_0 0.0))
    (assert (<= X_0 1.0))
    (assert (>= Y_0 0.5))
    """
    spec = parse_vnnlib_text(txt)
    assert len(spec.disjuncts) == 1
    assert spec.disjuncts[0].input_lo is None
    assert float(spec.x_lo[0]) == 0.0 and float(spec.x_hi[0]) == 1.0


def test_dnf_mixed_xy_blocks_unchanged():
    """nn4sys lindex shape: each (and ...) block carries BOTH its own X subbox
    and Y constraints. Must remain one disjunct per block with that box."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0) (>= Y_0 5.0))
        (and (>= X_0 2.0) (<= X_0 3.0) (>= Y_0 9.0))))
    """
    spec = parse_vnnlib_text(txt)
    assert len(spec.disjuncts) == 2
    assert all(d.input_lo is not None for d in spec.disjuncts)
    # x=0.5,y=6: in box0 [0,1] AND Y_0=6>=5 -> CEX.
    assert spec.check_witness(np.array([0.5]), np.array([6.0]))[0]
    # x=0.5,y=6 must NOT match box1's Y_0>=9; x=2.5,y=6 in box1 but 6<9 -> no.
    assert not spec.check_witness(np.array([2.5]), np.array([6.0]))[0]


def test_disjunctive_output_single_box():
    """Single input box, disjunctive output (OR of two conditions). No input
    sub-boxes needed; both output conditions become disjuncts."""
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (assert (>= X_0 -1.0))
    (assert (<= X_0 1.0))
    (assert (or
        (and (>= Y_0 0.5))
        (and (>= Y_1 0.5))))
    """
    spec = parse_vnnlib_text(txt)
    assert len(spec.disjuncts) == 2


# ---------------------------------------------------------------------------
# End-to-end: identity net + verify must NOT return 'sat' on the gap spec
# ---------------------------------------------------------------------------

def _identity_onnx(tmp_path, n=1):
    """Y = ReLU(I @ X) — identity on the (all-positive) input regions used by
    these tests, via Gemm(W=I, b=0) -> ReLU. The ReLU is semantically a no-op on
    x >= 0 but gives the verifier an activation layer (so the alpha-CROWN slope
    optimiser has parameters; a pure-Gemm net has none)."""
    W = np.eye(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    inp = oh.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, n])
    out = oh.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, n])
    inits = [
        oh.make_tensor('W', onnx.TensorProto.FLOAT, [n, n], W.flatten()),
        oh.make_tensor('B', onnx.TensorProto.FLOAT, [n], b),
    ]
    nodes = [
        oh.make_node('Gemm', ['x', 'W', 'B'], ['z'], transB=1),
        oh.make_node('Relu', ['z'], ['y']),
    ]
    graph = oh.make_graph(nodes, 'identity', [inp], [out], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 14)])
    model.ir_version = 7
    p = tmp_path / f'identity_{n}.onnx'
    onnx.save(model, str(p))
    return str(p)


def test_identity_net_gap_spec_not_sat(tmp_path):
    """End-to-end: identity net, disjunctive input with a gap, output reachable
    only in the gap. The verifier must NOT return 'sat' (the only "witness"
    lives in the excluded gap)."""
    from vibecheck.network import ComputeGraph
    from vibecheck.settings import default_settings
    from vibecheck.verify_graph import verify_graph

    net = _identity_onnx(tmp_path, n=1)
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0))
        (and (>= X_0 2.0) (<= X_0 3.0))))
    (assert (or
        (and (>= Y_0 1.2) (<= Y_0 1.8))))
    """
    spec = parse_vnnlib_text(txt)
    graph = ComputeGraph.from_onnx(net, dtype=np.float32)
    settings = default_settings(device='cpu', bits=32, total_timeout=20)
    settings.print_progress = False
    graph.optimize(settings)
    result, _ = verify_graph(graph, spec, settings)
    assert result != 'sat', (
        f'identity net returned {result!r} on a spec whose only output-violating '
        'points are in the excluded input gap -> false SAT (unsound)')


def test_identity_net_real_sat(tmp_path):
    """Control: same net/input region but the unsafe output IS reachable inside
    a real input box -> verifier should find it (sat), confirming the fix did
    not break genuine CEX detection."""
    from vibecheck.network import ComputeGraph
    from vibecheck.settings import default_settings
    from vibecheck.verify_graph import verify_graph

    net = _identity_onnx(tmp_path, n=1)
    txt = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0))
        (and (>= X_0 2.0) (<= X_0 3.0))))
    (assert (or
        (and (>= Y_0 2.5))))
    """
    spec = parse_vnnlib_text(txt)
    graph = ComputeGraph.from_onnx(net, dtype=np.float32)
    settings = default_settings(device='cpu', bits=32, total_timeout=20)
    settings.print_progress = False
    graph.optimize(settings)
    result, _ = verify_graph(graph, spec, settings)
    assert result == 'sat', f'genuine CEX in box [2,3] not found (got {result!r})'
