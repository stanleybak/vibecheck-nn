"""Verification orchestration — zonotope forward propagation + spec check."""

import numpy as np
from .zonotope import DenseZonotope
from .network import _prod


def zonotope_verify(graph, spec, dtype='graph'):
    """Run zonotope verification on a ComputeGraph with a VNNSpec.

    The algorithm:
    1. Create an initial zonotope from the spec's input bounds.
       - Center = midpoint of [x_lo, x_hi]
       - One generator column per input dimension with nonzero radius
       - A point input (x_lo == x_hi) produces 0 generators

    2. Propagate the zonotope through the graph in topological order.
       - Each node's zonotope_propagate() transforms center + generators
       - At fork points (node feeds multiple consumers), zonotopes are copied
         so each branch gets independent state
       - At merge points (Add with two computed inputs), shared generator
         columns are added element-wise, branch-specific columns concatenated

    3. Extract output bounds from the final zonotope.
       - lo = center - sum(|generators|), hi = center + sum(|generators|)
       - For point zonotopes these are exact (lo == hi == center)

    4. Check the spec's output constraints against these bounds.
       - Each constraint computes a margin (positive = verified safe)
       - All constraints must have positive margin for 'verified'

    Args:
        graph: ComputeGraph from onnx_loader
        spec: VNNSpec with x_lo, x_hi, and output constraints (disjuncts)
        dtype: numpy dtype for computation (np.float32 or np.float64, default float64)

    Returns:
        result: 'verified' or 'unknown'
        details: dict with output_lo, output_hi, margins, worst_margin
    """
    # Identify fork points — nodes whose output feeds multiple consumers.
    # These need zonotope copies so branches don't share mutable state.
    forks = graph.fork_points()

    # Initialize zonotope state for the graph input.
    # from_input_bounds creates generators only for dimensions with nonzero
    # radius, so x_lo == x_hi gives 0 generators (point propagation).
    zono_state = {}
    gen_count = {}
    dt = graph.dtype if dtype == 'graph' else dtype
    zono_state[graph.input_name] = DenseZonotope.from_input_bounds(
        spec.x_lo, spec.x_hi, dtype=dt)
    gen_count[graph.input_name] = zono_state[graph.input_name].generators.shape[1]

    # Helper: get input zonotope, copying at fork points to avoid aliasing.
    def _get_input(inp_name):
        if inp_name in forks:
            return zono_state[inp_name].copy()
        return zono_state[inp_name]

    # Forward propagation in topological order.
    # Each node reads its inputs from zono_state and writes its output.
    # SplitNode may pre-populate its children's entries in zono_state,
    # so we skip nodes that already have state.
    for name in graph.topo_order:
        if name in zono_state:
            continue
        node = graph.nodes[name]
        node.zonotope_propagate(
            zono_state, gen_count, _get_input, 'std', graph)
        gen_count[name] = zono_state[name].generators.shape[1]

    # Extract output bounds from the final zonotope.
    # For a DenseZonotope: lo = center - |G|.sum(axis=1),
    #                      hi = center + |G|.sum(axis=1)
    z_out = zono_state[graph.output_name]
    output_lo, output_hi = z_out.bounds()

    # Check the spec's output constraints against our computed bounds.
    # The spec defines the UNSAFE region as a disjunction of conjuncts.
    # Each conjunct computes a margin; positive means that unsafe region
    # is provably unreachable. All must be positive for 'verified'.
    result, check_details = spec.check(output_lo, output_hi)

    return result, {
        'output_lo': output_lo,
        'output_hi': output_hi,
        'margins': check_details['margins'],
        'worst_margin': check_details['worst_margin'],
    }
