"""CLI entry point for zonotope-based neural network verification."""

import argparse
import sys
import time

import numpy as np

from .network import ComputeGraph
from .vnnlib_loader import load_vnnlib
from .verify import zonotope_verify

_DTYPES = {'float32': np.float32, 'float64': np.float64,
           'f32': np.float32, 'f64': np.float64}


def main():
    parser = argparse.ArgumentParser(
        description='VibeCheck — Neural Network Verification via Zonotope Analysis')
    parser.add_argument('--net', required=True, help='Path to ONNX network')
    parser.add_argument('--spec', required=True, help='Path to VNNLIB specification')
    parser.add_argument('--dtype', default='float32', choices=list(_DTYPES),
                        help='Computation dtype (default: float32)')
    args = parser.parse_args()

    dtype = _DTYPES[args.dtype]
    t_start = time.time()

    print(f'Loading network: {args.net}')
    graph = ComputeGraph.from_onnx(args.net, dtype=dtype)
    n_relu = len(graph.relu_nodes())
    forks = graph.fork_points()
    print(f'  {len(graph.nodes)} ops, {n_relu} ReLU layers, '
          f'{len(forks)} fork points, input shape: {graph.input_shape}')

    print(f'Loading spec: {args.spec}')
    spec = load_vnnlib(args.spec)
    print(f'  {spec.n_constraints} constraint(s), '
          f'{len(spec.disjuncts)} disjunct(s)')

    print('Running zonotope analysis...')
    result, details = zonotope_verify(graph, spec)

    t_total = time.time() - t_start

    print(f'\nResult: {result}')
    print(f'  Worst margin: {details["worst_margin"]:.6f}')
    for i, margin in details['margins'].items():
        status = 'SAFE' if margin > 0 else 'UNKNOWN'
        print(f'  Disjunct {i}: margin={margin:.6f} [{status}]')
    print(f'  Time: {t_total:.2f}s')

    sys.exit(0 if result == 'verified' else 1)
