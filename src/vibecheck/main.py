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
    parser.add_argument('--mode', default='zonotope',
                        choices=['zonotope', 'bnb', 'milp'],
                        help='Verification mode (default: zonotope)')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                        help='Device for BnB mode (default: gpu)')
    parser.add_argument('--bits', type=int, default=32, choices=[16, 32, 64],
                        help='Float precision for BnB mode (default: 32)')
    parser.add_argument('--bnb-order', default='bfs', choices=['bfs', 'dfs'],
                        help='BnB search order (default: bfs)')
    parser.add_argument('--timeout', type=float, default=30,
                        help='BnB timeout in seconds (default: 30)')
    parser.add_argument('--pgd-restarts', type=int, default=100,
                        help='PGD restarts for BnB (default: 100)')
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

    if args.mode == 'bnb':
        from .settings import default_settings
        from .verify_zono_bnb import zonotope_bnb_verify
        settings = default_settings(
            device=args.device,
            bits=args.bits,
            bnb_order=args.bnb_order,
            bnb_timeout=args.timeout,
            pgd_restarts=args.pgd_restarts,
        )
        print(f'Running BnB verification (device={args.device}, '
              f'bits={args.bits}, order={args.bnb_order})...')
        result, details = zonotope_bnb_verify(graph, spec, settings)
    elif args.mode == 'milp':
        from .settings import default_settings
        from .verify_milp import milp_verify
        settings = default_settings(
            device=args.device,
            bits=args.bits,
            total_timeout=args.timeout,
            pgd_restarts=args.pgd_restarts,
        )
        print(f'Running MILP verification (device={args.device}, '
              f'timeout={args.timeout}s)...')
        result, details = milp_verify(graph, spec, settings)
    else:
        print('Running zonotope analysis...')
        result, details = zonotope_verify(graph, spec)

    t_total = time.time() - t_start

    print(f'\nResult: {result}')
    if 'worst_margin' in details:
        print(f'  Worst margin: {details["worst_margin"]:.6f}')
        for i, margin in details['margins'].items():
            status = 'SAFE' if margin > 0 else 'UNKNOWN'
            print(f'  Disjunct {i}: margin={margin:.6f} [{status}]')
    if 'n_evals' in details:
        print(f'  BnB evals: {details["n_evals"]}')
    if 'volume_proven' in details:
        print(f'  Volume proven: {details["volume_proven"]:.1%}')
    print(f'  Time: {t_total:.2f}s')

    sys.exit(0 if result == 'verified' else 1)
