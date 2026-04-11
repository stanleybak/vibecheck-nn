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
                        choices=['zonotope', 'bnb', 'milp', 'graph'],
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
    elif args.mode == 'graph':
        from .settings import default_settings
        from .verify_graph import verify_graph
        settings = default_settings(
            device=args.device,
            bits=args.bits,
            total_timeout=args.timeout,
            pgd_restarts=args.pgd_restarts,
        )
        print(f'Running graph verification (device={args.device}, '
              f'impl={settings.graph_impl}, timeout={args.timeout}s)...')
        result, details = verify_graph(graph, spec, settings)
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
    if args.mode == 'graph':
        phase_timing = details.get('timing', {})
        if phase_timing:
            parts = [f'{k.replace("phase", "p").split("_", 1)[0]}={v:.2f}s'
                     if k.startswith('phase') else f'{k}={v:.2f}s'
                     for k, v in phase_timing.items()]
            print('  Timing: ' + '  '.join(parts))
        n_splits = details.get('n_splits', {})
        if n_splits:
            split_str = ' '.join(f'{k}:{v}' for k, v in n_splits.items() if v > 0)
            print(f'  Splits: {split_str}')
        if 'build_time_total' in details:
            print(f'  Build time total: {details["build_time_total"]:.2f}s')
        per_layer = details.get('per_layer_timing', {})
        if per_layer:
            print('  Per-layer tightening:')
            for name, t in per_layer.items():
                wz = t.get('width_zono', 0.0)
                wa = t.get('width_adapt', 0.0)
                wl = t.get('width_lp', 0.0)
                wf = t.get('width_final', 0.0)
                print(f'    {name}: '
                      f'build={t["build"]:.2f}s '
                      f'probe={t["probe"]:.2f}s '
                      f'solve={t["solve"]:.2f}s  '
                      f'width zono={wz:.3f}→adapt={wa:.3f}'
                      f'→lp={wl:.3f}→final={wf:.3f}')
    print(f'  Time: {t_total:.2f}s')

    sys.exit(0 if result == 'verified' else 1)
