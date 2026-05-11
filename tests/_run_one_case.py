"""Subprocess-side driver: run one verify_graph case and emit a JSON result.

Used by tests/sweep_relusplitter.py to isolate each verification in its own
process (so OOM / Gurobi crashes / runaway tensor allocs don't take down
the orchestrator). The harness invokes this with:

    .venv/bin/python -m tests._run_one_case
        --net <onnx>
        --spec <vnnlib>
        --timeout <seconds>
        --out <result.json>
        [--override-json '{"key": value, ...}']
        [--id <case-id>]

Exit code: 0 if verdict is 'verified', 1 otherwise (matches main.py).
The JSON record holds {id, status, wall_s, error, timing, phase, remaining,
settings_hash, expected, override_json}. `status` is 'verified' / 'sat' /
'unknown' / 'error' / 'timeout' (timeout = wall exceeded; verify_graph
itself returns 'unknown' on internal timeout).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _settings_hash(s) -> str:
    """Stable short hash over a DotMap-like settings object.

    Skips callables (e.g. milp_callback) and torch tensors. Used purely
    to flag baseline drift in --diff comparisons; not for soundness.
    """
    items = []
    for k in sorted(s.keys() if hasattr(s, 'keys') else []):
        v = s[k]
        if callable(v):
            continue
        try:
            items.append((k, repr(v)))
        except Exception:
            items.append((k, '<unrepr>'))
    blob = repr(items).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--net', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--out', required=True)
    p.add_argument('--override-json', default='{}')
    p.add_argument('--id', default='')
    p.add_argument('--expected', default='')
    args = p.parse_args()

    # Defer heavy imports until after argparse so --help is fast.
    from vibecheck.settings import default_settings
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.verify_graph import verify_graph

    overrides = json.loads(args.override_json)
    settings = default_settings()
    settings.total_timeout = float(args.timeout)
    settings.print_progress = False
    settings.verbose = False
    for k, v in overrides.items():
        settings[k] = v

    record = {
        'id': args.id or f'{Path(args.net).stem}::{Path(args.spec).stem}',
        'net': args.net,
        'spec': args.spec,
        'timeout': args.timeout,
        'override_json': overrides,
        'expected': args.expected,
        'settings_hash': _settings_hash(settings),
        'status': 'error',
        'wall_s': 0.0,
        'error': None,
        'timing': {},
        'phase': None,
        'remaining': None,
    }

    t0 = time.perf_counter()
    try:
        graph = load_onnx(args.net)
        graph.optimize(settings)
        spec = load_vnnlib(args.spec)
        result, details = verify_graph(graph, spec, settings)
        wall = time.perf_counter() - t0

        record['wall_s'] = wall
        record['status'] = result
        record['phase'] = details.get('phase')
        record['remaining'] = details.get('remaining')
        timing = details.get('timing', {}) or {}
        record['timing'] = {k: float(v) for k, v in timing.items()
                            if isinstance(v, (int, float))}
    except Exception as e:
        wall = time.perf_counter() - t0
        record['wall_s'] = wall
        record['status'] = 'error'
        record['error'] = f'{type(e).__name__}: {e}'
        record['traceback'] = traceback.format_exc()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(record, f, indent=2, default=str)

    # Empty cuda cache before exit. Prevents the next subprocess in a
    # sweep from hitting `torch.AcceleratorError: out of memory` at its
    # very first GPU allocation — observed reliably on RTX 3080 when the
    # CUDA driver lazily reclaims pages after process exit. Without this,
    # cifar_biasfield-class workloads (~9 GB peak) leave the next
    # subprocess starved for ~100 ms post-exit.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception:
        pass

    sys.exit(0 if record['status'] == 'verified' else 1)


if __name__ == '__main__':
    main()
