"""vnncomp benchmark tests — load, propagate, compare vs onnxruntime.

By default only the regular track is run. To also run the extended track:
    pytest tests/test_vnncomp_nets.py -k "regular or extended" -v -s
"""

import glob
import gzip
import os
import time
import numpy as np
import pytest
from pathlib import Path

import onnxruntime as ort

from vibecheck.network import ComputeGraph
from vibecheck.verify import zonotope_verify
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.zonotope import DenseZonotope


# ---- vnncomp tracks (from scoring repo settings.py) ----

_REGULAR_BENCHMARKS = {
    'safenlp_2024', 'nn4sys', 'cora_2024', 'linearizenn_2024',
    'dist_shift_2023', 'cifar100_2024', 'tinyimagenet_2024',
    'acasxu_2023', 'cgan_2023', 'collins_rul_cnn_2022',
    'metaroom_2023', 'tllverifybench_2023', 'cersyve',
    'malbeware', 'sat_relu', 'soundnessbench',
}
_EXTENDED_BENCHMARKS = {
    'ml4acopf_2024', 'collins_aerospace_benchmark', 'lsnc_relu',
    'yolo_2023', 'cctsdb_yolo_2023', 'traffic_signs_recognition_2023',
    'vggnet16_2022', 'vit_2023', 'relusplitter',
}

# Networks that cannot be tested (corrupt ONNX, spec/network mismatch, no files)
_MISSING = {
    'vggnet16_2022',
    'nn4sys/mscn_2048d_dual',
    'nn4sys/pensieve_big_parallel',
    'nn4sys/pensieve_small_parallel',
}

# Extended track networks that currently fail
_HARD_EXTENDED = {
    'cctsdb_yolo_2023',
    'collins_aerospace_benchmark',
    'ml4acopf_2024',
    # vit_2023: the gpu_graph path is point-prop EXACT (pinned by
    # tests/test_vit_gg_pointprop.py) but the basic per-node
    # DenseZonotope path still mis-propagates the N-D attention ops
    # this test exercises (~1.2 output error) — keep it out until the
    # basic path either supports or loudly refuses those ops.
    'vit_2023',
}


# ---- Discovery ----

def _discover_cases(vnncomp_path, include=None, exclude=None):
    """Discover (test_id, onnx_path, spec_path) from instances.csv.

    Picks the first instance row for each unique network.
    """
    import csv
    base = str(vnncomp_path)
    cases = []
    for d in sorted(os.listdir(base)):
        if d in _MISSING:
            continue
        if exclude is not None and d in exclude:
            continue
        if include is not None and d not in include:
            if not any(i.startswith(d + '/') for i in include):
                continue
        csv_path = os.path.join(base, d, 'instances.csv')
        if not os.path.exists(csv_path):
            continue
        seen_nets = set()
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                onnx_rel = row[0].lstrip('./')
                spec_rel = row[1].lstrip('./')
                onnx_path = os.path.join(base, d, onnx_rel)
                spec_path = os.path.join(base, d, spec_rel)
                if not os.path.exists(onnx_path):
                    if os.path.exists(onnx_path + '.gz'):
                        onnx_path += '.gz'
                if not os.path.exists(spec_path):
                    if os.path.exists(spec_path + '.gz'):
                        spec_path += '.gz'
                if not os.path.exists(onnx_path):
                    continue
                net_name = os.path.basename(onnx_path).replace(
                    '.onnx.gz', '').replace('.onnx', '')
                if net_name in seen_nets:
                    continue
                seen_nets.add(net_name)
                test_id = f'{d}/{net_name}'
                if test_id in _MISSING:
                    continue
                if exclude is not None and test_id in exclude:
                    continue
                if include is not None and d not in include and test_id not in include:
                    continue
                cases.append((test_id, onnx_path, spec_path))
    return cases


def _resolve_vnncomp_path():
    """Resolve benchmarks path from paths.yaml at collection time."""
    paths_file = Path(__file__).parent / "paths.yaml"
    if not paths_file.exists():
        return None
    import yaml
    with open(paths_file) as f:
        paths = yaml.safe_load(f) or {}
    p = paths.get("vnncomp_benchmarks")
    if not p or not os.path.exists(p):
        return None
    base = Path(p)
    if (base / "benchmarks").is_dir():
        base = base / "benchmarks"
    return base


def _get_ids(include=None, exclude=None):
    base = _resolve_vnncomp_path()
    if base is None:
        return []
    return [c[0] for c in _discover_cases(base, include=include, exclude=exclude)]


_REGULAR_IDS = _get_ids(include=_REGULAR_BENCHMARKS)
_EXTENDED_IDS = _get_ids(include=_EXTENDED_BENCHMARKS, exclude=_HARD_EXTENDED)
_HARD_EXTENDED_IDS = _get_ids(include=_HARD_EXTENDED)


# ---- Per-node diagnostic (on soundness failure) ----

def _ort_node_compare(onnx_path, graph, center):
    """Run ort with all intermediate outputs and compare per-node."""
    import onnx

    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model = onnx.load_from_string(f.read())
    else:
        model = onnx.load(onnx_path)

    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing:
                model.graph.output.append(
                    onnx.helper.make_tensor_value_info(
                        out, onnx.TensorProto.FLOAT, None))

    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    inp_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    feed = {inp.name: center.astype(np.float32).reshape(inp_shape)}
    out_names = [o.name for o in sess.get_outputs()]
    results = sess.run(out_names, feed)
    ort_vals = {name: val.flatten().astype(np.float64)
                for name, val in zip(out_names, results)}

    # Our point propagation
    forks = graph.fork_points()
    zono_state = {graph.input_name: DenseZonotope(
        center, np.zeros((len(center), 0)))}
    gen_count = {graph.input_name: 0}
    def _get(name):
        return zono_state[name].copy() if name in forks else zono_state[name]
    for name in graph.topo_order:
        if name in zono_state:
            continue
        graph.nodes[name].zonotope_propagate(
            zono_state, gen_count, _get, 'std', graph)
        gen_count[name] = 0

    lines = [f"{'idx':>4s}  {'op':15s}  {'size':>6s}  {'max_err':>10s}  "
             f"{'ort_range':>24s}  {'our_range':>24s}",
             "-" * 100]

    first_bad = None
    for i, name in enumerate(graph.topo_order):
        node = graph.nodes[name]
        our_val = zono_state[name].center
        if name not in ort_vals:
            lines.append(f"[{i:>3d}]  {node.op_type:15s}  {len(our_val):>6d}  "
                         f"{'(no ort)':>10s}")
            continue
        ort_val = ort_vals[name]
        n = min(len(ort_val), len(our_val))
        err = np.max(np.abs(ort_val[:n] - our_val[:n]))
        ort_rng = f"[{ort_val[:n].min():.4f}, {ort_val[:n].max():.4f}]"
        our_rng = f"[{our_val.min():.4f}, {our_val.max():.4f}]"
        marker = " <<< FIRST" if first_bad is None and err > 1e-3 else ""
        if first_bad is None and err > 1e-3:
            first_bad = (i, node.op_type, name)
        lines.append(f"[{i:>3d}]  {node.op_type:15s}  {n:>6d}  {err:>10.2e}  "
                     f"{ort_rng:>24s}  {our_rng:>24s}{marker}")

    report = '\n'.join(lines)
    if first_bad:
        idx, op, name = first_bad
        return f"Soundness diverges at [{idx}] {op} ({name[:60]})\n\n{report}"
    return f"No per-node divergence found\n\n{report}"


# ---- Test runner (in-process, no subprocess) ----

def _run_benchmark(vnncomp_benchmarks, case_id, track_benchmarks):
    """Load network + spec, run point propagation, compare vs onnxruntime."""
    cases = _discover_cases(vnncomp_benchmarks)
    case = next((c for c in cases if c[0] == case_id), None)
    assert case is not None, f'case {case_id} not found'
    _, onnx_path, spec_path = case

    # Load network
    t0 = time.perf_counter()
    g = ComputeGraph.from_onnx(onnx_path)
    t_load = time.perf_counter() - t0

    flat_input = 1
    for d in g.input_shape:
        flat_input *= d

    # Parse spec
    spec = load_vnnlib(spec_path)
    assert len(spec.x_lo) == flat_input, (
        f'spec input size {len(spec.x_lo)} != network {flat_input}')

    # Point zonotope at center of spec box
    center = (spec.x_lo + spec.x_hi) / 2
    spec.x_lo = center.copy()
    spec.x_hi = center.copy()

    # Verify
    t0 = time.perf_counter()
    result, details = zonotope_verify(g, spec)
    t_verify = time.perf_counter() - t0

    # Compare against onnxruntime
    if onnx_path.endswith('.gz'):
        with gzip.open(onnx_path, 'rb') as f:
            model_bytes = f.read()
        sess = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
    else:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    feed = {inp_name: center.astype(np.float32).reshape(
        [d if isinstance(d, int) and d > 0 else 1 for d in inp_shape])}
    t0 = time.perf_counter()
    ort_out = sess.run(None, feed)[0].flatten().astype(np.float64)
    t_ort = time.perf_counter() - t0

    out_lo = details['output_lo']
    out_hi = details['output_hi']
    our_out = (out_lo + out_hi) / 2
    n = min(len(ort_out), len(our_out))
    max_err = float(np.max(np.abs(ort_out[:n] - our_out[:n])))

    # Print summary
    print(f'  net:  {os.path.basename(onnx_path)}')
    print(f'  spec: {os.path.basename(spec_path)}')
    print(f'  {len(g.topo_order)} ops  in={g.input_shape}  '
          f'load={t_load:.3f}s  verify={t_verify:.3f}s  '
          f'ort={t_ort:.3f}s  err={max_err:.2e}')

    # Soundness check
    tol = 1e-4
    if np.any(ort_out[:n] < out_lo[:n] - tol) or np.any(ort_out[:n] > out_hi[:n] + tol):
        diag = _ort_node_compare(onnx_path, g, center)
        pytest.fail(f'{case_id}: {diag}')


# ---- Regular track (run by default) ----

@pytest.mark.parametrize('case_id', _REGULAR_IDS)
def test_vnncomp_regular(vnncomp_benchmarks, case_id):
    """Regular track: load, point-propagate, compare vs onnxruntime."""
    _run_benchmark(vnncomp_benchmarks, case_id, _REGULAR_BENCHMARKS)


# ---- Extended track ----
# Run with: pytest tests/test_vnncomp_nets.py -k "extended" -v -s

@pytest.mark.parametrize('case_id', _EXTENDED_IDS)
def test_vnncomp_extended(vnncomp_benchmarks, case_id):
    """Extended track: load, point-propagate, compare vs onnxruntime."""
    _run_benchmark(vnncomp_benchmarks, case_id, _EXTENDED_BENCHMARKS)


# ---- Hard extended (currently failing) ----
# Run with: pytest tests/test_vnncomp_nets.py -k "hard" -v -s

@pytest.mark.parametrize('case_id', _HARD_EXTENDED_IDS)
def test_vnncomp_hard_extended(vnncomp_benchmarks, case_id):
    """Extended track networks that currently fail."""
    _run_benchmark(vnncomp_benchmarks, case_id, _EXTENDED_BENCHMARKS)
