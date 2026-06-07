#!/usr/bin/env python3
"""Batch-run vibecheck over VNNCOMP benchmark instances, emitting a results.csv
in the same format the other tools use in the vnncomp2025_results repo.

For each instance (onnx, vnnlib, timeout) in a benchmark's instances.csv this
runs the competition scripts faithfully — prepare_instance.sh (pkl + warmup)
then run_instance.sh — and records one results.csv row:

    <category>,<onnx_path>,<vnnlib_path>,<prepare_s>,<verdict>,<runtime_s>

(paths in the `vnncomp2025_benchmarks/benchmarks/<cat>/...` style the other
tools use). On a `sat` verdict the counterexample (the s-expression the run
writes after the verdict line) is gzipped to the sidecar the scoring harness
expects: <results_dir>/vibecheck/2025_<cat>/<net>_<prop>.counterexample.gz.

Resumable: instances already present in results.csv are skipped, so a crash /
GPU drop / reboot can just re-run the same command.

Usage:
    run_benchmarks.py <category|all> [--benchmarks-dir DIR] [--results-dir DIR]

Env:
    VNNCOMP_BENCHMARKS  default benchmarks dir (…/vnncomp2025_benchmarks)
    VIBECHECK_RESULTS_DIR  default results dir (…/vnncomp2025_results)
    VNNCOMP_PYTHON_PATH, VIBECHECK_PKL_CACHE_DIR  passed through to the scripts
"""
import argparse
import csv
import gzip
import os
import subprocess
import sys
import time

REGULAR_TRACK = [
    'acasxu_2023', 'cersyve', 'cgan_2023', 'cifar100_2024',
    'collins_rul_cnn_2022', 'cora_2024', 'dist_shift_2023', 'linearizenn_2024',
    'malbeware', 'metaroom_2023', 'nn4sys', 'safenlp_2024', 'sat_relu',
    'soundnessbench', 'tinyimagenet_2024', 'tllverifybench_2023',
]

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _strip_ext(name):
    for ext in ('.gz', '.onnx', '.vnnlib'):
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name


def _ce_stem(onnx_rel, vnnlib_rel):
    """`<net>_<prop>` stem for the .counterexample.gz, matching the scoring
    harness's `f"{net}_{prop}.counterexample.gz"` construction."""
    net = _strip_ext(os.path.basename(onnx_rel))
    prop = _strip_ext(os.path.basename(vnnlib_rel))
    return f'{net}_{prop}'


def _drop_instance_pkl(onnx_abs, vnnlib_abs):
    """Delete this instance's pre-parse .pkl cache (disk hygiene). Best-effort:
    import lazily and ignore any failure — a stale cache only wastes space."""
    try:
        import numpy as np
        from vibecheck.preparse import pkl_cache_path
        p = pkl_cache_path(onnx_abs, vnnlib_abs, np.float32)
        if os.path.isfile(p):
            os.remove(p)
    except (ImportError, OSError):
        pass


def _done_keys(results_csv):
    """(onnx, vnnlib) pairs already recorded, for resumability."""
    done = set()
    if os.path.isfile(results_csv):
        with open(results_csv, newline='') as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    done.add((row[1], row[2]))
    return done


def run_category(cat, benchmarks_dir, results_dir, version='v1'):
    bench = os.path.join(benchmarks_dir, 'benchmarks', cat)
    instances = os.path.join(bench, 'instances.csv')
    if not os.path.isfile(instances):
        print(f'[skip] {cat}: no instances.csv at {instances}')
        return
    out_dir = os.path.join(results_dir, 'vibecheck', f'2025_{cat}')
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'results.csv')
    done = _done_keys(results_csv)

    with open(instances, newline='') as f:
        rows = [r for r in csv.reader(f) if r and len(r) >= 3]
    print(f'=== {cat}: {len(rows)} instances ({len(done)} already done) → '
          f'{results_csv} ===', flush=True)

    prepare_sh = os.path.join(SCRIPT_DIR, 'prepare_instance.sh')
    run_sh = os.path.join(SCRIPT_DIR, 'run_instance.sh')

    for i, (onnx_rel, vnnlib_rel, timeout) in enumerate(rows):
        # Path style the other tools' results.csv use (results-repo relative).
        onnx_repo = f'vnncomp2025_benchmarks/benchmarks/{cat}/{onnx_rel}'
        vnnlib_repo = f'vnncomp2025_benchmarks/benchmarks/{cat}/{vnnlib_rel}'
        if (onnx_repo, vnnlib_repo) in done:
            continue
        onnx_abs = os.path.join(bench, onnx_rel)
        vnnlib_abs = os.path.join(bench, vnnlib_rel)

        # prepare (un-timed by the competition, but we record its wall).
        t0 = time.perf_counter()
        subprocess.run([prepare_sh, version, cat, onnx_abs, vnnlib_abs],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        prepare_s = time.perf_counter() - t0

        # run (the verdict file is authoritative).
        res_file = f'/tmp/vibecheck_res_{cat}_{i}.txt'
        if os.path.exists(res_file):
            os.remove(res_file)
        t0 = time.perf_counter()
        subprocess.run([run_sh, version, cat, onnx_abs, vnnlib_abs,
                        res_file, str(timeout)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        runtime_s = time.perf_counter() - t0

        verdict, ce_lines = 'unknown', []
        if os.path.isfile(res_file):
            with open(res_file) as rf:
                lines = rf.read().splitlines()
            if lines:
                verdict = lines[0].strip() or 'unknown'
                ce_lines = lines[1:]
            os.remove(res_file)

        # On sat, gzip the counterexample to the scoring sidecar.
        if verdict == 'sat' and ce_lines:
            ce_path = os.path.join(out_dir,
                                   f'{_ce_stem(onnx_rel, vnnlib_rel)}.counterexample.gz')
            with gzip.open(ce_path, 'wb') as gz:
                gz.write(('\n'.join(ce_lines) + '\n').encode('utf-8'))

        with open(results_csv, 'a', newline='') as out:
            csv.writer(out).writerow(
                [cat, onnx_repo, vnnlib_repo,
                 f'{prepare_s:.6f}', verdict, f'{runtime_s:.6f}'])

        # Disk hygiene (the run host can be tight on space over thousands of
        # instances): the per-instance pre-parse .pkl is only needed for THIS
        # instance's run, so drop it now. Best-effort; a leftover just wastes
        # space, never correctness.
        _drop_instance_pkl(onnx_abs, vnnlib_abs)

        print(f'  [{i+1}/{len(rows)}] {os.path.basename(onnx_rel)} / '
              f'{os.path.basename(vnnlib_rel)} → {verdict} '
              f'({runtime_s:.1f}s, prep {prepare_s:.1f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('category', help='benchmark category, or "all" (regular track)')
    ap.add_argument('--benchmarks-dir',
                    default=os.environ.get('VNNCOMP_BENCHMARKS',
                                           os.path.expanduser('~/vnncomp2025_benchmarks')))
    ap.add_argument('--results-dir',
                    default=os.environ.get('VIBECHECK_RESULTS_DIR',
                                           os.path.expanduser('~/vnncomp2025_results')))
    ap.add_argument('--version', default='v1')
    args = ap.parse_args()

    cats = REGULAR_TRACK if args.category == 'all' else [args.category]
    for cat in cats:
        run_category(cat, args.benchmarks_dir, args.results_dir, args.version)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
