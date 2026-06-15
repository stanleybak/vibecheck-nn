#!/usr/bin/env python3
"""Batch-run vibecheck over VNNCOMP benchmark instances, emitting a results.csv
in the same format the other tools use in the vnncomp results repos.

For each instance (onnx, vnnlib, timeout) in a benchmark's instances.csv this
runs the competition scripts faithfully - prepare_instance.sh (pkl + warmup)
then run_instance.sh - and records one results.csv row:

    <category>,<onnx_path>,<vnnlib_path>,<prepare_s>,<verdict>,<runtime_s>

On a `sat` verdict the counterexample (the s-expression the run writes after the
verdict line) is gzipped to the sidecar the scoring harness expects:
<results_dir>/vibecheck/<year>_<cat>/<net>_<prop>.counterexample.gz.

VNNCOMP 2026 layout: benchmarks are versioned, `benchmarks/<cat>/<version>/`
(e.g. `1.0`, `2.0`), each version dir holding its own instances.csv + onnx/ +
vnnlib/. `--bench-version` selects which (default `1.0`, i.e. the v1 specs;
v2 spec parsing is not wired up yet). The flat 2025 layout
(`benchmarks/<cat>/instances.csv`) is still accepted as a fallback. The
results-repo path style and the `<year>_<cat>` output tag are derived from the
benchmarks-dir name, so pointing `--benchmarks-dir` at a 2025 or 2026 clone
produces the matching paths automatically.

Resumable: instances already present in results.csv are skipped, so a crash /
GPU drop / reboot can just re-run the same command.

Usage:
    run_benchmarks.py <category|all|regular> [--bench-version 1.0]
                      [--benchmarks-dir DIR] [--results-dir DIR] [--log-dir DIR]

`all` runs every category on disk that has a runnable instances.csv for the
chosen version; `regular` runs the 2026 regular-track list below.

Debug logs: pass `--log-dir DIR` to capture each instance's prepare+run output
(with vibecheck `--verbose` enabled) to `DIR/<cat>/<net>__<prop>.{prepare,run}.log`
instead of discarding it - handy for diagnosing a verdict. Add `--heartbeat N`
to also emit per-phase heartbeat lines (stall detection).

Env:
    VNNCOMP_BENCHMARKS  default benchmarks dir (…/vnncomp2026_benchmarks)
    VIBECHECK_RESULTS_DIR  default results dir (…/vnncomp2026_results)
    VNNCOMP_PYTHON_PATH, VIBECHECK_PKL_CACHE_DIR  passed through to the scripts
"""
import argparse
import csv
import gzip
import os
import re
import subprocess
import sys
import time

# 2026 regular track (scored). `all` instead discovers categories from disk.
REGULAR_TRACK = [
    'acasxu_2023', 'cersyve', 'cgan2026', 'challenging_certified_training_2026',
    'cifar100_2024', 'collins_rul_cnn_2022', 'cora_2024', 'dist_shift_2023',
    'linearizenn_2024', 'lsnc_relu', 'malbeware', 'metaroom_2023',
    'ml4acopf_2024', 'nn4sys', 'relusplitter_2026', 'safenlp_2024', 'sat_relu',
    'soundnessbench_2026', 'tinyimagenet_2024', 'tllverifybench_2023',
    'traffic_signs_recognition_2023', 'vggnet16_2022', 'vit_2023', 'yolo_2023',
]

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _strip_ext(name):
    for ext in ('.gz', '.onnx', '.vnnlib'):
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name


def _ce_stem(cat, onnx_rel, vnnlib_rel):
    """CE filename stem, matching the scoring harness EXACTLY (process_results.py).

    Default is `<net>_<prop>`. safenlp is special-cased there: its onnx/vnnlib
    live in `medical/` and `ruarobot/` subdirs whose basenames collide across the
    two, so the harness prefixes the CE with the subdir
    (`medical_<net>_<prop>` / `ruarobot_<net>_<prop>`). We MUST match, or every
    safenlp `sat` counterexample reads as NO_CE -> a -150 penalty per instance.
    """
    net = _strip_ext(os.path.basename(onnx_rel))
    prop = _strip_ext(os.path.basename(vnnlib_rel))
    if cat == 'safenlp_2024':
        if 'medical' in onnx_rel:
            return f'medical_{net}_{prop}'
        # harness asserts the only other safenlp subdir is ruarobot
        return f'ruarobot_{net}_{prop}'
    return f'{net}_{prop}'


def _drop_instance_pkl(onnx_abs, vnnlib_abs):
    """Delete this instance's pre-parse .pkl cache (disk hygiene). Best-effort:
    import lazily and ignore any failure - a stale cache only wastes space."""
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


def _repo_name_and_year(benchmarks_dir):
    """Derive the results-repo path prefix + `<year>_<cat>` tag from the
    benchmarks-dir name, so a 2025 or 2026 clone yields matching paths."""
    repo = os.path.basename(os.path.normpath(benchmarks_dir)) or 'vnncomp_benchmarks'
    m = re.search(r'(20\d\d)', repo)
    return repo, (m.group(1) if m else '2026')


def _resolve_instances(bench_root, cat, bench_version):
    """Return (instances_csv, instance_dir, version_subpath) for this category.

    Prefers the versioned 2026 layout `<cat>/<bench_version>/instances.csv`;
    falls back to the flat 2025 layout `<cat>/instances.csv`. version_subpath is
    the path segment to splice into the results-repo path (e.g. '1.0/' or '')."""
    cat_dir = os.path.join(bench_root, cat)
    versioned = os.path.join(cat_dir, bench_version, 'instances.csv')
    if os.path.isfile(versioned):
        return versioned, os.path.join(cat_dir, bench_version), f'{bench_version}/'
    flat = os.path.join(cat_dir, 'instances.csv')
    if os.path.isfile(flat):
        return flat, cat_dir, ''
    return None, None, None


def _discover_categories(bench_root, bench_version):
    """Every category dir that has a runnable instances.csv for this version."""
    cats = []
    for name in sorted(os.listdir(bench_root)):
        if not os.path.isdir(os.path.join(bench_root, name)):
            continue
        ic, _, _ = _resolve_instances(bench_root, name, bench_version)
        if ic:
            cats.append(name)
    return cats


def run_category(cat, benchmarks_dir, results_dir, version='v1',
                 bench_version='1.0', log_dir=None, heartbeat=None):
    bench_root = os.path.join(benchmarks_dir, 'benchmarks')
    instances, instance_dir, ver_sub = _resolve_instances(bench_root, cat, bench_version)
    if not instances:
        print(f'[skip] {cat}: no instances.csv for version {bench_version} '
              f'(2.0-only category under v1?) under {os.path.join(bench_root, cat)}')
        return

    repo_name, year = _repo_name_and_year(benchmarks_dir)
    out_dir = os.path.join(results_dir, 'vibecheck', f'{year}_{cat}')
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'results.csv')
    done = _done_keys(results_csv)

    cat_log_dir = None
    if log_dir:
        cat_log_dir = os.path.join(log_dir, cat)
        os.makedirs(cat_log_dir, exist_ok=True)

    with open(instances, newline='') as f:
        rows = [r for r in csv.reader(f) if r and len(r) >= 3]
    print(f'=== {cat} [{ver_sub or "flat"}]: {len(rows)} instances '
          f'({len(done)} already done) → {results_csv} ===', flush=True)

    prepare_sh = os.path.join(SCRIPT_DIR, 'prepare_instance.sh')
    run_sh = os.path.join(SCRIPT_DIR, 'run_instance.sh')

    # run/prepare scripts are verbose by default; just pass heartbeat through
    # when requested (only meaningful when we're capturing the logs).
    child_env = dict(os.environ)
    if cat_log_dir and heartbeat:
        child_env['VIBECHECK_HEARTBEAT'] = str(heartbeat)

    def _sink(stem, phase):
        """stdout/stderr target for a child: a log file, or DEVNULL."""
        if not cat_log_dir:
            return subprocess.DEVNULL, None
        path = os.path.join(cat_log_dir, f'{stem}.{phase}.log')
        return open(path, 'wb'), path

    for i, (onnx_rel, vnnlib_rel, timeout) in enumerate(rows):
        # Path style the other tools' results.csv use (results-repo relative),
        # now including the version segment for the 2026 layout.
        onnx_repo = f'{repo_name}/benchmarks/{cat}/{ver_sub}{onnx_rel}'
        vnnlib_repo = f'{repo_name}/benchmarks/{cat}/{ver_sub}{vnnlib_rel}'
        if (onnx_repo, vnnlib_repo) in done:
            continue
        onnx_abs = os.path.join(instance_dir, onnx_rel)
        vnnlib_abs = os.path.join(instance_dir, vnnlib_rel)
        stem = f'{_strip_ext(os.path.basename(onnx_rel))}__{_strip_ext(os.path.basename(vnnlib_rel))}'

        # prepare (un-timed by the competition, but we record its wall).
        out, _ = _sink(stem, 'prepare')
        t0 = time.perf_counter()
        subprocess.run([prepare_sh, version, cat, onnx_abs, vnnlib_abs],
                       stdout=out, stderr=subprocess.STDOUT, env=child_env)
        prepare_s = time.perf_counter() - t0
        if cat_log_dir:
            out.close()

        # run (the verdict file is authoritative).
        res_file = f'/tmp/vibecheck_res_{cat}_{i}.txt'
        if os.path.exists(res_file):
            os.remove(res_file)
        out, run_log = _sink(stem, 'run')
        t0 = time.perf_counter()
        subprocess.run([run_sh, version, cat, onnx_abs, vnnlib_abs,
                        res_file, str(timeout)],
                       stdout=out, stderr=subprocess.STDOUT, env=child_env)
        runtime_s = time.perf_counter() - t0
        if cat_log_dir:
            out.close()

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
                                   f'{_ce_stem(cat, onnx_rel, vnnlib_rel)}.counterexample.gz')
            with gzip.open(ce_path, 'wb') as gz:
                gz.write(('\n'.join(ce_lines) + '\n').encode('utf-8'))

        with open(results_csv, 'a', newline='') as out_csv:
            csv.writer(out_csv).writerow(
                [cat, onnx_repo, vnnlib_repo,
                 f'{prepare_s:.6f}', verdict, f'{runtime_s:.6f}'])

        # Disk hygiene (the run host can be tight on space over thousands of
        # instances): the per-instance pre-parse .pkl is only needed for THIS
        # instance's run, so drop it now. Best-effort; a leftover just wastes
        # space, never correctness.
        _drop_instance_pkl(onnx_abs, vnnlib_abs)

        log_hint = f'  [log: {run_log}]' if run_log else ''
        print(f'  [{i+1}/{len(rows)}] {os.path.basename(onnx_rel)} / '
              f'{os.path.basename(vnnlib_rel)} → {verdict} '
              f'({runtime_s:.1f}s, prep {prepare_s:.1f}s){log_hint}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('category',
                    help='a benchmark category, "all" (every runnable category '
                         'on disk), or "regular" (the 2026 regular-track list)')
    ap.add_argument('--benchmarks-dir',
                    default=os.environ.get('VNNCOMP_BENCHMARKS',
                                           os.path.expanduser('~/repositories/vnncomp2026_benchmarks')))
    ap.add_argument('--results-dir',
                    default=os.environ.get('VIBECHECK_RESULTS_DIR',
                                           os.path.expanduser('~/repositories/vnncomp2026_results')))
    ap.add_argument('--bench-version', default='1.0',
                    help='spec-version dir to run (2026 layout): 1.0 (v1) or 2.0 '
                         '(v2; not yet parseable). Default 1.0.')
    ap.add_argument('--version', default='v1',
                    help='version string passed to the install/prepare/run '
                         'scripts (the VNNCOMP script ABI, unrelated to '
                         '--bench-version). Default v1.')
    ap.add_argument('--log-dir', default=None,
                    help='capture per-instance prepare+run logs here with '
                         'vibecheck --verbose enabled (for debugging)')
    ap.add_argument('--heartbeat', type=float, default=None,
                    help='with --log-dir, also pass --heartbeat N to vibecheck '
                         '(per-phase stall detection)')
    args = ap.parse_args()

    bench_root = os.path.join(args.benchmarks_dir, 'benchmarks')
    if not os.path.isdir(bench_root):
        sys.exit(f'no benchmarks dir at {bench_root} '
                 f'(set --benchmarks-dir or $VNNCOMP_BENCHMARKS)')

    if args.category == 'all':
        cats = _discover_categories(bench_root, args.bench_version)
        print(f'[all] {len(cats)} categories with v={args.bench_version}: '
              f'{", ".join(cats)}', flush=True)
    elif args.category == 'regular':
        cats = REGULAR_TRACK
    else:
        cats = [args.category]

    for cat in cats:
        run_category(cat, args.benchmarks_dir, args.results_dir, args.version,
                     args.bench_version, args.log_dir, args.heartbeat)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
