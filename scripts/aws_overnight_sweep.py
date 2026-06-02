#!/usr/bin/env python
"""Autonomous overnight sweep on the AWS g5 box.

Per the operational rules: every verdict comes from the CLI `--results-file`,
NEVER exit code/stdout. Each case is its own `vibecheck.main` subprocess (fresh
CUDA context). Before every case we `rm -f /tmp/idle_since` so the AWS idle-
shutdown cron can't fire mid-sweep.

Three phases, run in this order (soundness first):
  A  SOUNDNESS  — every case AB-CROWN labelled `sat`, run with
                  --disable-sat-finding (no PGD / MILP witness). A SOUND
                  verifier must NEVER return `unsat` here; any that does is
                  flagged UNSOUND. Capped at 30s (a false-unsat surfaces in the
                  bounds/MILP pass, not the deep BaB tail).
  B  COMPLETE   — every case AB-CROWN labelled `unsat`, run normally at the
                  FULL competition timeout. Did we verify within budget? vs ABC.
  C  STRETCH    — cases AB-CROWN did NOT solve (timeout/unknown); can we?

Results stream to results.csv (resumable: completed (phase,onnx,vnnlib) skipped).
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
BENCH = HOME / 'vnncomp2025_benchmarks' / 'benchmarks'
ABCDIR = HOME / 'vnncomp2025_results' / 'alpha_beta_crown'
VIBE = HOME / 'vibe' / 'bin' / 'python'
VC = HOME / 'vibecheck'
OUT = HOME / 'persistent_runs' / 'aws_sweep'
LOGS = OUT / 'logs'
RESULTS = OUT / 'results.csv'
PROGRESS = OUT / 'progress.log'

# Fast benchmarks first so partial overnight coverage is maximised.
BENCHMARKS = [
    # Benchmarks that had misses in the previous sweep go FIRST, so their
    # re-check surfaces quickly (main() iterates benchmark-first, A->B->C each).
    'nn4sys', 'acasxu_2023', 'dist_shift_2023',
    # Then the rest (previous sweep: 0 misses).
    'cersyve', 'cgan_2023', 'collins_rul_cnn_2022', 'linearizenn_2024',
    'malbeware', 'metaroom_2023', 'safenlp_2024', 'cora_2024', 'cifar100_2024',
]  # tinyimagenet deliberately excluded (under debugging)

PHASE_A_CAP = 20.0          # soundness-probe timeout cap (s)
KILL_BUFFER = 90.0          # external kill = case_timeout + this
FILE_TO_VC = {'unsat': 'verified', 'sat': 'sat',
              'unknown': 'unknown', 'timeout': 'timeout'}


def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(PROGRESS, 'a') as f:
        f.write(line + '\n')


def poke():
    """Reset the AWS idle-shutdown counter."""
    os.system('sudo rm -f /tmp/idle_since 2>/dev/null')


def resolve(p):
    """Existing path, gunzipping a .gz-only file."""
    if p.exists():
        return p
    gz = Path(str(p) + '.gz')
    if gz.exists():
        subprocess.run(['gunzip', '-kf', str(gz)], check=False)
        if p.exists():
            return p
    return p


def load_instances(b):
    out = []
    fp = BENCH / b / 'instances.csv'
    if not fp.exists():
        return out
    for line in fp.read_text().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 3 and parts[0]:
            try:
                out.append((parts[0], parts[1], float(parts[2])))
            except ValueError:
                pass
    return out


def _relkey(path, kind):
    """Normalise an onnx/vnnlib path to its 'onnx/...' or 'vnnlib/...' relative
    form. Benchmarks with subdirs (safenlp/medical) have colliding basenames, so
    matching ABC results to instances MUST use the relative path, not basename."""
    path = path.lstrip('./')
    marker = f'/{kind}/'
    if marker in path:
        return kind + '/' + path.split(marker, 1)[1]
    if path.startswith(f'{kind}/'):
        return path
    return os.path.basename(path)


def load_abc(b):
    m = {}
    fp = ABCDIR / f'2025_{b}' / 'results.csv'
    if not fp.exists():
        return m
    for line in fp.read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 6:
            key = (_relkey(p[1], 'onnx'), _relkey(p[2], 'vnnlib'))
            m[key] = (p[4], p[5])
    return m


def already_done():
    done = set()
    if RESULTS.exists():
        with open(RESULTS) as f:
            for row in csv.reader(f):
                if len(row) >= 4 and row[0] != 'phase':
                    done.add((row[0], row[2], row[3]))
    return done


def run_case(phase, b, onnx_rel, vnn_rel, timeout, abc_res, abc_time):
    onnx = resolve(BENCH / b / onnx_rel)
    vnn = resolve(BENCH / b / vnn_rel)
    config = VC / 'configs' / f'{b}.yaml'
    tag = f'{b}__{os.path.basename(vnn_rel)}'.replace('/', '_')[:150]
    rfile = OUT / f'verdict_{phase}_{tag}.txt'
    logf = LOGS / f'{phase}_{tag}.log'
    if rfile.exists():
        rfile.unlink()
    cmd = [str(VIBE), '-m', 'vibecheck.main',
           '--net', str(onnx), '--spec', str(vnn),
           '--timeout', str(timeout), '--device', 'gpu', '--bits', '32',
           '--results-file', str(rfile)]
    if config.exists():
        cmd += ['--config', str(config)]
    if phase == 'A':
        cmd += ['--disable-sat-finding']
    poke()
    t0 = time.time()
    killed = False
    with open(logf, 'w') as lf:
        try:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=str(VC), timeout=timeout + KILL_BUFFER)
        except subprocess.TimeoutExpired:
            killed = True
    wall = time.time() - t0
    # Verdict from the FILE ONLY.
    raw = ''
    if rfile.exists():
        txt = rfile.read_text().strip()
        raw = txt.split()[0] if txt else ''
    vc = FILE_TO_VC.get(raw, 'error' if not raw else f'other({raw})')
    if killed and vc in ('error', 'other()'):
        vc = 'killed'
    # Interpretation
    flag = ''
    if phase == 'A':
        agree = (vc != 'verified')          # sound iff NOT verified on a sat case
        if vc == 'verified':
            flag = 'UNSOUND'
    elif phase == 'B':
        agree = (vc == 'verified')          # complete iff we also verified
    else:
        agree = (vc == 'verified')          # phase C: a bonus win
    with open(RESULTS, 'a', newline='') as f:
        csv.writer(f).writerow([phase, b, onnx_rel, vnn_rel, f'{timeout:g}',
                                abc_res, abc_time, vc, f'{wall:.1f}',
                                'Y' if agree else 'N', flag])
    if flag:
        log(f'*** {flag} *** {phase} {b} {os.path.basename(vnn_rel)} '
            f'-> vc={vc} (abc={abc_res})')
    return vc, wall, flag


def build_worklist():
    A, B, C = [], [], []
    for b in BENCHMARKS:
        abc = load_abc(b)
        for onnx_rel, vnn_rel, tmo in load_instances(b):
            key = (_relkey(onnx_rel, 'onnx'), _relkey(vnn_rel, 'vnnlib'))
            ar, at = abc.get(key, ('unknown', ''))
            if ar == 'sat':
                A.append((b, onnx_rel, vnn_rel, min(tmo, PHASE_A_CAP), ar, at))
            elif ar == 'unsat':
                B.append((b, onnx_rel, vnn_rel, tmo, ar, at))
            else:
                C.append((b, onnx_rel, vnn_rel, tmo, ar, at))
    return A, B, C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if not RESULTS.exists():
        with open(RESULTS, 'w', newline='') as f:
            csv.writer(f).writerow(['phase', 'benchmark', 'onnx', 'vnnlib',
                                    'timeout', 'abc_result', 'abc_time',
                                    'vc_verdict', 'vc_wall', 'agree', 'flag'])
    done = already_done()
    A, B, C = build_worklist()
    log(f'worklist: A(sat-soundness)={len(A)} B(unsat-complete)={len(B)} '
        f'C(abc-unsolved)={len(C)}  (already done: {len(done)})')
    # Benchmark-FIRST iteration: each benchmark runs all its phases (A soundness
    # -> B completeness -> C stretch) before the next benchmark. With BENCHMARKS
    # ordered problem-first, the benchmarks that missed last sweep (nn4sys,
    # acasxu, dist_shift) are fully re-checked before the clean ones — and
    # soundness still precedes completeness WITHIN each benchmark.
    by_bench = {}
    for phase, work in (('A', A), ('B', B), ('C', C)):
        for case in work:
            by_bench.setdefault(case[0], {'A': [], 'B': [], 'C': []})[phase].append(case)
    ordered = ([b for b in BENCHMARKS if b in by_bench]
               + [b for b in by_bench if b not in BENCHMARKS])
    n_unsound = 0
    for b in ordered:
        for phase in ('A', 'B', 'C'):
            work = by_bench[b][phase]
            if not work:
                continue
            log(f'=== {b} PHASE {phase}: {len(work)} cases ===')
            for i, (bb, onnx_rel, vnn_rel, tmo, ar, at) in enumerate(work):
                if (phase, onnx_rel, vnn_rel) in done:
                    continue
                vc, wall, flag = run_case(phase, bb, onnx_rel, vnn_rel, tmo, ar, at)
                if flag == 'UNSOUND':
                    n_unsound += 1
                if i % 20 == 0:
                    log(f'  [{b} {phase}] {i}/{len(work)} last={vc} '
                        f'({wall:.0f}s) unsound_so_far={n_unsound}')
    log(f'SWEEP COMPLETE. unsound_total={n_unsound}')


if __name__ == '__main__':
    main()
