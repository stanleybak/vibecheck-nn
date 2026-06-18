#!/usr/bin/env python3
"""Parse a full VNNCOMP run log into a per-instance breakdown under parsed_log/.

Input: a single log file (default: `log.txt` next to this script) -- e.g. the
website's select-all dump of a toolkit run. It contains the verbose per-instance
output (with [vibecheck:prepare_instance] / [vibecheck:run_instance] banners) AND
the per-category results.csv blocks the competition harness wrote.

The log is `set -x` xtrace, so every shell `echo` appears twice (a `+ echo '...'`
trace line and the real output line). We parse only the real OUTPUT lines
(anchored to the line start), ignoring the `+ ...` traces.

Output (all under parsed_log/, which is gitignored):
  results.csv           -- RECONSTRUCTED from the verbose run banners (indexed).
  results_official.csv  -- the results.csv rows embedded in the log (what the
                           harness recorded).
  compare.txt           -- sanity check: do the two agree per instance? Lists
                           agreements, disagreements, and instances present in
                           one but not the other (e.g. verbose truncated).
  <NNNN>_<cat>__<net>__<prop>/log.txt -- each instance's raw log slice, by index,
                           so any results.csv row is easy to open.

Usage:  parse_log.py [logfile]
"""
import csv
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# OUTPUT banner anchors (line start; the `+ echo '...'` xtrace dupes start with
# '+' and are skipped).
PREP_BEGIN = re.compile(r'^\[vibecheck:prepare_instance\] BEGIN')
PREP_END = re.compile(r'^\[vibecheck:prepare_instance\] END\s+(.*)')
RUN_BEGIN = re.compile(r'^\[vibecheck:run_instance\] BEGIN\s+(.*)')
RUN_END = re.compile(r'^\[vibecheck:run_instance\] END\s+(.*)')
FIELD = re.compile(r'^\s+(onnx|vnnlib|config)=(.+?)\s*$')
KV = re.compile(r'(\w+)=(\S+)')
TIME = re.compile(r'^\s*Time:\s+([\d.]+)s')
NET = re.compile(r'^\s*(\d+)\s+ops,\s+(\d+)\s+ReLU layers.*input shape:\s*(.+)')
HB = re.compile(r'^\[heartbeat\]\s+phase=(\S+)\s+in-phase=(\S+)')
# Embedded official results.csv row: cat,onnx,vnnlib,prepare_s,verdict,runtime_s
# (onnx path may be `./benchmarks/...` or `<repo>/benchmarks/...`).
CSV_ROW = re.compile(
    r'^([A-Za-z0-9_]+),'
    r'((?:\./|[\w.-]+/)*benchmarks/\S+?\.onnx(?:\.gz)?),'
    r'(\S+?\.vnnlib(?:\.gz)?),'
    r'([\d.]+),(unsat|sat|unknown|timeout|error),([\d.]+)\s*$')


def _stem(p):
    b = os.path.basename(p or '')
    for ext in ('.gz', '.onnx', '.vnnlib'):
        if b.endswith(ext):
            b = b[:-len(ext)]
    return b


def _version(p):
    """Benchmark spec version dir (e.g. '1.0'/'2.0') from a path, or ''."""
    m = re.search(r'/(\d+\.\d+)/', p or '')
    return m.group(1) if m else ''


def parse_instance(slice_lines):
    """Pull verdict/timing/identity out of one instance's log slice."""
    d = {'category': '?', 'onnx': '', 'vnnlib': '', 'verdict': None,
         'run_elapsed': None, 'prep_elapsed': None, 'solve': None,
         'net': None, 'hung_phase': None, 'status': 'completed'}
    saw_run_begin = False
    for ln in slice_lines:
        m = FIELD.match(ln)
        if m:
            d[m.group(1)] = m.group(2)
            continue
        m = PREP_END.match(ln)
        if m:
            d['prep_elapsed'] = (dict(KV.findall(m.group(1))).get('elapsed')
                                 or '').rstrip('s') or None
            continue
        m = RUN_BEGIN.match(ln)
        if m:
            saw_run_begin = True
            d['category'] = dict(KV.findall(m.group(1))).get('category', d['category'])
            continue
        m = RUN_END.match(ln)
        if m:
            kv = dict(KV.findall(m.group(1)))
            d['verdict'] = kv.get('verdict')
            d['run_elapsed'] = (kv.get('elapsed') or '').rstrip('s') or None
            continue
        m = TIME.match(ln)
        if m:
            d['solve'] = m.group(1)
            continue
        m = NET.match(ln)
        if m:
            d['net'] = f'{m.group(1)} ops, {m.group(2)} ReLU, input {m.group(3).strip()}'
            continue
        m = HB.match(ln)
        if m:
            d['hung_phase'] = f'{m.group(1)} (in-phase {m.group(2)})'
    if saw_run_begin and d['verdict'] is None:
        d['status'] = 'KILLED'    # run started but never emitted an END banner
    elif not saw_run_begin:
        d['status'] = 'NO_RUN'    # prepare only (run never started)
    return d


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def main():
    logpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, 'log.txt')
    if not os.path.isfile(logpath):
        sys.exit(f'log file not found: {logpath} (drop the run log there, or pass a path)')
    lines = open(logpath, errors='replace').read().splitlines()
    outdir = os.path.join(SCRIPT_DIR, 'parsed_log')
    os.makedirs(outdir, exist_ok=True)

    # --- slice into instances: each starts at an OUTPUT prepare-BEGIN banner ---
    starts = [i for i, l in enumerate(lines) if PREP_BEGIN.match(l)]
    bounds = starts + [len(lines)]
    instances = []
    for idx, s in enumerate(starts):
        sl = lines[s:bounds[idx + 1]]
        d = parse_instance(sl)
        d['index'] = idx
        d['slice'] = sl
        instances.append(d)

    # --- embedded official results.csv rows ---
    official = []
    for l in lines:
        m = CSV_ROW.match(l)
        if m:
            official.append(dict(category=m.group(1), onnx=m.group(2), vnnlib=m.group(3),
                                 prep_s=m.group(4), verdict=m.group(5), runtime_s=m.group(6)))

    # --- per-instance log slices (set d['_dir'] as we go) ---
    for d in instances:
        d['_dir'] = f"{d['index']:04d}_{_safe(d['category'])}__{_safe(_stem(d['onnx']))}__{_safe(_stem(d['vnnlib']))}"
        idir = os.path.join(outdir, d['_dir'])
        os.makedirs(idir, exist_ok=True)
        with open(os.path.join(idir, 'log.txt'), 'w') as f:
            f.write('\n'.join(d['slice']) + '\n')

    # --- write reconstructed results.csv (uses _dir) ---
    recon_csv = os.path.join(outdir, 'results.csv')
    with open(recon_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'category', 'version', 'net', 'prop', 'verdict',
                    'status', 'run_s', 'prep_s', 'solve_s', 'log_dir'])
        for d in instances:
            w.writerow([d['index'], d['category'], _version(d['onnx']),
                        _stem(d['onnx']), _stem(d['vnnlib']), d['verdict'] or '',
                        d['status'], d['run_elapsed'] or '', d['prep_elapsed'] or '',
                        d['solve'] or '', d['_dir']])

    # --- write official results.csv ---
    with open(os.path.join(outdir, 'results_official.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'onnx', 'vnnlib', 'prep_s', 'verdict', 'runtime_s'])
        for r in official:
            w.writerow([r['category'], r['onnx'], r['vnnlib'], r['prep_s'],
                        r['verdict'], r['runtime_s']])

    # --- sanity check: reconstructed (B) vs official (A) by (cat,version,net,prop) ---
    def key(cat, onnx, vnnlib):
        return (cat, _version(onnx), _stem(onnx), _stem(vnnlib))

    b_map, a_map = {}, {}
    for d in instances:
        b_map.setdefault(key(d['category'], d['onnx'], d['vnnlib']), []).append(d['verdict'])
    for r in official:
        a_map.setdefault(key(r['category'], r['onnx'], r['vnnlib']), []).append(r['verdict'])

    agree, disagree, killed, only_a, only_b = [], [], [], [], []
    for k in sorted(set(a_map) | set(b_map)):
        av = a_map.get(k)
        bv = b_map.get(k)
        if av and bv:
            bv_real = [v for v in bv if v]   # drop None (killed/truncated verbose)
            if not bv_real:
                killed.append((k, av))       # official has a verdict, verbose was cut off
            elif set(av) == set(bv_real):
                agree.append((k, av, bv_real))
            else:
                disagree.append((k, av, bv_real))
        elif av:
            only_a.append((k, av))
        else:
            only_b.append((k, bv))

    # per-category coverage: how many official rows have a captured verbose log
    cats = sorted({r['category'] for r in official} | {d['category'] for d in instances})
    coverage = []
    for c in cats:
        v = sum(1 for d in instances if d['category'] == c)
        o = sum(1 for r in official if r['category'] == c)
        coverage.append((c, v, o))

    cmp_path = os.path.join(outdir, 'compare.txt')
    with open(cmp_path, 'w') as f:
        f.write('Sanity check: reconstructed (verbose) vs official (results.csv)\n')
        f.write(f'  instances with verbose logs : {len(instances)}\n')
        f.write(f'  official results.csv rows   : {len(official)}\n')
        f.write('  per-category coverage (verbose captured / official rows):\n')
        for c, v, o in coverage:
            flag = '' if v == o else '   <-- verbose truncated' if v < o else ''
            f.write(f'    {c:<22} {v:>4} / {o:<4}{flag}\n')
        f.write(f'  agree (verdict matches)     : {len(agree)}\n')
        f.write(f'  DISAGREE (real verdict conflict): {len(disagree)}\n')
        f.write(f'  killed/truncated verbose (official has verdict): {len(killed)}\n')
        f.write(f'  only in official (no verbose captured): {len(only_a)}\n')
        f.write(f'  only in verbose (no results row)     : {len(only_b)}\n\n')
        if killed:
            f.write('=== verbose killed/truncated (official verdict shown) ===\n')
            for k, av in killed:
                f.write(f'  {k} : official={av}\n')
            f.write('\n')
        if disagree:
            f.write('=== DISAGREEMENTS (cat, ver, net, prop : official vs verbose) ===\n')
            for k, av, bv in disagree:
                f.write(f'  {k} : official={av} verbose={bv}\n')
            f.write('\n')
        if only_a:
            f.write('=== in official only (verbose truncated/missing) ===\n')
            for k, av in only_a:
                f.write(f'  {k} : {av}\n')
            f.write('\n')
        if only_b:
            f.write('=== in verbose only (no official results row) ===\n')
            for k, bv in only_b:
                f.write(f'  {k} : {bv}\n')

    # --- console summary ---
    print(f'Parsed {logpath}')
    print(f'  instances (verbose): {len(instances)}  '
          f'[{sum(1 for d in instances if d["status"]=="completed")} ok, '
          f'{sum(1 for d in instances if d["status"]=="KILLED")} killed]')
    print(f'  official rows      : {len(official)}')
    for c, v, o in coverage:
        print(f'    {c:<22} verbose {v:>4} / {o:<4} official'
              + ('  <-- verbose truncated' if v < o else ''))
    print(f'  -> {recon_csv}')
    print(f'  -> {outdir}/results_official.csv')
    print(f'  -> {len(instances)} per-instance log dirs under {outdir}/')
    print(f'  sanity: agree={len(agree)} disagree={len(disagree)} '
          f'killed/truncated={len(killed)} only-official={len(only_a)} '
          f'only-verbose={len(only_b)}  (-> compare.txt)')
    if disagree:
        print('  WARNING: real verdict DISAGREEMENTS found -- see compare.txt')


if __name__ == '__main__':
    main()
