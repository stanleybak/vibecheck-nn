#!/usr/bin/env python3
"""Summarize vibecheck VNNCOMP run logs - the stdout the competition captures
per instance (and what `run_benchmarks.py --log-dir` saves locally).

The install/prepare/run scripts emit stable banner anchors:

    [vibecheck:run_instance] BEGIN category=<cat> timeout=<t>s
        onnx=...   vnnlib=...   config=...
    [vibecheck:run_instance] END verdict=<v> elapsed=<s>s rc=<rc> category=<cat>

plus vibecheck's own `Result:`, `Time:`, the `N ops, N ReLU layers` net line,
`N constraint(s), N disjunct(s)`, and `[heartbeat] phase=...` lines. This reads
one log, a directory tree (the --log-dir layout), or stdin, and prints status:
verdict, timing, the net/spec shape, and - crucially for a run that timed out -
whether it CLOSED (saw its END banner) or was KILLED, and the last heartbeat
phase it was stuck in.

Usage:
    parse_log.py LOG [--full]       # one captured stdout log
    parse_log.py DIR [--full]       # a run_benchmarks.py --log-dir tree
    cat run.log | parse_log.py      # from stdin
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter

BEGIN_RE = re.compile(r'\[vibecheck:(\w+)\]\s+BEGIN\s+(.*)')
END_RE = re.compile(r'\[vibecheck:(\w+)\]\s+END\s+(.*)')
KV_RE = re.compile(r'(\w+)=(\S+)')
FIELD_RE = re.compile(r'^\s+(onnx|vnnlib|config|tool_dir)=(.+)$')
RESULT_RE = re.compile(r'^\s*Result:\s+(\w+)')
TIME_RE = re.compile(r'^\s*Time:\s+([\d.]+)s')
NET_RE = re.compile(r'^\s*(\d+)\s+ops,\s+(\d+)\s+ReLU layers.*input shape:\s*(.+)')
SPEC_RE = re.compile(r'^\s*(\d+)\s+constraint\(s\),\s+(\d+)\s+disjunct')
HB_RE = re.compile(r'\[heartbeat\]\s+phase=(\S+)\s+in-phase=(\S+)\s+total=(\S+)\s+gpu=(\S+)')
ERR_RE = re.compile(r'Traceback \(most recent call last\)|^[\w.]*(?:Error|Exception):')


def parse_blocks(text):
    """Split a log into [vibecheck:<script>] BEGIN..END blocks. A BEGIN with no
    matching END (next BEGIN or EOF first) is left `closed=False` → the run was
    killed mid-flight (timeout SIGKILL / crash), which is itself the diagnosis."""
    blocks, cur = [], None
    for ln in text.splitlines():
        mb = BEGIN_RE.search(ln)
        if mb:
            if cur:
                blocks.append(cur)
            cur = {'script': mb.group(1), 'kv': dict(KV_RE.findall(mb.group(2))),
                   'closed': False, 'verdict': None, 'elapsed': None, 'rc': None,
                   'status': None, 'solve': None, 'net': None, 'spec': None,
                   'last_phase': None, 'last_inphase': None, 'errors': []}
            continue
        if cur is None:
            continue
        me = END_RE.search(ln)
        if me and me.group(1) == cur['script']:
            kv = dict(KV_RE.findall(me.group(2)))
            # END verdict is from the authoritative results file
            # (unsat/sat/unknown/timeout) and WINS over the stdout `Result:`
            # line, which uses verified/unknown wording.
            cur.update(closed=True, verdict=kv.get('verdict') or cur['verdict'],
                       elapsed=(kv.get('elapsed') or '').rstrip('s') or None,
                       rc=kv.get('rc'), status=kv.get('status'))
            blocks.append(cur)
            cur = None
            continue
        mf = FIELD_RE.match(ln)
        if mf:
            cur['kv'][mf.group(1)] = mf.group(2).strip()
        if (m := RESULT_RE.match(ln)) and not cur['verdict']:
            cur['verdict'] = m.group(1)
        if m := TIME_RE.match(ln):
            cur['solve'] = m.group(1)
        if m := NET_RE.match(ln):
            cur['net'] = (m.group(1), m.group(2), m.group(3).strip())
        if m := SPEC_RE.match(ln):
            cur['spec'] = (m.group(1), m.group(2))
        if m := HB_RE.search(ln):
            cur['last_phase'], cur['last_inphase'] = m.group(1), m.group(2)
        if ERR_RE.match(ln.strip()):
            cur['errors'].append(ln.strip()[:200])
    if cur:
        blocks.append(cur)
    return blocks


def _stem(block, key):
    p = block['kv'].get(key)
    if not p:
        return None
    b = os.path.basename(p)
    for ext in ('.gz', '.onnx', '.vnnlib'):
        if b.endswith(ext):
            b = b[:-len(ext)]
    return b


def status_of(run):
    """One-word status from a run_instance block."""
    if not run['closed']:
        return 'KILLED'           # no END banner → SIGKILL/crash mid-run
    v = (run['verdict'] or 'unknown').lower()
    if v == 'error':
        return 'ERROR'
    if v in ('sat', 'unsat', 'unknown', 'timeout'):
        return v
    return v or 'unknown'


def fmt_instance(run, prep=None, full=False):
    net = _stem(run, 'onnx') or '?'
    prop = _stem(run, 'vnnlib') or '?'
    cat = run['kv'].get('category', '?')
    st = status_of(run)
    line = f'{st:<8} {cat}/{net} :: {prop}'
    bits = []
    if run['elapsed']:
        bits.append(f"elapsed={run['elapsed']}s")
    if run['solve']:
        bits.append(f"solve={run['solve']}s")
    if prep and prep.get('elapsed'):
        bits.append(f"prep={prep['elapsed']}s")
    if run['kv'].get('rc') is not None and run['rc']:
        bits.append(f"rc={run['rc']}")
    if bits:
        line += '   (' + ', '.join(bits) + ')'
    out = [line]
    if st == 'KILLED' and run['last_phase']:
        out.append(f'         ^ hung in phase={run["last_phase"]} '
                   f'in-phase={run["last_inphase"]} (last heartbeat before kill)')
    elif st == 'KILLED':
        out.append('         ^ no END banner and no heartbeat - died before/at load '
                   '(enable VIBECHECK_HEARTBEAT=N to localize a hang)')
    if run['errors']:
        out.append(f'         ! {run["errors"][-1]}')
    if full:
        if run['net']:
            out.append(f'         net: {run["net"][0]} ops, {run["net"][1]} ReLU, '
                       f'input {run["net"][2]}')
        if run['spec']:
            out.append(f'         spec: {run["spec"][0]} constraints, '
                       f'{run["spec"][1]} disjunct(s)')
        if run['kv'].get('config'):
            out.append(f'         config: {run["kv"]["config"]}')
    return '\n'.join(out)


def collect_dir(path):
    """Group a --log-dir tree into (run_block, prepare_block) per instance."""
    runs, preps = {}, {}
    for fp in sorted(glob.glob(os.path.join(path, '**', '*.log'), recursive=True)):
        base = os.path.basename(fp)
        with open(fp, errors='replace') as f:
            blocks = parse_blocks(f.read())
        if base.endswith('.prepare.log'):
            key = (os.path.dirname(fp), base[:-len('.prepare.log')])
            for b in blocks:
                if b['script'] == 'prepare_instance':
                    preps[key] = b
        elif base.endswith('.run.log'):
            key = (os.path.dirname(fp), base[:-len('.run.log')])
            for b in blocks:
                if b['script'] == 'run_instance':
                    runs[key] = b
        else:  # generic combined log: take any run_instance blocks
            for i, b in enumerate(blocks):
                if b['script'] == 'run_instance':
                    runs[(fp, i)] = b
    return runs, preps


def summarize(runs, preps, full=False):
    keys = sorted(runs, key=lambda k: (status_of(runs[k]) != 'KILLED',
                                       runs[k]['kv'].get('category', ''),
                                       _stem(runs[k], 'onnx') or ''))
    counts = Counter(status_of(runs[k]) for k in keys)
    issues, slow = [], []
    for k in keys:
        r = runs[k]
        print(fmt_instance(r, preps.get(k), full=full))
        st = status_of(r)
        if st in ('KILLED', 'ERROR', 'unknown', 'timeout'):
            issues.append(k)
        try:
            slow.append((float(r['elapsed']), k))
        except (TypeError, ValueError):
            pass
    print('\n=== summary ===')
    print(f'instances: {len(keys)}')
    for st, n in counts.most_common():
        print(f'  {st:<8} {n}')
    if slow:
        slow.sort(reverse=True)
        print('slowest:')
        for s, k in slow[:5]:
            print(f'  {s:6.2f}s  {runs[k]["kv"].get("category","?")}/'
                  f'{_stem(runs[k], "onnx")}')
    if issues:
        print(f'needs-attention: {len(issues)} '
              f'(KILLED / ERROR / unknown / timeout - listed first above)')
    else:
        print('needs-attention: none')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?', help='log file, --log-dir tree, or omit for stdin')
    ap.add_argument('--full', action='store_true',
                    help='also print net/spec shape + config per instance')
    args = ap.parse_args()

    if args.path and os.path.isdir(args.path):
        runs, preps = collect_dir(args.path)
        if not runs:
            sys.exit(f'no [vibecheck:run_instance] blocks found under {args.path}')
        summarize(runs, preps, full=args.full)
        return

    text = open(args.path, errors='replace').read() if args.path else sys.stdin.read()
    blocks = parse_blocks(text)
    runs = [b for b in blocks if b['script'] == 'run_instance']
    preps = [b for b in blocks if b['script'] == 'prepare_instance']
    if not runs and not blocks:
        sys.exit('no [vibecheck:*] banner blocks found - was the log captured '
                 'with the run/prepare scripts (verbose, banners on)?')
    if not runs:  # only prepare/install blocks present
        for b in blocks:
            print(f"{b['script']}: status={b.get('status') or ('closed' if b['closed'] else 'KILLED')}"
                  f" elapsed={b.get('elapsed')}")
        return
    for i, r in enumerate(runs):
        print(fmt_instance(r, preps[i] if i < len(preps) else None, full=args.full or len(runs) == 1))


if __name__ == '__main__':
    main()
