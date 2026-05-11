"""Soundness probe: every AB-CROWN-confirmed SAT case, with PGD disabled.

With `disable_sat_finding=True` the only legitimate verdicts are
'unknown' or 'sat-unreachable'. If vibecheck returns 'verified' on a
case AB-CROWN says is SAT, that's a soundness bug — the LP relaxation
incorrectly claims UNSAT for an instance whose negation is satisfiable.

This is a thin wrapper around `sweep_relusplitter.py --set soundness`
so the soundness probe has its own one-line invocation and a stable
exit code (0 = clean, 3 = soundness break).

Usage:
    .venv/bin/python tests/soundness_spotcheck.py [--remote stan@HOST]
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--remote', default='')
    p.add_argument('--out', default='/tmp/vibecheck_runs/soundness.json')
    p.add_argument('--memory-max', default='14G')
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from tests.sweep_relusplitter import main as sweep_main

    argv = ['--set', 'soundness',
            '--out', args.out,
            '--memory-max', args.memory_max]
    if args.remote:
        argv += ['--remote', args.remote]

    saved = sys.argv
    sys.argv = ['sweep_relusplitter.py'] + argv
    try:
        return sweep_main()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    sys.exit(main())
