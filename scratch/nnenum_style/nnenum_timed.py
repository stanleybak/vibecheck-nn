"""Clean nnenum timing: wall time + num LPs, configurable NUM_PROCESSES and mode.
No monkeypatching (so timing is clean). LP count from res.total_lps.
Usage: nnenum_timed.py NET PROP MODE NPROC TIMEOUT   (MODE = default | matched)"""
import sys, time
import numpy as np
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'

LP = [0]  # lean LP counter (single-process only; multi uses res.total_lps)


def patch_lean_counter():
    """Count LPs in-process (single-process only). Minimal overhead: just +=1."""
    from nnenum import lp_star
    S = lp_star.LpStar
    _mo, _mv, _ubb = S.minimize_output, S.minimize_vec, S.update_input_box_bounds

    def mo(self, oi, maximize=False):
        if self.a_mat.size:
            LP[0] += 1
        return _mo(self, oi, maximize=maximize)

    def mv(self, vec, return_io=False, fail_on_unsat=True):
        b = self.num_lps
        rv = _mv(self, vec, return_io=return_io, fail_on_unsat=fail_on_unsat)
        LP[0] += self.num_lps - b
        return rv

    def ubb(self, hv, rhs, count_lps=True):
        b = self.num_lps
        rv = _ubb(self, hv, rhs, count_lps=count_lps)
        LP[0] += self.num_lps - b
        return rv

    S.minimize_output, S.minimize_vec, S.update_input_box_bounds = mo, mv, ubb


def make_spec(vnnlib_filename, onnx_filename):
    ni, no, _ = get_num_inputs_outputs(onnx_filename)
    vspec = read_vnnlib_simple(vnnlib_filename, ni, no)
    rv = []
    for box, spec_list in vspec:
        if len(spec_list) == 1:
            spec = Specification(*spec_list[0])
        else:
            spec = [Specification(m, r) for m, r in spec_list]
        rv.append((box, spec))
    return rv


def set_matched(nproc, TO):
    Settings.TIMEOUT = TO
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = False
    Settings.PRINT_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = False
    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = (nproc > 1)
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = np.inf
    Settings.OVERAPPROX_MIN_GEN_LIMIT = 10 ** 9


def set_default(nproc, TO):
    # nnenum's competition config for acasxu (num_inputs<700 -> control)
    Settings.TIMING_STATS = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = (nproc > 1)
    Settings.PRINT_OVERAPPROX_OUTPUT = False
    Settings.PRINT_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = True
    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True
    Settings.OVERAPPROX_BOTH_BOUNDS = False
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = 1.5
    Settings.OVERAPPROX_LP_TIMEOUT = 0.02
    Settings.OVERAPPROX_MIN_GEN_LIMIT = 70
    Settings.TIMEOUT = TO


def main():
    net, prop, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]); TO = float(sys.argv[5])
    onnx = f'{BENCH}/onnx/ACASXU_run2a_{net}_batch_2000.onnx'
    vnnlib = f'{BENCH}/vnnlib/{prop}.vnnlib'
    spec_list = make_spec(vnnlib, onnx)
    network = load_onnx_network_optimized(onnx)

    Settings.NUM_PROCESSES = nproc
    (set_matched if mode == 'matched' else set_default)(nproc, TO)
    if nproc == 1:
        patch_lean_counter()

    box, spec = spec_list[0]
    box = np.array(box, dtype=np.float32)
    t0 = time.time()
    res = enumerate_network(box, network, spec)
    wall = time.time() - t0
    nlp = LP[0] if nproc == 1 else res.total_lps   # single: counter; multi: shared Value
    rs = res.result_str
    verdict = ('unsat' if rs in ('safe', 'holds') else
               'sat' if 'unsafe' in rs or 'violated' in rs else
               'timeout' if rs == 'timeout' else rs)
    results_file = sys.argv[6] if len(sys.argv) > 6 else None
    if results_file:
        with open(results_file, 'w') as f:
            f.write(verdict + '\n')
    line = (f"nnenum {net} {prop} mode={mode} nproc={nproc:2d}: {verdict:8s} "
            f"wall={wall:7.1f}s LP={nlp:8d} stars={getattr(res,'total_stars','?')}")
    with open(f'nnenum_timed_{net}_{prop}.out', 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)


if __name__ == '__main__':
    main()
