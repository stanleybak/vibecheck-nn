"""Categorize nnenum's LP solves into contraction / bounding / spec on one
acasxu case, single-process and in-process, so we can compare the per-category
LP count against v3's (CONTRACT_SOLVES / bounding / spec).

CONTRACT = LpStar.update_input_box_bounds (the contract_lp on a split + root box)
BOUND    = LpStar.minimize_output         (update_bounds_lp per-neuron tightening)
SPEC     = LpStar.minimize_vec            (violation / spec LP at a leaf)

Run:  .venv/bin/python trace_lp_categories.py <net> <prop> [both_bounds] [nocap]
"""
import sys
import numpy as np

from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from nnenum import lp_star

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'

CONTRACT = [0]
BOUND = [0]
SPEC = [0]
CONTRACT_CALLS = []   # LPs per contract_lp call (per child contraction at a split)
BOUND_CALLS = []      # LPs per update_bounds_lp_serial invocation (per child)
CONTRACT_DETAIL = []  # per contract_lp call: (depth, n_cut_lo, n_cut_hi, nlps)
SPLIT_TRACE = []      # per split: (cur_layer, neuron_index, score=min(hi,-lo))
MO8_TRACE = []        # every minimize_output for layer-0 neuron 8: (depth, side, value)
SPLIT8_DETAIL = []    # the exact node where L0 neuron 8 is split
FORCE_BRANCH_BOTH = [False]  # emulate a BRANCH_BOTH_BOUNDS setting


def make_spec(vnnlib_filename, onnx_filename):
    num_inputs, num_outputs, _ = get_num_inputs_outputs(onnx_filename)
    vnnlib_spec = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)
    rv = []
    for box, spec_list in vnnlib_spec:
        if len(spec_list) == 1:
            mat, rhs = spec_list[0]
            spec = Specification(mat, rhs)
        else:
            spec = [Specification(mat, rhs) for mat, rhs in spec_list]
        rv.append((box, spec))
    return rv


def patch():
    LpStar = lp_star.LpStar
    _ubb = LpStar.update_input_box_bounds
    _mo = LpStar.minimize_output
    _mv = LpStar.minimize_vec

    def ubb(self, hv, rhs, count_lps=True):
        before = self.num_lps
        # replicate nnenum's witness cut-count (lp_star.py:260-269) for the trace
        n_cut_lo = n_cut_hi = -1
        depth = -1
        if not isinstance(hv, list) and self.input_bounds_witnesses is not None:
            n_cut_lo = sum(1 for d in range(len(self.input_bounds_witnesses))
                           if float(np.dot(hv, self.input_bounds_witnesses[d][0])) > rhs)
            n_cut_hi = sum(1 for d in range(len(self.input_bounds_witnesses))
                           if float(np.dot(hv, self.input_bounds_witnesses[d][1])) > rhs)
            depth = self.lpi.get_num_rows()
        rv = _ubb(self, hv, rhs, count_lps=count_lps)
        d = self.num_lps - before
        CONTRACT[0] += d
        CONTRACT_CALLS.append(d)
        CONTRACT_DETAIL.append((depth, n_cut_lo, n_cut_hi, d))
        return rv

    def mo(self, output_index, maximize=False):
        did = self.a_mat.size != 0
        rv = _mo(self, output_index, maximize=maximize)
        if did:
            BOUND[0] += 1
            if output_index == 8 and self.a_mat.shape[0] == 50:  # layer-0 neuron 8
                MO8_TRACE.append((self.lpi.get_num_rows(),
                                  'max' if maximize else 'min', round(float(rv), 4)))
        return rv

    def mv(self, vec, return_io=False, fail_on_unsat=True):
        before = self.num_lps
        rv = _mv(self, vec, return_io=return_io, fail_on_unsat=fail_on_unsat)
        SPEC[0] += self.num_lps - before
        return rv

    LpStar.update_input_box_bounds = ubb
    LpStar.minimize_output = mo
    LpStar.minimize_vec = mv

    # trace which (layer, neuron) nnenum splits, + the score it sorted by
    from nnenum.lp_star_state import LpStarState
    _split = LpStarState.do_first_relu_split

    def split_trace(self, network, spec, start_time):
        ob = self.prefilter.output_bounds
        idx = int(ob.branching_neurons[0])
        lb, ub = ob.layer_bounds[idx]
        SPLIT_TRACE.append((self.cur_layer, idx, round(float(min(ub, -lb)), 4),
                            round(float(lb), 4), round(float(ub), 4)))
        if idx == 8 and self.cur_layer == 4 and not SPLIT8_DETAIL:
            zb = self.prefilter.zono.box_bounds()[8]
            sim8 = self.prefilter.simulation[1][8] if self.prefilter.simulation else None
            SPLIT8_DETAIL.append({
                'layer_bounds[8](used for split)': (round(float(lb), 4), round(float(ub), 4)),
                'live zono box_bounds[8]': (round(float(zb[0]), 4), round(float(zb[1]), 4)),
                'sim[8]': round(float(sim8), 4) if sim8 is not None else None,
            })
        return _split(self, network, spec, start_time)

    LpStarState.do_first_relu_split = split_trace

    from nnenum import lputil
    _ubl = lputil.update_bounds_lp_serial

    def ubl(layer_bounds, star, sim, split_indices, check_cancel_func=None,
            both_bounds=False):
        before = BOUND[0]
        if FORCE_BRANCH_BOTH[0]:
            both_bounds = True   # emulate a Settings.BRANCH_BOTH_BOUNDS=True knob
        rv = _ubl(layer_bounds, star, sim, split_indices,
                  check_cancel_func=check_cancel_func, both_bounds=both_bounds)
        BOUND_CALLS.append(BOUND[0] - before)
        return rv

    lputil.update_bounds_lp_serial = ubl
    # prefilter.py imports the name directly -> patch there too
    from nnenum import prefilter
    if hasattr(prefilter, 'update_bounds_lp_serial'):
        prefilter.update_bounds_lp_serial = ubl


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else '1_1'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_3'
    both_bounds = 'both_bounds' in sys.argv
    nocap = 'nocap' in sys.argv
    FORCE_BRANCH_BOTH[0] = 'branch_both' in sys.argv

    onnx_path = f'{BENCH}/onnx/ACASXU_run2a_{net}_batch_2000.onnx'
    spec_path = f'{BENCH}/vnnlib/{prop}.vnnlib'

    spec_list = make_spec(spec_path, onnx_path)
    network = load_onnx_network_optimized(onnx_path)

    # ----- settings: single-process, pure BaB (no try-quick), match v3 -----
    Settings.NUM_PROCESSES = 1
    Settings.TIMEOUT = 300.0
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = False
    Settings.PRINT_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = False   # isolate the BaB loop (v3 has none)
    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = False
    if both_bounds:
        Settings.OVERAPPROX_BOTH_BOUNDS = True
    if nocap:
        Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = np.inf
        Settings.OVERAPPROX_MIN_GEN_LIMIT = 10 ** 9

    patch()

    import time
    t0 = time.perf_counter()
    init_box, spec = spec_list[0]
    init_box = np.array(init_box, dtype=np.float32)
    res = enumerate_network(init_box, network, spec)
    dt = time.perf_counter() - t0

    total = res.total_lps
    cat = CONTRACT[0] + BOUND[0] + SPEC[0]
    stars = getattr(res, 'total_stars', None)
    print(f"nnenum {net} {prop} (both_bounds={both_bounds} nocap={nocap}): "
          f"{res.result_str} {dt:.2f}s stars={stars}")
    print(f"  total LP   = {total}")
    print(f"  contraction= {CONTRACT[0]}")
    print(f"  bounding   = {BOUND[0]}")
    print(f"  spec       = {SPEC[0]}")
    print(f"  (sum cats  = {cat}, unattributed = {total - cat})")
    cc = np.array(CONTRACT_CALLS) if CONTRACT_CALLS else np.array([0])
    bc = np.array(BOUND_CALLS) if BOUND_CALLS else np.array([0])
    print(f"  contract_lp calls = {len(CONTRACT_CALLS)}  LP/call: "
          f"min={cc.min()} mean={cc.mean():.2f} max={cc.max()}  "
          f"hist0..10={np.bincount(cc, minlength=11)[:11].tolist()}")
    print(f"  bound_lp invocations = {len(BOUND_CALLS)}  LP/call: "
          f"min={bc.min()} mean={bc.mean():.2f} max={bc.max()}")
    print(f"  first 6 contract_lp LP counts: {CONTRACT_CALLS[:6]}")
    print(f"  first 6 bound invocations:    {BOUND_CALLS[:6]}")
    print(f"  first 10 contract detail (depth, cut_lo, cut_hi, lps):")
    for row in CONTRACT_DETAIL[:10]:
        print(f"      {row}")
    # contraction LP/call grouped by depth (first constraints)
    from collections import defaultdict
    byd = defaultdict(list)
    for depth, cl, ch, lp in CONTRACT_DETAIL:
        byd[depth].append(lp)
    print(f"  contract LP/call by depth: "
          + ", ".join(f"d{d}:n{len(byd[d])} mean{np.mean(byd[d]):.1f}"
                      for d in sorted(byd)[:8]))
    print(f"  first 12 SPLITS (cur_layer, neuron, score): {SPLIT_TRACE[:12]}")
    if SPLIT8_DETAIL:
        print(f"  neuron 8 SPLIT NODE detail: {SPLIT8_DETAIL[0]}")
    print(f"  minimize_output calls for L0 neuron 8 (depth, side, value):")
    for row in MO8_TRACE[:20]:
        print(f"      {row}")


if __name__ == '__main__':
    main()
