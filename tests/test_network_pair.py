"""Unit tests for network_pair.py — the GENERIC multi-network (iso/mono) merger.

Synthetic tiny acasxu-like nets (5-in, 5-out, Gemm+Relu+Gemm) + synthetic v2 pair
specs; checks parse_multinet's IR, that the merged ONNX computes the atom LHS exactly
(the onnxruntime oracle inside build_merged_instance raises on a bad merge), and the
io/detect edge cases.
"""
import os
import gzip
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

from vibecheck import network_pair as npair


def _tiny_acasxu(path, seed=0, scale=1.0):
    """A small [1,1,1,5] -> [1,5] ReLU net (Gemm-Relu-Gemm), like an acasxu net."""
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((5, 8)) * scale).astype(np.float32)
    b1 = rng.standard_normal(8).astype(np.float32)
    W2 = (rng.standard_normal((8, 5)) * scale).astype(np.float32)
    b2 = rng.standard_normal(5).astype(np.float32)
    nodes = [
        helper.make_node('Reshape', ['X', 'flat_shape'], ['xf']),
        helper.make_node('MatMul', ['xf', 'W1'], ['h1']),
        helper.make_node('Add', ['h1', 'b1'], ['h1b']),
        helper.make_node('Relu', ['h1b'], ['a1']),
        helper.make_node('MatMul', ['a1', 'W2'], ['h2']),
        helper.make_node('Add', ['h2', 'b2'], ['Y']),
    ]
    inits = [
        numpy_helper.from_array(np.array([-1, 5], np.int64), 'flat_shape'),  # batch-safe
        numpy_helper.from_array(W1, 'W1'), numpy_helper.from_array(b1, 'b1'),
        numpy_helper.from_array(W2, 'W2'), numpy_helper.from_array(b2, 'b2'),
    ]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 1, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
    g = helper.make_graph(nodes, 'tiny', [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.checker.check_model(m)
    onnx.save(m, path)


def _iso_spec(path, eps=0.05):
    L = ['(vnnlib-version <2.0>)',
         '(declare-network f (declare-input X_f float32 [1,1,1,5]) (declare-output Y_f float32 [1,5]))',
         '(declare-network g (isomorphic-to f))']
    for i in range(5):
        L.append(f'(assert (and (<= X_f[{i}] 1.0) (>= X_f[{i}] -1.0)))')
    for i in range(5):
        L.append(f'(assert (== X_f[{i}] X_g[{i}]))')
    # real iso output: OR_i [ Y_g[i] > Y_f[i]+eps  OR  Y_g[i] < Y_f[i]-eps ]
    L.append('(assert (or')
    for i in range(5):
        L.append(f' (or (> Y_g[{i}] (+ Y_f[{i}] {eps})) (< Y_g[{i}] (- Y_f[{i}] {eps})))')
    L.append('))')
    open(path, 'w').write('\n'.join(L) + '\n')


def _mono_spec(path, pins=False):
    L = ['(vnnlib-version <2.0>)',
         '(declare-network f (declare-input X_f float32 [1,1,1,5]) (declare-output Y_f float32 [1,5]))',
         '(declare-network g (equal-to f))']
    for i in range(5):
        L.append(f'(assert (and (<= X_f[{i}] 1.0) (>= X_f[{i}] -1.0)))')
    for i in range(1, 5):
        L.append(f'(assert (== X_f[{i}] X_g[{i}]))')
    if pins:                                           # const-pinned coords (real specs have these)
        L.append('(assert (== X_f[3] 0.5))')
        L.append('(assert (== X_f[4] -0.5))')
    L.append('(assert (>= X_f[0] X_g[0]))')            # relational coord 0
    L.append('(assert (>= X_g[0] -1.0))')
    L.append('(assert (< Y_f[3] Y_g[3]))')             # violation: Y_f[3] < Y_g[3]
    open(path, 'w').write('\n'.join(L) + '\n')


def test_detect_and_field():
    assert npair.is_network_pair_net_field("[('f', 'a.onnx'), ('g', 'b.onnx')]")
    assert not npair.is_network_pair_net_field("onnx/a.onnx")
    assert npair._onnx_paths_from_field("[('f', 'a.onnx'), ('g', 'b.onnx')]") == ['a.onnx', 'b.onnx']
    assert npair.detect_kind('(declare-network g (isomorphic-to f))') == 'iso'
    assert npair.detect_kind('(declare-network g (equal-to f))') == 'mono'
    assert npair.detect_kind('(assert (<= X_0 1))') is None


def test_parse_iso_ir(tmp_path):
    spec = str(tmp_path / 's.vnnlib'); _iso_spec(spec, eps=0.05)
    ir = npair.parse_multinet(open(spec).read())
    assert ir['n'] == 5 and ir['rel'] is None and ir['dnf'] == 'or'
    assert len(ir['atoms']) == 10                       # 2 per coord
    # each atom is Yg[i]-Yf[i] (>= eps) or (<= -eps)
    a = ir['atoms'][0]
    assert a['lhs'][('g', 0)] == 1.0 and a['lhs'][('f', 0)] == -1.0
    assert {at['rhs'] for at in ir['atoms']} == {0.05, -0.05}


def test_parse_mono_ir(tmp_path):
    spec = str(tmp_path / 's.vnnlib'); _mono_spec(spec)
    ir = npair.parse_multinet(open(spec).read())
    assert ir['n'] == 5 and ir['dnf'] == 'and' and len(ir['atoms']) == 1
    assert ir['rel']['k'] == 0 and ir['rel']['dmax'] == 2.0
    at = ir['atoms'][0]
    # strict `< 0` -> NON-STRICT closure `<= 0` (sound bound); strictness is
    # enforced downstream in SAT-detection (clear CE vs within-tol boundary).
    assert at['op'] == '<=' and at['rhs'] == 0.0
    assert at['lhs'][('f', 3)] == 1.0 and at['lhs'][('g', 3)] == -1.0


def test_iso_merge_oracle(tmp_path):
    f = str(tmp_path / 'f.onnx'); g = str(tmp_path / 'g.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=1); _tiny_acasxu(g, seed=2); _iso_spec(spec, eps=0.05)
    field = f"[('f', '{f}'), ('g', '{g}')]"
    mo, mv = npair.build_merged_instance(field, spec)   # raises if oracle fails
    assert os.path.isfile(mo) and os.path.isfile(mv)
    # merged outputs the 5 distinct atom LHS = Yg[i]-Yf[i]; input is [batch,5]
    sm = ort.InferenceSession(mo); sf = ort.InferenceSession(f); sg = ort.InferenceSession(g)
    x = np.random.default_rng(0).random((1, 5)).astype(np.float32) * 2 - 1
    ym = sm.run(None, {'X': x})[0].flatten()
    xr = x.reshape(1, 1, 1, 5)
    ref = sg.run(None, {'X': xr})[0].flatten() - sf.run(None, {'X': xr})[0].flatten()
    assert ym.shape == (5,) and np.abs(ym - ref).max() < 1e-4
    txt = open(mv).read()
    assert txt.count('(declare-const X_') == 5 and '(assert (or' in txt


def test_mono_merge_oracle(tmp_path):
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=3); _mono_spec(spec)
    field = f"[('f', '{f}'), ('g', '{f}')]"
    mo, mv = npair.build_merged_instance(field, spec)
    assert os.path.isfile(mo) and os.path.isfile(mv)
    # merged input is [base(5), delta] = 6; output is f(x_f)[3]-f(x_g)[3] (1 atom)
    sm = ort.InferenceSession(mo); sf = ort.InferenceSession(f)
    rng = np.random.default_rng(0)
    xg = (rng.random((1, 5)).astype(np.float32) * 2 - 1)
    delta = np.array([[0.7]], np.float32)
    Z = np.concatenate([xg, delta], axis=1).astype(np.float32)
    ym = sm.run(None, {'X': Z})[0].flatten()
    xf = xg.copy(); xf[0, 0] = np.clip(xg[0, 0] + delta[0, 0], -1.0, 1.0)
    yf = sf.run(None, {'X': xf.reshape(1, 1, 1, 5)})[0].flatten()
    yg = sf.run(None, {'X': xg.reshape(1, 1, 1, 5)})[0].flatten()
    assert ym.shape == (1,) and abs(ym[0] - (yf[3] - yg[3])) < 1e-4
    txt = open(mv).read()
    assert txt.count('(declare-const X_') == 6


def test_reconstruct_pair_cex_mono(tmp_path):
    # reconstruct_pair_cex maps a MERGED-net witness [base(5), delta] back to the
    # pair's declared per-network tensors: x_g=base, x_f=base with coord 0 clamped,
    # y_f/y_g from the ORIGINAL net. Returns (concat(x_f,x_g), concat(y_f,y_g)).
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=3); _mono_spec(spec)
    field = f"[('f', '{f}'), ('g', '{f}')]"
    npair.build_merged_instance(field, spec)
    ir = npair.parse_multinet(open(spec).read())
    rng = np.random.default_rng(2)
    base = (rng.random(5).astype(np.float32) * 2 - 1)
    delta = 0.7
    z = np.concatenate([base, [delta]]).astype(np.float64)
    x_flat, y_flat = npair.reconstruct_pair_cex(f, f, ir, z)
    # expected x_f / x_g
    x_f = base.copy().astype(np.float64); x_g = base.copy().astype(np.float64)
    x_f[0] = np.clip(base[0] + delta, ir['xf_box'][0][0], ir['xf_box'][0][1])
    sf = ort.InferenceSession(f)
    y_f = sf.run(None, {'X': x_f.astype(np.float32).reshape(1, 1, 1, 5)})[0].flatten()
    y_g = sf.run(None, {'X': x_g.astype(np.float32).reshape(1, 1, 1, 5)})[0].flatten()
    assert np.allclose(x_flat, np.concatenate([x_f, x_g]))
    assert np.allclose(y_flat, np.concatenate([y_f, y_g]), atol=1e-6)
    # input group is 10 (5+5), output group is 10 (5+5)
    assert x_flat.shape == (10,) and y_flat.shape == (10,)


def test_mono_constpins_build(tmp_path):
    # constant-pinned coords (== X_f[i] <const>) -> degenerate box; oracle still exact
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=5); _mono_spec(spec, pins=True)
    ir = npair.parse_multinet(open(spec).read())
    assert ir['base_box'][3] == (0.5, 0.5) and ir['base_box'][4] == (-0.5, -0.5)
    mo, mv = npair.build_merged_instance(f"[('f', '{f}'), ('g', '{f}')]", spec)
    assert os.path.isfile(mo) and os.path.isfile(mv)


def test_output_const_atom(tmp_path):
    # generic output form 3: `OP Y_a[i] const` (single output vs a constant). Not used
    # by iso/mono, but a valid general atom — build + oracle must handle it.
    f = str(tmp_path / 'f.onnx'); g = str(tmp_path / 'g.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=1); _tiny_acasxu(g, seed=2)
    L = ['(vnnlib-version <2.0>)',
         '(declare-network f (declare-input X_f float32 [1,1,1,5]) (declare-output Y_f float32 [1,5]))',
         '(declare-network g (isomorphic-to f))']
    for i in range(5):
        L.append(f'(assert (and (<= X_f[{i}] 1.0) (>= X_f[{i}] -1.0)))')
        L.append(f'(assert (== X_f[{i}] X_g[{i}]))')
    L.append('(assert (<= Y_f[0] 0.5))')               # form 3
    open(spec, 'w').write('\n'.join(L) + '\n')
    ir = npair.parse_multinet(open(spec).read())
    assert ir['dnf'] == 'and' and ir['atoms'][0]['lhs'] == {('f', 0): 1.0}
    assert ir['atoms'][0]['op'] == '<=' and ir['atoms'][0]['rhs'] == 0.5
    mo, mv = npair.build_merged_instance(f"[('f', '{f}'), ('g', '{g}')]", spec)
    assert os.path.isfile(mo) and os.path.isfile(mv)


def test_loaders_missing_file_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        npair._load_onnx(str(tmp_path / 'nope.onnx'))
    with pytest.raises(FileNotFoundError):
        npair._read_vnnlib_text(str(tmp_path / 'nope.vnnlib'))


def test_build_from_gz_no_oracle(tmp_path):
    # gzip the onnx + vnnlib (the authoritative repo form); loaders must prefer .gz.
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=7); _mono_spec(spec)
    with open(f, 'rb') as fh, gzip.open(f + '.gz', 'wb') as gz:
        gz.write(fh.read())
    with open(spec, 'rb') as fh, gzip.open(spec + '.gz', 'wb') as gz:
        gz.write(fh.read())
    os.remove(f); os.remove(spec)                      # only the .gz remain
    mo, mv = npair.build_merged_instance(f"[('f', '{f}'), ('g', '{f}')]", spec, run_oracle=False)
    assert os.path.isfile(mo) and os.path.isfile(mv)


def test_build_rejects_non_pair(tmp_path):
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f); open(spec, 'w').write('(assert (<= X_0 1))\n')
    with pytest.raises(AssertionError):
        npair.build_merged_instance(f"[('f', '{f}')]", spec)


# --------------------------------------------------------------------------- #
# --net argument parsing: single path vs network-pair list-string (the harness
# form), ast-based, label-matched, CWD/version-dir tolerant resolution.
# --------------------------------------------------------------------------- #

def test_parse_network_field_single_is_none():
    assert npair.parse_network_field('onnx/a.onnx') is None
    assert npair.parse_network_field('/abs/a.onnx') is None


def test_parse_network_field_pair_ast():
    p = npair.parse_network_field("[('f', 'a.onnx'), ('g', 'b.onnx')]")
    assert p == [('f', 'a.onnx'), ('g', 'b.onnx')]
    # robust to real paths with / . _ and a perturbed-N suffix (the harness form)
    real = ("[('f', './benchmarks/iso/2.0/onnx/original/ACASXU_2_9.onnx'), "
            "('g', './benchmarks/iso/2.0/onnx/perturbed/ACASXU_2_9_perturbed_15.onnx')]")
    p2 = npair.parse_network_field(real)
    assert p2[0][0] == 'f' and p2[1][0] == 'g'
    assert p2[1][1].endswith('perturbed_15.onnx')
    # back-compat helper still returns the ordered paths
    assert npair._onnx_paths_from_field("[('f', 'a.onnx'), ('g', 'b.onnx')]") == ['a.onnx', 'b.onnx']


@pytest.mark.parametrize('bad', [
    "[('f',)]", "[('f', 'a', 'x')]", "[]", "[('f', 5)]", "['a.onnx']", "[oops"])
def test_parse_network_field_malformed_raises(bad):
    with pytest.raises(ValueError):
        npair.parse_network_field(bad)


def test_declared_networks():
    txt = ('(declare-network f (declare-input X_f real [5]))\n'
           '(declare-network g (equal-to f))')
    assert npair.declared_networks(txt) == ['f', 'g']


def test_resolve_onnx_path_cwd_vs_basedir(tmp_path):
    assert npair._resolve_onnx_path('/abs/x.onnx', '/base') == '/abs/x.onnx'   # absolute
    f = tmp_path / 'x.onnx'; f.write_text('x')
    here = os.getcwd()
    try:                                                # CWD-relative + EXISTS -> as-is
        os.chdir(tmp_path)
        assert npair._resolve_onnx_path('x.onnx', '/unused/base') == 'x.onnx'
    finally:
        os.chdir(here)
    # relative + not at CWD -> joined to the benchmark version dir (instances.csv form)
    assert npair._resolve_onnx_path('onnx/x.onnx', '/b/2.0') == os.path.join('/b/2.0', 'onnx/x.onnx')


def test_resolve_pair_paths(tmp_path):
    vdir = tmp_path / '2.0' / 'vnnlib'; vdir.mkdir(parents=True)
    spec = str(vdir / 's.vnnlib'); open(spec, 'w').write('x')
    m = npair.resolve_pair_paths("[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx')]", spec)
    base = str(tmp_path / '2.0')
    assert m == {'f': os.path.join(base, 'onnx/f.onnx'),
                 'g': os.path.join(base, 'onnx/g.onnx')}
    assert npair.resolve_pair_paths('single.onnx', spec) is None


def test_build_matches_by_name_not_position(tmp_path):
    """The list-string can come in any order; networks are matched to the spec's
    declared (f, g) BY LABEL, not by position — reversed order merges identically."""
    f = str(tmp_path / 'f.onnx'); g = str(tmp_path / 'g.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=1); _tiny_acasxu(g, seed=2); _iso_spec(spec)
    mo_rev, _ = npair.build_merged_instance(f"[('g', '{g}'), ('f', '{f}')]", spec)   # g first
    mo_fwd, _ = npair.build_merged_instance(f"[('f', '{f}'), ('g', '{g}')]", spec)
    sm_rev = ort.InferenceSession(mo_rev); sm_fwd = ort.InferenceSession(mo_fwd)
    x = np.random.default_rng(0).random((1, 5)).astype(np.float32) * 2 - 1
    assert np.abs(sm_rev.run(None, {'X': x})[0] - sm_fwd.run(None, {'X': x})[0]).max() < 1e-5


def test_build_missing_declared_network_raises(tmp_path):
    f = str(tmp_path / 'f.onnx'); spec = str(tmp_path / 's.vnnlib')
    _tiny_acasxu(f, seed=1); _iso_spec(spec)            # iso declares BOTH f and g
    with pytest.raises(AssertionError):                # list provides only f -> g missing
        npair.build_merged_instance(f"[('f', '{f}')]", spec)
