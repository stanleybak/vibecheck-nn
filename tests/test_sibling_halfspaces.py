"""Sibling-halfspace dual cuts (INVPROP-lite) in the Phase-8 fast BnB.

For a CONJUNCTIVE disjunct, refutation may assume the SAT set, so every
sibling conjunct ``w_j·y + b_j <= 0`` is a valid extra constraint. It is
projected through the output zonotope map ``y = c_out + G e`` into the
generator-space halfspace ``(w_j @ G)·e <= -(w_j·c_out + b_j)`` and dualized
(nu >= 0) into every node bound — sound by weak duality for any nu >= 0.

Pins, on a hand-built 2-input toy (query min = e1, one dummy unstable split):
  1. ``parse_problem(extra_hs=...)`` builds hs_a/hs_b exactly as the formula.
  2. Binding sibling (forces e1 >= 0.5, joint min = +0.5) -> 'unsat'; the
     same problem WITHOUT the sibling is structurally unverifiable (true
     min = -1) -> 'unknown'.
  3. Soundness: a loose sibling (e1 >= -0.5, joint min = -0.5 < 0) must NOT
     be certified -> 'unknown'.
  4. M=0 / non-binding parity: empty extra_hs goes through the no-hs path;
     a fully slack sibling (e1 >= -10) changes neither verdict nor node
     count vs the no-hs run.
"""
import numpy as np
import scipy.sparse as sp

from vibecheck.fast_dual_ascent import parse_problem, Verifier


def _toy_state():
    """y = c_out + G e over e in [-1,1]^2 plus one (irrelevant) unstable
    neuron's new column: y0 = e0 (the query output), y1 = e0 (the sibling
    output). The unstable split exists only so the BnB has a level to
    process — its column coefficient in the outputs is 0, and its pre-ReLU
    row coefficient is TINY (1e-3) so the split multipliers cannot dominate
    the kernel's single coupled line-search direction (with a 1.0 row the
    ON-side lambda drags the joint ascent ray's peak below 0 and the
    certification needs more levels than the toy has)."""
    n_input, n_gens = 2, 3
    G = np.zeros((2, n_gens))
    G[0, 0] = 1.0   # y0 = e0
    G[1, 0] = 1.0   # y1 = e0
    unstable = [dict(
        layer_idx=1, neuron_idx=0, c_in=0.0, lo=-1.0, hi=1.0,
        lam=0.5, mu=0.25, e_new_col=2, form='alpha_zono',
        row_indices=np.array([1], np.int64), row_values=np.array([1e-3]))]
    return dict(
        n_gens=n_gens, n_input=n_input, formulation='sparse',
        unstable_list=unstable, stable_list=[],
        obj_c_out=np.zeros(2), obj_G_out_csr=sp.csr_matrix(G),
        sigmoid_tanh_layer_ids=set())


QW = np.array([1.0, 0.0])   # query: min y0 = min e0  (refuted iff lb > 0)
QB = 0.0
KEYS = [(1, 0)]


def _verifier():
    return Verifier(device='cpu', compile=False)


def test_parse_problem_builds_hs_rows():
    state = _toy_state()
    rng = np.random.default_rng(0)
    G = rng.standard_normal((2, 5))
    state['obj_G_out_csr'] = sp.csr_matrix(G)
    state['obj_c_out'] = rng.standard_normal(2)
    state['n_gens'] = 5
    sibs = [(rng.standard_normal(2), float(rng.standard_normal()))
            for _ in range(3)]
    prob = parse_problem(state, QW, QB, KEYS, extra_hs=sibs)
    assert prob.hs_a.shape == (3, 5) and prob.hs_b.shape == (3,)
    for j, (wj, bj) in enumerate(sibs):
        np.testing.assert_allclose(prob.hs_a[j], wj @ G, atol=1e-12)
        np.testing.assert_allclose(prob.hs_b[j],
                                   -(wj @ state['obj_c_out'] + bj), atol=1e-12)


def test_binding_sibling_certifies_and_without_it_cannot():
    state = _toy_state()
    v = _verifier()
    # no sibling: true min y0 = -1 over the box -> must stay unknown
    verdict, info = v.verify_query(state, QW, QB, KEYS)
    assert verdict == 'unknown'
    # sibling w=( 0,-1), b=0.5 assumes -y1 + 0.5 <= 0  <=>  e0 >= 0.5,
    # so the joint min is +0.5 > 0 -> the nu-ascent must certify
    verdict, info = v.verify_query(state, QW, QB, KEYS,
                                   extra_hs=[(np.array([0.0, -1.0]), 0.5)])
    assert verdict == 'unsat'


def test_loose_sibling_is_not_certified():
    # e0 >= -0.5 leaves the joint min at -0.5 < 0: certifying would be UNSOUND
    state = _toy_state()
    verdict, info = _verifier().verify_query(
        state, QW, QB, KEYS, extra_hs=[(np.array([0.0, -1.0]), -0.5)])
    assert verdict == 'unknown'


def test_slack_sibling_is_node_parity_noop():
    state = _toy_state()
    v = _verifier()
    verdict0, info0 = v.verify_query(state, QW, QB, KEYS, extra_hs=())
    # e0 >= -10 never binds inside the unit box
    verdict1, info1 = v.verify_query(
        state, QW, QB, KEYS, extra_hs=[(np.array([0.0, -1.0]), -10.0)])
    assert (verdict0, info0['nodes']) == (verdict1, info1['nodes'])
