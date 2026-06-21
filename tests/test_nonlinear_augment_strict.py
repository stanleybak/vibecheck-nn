"""Soundness regression: the nonlinear-augment v1 emitter must NOT shrink the
unsafe set of a STRICT constraint.

A strict original atom `p_c(X,Y) < 0` has unsafe set {p_c < 0}. The augmented
v1 spec encodes it as `Y_c <= threshold`. If threshold is pushed to -MARGIN
(-1e-4) the encoded unsafe set becomes {p_c <= -1e-4} — a STRICT SUBSET of the
true unsafe set, missing the band (-1e-4, 0). A real shallow counterexample with
p_c = -5e-5 (genuinely < 0) is then declared SAFE, so the verifier can prove
`unsat` on an instance that is actually `sat` — a false-unsat (unsound).

The sound encoding uses threshold 0 (a SUPERSET of {p_c < 0}); the measure-zero
over-inclusion at p_c == 0 is caught downstream by the ORT witness re-validation,
so it cannot cause a false-sat. This test pins that a shallow strict CE is still
recognized as unsafe by the emitted spec.
"""
import tempfile, os

from vibecheck.nonlinear_augment import emit_v1
from vibecheck.vnnlib_loader import load_vnnlib


def test_strict_shallow_ce_not_dropped():
    # one STRICT constraint p_0 < 0 (cons entry: row, bias, strict=True)
    cons = [({0: 1.0}, 0.0, True)]
    clauses = [[0]]
    xbox = {0: [-1.0, 1.0]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'aug.vnnlib')
        emit_v1(cons, clauses, xbox, n_in=1, out_path=path)
        spec = load_vnnlib(path)

    # a point whose constraint polynomial p_0 = -5e-5 is a genuine strict CE
    # (p_0 < 0). The emitted spec must classify it as unsafe (conjunct margin<=0).
    y = [-5e-5]
    margin = min(conj.margin(y, y) for conj in spec.disjuncts)
    assert margin <= 0.0, (
        f"shallow strict CE p_0=-5e-5 wrongly declared safe (margin={margin:+.2e}); "
        f"-MARGIN threshold shrinks the strict unsafe set -> false-unsat risk")
