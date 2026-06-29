"""The nonlinear-augment SAT chokepoint must commit `sat` ONLY for a witness that
STRICTLY violates the ORIGINAL v2 spec — a boundary CE that meets the augmented
CLOSURE (output exactly on a strict `>`/`<` threshold) must NOT be committed, so the
search keeps going. This is centralized in `verify_graph._sat_disposition`, which
every graph sat path (Phase-0 PGD, trig_bab, the nominal probe, MILP via _finalize)
funnels through. See main.py's `augment_cex_validator` stash.

Regression: before the fix, adaptive_cruise emitted boundary witnesses (Y landing
exactly on the 100.001 threshold) that the official VNN-COMP 2026 scorer rejects as
`spec_not_violated` -> a scoring PENALTY. The strict-disposition rule turns those
into either a deeper genuine `sat` (the search finds the strict CE) or a sound
non-commit (never a penalty).
"""
import numpy as np

from vibecheck.settings import default_settings
from vibecheck.verify_graph import _sat_disposition


def test_standard_run_commits_real():
    """No augment validator set (the normal case): any ORT-validated witness is a
    genuine violation, so disposition is 'real'."""
    s = default_settings()
    assert getattr(s, 'augment_cex_validator', None) is None
    assert _sat_disposition(None, None, s, np.array([1.0, 2.0]), {}) == 'real'


def test_augment_strict_violation_commits_real():
    """Augment run, witness STRICTLY violates the original v2 spec -> commit 'sat'."""
    s = default_settings()
    s.augment_cex_validator = lambda w: True       # scorer would award it
    assert _sat_disposition(None, None, s, np.array([80.08, -38.99]), {}) == 'real'


def test_augment_boundary_does_not_commit():
    """Augment run, witness only meets the closure boundary (scorer would NOT award
    it) -> 'within_tol' so callers keep searching, never an emitted penalty."""
    s = default_settings()
    seen = {}

    def _val(w):
        seen['w'] = np.asarray(w)
        return False                                # boundary: not strictly violating

    s.augment_cex_validator = _val
    assert _sat_disposition(None, None, s, np.array([94.5, -40.0]), {}) == 'within_tol'
    # the witness was actually passed to the original-spec validator
    assert np.allclose(seen['w'], [94.5, -40.0])
