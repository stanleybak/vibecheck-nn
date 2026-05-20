"""Tests for `_build_constraint_matrices` margin sign convention.

PGD's margin sign MUST match `spec.py`'s Constraint.margin convention:
positive margin = safe; PGD descends toward margin ≤ 0 = unsafe. A
sign flip on threshold constraints (`<=`, `>=`) silently broke cersyve
SAT detection — all 6 SAT cases stuck at "unknown" while brute-force
sampling found witnesses in <100k samples. Pairwise constraints (used
by cifar100/tinyimagenet) were correct, so the bug was silent on
benchmarks merged before cersyve.
"""

import numpy as np
import torch
import pytest

from vibecheck.pgd import _build_constraint_matrices
from vibecheck.spec import Constraint, PairwiseConstraint


def _pgd_margin_point(c, y):
    """Run a constraint's margin through `_build_constraint_matrices`
    against a single output point `y` (1-D numpy array). Returns the
    margin as a Python float."""
    n_out = len(y)
    mats = _build_constraint_matrices(
        [[c]], n_out, torch.float64, torch.device('cpu'))
    W, b = mats[0]
    out = torch.tensor(y, dtype=torch.float64).unsqueeze(0)
    return float((out @ W.T + b).squeeze(0).item())


def _spec_margin_point(c, y):
    """Spec.py canonical margin evaluated at a single point (lo=hi=y)."""
    return float(c.margin(np.asarray(y), np.asarray(y)))


# --- Sign equivalence: pgd-margin == spec-margin for every op ------------

def test_pairwise_sign_matches_spec():
    """Pairwise: pgd should compute y[pred] - y[comp]."""
    c = PairwiseConstraint(pred=0, comp=1)
    y = np.array([0.3, 0.7])
    assert _pgd_margin_point(c, y) == pytest.approx(_spec_margin_point(c, y))
    assert _pgd_margin_point(c, y) == pytest.approx(-0.4)


def test_geq_sign_matches_spec_unsafe_witness():
    """`Y_i >= val` unsafe. spec margin = val - y[i]; PGD must match."""
    c = Constraint(index=0, op='>=', value=0.5)
    # Point where Y_0 = 0.7 ≥ 0.5 → unsafe (margin should be NEGATIVE).
    y_unsafe = np.array([0.7, 0.0])
    spec_m = _spec_margin_point(c, y_unsafe)
    pgd_m = _pgd_margin_point(c, y_unsafe)
    assert spec_m == pytest.approx(-0.2)
    assert pgd_m == pytest.approx(spec_m), (
        f"pgd-margin {pgd_m} != spec-margin {spec_m} for '>=': "
        f"sign-flip bug regression. With a positive PGD margin on an "
        f"unsafe point, the PGD loss `sum(clamp(margin, min=-1e-5))` "
        f"would gradient-descend AWAY from this witness.")


def test_geq_sign_matches_spec_safe_witness():
    c = Constraint(index=0, op='>=', value=0.5)
    y_safe = np.array([0.3, 0.0])
    assert _pgd_margin_point(c, y_safe) == pytest.approx(
        _spec_margin_point(c, y_safe))
    assert _pgd_margin_point(c, y_safe) == pytest.approx(0.2)


def test_leq_sign_matches_spec_unsafe_witness():
    """`Y_i <= val` unsafe. spec margin = y[i] - val; PGD must match."""
    c = Constraint(index=1, op='<=', value=0.3)
    # Y_1 = 0.1 ≤ 0.3 → unsafe.
    y_unsafe = np.array([0.0, 0.1])
    spec_m = _spec_margin_point(c, y_unsafe)
    pgd_m = _pgd_margin_point(c, y_unsafe)
    assert spec_m == pytest.approx(-0.2)
    assert pgd_m == pytest.approx(spec_m)


def test_leq_sign_matches_spec_safe_witness():
    c = Constraint(index=1, op='<=', value=0.3)
    y_safe = np.array([0.0, 0.5])
    assert _pgd_margin_point(c, y_safe) == pytest.approx(
        _spec_margin_point(c, y_safe))
    assert _pgd_margin_point(c, y_safe) == pytest.approx(0.2)


# --- Cersyve pendulum spec end-to-end ------------------------------------

def test_cersyve_pendulum_witness_margins_negative():
    """cersyve pendulum spec: unsafe iff (Y_0 <= 0) AND (Y_1 >= 0). A
    SAT witness (found by random sampling) has y ≈ [-0.0003, 0.0009].
    Both PGD margins must be NEGATIVE (= unsafe direction) on this
    witness, else PGD's `sum(clamp(margin))` loss descends AWAY."""
    c0 = Constraint(index=0, op='<=', value=0.0)
    c1 = Constraint(index=1, op='>=', value=0.0)
    y_wit = np.array([-0.00029, 0.00093])
    m0 = _pgd_margin_point(c0, y_wit)
    m1 = _pgd_margin_point(c1, y_wit)
    assert m0 < 0, f'PGD margin on unsafe witness must be ≤ 0; got {m0}'
    assert m1 < 0, f'PGD margin on unsafe witness must be ≤ 0; got {m1}'
    # Specific values (sanity):
    assert m0 == pytest.approx(-0.00029)
    assert m1 == pytest.approx(-0.00093)


def test_cersyve_pendulum_safe_margins_positive():
    """Inverse: a clearly-safe point (both Y_0 > 0 and Y_1 < 0) must
    give POSITIVE margins."""
    c0 = Constraint(index=0, op='<=', value=0.0)
    c1 = Constraint(index=1, op='>=', value=0.0)
    y_safe = np.array([0.5, -0.5])
    m0 = _pgd_margin_point(c0, y_safe)
    m1 = _pgd_margin_point(c1, y_safe)
    assert m0 > 0
    assert m1 > 0
    assert m0 == pytest.approx(0.5)
    assert m1 == pytest.approx(0.5)


def test_loss_direction_drives_unsafe():
    """PGD's loss = `sum(clamp(margin, min=hinge_thr))`. Minimizing this
    must move y from a safe configuration TOWARD an unsafe one. Verify
    via direct numeric check: gradient of loss w.r.t. y points in the
    unsafe-witness direction."""
    c0 = Constraint(index=0, op='<=', value=0.0)
    c1 = Constraint(index=1, op='>=', value=0.0)
    mats = _build_constraint_matrices(
        [[c0, c1]], n_out=2, dtype=torch.float64, device=torch.device('cpu'))
    W, b = mats[0]
    y = torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)
    margins = (y.unsqueeze(0) @ W.T + b).squeeze(0)
    loss = torch.clamp(margins, min=-1e-5).sum()
    loss.backward()
    grad = y.grad
    # Gradient of margin_0 = y[0] - 0 w.r.t. y[0] is +1 (margin
    # increases with y[0]). To DECREASE margin (push unsafe), y[0]
    # should DECREASE → step direction is -grad (PGD descends), so y[0]
    # should move DOWN (toward 0 then negative). grad[0] = +1.
    assert grad[0].item() > 0, (
        f'expected ∂loss/∂y[0] > 0 (so PGD descent decreases y[0] '
        f'toward 0); got {grad[0].item()}')
    # Similarly margin_1 = -y[1] + 0; ∂margin/∂y[1] = -1 → grad[1] = -1.
    # PGD descent moves y[1] UP toward 0+. ✓
    assert grad[1].item() < 0, (
        f'expected ∂loss/∂y[1] < 0; got {grad[1].item()}')
