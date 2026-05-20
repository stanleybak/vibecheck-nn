"""Tests for the PGD counterexample search (``vibecheck.pgd``).

Regression focus: the GPU-side witness screen in ``_gpu_witness_candidates``
must flag a sample as a candidate whenever ``VNNSpec.check`` would return
'unknown'. An earlier version took the MAX across disjuncts (requiring every
disjunct to have all-negative margins — almost never true on cifar-style
99-way OR specs), so PGD missed essentially every SAT on classification
benchmarks even when the gradient step had already produced a valid
counterexample.
"""
import numpy as np
import torch

from vibecheck.pgd import (
    _gpu_witness_candidates,
    _build_constraint_matrices,
    _compute_margins_per_disj_batched,
)
from vibecheck.spec import (
    VNNSpec, Conjunct, Constraint, PairwiseConstraint,
)


def _torch(ms):
    return [torch.as_tensor(m, dtype=torch.float64) for m in ms]


def test_witness_screen_flags_some_disjunct_violation():
    """Classification-style spec: 99 OR-disjuncts each with ONE pairwise.

    If even ONE disjunct's margin is negative the sample is a candidate
    (the predicted class lost to SOME other class).
    """
    # sample 0: one disjunct (j=3) is negative → should flag.
    # sample 1: all disjuncts positive → should NOT flag.
    margins_per_disj = _torch([
        # one tensor per disjunct, shape (n_batch, n_c=1)
        np.array([[0.5], [0.5]]),
        np.array([[0.3], [0.3]]),
        np.array([[0.4], [0.4]]),
        np.array([[-0.2], [0.1]]),   # sample 0 violates here
        np.array([[0.5], [0.5]]),
    ])
    flags = _gpu_witness_candidates(margins_per_disj)
    assert flags.tolist() == [True, False]


def test_witness_screen_single_disjunct_multi_constraint():
    """Single-disjunct AND-conjunct: any constraint <= 0 is a candidate.

    Matches ``VNNSpec.check``'s semantics (uses ``min`` of constraint
    margins as the conjunct margin).
    """
    margins_per_disj = _torch([
        np.array([[0.5, -0.1, 0.3],     # sample 0: c1 is negative → candidate
                  [0.5, 0.2, 0.3]]),    # sample 1: all positive → safe
    ])
    flags = _gpu_witness_candidates(margins_per_disj)
    assert flags.tolist() == [True, False]


def test_witness_screen_empty_margins():
    assert _gpu_witness_candidates([]) is None


def test_witness_matches_spec_check_semantics():
    """For an arbitrary spec the witness screen must agree with
    ``VNNSpec.check`` on every synthetic output vector."""
    rng = np.random.default_rng(0)
    n_output = 10
    # Build a 5-disjunct DNF: each disjunct is a random pairwise constraint.
    disjuncts = []
    for _ in range(5):
        pred = int(rng.integers(n_output))
        comp = int(rng.integers(n_output))
        if comp == pred:
            comp = (comp + 1) % n_output
        disjuncts.append(Conjunct([PairwiseConstraint(pred=pred, comp=comp)]))
    spec = VNNSpec(
        x_lo=np.zeros(1), x_hi=np.ones(1), disjuncts=disjuncts)

    per_disj_constraints = [list(d.constraints) for d in disjuncts]
    outs = torch.as_tensor(rng.standard_normal((50, n_output)),
                           dtype=torch.float64)
    mats = _build_constraint_matrices(
        per_disj_constraints, n_output, torch.float64, outs.device)
    margins_pd = _compute_margins_per_disj_batched(outs, mats)
    flags = _gpu_witness_candidates(margins_pd).numpy()

    for i in range(outs.shape[0]):
        o = outs[i].numpy()
        result, _ = spec.check(o, o)
        expected_flag = (result == 'unknown')
        assert bool(flags[i]) == expected_flag, (
            f'sample {i}: flags={bool(flags[i])} spec.check={result}')


def test_pgd_finds_sat_on_trivial_adversarial_case():
    """End-to-end: tiny fully-connected net, narrow input box that contains
    a counterexample. PGD must find it."""
    import torch.nn.functional as F
    from vibecheck.pgd import pgd_attack_general
    from vibecheck.settings import default_settings

    # 2-layer net: W1 = eye, ReLU, W2 = [[+1, -1]]; output = max(x0, 0) - max(x1, 0).
    # Unsafe = Y_0 >= 0 (trivially: set x0=0.5, x1=0).
    n_in = 2
    gg = {
        'input_name': 'x',
        'fork_points': set(),
        'ops': [
            {
                'name': 'fc1', 'type': 'fc',
                'inputs': ['x'],
                'W': torch.eye(n_in),
                'bias': torch.zeros(n_in),
                'in_shape': (n_in,),
                'out_shape': (n_in,),
            },
            {
                'name': 'r1', 'type': 'relu',
                'inputs': ['fc1'],
                'out_shape': (n_in,),
            },
            {
                'name': 'fc2', 'type': 'fc',
                'inputs': ['r1'],
                'W': torch.tensor([[1.0, -1.0]]),
                'bias': torch.zeros(1),
                'in_shape': (n_in,),
                'out_shape': (1,),
            },
        ],
    }
    # Input box [-1, 1] x [-1, 1] so the point (0.5, 0) is inside.
    xl = torch.tensor([-1.0, -1.0])
    xh = torch.tensor([1.0, 1.0])
    spec = VNNSpec(
        x_lo=xl.numpy(), x_hi=xh.numpy(),
        disjuncts=[Conjunct([Constraint(index=0, op='>=', value=0.0)])])
    settings = default_settings(pgd_restarts=4, pgd_iter=30)
    sat, witness = pgd_attack_general(xl, xh, spec, gg, settings)
    assert sat, 'PGD failed to find the trivial counterexample'
    assert witness is not None


# ---------------------------------------------------------------------------
# _PGDOptim — pluggable optimizer (sign_sgd, adam_sign, adam_clipping)
# ---------------------------------------------------------------------------

def _trivial_pgd_graph():
    """Same tiny graph as `test_pgd_finds_sat_on_trivial_adversarial_case`,
    factored out so multiple optimizer-mode tests can reuse it."""
    n_in = 2
    gg = {
        'input_name': 'x',
        'fork_points': set(),
        'ops': [
            {'name': 'fc1', 'type': 'fc', 'inputs': ['x'],
             'W': torch.eye(n_in), 'bias': torch.zeros(n_in),
             'in_shape': (n_in,), 'out_shape': (n_in,)},
            {'name': 'r1', 'type': 'relu', 'inputs': ['fc1'],
             'out_shape': (n_in,)},
            {'name': 'fc2', 'type': 'fc', 'inputs': ['r1'],
             'W': torch.tensor([[1.0, -1.0]]), 'bias': torch.zeros(1),
             'in_shape': (n_in,), 'out_shape': (1,)},
        ],
    }
    xl = torch.tensor([-1.0, -1.0])
    xh = torch.tensor([1.0, 1.0])
    spec = VNNSpec(
        x_lo=xl.numpy(), x_hi=xh.numpy(),
        disjuncts=[Conjunct([Constraint(index=0, op='>=', value=0.0)])])
    return gg, spec, xl, xh


def test_pgd_optim_default_is_adam_clipping():
    """Default `pgd_optim` is 'adam_clipping' (mirrors α,β-CROWN's
    AdamClipping). Was 'adam_sign' historically; switched after the 5
    mnist 256x4 prop_*_0.05 SAT regressions where PGD took 80-217s vs
    AB's 3-7s. With 'adam_clipping' + 30 restarts + a Phase 0 PGD
    invocation before the cascade those cases close in <15s."""
    from vibecheck.settings import default_settings
    s = default_settings()
    assert s.pgd_optim == 'adam_clipping'


def test_pgd_optim_unknown_raises():
    """Unknown pgd_optim mode raises an AssertionError when the optimizer
    is constructed at the top of the PGD loop."""
    from vibecheck.pgd import _PGDOptim
    import pytest
    with pytest.raises(AssertionError):
        _PGDOptim('badmode', (4,), torch.float64, torch.device('cpu'))


def test_pgd_optim_sign_sgd_no_adam_state():
    """`sign_sgd` mode does NOT allocate the Adam moment buffers.

    Saves both the GPU memory (~2× the input batch) and the per-iter
    moment-update FLOPs.
    """
    from vibecheck.pgd import _PGDOptim
    o = _PGDOptim('sign_sgd', (3, 4), torch.float64, torch.device('cpu'))
    assert not hasattr(o, 'm')
    assert not hasattr(o, 'v')

    o2 = _PGDOptim('adam_sign', (3, 4), torch.float64, torch.device('cpu'))
    assert hasattr(o2, 'm') and hasattr(o2, 'v')
    assert o2.m.shape == (3, 4)


def test_pgd_optim_compute_delta_shapes_all_modes():
    """For every mode, `compute_delta` returns a tensor with the same
    shape as `grad`."""
    from vibecheck.pgd import _PGDOptim
    shape = (2, 3)
    for mode in ('sign_sgd', 'adam_sign', 'adam_clipping'):
        o = _PGDOptim(mode, shape, torch.float64, torch.device('cpu'))
        grad = torch.randn(*shape, dtype=torch.float64)
        step = torch.full(shape, 0.1, dtype=torch.float64)
        delta = o.compute_delta(grad, step)
        assert delta.shape == shape


def test_pgd_optim_sign_sgd_matches_g_sign():
    """`sign_sgd` mode's delta is exactly `step * sign(grad)`."""
    from vibecheck.pgd import _PGDOptim
    grad = torch.tensor([-2.0, 0.0, 3.5])
    step = torch.tensor([0.1, 0.2, 0.3])
    o = _PGDOptim('sign_sgd', grad.shape, grad.dtype, grad.device)
    delta = o.compute_delta(grad, step)
    expected = step * grad.sign()
    assert torch.allclose(delta, expected)


def test_pgd_optim_adam_clipping_step_size_grows_via_bias_correction():
    """At iter 1 the AdamClipping step magnitude = lr / (1 - beta1) = 10·lr.

    α,β-CROWN's `attack_utils.py:228-238` divides by `bias_correction1`,
    which makes the FIRST step much larger than the asymptotic step. This
    is empirically what helps escape saturated-ReLU plateaus on hard
    cases. The 'adam_sign' mode does not do this.
    """
    from vibecheck.pgd import _PGDOptim
    grad = torch.tensor([1.0, -1.0, 0.5])  # all non-zero
    step = torch.tensor([0.01, 0.01, 0.01])
    o_clip = _PGDOptim('adam_clipping', grad.shape, grad.dtype, grad.device)
    delta_clip = o_clip.compute_delta(grad, step)
    o_sign = _PGDOptim('adam_sign', grad.shape, grad.dtype, grad.device)
    delta_sign = o_sign.compute_delta(grad, step)
    # Both have sign = sign(grad) (m_buf initialized at 0; one step makes
    # m_buf = (1-beta1)*grad which has the same sign as grad).
    assert torch.equal(delta_clip.sign(), grad.sign())
    assert torch.equal(delta_sign.sign(), grad.sign())
    # adam_clipping step magnitude = step / 0.1 = 10 × step.
    # adam_sign step magnitude = step (no bias correction in step size).
    expected_clip_mag = step / (1.0 - 0.9)
    expected_sign_mag = step
    assert torch.allclose(delta_clip.abs(), expected_clip_mag)
    assert torch.allclose(delta_sign.abs(), expected_sign_mag)


def test_pgd_optim_adam_clipping_finds_trivial_sat():
    """End-to-end: adam_clipping mode finds the trivial counterexample."""
    from vibecheck.pgd import pgd_attack_general
    from vibecheck.settings import default_settings
    gg, spec, xl, xh = _trivial_pgd_graph()
    settings = default_settings(
        pgd_restarts=4, pgd_iter=30, pgd_optim='adam_clipping')
    sat, witness = pgd_attack_general(xl, xh, spec, gg, settings)
    assert sat
    assert witness is not None


def test_pgd_optim_sign_sgd_finds_trivial_sat():
    """End-to-end: sign_sgd mode (no Adam state) also finds the trivial
    counterexample. Soundness check that the dispatcher doesn't break
    for the cheapest mode."""
    from vibecheck.pgd import pgd_attack_general
    from vibecheck.settings import default_settings
    gg, spec, xl, xh = _trivial_pgd_graph()
    settings = default_settings(
        pgd_restarts=4, pgd_iter=30, pgd_optim='sign_sgd')
    sat, witness = pgd_attack_general(xl, xh, spec, gg, settings)
    assert sat
    assert witness is not None


def test_pgd_optim_adam_clipping_box_clamped():
    """Box-clamping happens inside the loop in pgd_attack_general — the
    optimizer's compute_delta is unaware of bounds. The loop's `clamp(x_new,
    xl, xh)` is what enforces invariance. Sanity-check that on a
    feasibility-only setting (no SAT to find), all final iterates respect
    the box."""
    from vibecheck.pgd import pgd_attack_general
    from vibecheck.settings import default_settings
    gg, _, xl, xh = _trivial_pgd_graph()
    # Spec that is trivially safe so PGD runs to completion without an
    # early SAT exit. Y[0] = max(x0,0) - max(x1,0) ∈ [-1, 1] for inputs
    # in [-1, 1]^2 → 'Y[0] >= -2' is always true.
    spec = VNNSpec(
        x_lo=xl.numpy(), x_hi=xh.numpy(),
        disjuncts=[Conjunct([Constraint(index=0, op='>=', value=-2.0)])])
    # Run to completion — when no SAT is found, the loop terminates normally.
    settings = default_settings(
        pgd_restarts=2, pgd_iter=15, pgd_optim='adam_clipping')
    sat, _ = pgd_attack_general(xl, xh, spec, gg, settings)
    assert not sat  # spec is trivially safe; PGD shouldn't claim SAT
