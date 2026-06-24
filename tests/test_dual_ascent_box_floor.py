"""Dual-ascent node bound must never fall below the box-corner bound (λ=0).

A BaB split adds a halfspace constraint (z≥0 or z≤0) to the node's LP, which can
only SHRINK the feasible region and therefore RAISE the min objective. The dual
bound g(λ) is a valid lower bound for ANY λ≥0; at λ=0 it equals the unconstrained
box bound. So a node's reported bound at an inherited (possibly bad) warm-start λ
must never be reported BELOW its λ=0 value — otherwise deep BaB nodes get garbage
bounds, never certify, and the frontier explodes (observed: worst_lb −0.0016 →
−9969 with depth on soundnessbench property_012).

This pins that invariant directly on the node-bound kernels.
"""
import torch

from vibecheck.fast_dual_ascent.fast_verify_topk import (
    node_bound_logbucket, node_bound_topk)


def _tiny_F(n=2):
    """Minimal node geometry: D=1 split, n generators, M=0 sibling halfspaces.
    Chosen so a large λ on the split's `−λ·b` term drives g far below the box
    corner (c_in>0 → b0=−c_in<0 on the OFF side)."""
    return {
        'a_g': torch.tensor([[1.0, 0.0]]),        # (D,n) split halfspace coeff
        'el': torch.tensor([-1.0, -1.0]),
        'eh': torch.tensor([1.0, 1.0]),
        'ratio_off': torch.tensor([0.0]),
        'ratio_on': torch.tensor([0.0]),
        'c0': torch.tensor(0.0),
        'c0_off': torch.tensor([0.0]),
        'c0_on': torch.tensor([0.0]),
        'c_in': torch.tensor([0.5]),              # >0
        'z_lo': torch.tensor([-1.0]),
        'z_hi': torch.tensor([1.0]),
        'hs_a': torch.zeros(0, n),                # M=0
        'hs_b': torch.zeros(0),
        'd_base': torch.tensor([1.0, 1.0]),       # (n,) objective
    }


def _check(node_bound_fn):
    F = _tiny_F()
    sides = torch.tensor([[0]], dtype=torch.int8)   # one node, OFF side
    nu = torch.zeros(1, 0)
    zero = torch.zeros(1, 1)
    # box-corner bound: λ=0
    g_box, *_ = node_bound_fn(F, sides, zero.clone(), zero.clone(), nu.clone())
    # bound at a large, perfectly-valid (≥0) inherited λ
    big = torch.full((1, 1), 10.0)
    g_lam, *_ = node_bound_fn(F, sides, big.clone(), zero.clone(), nu.clone())
    assert float(g_lam) >= float(g_box) - 1e-6, (
        f"{node_bound_fn.__name__}: split bound {float(g_lam):.4f} is WORSE than "
        f"the box-corner bound {float(g_box):.4f} — a halfspace constraint can "
        f"only tighten; the reported bound must floor at the box bound")


def test_logbucket_bound_never_below_box_corner():
    _check(node_bound_logbucket)


def test_topk_bound_never_below_box_corner():
    _check(node_bound_topk)
