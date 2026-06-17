"""Dual-space corrections for a nonlinear split in the GPU dual-ascent BaB.

`nonlinear_split_planes` answers "how does the parallelogram relaxation CHANGE
on each side of a split". This module answers the next question the dual-ascent
BaB needs: "given that band change for neuron k, how do the LP objective
(c0, d) and constraints (A·e ≤ b) change?" — WITHOUT re-forwarding the network
(sound by the same fixed-root argument as the ReLU substitution path in
`dual_ascent_bab`).

Setup (one splittable neuron k, single scalar spec direction):
    pre-activation   z_k = c_in + a·e            (a = row over error symbols e)
    affine band      y_k = (λ·c_in + μ) + λ·(a·e) + δ·e_new ,   e_new ∈ [-1,1]
    objective        obj(e) = c0 + d·e   (minimised over the box; safe iff >0)

Because y_k enters the (affine-after-relaxation) network linearly, it hits the
objective with a single scalar BACKWARD SENSITIVITY g_k, and the parent objective
necessarily has
    d[e_new_col] = g_k · δ                       (e_new is k's private symbol)
    d[a columns] includes g_k · λ · a            (k's share of the input columns)
    c0           includes g_k · (λ·c_in + μ)     (k's constant share)
so  g_k = d[e_new_col] / δ   (the same quantity the ReLU path calls
d_at_e_new / mu_k; see verify_gen_lp band convention μ==δ for ReLU).

Splitting neuron k at pre-activation point p:
  * each side re-evaluates the band over its sub-interval → (λ',μ',δ')
    (`nonlinear_split_planes.split_planes` / `op_planes`);
  * the objective is corrected by `band_change_correction` below — replace k's
    (λ,μ,δ) share with (λ',μ',δ'). Everything else is unchanged, so we only add
    deltas on the row columns, the e_new column, and c0;
  * a single halfspace pins the side's pre-activation: z_k ≤ p (left) or
    z_k ≥ p (right), i.e. a·e ≤ p−c_in or −a·e ≤ −(p−c_in).

Soundness: keeping every OTHER neuron's parent relaxation fixed is sound because
each parent band is valid over the parent interval, which contains the child
sub-interval; tightening only neuron k's band and adding the side halfspace
yields a valid over-approximation of the child sub-domain (looser than a full
re-forward, but sound — identical argument to the ReLU substitution and the
forward op_clamps path). The correction is EXACT relative to an honest
re-propagation with the child band (validated in tests/test_nonlinear_split_dual.py
against fresh zonotope propagation), so no tightness is lost beyond the
fixed-downstream choice.

ReLU falls out as the p=0 special case: the sub-bands are exact (left λ'=μ'=δ'=0,
right λ'=1,μ'=δ'=0), so band_change_correction reproduces the y=0 / y=z
substitution objective (minus the substitution's extra e_new box-feasibility
cut, which the dedicated ReLU path keeps for a touch more tightness).
"""
import torch


def backward_sensitivity(d_e_new, delta_parent, eps=1e-30):
    """g_k = d[e_new_col] / δ_parent — neuron k's scalar objective sensitivity.

    δ_parent is the parent band half-width (the e_new column magnitude). A
    near-zero δ means y_k is effectively exact already (no fresh noise); g_k is
    then irrelevant (its band terms vanish), so we guard the divide and the
    caller's (δ'−δ)·g_k / (λ'−λ)·g_k corrections stay finite."""
    d_e_new = torch.as_tensor(d_e_new, dtype=torch.float64)
    delta_parent = torch.as_tensor(delta_parent, dtype=torch.float64)
    safe = delta_parent.abs() > eps
    denom = torch.where(safe, delta_parent, torch.ones_like(delta_parent))
    g = d_e_new / denom
    return torch.where(safe, g, torch.zeros_like(g))


def band_change_correction(g_k, c_in, row_values,
                           lam_old, mu_old, delta_old,
                           lam_new, mu_new, delta_new):
    """Objective correction for replacing neuron k's band (λ,μ,δ)_old with
    (λ,μ,δ)_new, given k's backward sensitivity g_k.

    All args broadcast; `row_values` is the [..., R] tensor of a's nonzero
    coefficients (the R entries at `row_indices`). Returns
        (d_corr_rows [..., R], d_corr_e_new [...], c0_corr [...])
    to be ADDED to d at row_indices, d at e_new_col, and c0 respectively.

    Derivation (y_k = (λ c_in + μ) + λ(a·e) + δ e_new, obj gains g_k·y_k):
        Δ(a-column coeff) = g_k (λ'−λ) a          -> d_corr_rows
        Δ(e_new coeff)    = g_k (δ'−δ)            -> d_corr_e_new
        Δ(constant)       = g_k ((λ'−λ) c_in + (μ'−μ))   -> c0_corr
    """
    g_k = torch.as_tensor(g_k, dtype=torch.float64)
    c_in = torch.as_tensor(c_in, dtype=torch.float64)
    row_values = torch.as_tensor(row_values, dtype=torch.float64)
    lam_old = torch.as_tensor(lam_old, dtype=torch.float64)
    mu_old = torch.as_tensor(mu_old, dtype=torch.float64)
    delta_old = torch.as_tensor(delta_old, dtype=torch.float64)
    lam_new = torch.as_tensor(lam_new, dtype=torch.float64)
    mu_new = torch.as_tensor(mu_new, dtype=torch.float64)
    delta_new = torch.as_tensor(delta_new, dtype=torch.float64)

    dlam = lam_new - lam_old
    d_corr_rows = (g_k * dlam).unsqueeze(-1) * row_values
    d_corr_e_new = g_k * (delta_new - delta_old)
    c0_corr = g_k * (dlam * c_in + (mu_new - mu_old))
    return d_corr_rows, d_corr_e_new, c0_corr


def split_halfspace(c_in, row_values, p, side):
    """The single halfspace pinning a split side's pre-activation z_k = c_in+a·e.

    Returns (hs_row_values [..., R], hs_b [...]) for the constraint
    hs_row · e ≤ hs_b (the dual-ascent A·e ≤ b convention), where:
        side='left'  : z_k ≤ p   ⟺   a·e ≤ p − c_in
        side='right' : z_k ≥ p   ⟺  −a·e ≤ −(p − c_in)
    The row coefficients live at the SAME `row_indices` as `row_values`.
    """
    c_in = torch.as_tensor(c_in, dtype=torch.float64)
    row_values = torch.as_tensor(row_values, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    rhs = p - c_in
    if side == 'left':
        return row_values, rhs
    if side == 'right':
        return -row_values, -rhs
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")
