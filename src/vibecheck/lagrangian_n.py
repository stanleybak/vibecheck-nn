"""N-halfspace Lagrangian dual subgradient for the bound problem

    min  d·e + c0
    s.t. e ∈ [-1, 1]^n
         A·e ≤ β        (N halfspaces, A is N×n, β is N-vector)

The dual is g(λ) = c0 - λ·β - Σ_i |d_i + (Aᵀλ)_i|, λ ≥ 0. Always concave
in λ, so any λ ≥ 0 gives a sound LOWER BOUND on the primal min. Strong
duality (LP) means max_{λ≥0} g(λ) equals the primal min.

For N=1 this matches `box_halfspace.lagrangian_min`'s closed-form O(n log n)
walk over breakpoints. For N≥2 we use diminishing-step subgradient ascent;
each iteration is O(N·n + n) (one A·sign matvec + one Aᵀ·λ matvec).
"""
import numpy as np


def lag_subgrad_min(d, c0, A, beta, n_iters=100, alpha0=None, return_history=False):
    """Maximize the dual g(λ) via projected subgradient ascent on λ ≥ 0.

    Returns
        (best_g, best_lam) — best dual value seen across iterations, and
        the λ that achieved it. best_g ≤ primal_min (sound LB).

    With return_history=True, also returns dict {'g_per_iter': [...]}.

    Sign convention: dual is g(λ) = c0 − λ·β − Σ |d + Aᵀλ|.
    Subgradient: ∂g/∂λ_k = −β_k − Σ_i A_{k,i} · sign(d + Aᵀλ)_i.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size

    # Allow A=None or empty for the box-only case (no halfspaces).
    if A is None or (hasattr(A, 'size') and A.size == 0):
        g0 = float(c0) - float(np.sum(np.abs(d)))
        if return_history:
            return g0, np.zeros(0), {'g_per_iter': [g0]}
        return g0, np.zeros(0)

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    if A.shape[1] != n:
        # Caller passed (n,) for N=1 — fix to (1, n).
        A = A.reshape(1, -1)
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    N = A.shape[0]
    assert beta.size == N, f'beta size {beta.size} != N={N}'

    lam = np.zeros(N, dtype=np.float64)
    # g(0) is always sound — it's the box-only bound.
    best_g = float(c0) - float(np.sum(np.abs(d)))
    best_lam = lam.copy()
    hist = [best_g] if return_history else None

    # Step-size scaling.  Subgradient norms scale with the L1 mass of A
    # (since ∂g/∂λ_k ~ Σ_i a_{k,i}·sign(...)). Default alpha0 makes the
    # initial step move λ by ~ 1/||A||_1 per coordinate, avoiding the
    # overshoot-bounce problem from a fixed scalar step.
    if alpha0 is None:
        # Mean L1 norm of A's rows (avg of |Σ a_{k,i}| upper bounds).
        a_l1 = float(np.maximum(np.abs(A).sum(axis=1).mean(), 1e-9))
        alpha0 = 1.0 / a_l1

    for t in range(n_iters):
        # coefficient on e: d + Aᵀλ.  shape (n,)
        coef = d + A.T @ lam
        # Dual value at current λ.
        g_val = float(c0) - float(lam @ beta) - float(np.sum(np.abs(coef)))
        if g_val > best_g:
            best_g = g_val
            best_lam = lam.copy()
        if hist is not None:
            hist.append(g_val)
        # Subgradient: ∂g/∂λ_k = -β_k - Σ_i A_{k,i} · sign(coef_i)
        sub = -beta - A @ np.sign(coef)
        # Diminishing step (Robbins-Monro).
        alpha = alpha0 / np.sqrt(t + 1)
        lam = np.maximum(0.0, lam + alpha * sub)

    if return_history:
        return best_g, best_lam, {'g_per_iter': hist}
    return best_g, best_lam


def lag_subgrad_max(d, c0, A, beta, n_iters=100, alpha0=0.5, return_history=False):
    """Sound UPPER BOUND for max d·e + c0 over the same polytope.

    Just calls lag_subgrad_min with -d, -c0, then negates the result.
    """
    if return_history:
        g, lam, info = lag_subgrad_min(-d, -float(c0), A, beta, n_iters,
                                         alpha0, return_history=True)
        info['g_per_iter'] = [-x for x in info['g_per_iter']]
        return -g, lam, info
    g, lam = lag_subgrad_min(-d, -float(c0), A, beta, n_iters, alpha0)
    return -g, lam
