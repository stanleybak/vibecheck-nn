"""Stage A: validate the reverse-mode G construction on a raw DENSE chain
(fc-relu-fc-relu-fc). Proves the algorithm: G_backward == G_forward.

Forward (mirrors vibecheck's apply_relu_custom semantics exactly):
  z1 = W1 c0 + b1 ;  G_z1 = W1 G0
  a1 = lam1*z1 + shift1 ; G_a1 = [lam1*G_z1 | diag(mu1)[:,unstable1]]
  ... and so on. Pre-ReLU snapshot rows G_z{L}[unstable@L] are what the
  per-query state stores (row_indices/row_values per unstable neuron).

Reverse: for each unstable neuron k at layer L, backward from z_k through
  linear^T and ReLU (xlam) to the input, emitting slack columns = mu * the
  post-ReLU backward signal at each earlier unstable ReLU.
"""
import numpy as np
rng = np.random.default_rng(0)

def relu_relax(lo, hi):
    """min-area: lam=hi/(hi-lo), mu=-hi*lo/(2(hi-lo)) for unstable; else lam=0/1,mu=0."""
    ust = (lo < 0) & (hi > 0); dead = hi <= 0
    den = np.where(ust, hi - lo, 1.0)
    lam = np.where(ust, hi/den, np.where(dead, 0.0, 1.0))
    mu = np.where(ust, -hi*lo/(2*den), 0.0)
    return lam, mu, ust

# ---- toy net ----
n0, n1, n2, nout = 6, 8, 7, 4
W1 = rng.standard_normal((n1, n0)); b1 = rng.standard_normal(n1)*0.3
W2 = rng.standard_normal((n2, n1)); b2 = rng.standard_normal(n2)*0.3
W3 = rng.standard_normal((nout, n2)); b3 = rng.standard_normal(nout)*0.3
c0 = rng.standard_normal(n0)*0.5
r0 = np.abs(rng.standard_normal(n0))*0.5 + 0.2   # input radii (diagonal G0)
G0 = np.diag(r0)                                  # (n0, n0)

# ---- FORWARD ----
z1 = W1 @ c0 + b1; Gz1 = W1 @ G0                  # (n1,), (n1,n0)
lo1 = z1 - np.abs(Gz1).sum(1); hi1 = z1 + np.abs(Gz1).sum(1)
lam1, mu1, ust1 = relu_relax(lo1, hi1)
u1 = np.where(ust1)[0]
a1 = lam1*z1 + mu1                                # shift=mu (matches forward: center=lam*z+mu)
# G_a1: input cols scaled by lam1, plus a slack col per unstable neuron (mu at that neuron)
Ga1 = np.concatenate([lam1[:,None]*Gz1, np.zeros((n1, len(u1)))], 1)
for c, k in enumerate(u1): Ga1[k, n0 + c] = mu1[k]

z2 = W2 @ a1 + b2; Gz2 = W2 @ Ga1                 # (n2,), (n2, n0+nu1)
lo2 = z2 - np.abs(Gz2).sum(1); hi2 = z2 + np.abs(Gz2).sum(1)
lam2, mu2, ust2 = relu_relax(lo2, hi2)
u2 = np.where(ust2)[0]

# forward pre-ReLU snapshots (the rows the state stores)
fwd_Gz1 = Gz1[u1]                                 # (nu1, n0)
fwd_Gz2 = Gz2[u2]                                 # (nu2, n0+nu1)
print(f"net: n0={n0} n1={n1} n2={n2} | unstable: L1={len(u1)} L2={len(u2)} | n_gens after L2 = {n0+len(u1)}")

# ---- REVERSE: build the same rows backward ----
# layer-1 unstable rows: z1 = W1 c0 (+b1); only input gens. backward k: W1^T e_k, dot G0.
rev_Gz1 = np.zeros((len(u1), n0))
for i, k in enumerate(u1):
    b_in = W1[k, :]                # sensitivity of z1_k to input pre-perturb
    rev_Gz1[i] = b_in * r0         # input cols (G0 diagonal)

# layer-2 unstable rows: backward from z2_k through W2^T -> a1 (slack1 = mu1*b_a1[u1]) ->
#   xlam1 -> z1 -> W1^T -> input (cols = b_input * r0)
rev_Gz2 = np.zeros((len(u2), n0 + len(u1)))
for i, k in enumerate(u2):
    b_a1 = W2[k, :]                          # sensitivity of z2_k to a1 (post-relu1)
    # slack-1 columns: mu1 at unstable1, scaled by b_a1 at that neuron
    for c, j in enumerate(u1):
        rev_Gz2[i, n0 + c] = b_a1[j] * mu1[j]
    b_z1 = b_a1 * lam1                        # back through relu1 (a1 = lam1*z1)
    b_in = W1.T @ b_z1                        # back through W1
    rev_Gz2[i, :n0] = b_in * r0              # input cols

# ---- compare ----
e1 = np.abs(fwd_Gz1 - rev_Gz1).max()
e2 = np.abs(fwd_Gz2 - rev_Gz2).max()
print(f"L1 rows max|fwd-rev| = {e1:.2e}")
print(f"L2 rows max|fwd-rev| = {e2:.2e}")
print("RESULT:", "PASS" if max(e1, e2) < 1e-10 else "FAIL")
