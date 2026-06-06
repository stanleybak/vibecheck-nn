"""Stage A2: generalize to an N-relu dense chain. Critical test: a slack from
ReLU L' must reach a DEEP layer L'' (L''>L'+1) scaled by the intermediate path
(the lam's and W's between). Reverse must reproduce these multi-hop slack cols."""
import numpy as np
rng = np.random.default_rng(1)

def relu_relax(lo, hi):
    ust = (lo < 0) & (hi > 0); dead = hi <= 0
    den = np.where(ust, hi - lo, 1.0)
    lam = np.where(ust, hi/den, np.where(dead, 0.0, 1.0))
    mu = np.where(ust, -hi*lo/(2*den), 0.0)
    return lam, mu, ust

dims = [6, 9, 8, 10, 7, 5]            # input + 4 hidden (relu) + output (no relu)
n_relu = len(dims) - 2
Ws = [rng.standard_normal((dims[i+1], dims[i])) for i in range(len(dims)-1)]
bs = [rng.standard_normal(dims[i+1])*0.3 for i in range(len(dims)-1)]
c0 = rng.standard_normal(dims[0])*0.5
r0 = np.abs(rng.standard_normal(dims[0]))*0.5 + 0.2
G0 = np.diag(r0)

# ---- FORWARD: track full G, snapshot pre-relu unstable rows per layer ----
c = c0.copy(); G = G0.copy()
lams=[]; mus=[]; us=[]; fwd_rows=[]; slack_offsets=[]
ngen = dims[0]
for L in range(n_relu):
    c = Ws[L] @ c + bs[L]; G = Ws[L] @ G          # pre-relu z
    lo = c - np.abs(G).sum(1); hi = c + np.abs(G).sum(1)
    lam, mu, ust = relu_relax(lo, hi); u = np.where(ust)[0]
    fwd_rows.append(G[u].copy())                  # snapshot pre-relu rows
    slack_offsets.append(ngen)
    # apply relu: scale rows by lam, append slack cols
    c = lam*c + mu
    G = np.concatenate([lam[:,None]*G, np.zeros((dims[L+1], len(u)))], 1)
    for cc, k in enumerate(u): G[k, ngen+cc] = mu[k]
    ngen += len(u)
    lams.append(lam); mus.append(mu); us.append(u)
print(f"dims={dims} n_relu={n_relu} unstable={[len(u) for u in us]} n_gens={ngen}")

# ---- REVERSE: per layer L, backward from its unstable neurons ----
errs=[]
for L in range(n_relu):
    u = us[L]; row = np.zeros((len(u), ngen))
    for i, k in enumerate(u):
        b = np.zeros(dims[L+1]); b[k] = 1.0        # unit at pre-relu neuron k of layer L
        # walk back from layer L's pre-activation to the input
        # first hop: through W_L (pre-relu_L depends on post-relu_{L-1})
        b = Ws[L].T @ b                            # sensitivity at post-relu_{L-1} (or input if L==0)
        for Lp in range(L-1, -1, -1):              # earlier relu layers
            # slack cols of relu Lp: mu_Lp at unstable, scaled by current b
            for cc, j in enumerate(us[Lp]):
                row[i, slack_offsets[Lp]+cc] = b[j] * mus[Lp][j]
            b = b * lams[Lp]                       # back through relu Lp (post=lam*pre)
            b = Ws[Lp].T @ b                       # back through W_Lp
        row[i, :dims[0]] = b * r0                  # input cols (b is at input now)
    e = np.abs(fwd_rows[L] - row[:, :fwd_rows[L].shape[1]]).max() if len(u) else 0.0
    # also check the snapshot only has cols up to its own n_gens (later slacks are 0)
    later = np.abs(row[:, fwd_rows[L].shape[1]:]).max() if len(u) and row.shape[1]>fwd_rows[L].shape[1] else 0.0
    errs.append(max(e, later))
    print(f"  L{L}: {len(u)} unstable, max|fwd-rev|={e:.2e}, spurious-later-cols={later:.2e}")
print("RESULT:", "PASS" if max(errs) < 1e-9 else "FAIL")
