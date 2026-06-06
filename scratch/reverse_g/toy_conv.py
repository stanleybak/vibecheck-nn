"""Stage A3: conv layers. Forward convolves the generator batch (K,C,H,W);
reverse uses conv_transpose on the per-unstable-neuron backward signal. Net:
conv1 -> relu1 -> conv2 -> relu2 -> flatten -> fc. Validate pre-relu rows match."""
import torch, numpy as np
torch.manual_seed(0)
dt = torch.float64
def relu_relax(lo, hi):
    ust = (lo < 0) & (hi > 0); dead = hi <= 0
    den = torch.where(ust, hi - lo, torch.ones_like(hi))
    lam = torch.where(ust, hi/den, torch.where(dead, torch.zeros_like(hi), torch.ones_like(hi)))
    mu = torch.where(ust, -hi*lo/(2*den), torch.zeros_like(hi))
    return lam, mu, ust

# toy conv net (no padding, stride 1)
Cin, H, W = 2, 6, 6
C1, C2 = 3, 4; ksz = 3
k1 = torch.randn(C1, Cin, ksz, ksz, dtype=dt); bk1 = torch.randn(C1, dtype=dt)*0.2
k2 = torch.randn(C2, C1, ksz, ksz, dtype=dt); bk2 = torch.randn(C2, dtype=dt)*0.2
# spatial dims: 6 -> 4 -> 2 ; flat after conv2 = C2*2*2 = 16
nflat2 = C2*2*2; nout = 5
Wf = torch.randn(nout, nflat2, dtype=dt); bf = torch.randn(nout, dtype=dt)*0.2
c0 = torch.randn(Cin, H, W, dtype=dt)*0.5
r0 = (torch.rand(Cin, H, W, dtype=dt)*0.4 + 0.1)            # input radii
n0 = Cin*H*W
G0 = torch.zeros(n0, Cin, H, W, dtype=dt)                  # K=n0 generators, each one input pixel
flat0 = r0.reshape(-1)
for i in range(n0):
    G0.reshape(n0, -1)[i, i] = flat0[i]

def conv(x, k, b, stride=1):  # x: (K,Cin,H,W) -> (K,Cout,h,w)
    return torch.nn.functional.conv2d(x, k, bias=b, stride=stride)

# ---- FORWARD ----
c = c0.unsqueeze(0)                                         # (1,Cin,H,W)
G = G0                                                      # (K,Cin,H,W)
def fwd_conv(c, G, k, bk):
    c2 = conv(c, k, bk)[0]                                  # (Cout,h,w)
    G2 = conv(G, k, None)                                   # (K,Cout,h,w)
    return c2, G2
c1m, G1 = fwd_conv(c0.unsqueeze(0), G0, k1, bk1)            # pre-relu1
g1flat = G1.reshape(G1.shape[0], -1).t()                   # (nflat1, K)
c1flat = c1m.reshape(-1)
lo = c1flat - g1flat.abs().sum(1); hi = c1flat + g1flat.abs().sum(1)
lam1, mu1, ust1 = relu_relax(lo, hi); u1 = torch.where(ust1)[0]
fwd_rows1 = g1flat[u1].clone()                             # (nu1, n0)
# apply relu1
a1flat = lam1*c1flat + mu1
Ga1 = lam1[:,None]*g1flat                                  # (nflat1, K)
slack1 = torch.zeros(g1flat.shape[0], len(u1), dtype=dt)
for cc,k in enumerate(u1): slack1[k,cc] = mu1[k]
Ga1 = torch.cat([Ga1, slack1], 1)                          # (nflat1, K+nu1)
ng1 = G0.shape[0] + len(u1)
# conv2 on a1 (reshape to spatial)
C1_, h1, w1 = G1.shape[1:]
a1sp = a1flat.reshape(1, C1_, h1, w1)
Ga1_sp = Ga1.t().reshape(-1, C1_, h1, w1)                  # (K+nu1, C1,h,w)
c2m, G2 = fwd_conv(a1sp, Ga1_sp, k2, bk2)
g2flat = G2.reshape(G2.shape[0], -1).t()                   # (nflat2, K+nu1)
c2flat = c2m.reshape(-1)
lo2 = c2flat - g2flat.abs().sum(1); hi2 = c2flat + g2flat.abs().sum(1)
lam2, mu2, ust2 = relu_relax(lo2, hi2); u2 = torch.where(ust2)[0]
fwd_rows2 = g2flat[u2].clone()                             # (nu2, K+nu1)
print(f"conv net: n0={n0} nflat1={g1flat.shape[0]} nflat2={nflat2} | unstable L1={len(u1)} L2={len(u2)}")

# ---- REVERSE ----
def convT(grad, k, out_spatial, in_spatial):  # vector-Jacobian of conv
    return torch.nn.functional.conv_transpose2d(grad, k)

# layer1 unstable rows: pre-relu1 depends only on input. backward k: convT(unit_k) . G0
rev1 = torch.zeros(len(u1), n0, dtype=dt)
Cout1, h1_, w1_ = c1m.shape
for i,k in enumerate(u1):
    g = torch.zeros(Cout1*h1_*w1_, dtype=dt); g[k]=1.0
    bin_ = convT(g.reshape(1,Cout1,h1_,w1_), k1, None, None)[0]   # (Cin,H,W) sensitivity
    rev1[i] = (bin_.reshape(-1) * r0.reshape(-1))
e1 = (fwd_rows1 - rev1).abs().max().item()

# layer2 unstable rows: backward from z2_k through convT(k2) -> a1 (slack1) -> xlam1 -> z1 -> convT(k1) -> input
rev2 = torch.zeros(len(u2), ng1, dtype=dt)
Cout2, h2_, w2_ = c2m.shape
for i,k in enumerate(u2):
    g = torch.zeros(Cout2*h2_*w2_, dtype=dt); g[k]=1.0
    b_a1 = convT(g.reshape(1,Cout2,h2_,w2_), k2, None, None)[0].reshape(-1)  # at a1 (post-relu1), nflat1
    # slack1 cols
    for cc,j in enumerate(u1): rev2[i, G0.shape[0]+cc] = b_a1[j]*mu1[j]
    b_z1 = b_a1 * lam1                                       # back through relu1
    b_in = convT(b_z1.reshape(1,C1_,h1,w1), k1, None, None)[0].reshape(-1)
    rev2[i, :n0] = b_in * r0.reshape(-1)
e2 = (fwd_rows2 - rev2).abs().max().item()
print(f"  L1 rows max|fwd-rev|={e1:.2e}")
print(f"  L2 rows max|fwd-rev|={e2:.2e}")
print("RESULT:", "PASS" if max(e1,e2) < 1e-10 else "FAIL")
