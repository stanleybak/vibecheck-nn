"""Stage A4: skip connection (ResNet add). z_add = main(x) + skip(x), where
main = fc2(relu1(fc1(x))), skip = fc_s(x). Pre-relu2 = z_add. Its row depends on
the input via BOTH paths (summed) and on slack1 (relu1, main path only).
Reverse: fan the backward to both branches; slacks captured only where ReLUs are."""
import numpy as np
rng = np.random.default_rng(3)
def relu_relax(lo, hi):
    ust=(lo<0)&(hi>0); dead=hi<=0; den=np.where(ust,hi-lo,1.0)
    lam=np.where(ust,hi/den,np.where(dead,0.0,1.0)); mu=np.where(ust,-hi*lo/(2*den),0.0)
    return lam,mu,ust
n0,n1,n2 = 6,8,7      # n2 = add width (main fc2 out == skip out == n2)
W1=rng.standard_normal((n1,n0)); b1=rng.standard_normal(n1)*.3
W2=rng.standard_normal((n2,n1)); b2=rng.standard_normal(n2)*.3
Ws=rng.standard_normal((n2,n0)); bs=rng.standard_normal(n2)*.3   # skip
c0=rng.standard_normal(n0)*.5; r0=np.abs(rng.standard_normal(n0))*.5+.2; G0=np.diag(r0)

# ---- FORWARD ----
z1=W1@c0+b1; Gz1=W1@G0
lo=z1-np.abs(Gz1).sum(1); hi=z1+np.abs(Gz1).sum(1)
lam1,mu1,ust1=relu_relax(lo,hi); u1=np.where(ust1)[0]
a1=lam1*z1+mu1; Ga1=np.concatenate([lam1[:,None]*Gz1, np.zeros((n1,len(u1)))],1)
for cc,k in enumerate(u1): Ga1[k,n0+cc]=mu1[k]
ng1=n0+len(u1)
# main fc2 and skip, then add. shared gens = first n0 (input); main adds slack1 cols.
zmain=W2@a1+b2; Gmain=W2@Ga1                 # (n2, ng1)
zskip=Ws@c0+bs; Gskip=Ws@G0                  # (n2, n0)
z_add = zmain+zskip
# merge: shared first n0 cols add; main has extra slack1 cols (skip has none there)
Gadd = Gmain.copy(); Gadd[:, :n0] += Gskip   # shared input cols sum; slack1 cols stay
lo2=z_add-np.abs(Gadd).sum(1); hi2=z_add+np.abs(Gadd).sum(1)
lam2,mu2,ust2=relu_relax(lo2,hi2); u2=np.where(ust2)[0]
fwd_rows = Gadd[u2].copy()                    # (nu2, ng1)
print(f"skip net: n0={n0} n1={n1} n2={n2} | unstable L1={len(u1)} add={len(u2)} | n_gens={ng1}")

# ---- REVERSE from z_add neurons ----
rev=np.zeros((len(u2), ng1))
for i,k in enumerate(u2):
    # main branch backward
    b_a1 = W2[k,:]                            # at a1 (post-relu1)
    for cc,j in enumerate(u1): rev[i,n0+cc] = b_a1[j]*mu1[j]   # slack1
    b_z1 = b_a1*lam1
    b_in_main = W1.T @ b_z1
    # skip branch backward
    b_in_skip = Ws[k,:]
    # input cols = (main + skip) * r0
    rev[i,:n0] = (b_in_main + b_in_skip) * r0
e=np.abs(fwd_rows-rev).max()
print(f"  add-layer rows max|fwd-rev|={e:.2e}")
print("RESULT:", "PASS" if e<1e-10 else "FAIL")
