"""explore02: independent EXACT big-M MILP for a safenlp case, built straight
from onnx weights + valid interval pre-activation bounds. Ground-truth answer
for whether the unsafe region (Y_0 <= Y_1) is reachable, independent of
vibecheck's racing. Net: 30 -> W1(128x30)+b1 -> ReLU -> W2(2x128)+b2 -> 2."""
import sys, re
import numpy as np
import onnx
from onnx import numpy_helper
import gurobipy as grb

net = sys.argv[1]; spec = sys.argv[2]
g = onnx.load(net)
init = {i.name: numpy_helper.to_array(i) for i in g.graph.initializer}
# find the two weight matrices by shape: (30,128) and (128,2) (or transposed)
mats = [v for v in init.values() if v.ndim == 2]
biases = [v for v in init.values() if v.ndim == 1]
# W1: maps 30->128, W2: maps 128->2
def find(shape_set):
    for v in mats:
        if set(v.shape) == shape_set: return v
    raise SystemExit(f'no mat with shape {shape_set}')
W1 = find({30,128}); W2 = find({128,2})
# orient so y = x @ W  (row x). MatMul in onnx: x(1,30) @ W(30,128) -> (1,128)
if W1.shape != (30,128): W1 = W1.T
if W2.shape != (128,2): W2 = W2.T
b1 = next(v for v in biases if v.shape[0]==128)
b2 = next(v for v in biases if v.shape[0]==2)

# parse vnnlib input box
lo = np.full(30, -np.inf); hi = np.full(30, np.inf)
txt = open(spec).read()
for m in re.finditer(r'\(assert \(>= X_(\d+) ([-\d.eE]+)\)\)', txt):
    lo[int(m.group(1))] = float(m.group(2))
for m in re.finditer(r'\(assert \(<= X_(\d+) ([-\d.eE]+)\)\)', txt):
    hi[int(m.group(1))] = float(m.group(2))
assert np.isfinite(lo).all() and np.isfinite(hi).all(), 'incomplete box'

# valid interval pre-activation bounds for z = x@W1 + b1
W1p = np.maximum(W1,0); W1n = np.minimum(W1,0)
z_lo = lo @ W1p + hi @ W1n + b1
z_hi = hi @ W1p + lo @ W1n + b1

m = grb.Model(); m.Params.OutputFlag = 0
x = [m.addVar(lb=lo[i], ub=hi[i]) for i in range(30)]
a = []
for j in range(128):
    zj = grb.quicksum(x[i]*float(W1[i,j]) for i in range(30)) + float(b1[j])
    if z_hi[j] <= 0:            # dead
        a.append(0.0)
    elif z_lo[j] >= 0:         # active
        aj = m.addVar(lb=0); m.addConstr(aj == zj); a.append(aj)
    else:                       # unstable: exact big-M
        aj = m.addVar(lb=0, ub=float(z_hi[j])); s = m.addVar(vtype=grb.GRB.BINARY)
        m.addConstr(aj >= zj); m.addConstr(aj >= 0)
        m.addConstr(aj <= float(z_hi[j])*s)
        m.addConstr(aj <= zj - float(z_lo[j])*(1-s))
        a.append(aj)
# output Y = a@W2 + b2 ; minimize Y_0 - Y_1
y0 = grb.quicksum((a[j] if not isinstance(a[j],float) else a[j])*float(W2[j,0]) for j in range(128)) + float(b2[0])
y1 = grb.quicksum((a[j] if not isinstance(a[j],float) else a[j])*float(W2[j,1]) for j in range(128)) + float(b2[1])
m.setObjective(y0 - y1, grb.GRB.MINIMIZE)
m.Params.TimeLimit = 60
m.optimize()
n_unstable = int(((z_lo<0)&(z_hi>0)).sum())
print(f'n_unstable={n_unstable}  status={m.Status}  exact_min(Y0-Y1)={m.ObjVal:.6f}  bound={m.ObjBound:.6f}')
print('VERDICT:', 'unsat (Y0>Y1 always)' if m.ObjBound > 0 else 'sat (counterexample exists)')
