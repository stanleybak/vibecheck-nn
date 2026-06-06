"""Test the OOM-safe batched wrapper: inject a fake CUDA OOM for any chunk larger
than a cap, and verify the wrapper (a) halves down until it fits, (b) returns
states identical to an unconstrained batched build, (c) records the smaller chunk."""
import numpy as np, torch, onnx, sys
from onnx import helper, TensorProto
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
sys.path.insert(0,'/home/stan/repositories/vibecheck-nn/scratch/reverse_g')
import reverse_batched as RB

# residual toy
C,H,W,k=4,8,8,3; g=np.random.default_rng(11)
def f32(a): return a.astype(np.float32)
w1=f32(g.standard_normal((C,C,k,k)));b1=f32(g.standard_normal(C)*.1)
w2=f32(g.standard_normal((C,C,k,k)));b2=f32(g.standard_normal(C)*.1)
nflat=C*H*W;nout=5; wf=f32(g.standard_normal((nout,nflat)));bf=f32(g.standard_normal(nout)*.1)
def init(n,a): return helper.make_tensor(n,TensorProto.FLOAT,a.shape,a.flatten())
nodes=[helper.make_node('Conv',['X','w1','b1'],['c1'],kernel_shape=[k,k],pads=[1,1,1,1]),
 helper.make_node('Relu',['c1'],['r1']),
 helper.make_node('Conv',['r1','w2','b2'],['c2'],kernel_shape=[k,k],pads=[1,1,1,1]),
 helper.make_node('Add',['c2','X'],['a']), helper.make_node('Relu',['a'],['r2']),
 helper.make_node('Flatten',['r2'],['fl'],axis=1),
 helper.make_node('Gemm',['fl','wf','bf'],['Y'],transB=1)]
gr=helper.make_graph(nodes,'res',[helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,C,H,W])],
 [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,nout])],
 [init('w1',w1),init('b1',b1),init('w2',w2),init('b2',b2),init('wf',wf),init('bf',bf)])
m=helper.make_model(gr,opset_imports=[helper.make_opsetid('',13)]);m.ir_version=8
onnx.save(m,'/tmp/toy_res.onnx')
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dt=torch.float64
graph=load_onnx('/tmp/toy_res.onnx'); gg=graph.gpu_graph(dev,dt)
n_in=C*H*W; c0=g.standard_normal(n_in)*.3; eps=0.1
xl=torch.tensor(c0-eps,dtype=dt,device=dev); xh=torch.tensor(c0+eps,dtype=dt,device=dev)
bbr_t,_=_forward_zonotope_graph(xl,xh,gg,dev,dt)
bbr={L:(bbr_t[L][0].cpu().numpy(),bbr_t[L][1].cpu().numpy()) for L in bbr_t}
Dn=8
alphas=[{L:torch.tensor(np.random.default_rng(100*d+L+1).uniform(.05,.95,bbr[L][0].shape),dtype=dt,device=dev) for L in bbr} for d in range(Dn)]

# ground truth: unconstrained batched
truth=RB.build_states_reverse_batched(gg,xl,xh,bbr,alphas,dev,dt)

# inject fake OOM for chunks > CAP
CAP=3; _orig=RB.build_states_reverse_batched
def fake(gg,xl,xh,bbr,a,dev,dt):
    if len(a)>CAP:
        raise torch.cuda.OutOfMemoryError("injected")
    return _orig(gg,xl,xh,bbr,a,dev,dt)
RB.build_states_reverse_batched=fake
bench={}
safe=RB.build_states_reverse_batched_safe(gg,xl,xh,bbr,alphas,dev,dt,_bench=bench)
RB.build_states_reverse_batched=_orig

# verify: correct + chunk shrank to <= CAP
ng=truth[0]['n_gens']; worst=0.0
for d in range(Dn):
    T={(u['layer_idx'],u['neuron_idx']):u for u in truth[d]['unstable_list']}
    S={(u['layer_idx'],u['neuron_idx']):u for u in safe[d]['unstable_list']}
    assert set(T)==set(S), f"dir {d} neuron set mismatch"
    for key in T:
        da=np.zeros(ng);db=np.zeros(ng)
        da[T[key]['row_indices']]=T[key]['row_values']; db[S[key]['row_indices']]=S[key]['row_values']
        worst=max(worst,np.abs(da-db).max())
print(f"D={Dn} CAP={CAP} -> final_chunk={bench['final_chunk']} n_chunks={bench['n_chunks']}")
print(f"safe vs unconstrained-batched: max|Δrow|={worst:.2e}, all neuron sets match")
ok = (bench['final_chunk']<=CAP) and (bench['final_chunk']>=1) and (worst<1e-12) and (len(safe)==Dn)
print("RESULT:", "PASS" if ok else "FAIL")
