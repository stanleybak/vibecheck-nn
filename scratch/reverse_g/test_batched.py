import numpy as np, torch, onnx, sys
from onnx import helper, TensorProto
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
sys.path.insert(0,'/home/stan/repositories/vibecheck-nn/scratch/reverse_g')
from reverse_build import build_state_reverse
from reverse_batched import build_states_reverse_batched

# residual block (reuse builder from test_skip_gg)
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
Dn=5
alphas=[{L:torch.tensor(np.random.default_rng(100*d+L+1).uniform(.05,.95,bbr[L][0].shape),dtype=dt,device=dev) for L in bbr} for d in range(Dn)]

batched=build_states_reverse_batched(gg,xl,xh,bbr,alphas,dev,dt)
ng=batched[0]['n_gens']; worst=0.0; worst_obj=0.0; pat_ok=True
# reference per-direction
seq=[build_state_reverse(gg,xl,xh,bbr,alphas[d],dev,dt) for d in range(Dn)]
for d in range(Dn):
    Fb={(u['layer_idx'],u['neuron_idx']):u for u in batched[d]['unstable_list']}
    Fs={(u['layer_idx'],u['neuron_idx']):u for u in seq[d]['unstable_list']}
    for key in Fs:
        a=Fs[key]; b=Fb[key]
        da=np.zeros(ng);db=np.zeros(ng)
        da[a['row_indices']]=a['row_values']; db[b['row_indices']]=b['row_values']
        worst=max(worst,np.abs(da-db).max())
    oa=seq[d]['obj_G_out_csr'].toarray(); ob=batched[d]['obj_G_out_csr'].toarray()
    worst_obj=max(worst_obj, np.abs(oa-ob).max())
# static-pattern check: row_indices identical across directions (batched)
B0={(u['layer_idx'],u['neuron_idx']):set(u['row_indices'].tolist()) for u in batched[0]['unstable_list']}
for d in range(1,Dn):
    Bd={(u['layer_idx'],u['neuron_idx']):set(u['row_indices'].tolist()) for u in batched[d]['unstable_list']}
    for key in B0:
        if B0[key]!=Bd[key]: pat_ok=False
print(f"batched vs sequential: max|Δrow|={worst:.2e}  max|Δobj|={worst_obj:.2e}")
print(f"static pattern (row_indices identical across {Dn} directions): {pat_ok}")
print("RESULT:", "PASS" if max(worst,worst_obj)<1e-9 and pat_ok else "FAIL")
