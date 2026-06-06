"""Residual block through gg: X->conv1(pad1)->relu1->conv2(pad1)->Add(.,X)->relu2
->flatten->gemm. Build forward ground-truth state + reverse state, compare."""
import numpy as np, torch, onnx, sys
from onnx import helper, TensorProto
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck import alpha_crown as ac, verify_gen_lp
from vibecheck.verify_graph import _serialize_gg_ops
sys.path.insert(0,'/home/stan/repositories/vibecheck-nn/scratch/reverse_g')
from reverse_build import build_state_reverse

C,H,W=4,8,8; k=3
g=np.random.default_rng(11)
def f32(a): return a.astype(np.float32)
w1=f32(g.standard_normal((C,C,k,k))); b1=f32(g.standard_normal(C)*.1)
w2=f32(g.standard_normal((C,C,k,k))); b2=f32(g.standard_normal(C)*.1)
nflat=C*H*W; nout=5
wf=f32(g.standard_normal((nout,nflat))); bf=f32(g.standard_normal(nout)*.1)
def init(n,a): return helper.make_tensor(n,TensorProto.FLOAT,a.shape,a.flatten())
nodes=[
 helper.make_node('Conv',['X','w1','b1'],['c1'],kernel_shape=[k,k],pads=[1,1,1,1]),
 helper.make_node('Relu',['c1'],['r1']),
 helper.make_node('Conv',['r1','w2','b2'],['c2'],kernel_shape=[k,k],pads=[1,1,1,1]),
 helper.make_node('Add',['c2','X'],['a']),        # residual add
 helper.make_node('Relu',['a'],['r2']),
 helper.make_node('Flatten',['r2'],['fl'],axis=1),
 helper.make_node('Gemm',['fl','wf','bf'],['Y'],transB=1),
]
gr=helper.make_graph(nodes,'res',
 [helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,C,H,W])],
 [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,nout])],
 [init('w1',w1),init('b1',b1),init('w2',w2),init('b2',b2),init('wf',wf),init('bf',bf)])
m=helper.make_model(gr,opset_imports=[helper.make_opsetid('',13)]); m.ir_version=8
onnx.save(m,'/tmp/toy_res.onnx')

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dt=torch.float64
graph=load_onnx('/tmp/toy_res.onnx'); gg=graph.gpu_graph(dev,dt)
print("ops:",[(o['name'],o['type'],o.get('layer_idx'),o.get('is_merge')) for o in gg['ops']])
n_in=C*H*W; c0=g.standard_normal(n_in)*.3; eps=0.1
xl=torch.tensor(c0-eps,dtype=dt,device=dev); xh=torch.tensor(c0+eps,dtype=dt,device=dev)
bbr_t,_=_forward_zonotope_graph(xl,xh,gg,dev,dt)
bbr={L:(bbr_t[L][0].cpu().numpy(),bbr_t[L][1].cpu().numpy()) for L in bbr_t}
alpha={L:torch.tensor(np.random.default_rng(L+5).uniform(.05,.95,bbr[L][0].shape),dtype=dt,device=dev) for L in bbr}
unst={L:torch.as_tensor(np.where((bbr[L][0]<0)&(bbr[L][1]>0))[0],dtype=torch.long,device=dev) for L in bbr}
z,pre=ac.forward_zono_dir_adaptive(xl,xh,gg,alpha,bbr,dev,dt,settings=None,unstable_per_layer=unst)
st_f=verify_gen_lp.state_from_alpha_zono(z,pre,alpha,bbr,c0-eps,c0+eps,_serialize_gg_ops(gg),gg['input_name'],gg['ops'][-1]['name'],unstable_per_layer=unst)
st_r=build_state_reverse(gg,xl,xh,bbr,alpha,dev,dt)
print(f"fwd n_gens={st_f['n_gens']} nu={len(st_f['unstable_list'])} | rev n_gens={st_r['n_gens']} nu={len(st_r['unstable_list'])}")
ng=max(st_f['n_gens'],st_r['n_gens'])
F={(u['layer_idx'],u['neuron_idx']):u for u in st_f['unstable_list']}
R={(u['layer_idx'],u['neuron_idx']):u for u in st_r['unstable_list']}
keys=sorted(set(F)&set(R)); wl=wm=wc=wr=0.0; ne=0
for key in keys:
    a,b=F[key],R[key]
    wl=max(wl,abs(float(a['lam'])-float(b['lam']))); wm=max(wm,abs(float(a['mu'])-float(b['mu'])))
    wc=max(wc,abs(float(a['c_in'])-float(b['c_in'])))
    ne+= int(a['e_new_col'])!=int(b['e_new_col'])
    da=np.zeros(ng);db=np.zeros(ng)
    da[np.asarray(a['row_indices'],int)]=np.asarray(a['row_values'],float)
    db[np.asarray(b['row_indices'],int)]=np.asarray(b['row_values'],float)
    wr=max(wr,np.abs(da-db).max())
print(f"common {len(keys)} (F{len(F)} R{len(R)}) | max|Δlam|={wl:.2e} Δmu={wm:.2e} Δc_in={wc:.2e} enc_bad={ne} Δrow={wr:.2e}")
print("RESULT:","PASS" if max(wl,wm,wc,wr)<1e-6 and ne==0 and len(keys)==len(F)==len(R) else "FAIL")

# --- compare output objective (obj_c_out, obj_G_out) ---
import scipy.sparse as sp
oc_f=np.asarray(st_f['obj_c_out'],float); oc_r=np.asarray(st_r['obj_c_out'],float)
og_f=st_f['obj_G_out_csr'].toarray() if sp.issparse(st_f['obj_G_out_csr']) else np.asarray(st_f['obj_G_out_csr'])
og_r=st_r['obj_G_out_csr'].toarray() if sp.issparse(st_r['obj_G_out_csr']) else np.asarray(st_r['obj_G_out_csr'])
# pad to same n_gens
ngo=max(og_f.shape[1],og_r.shape[1])
def padc(a,n): 
    return np.pad(a,((0,0),(0,n-a.shape[1]))) if a.shape[1]<n else a
e_oc=np.abs(oc_f-oc_r).max()
e_og=np.abs(padc(og_f,ngo)-padc(og_r,ngo)).max()
print(f"obj: max|Δc_out|={e_oc:.2e} max|ΔG_out|={e_og:.2e}")
print("OBJ RESULT:", "PASS" if max(e_oc,e_og)<1e-6 else "FAIL")
