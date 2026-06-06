"""Build a small ONNX, load via vibecheck, get bbr + the forward ground-truth
alpha-zono state (forward_zono_dir_adaptive + state_from_alpha_zono). This is the
target the reverse walker must reproduce."""
import numpy as np, torch, onnx
from onnx import helper, TensorProto
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck import alpha_crown as ac
from vibecheck import verify_gen_lp
from vibecheck.verify_graph import _serialize_gg_ops

def build_onnx(path):
    # input 1x2x8x8 -> Conv(2->4,3x3) -> Relu -> Conv(4->4,3x3) -> Relu -> Flatten -> Gemm -> out
    Cin,H,W = 2,8,8; C1,C2=4,4; k=3
    g=np.random.default_rng(7)
    w1=g.standard_normal((C1,Cin,k,k)).astype(np.float32); b1=(g.standard_normal(C1)*.1).astype(np.float32)
    w2=g.standard_normal((C2,C1,k,k)).astype(np.float32); b2=(g.standard_normal(C2)*.1).astype(np.float32)
    spatial=(H-2)-2  # 8->6->4
    nflat=C2*spatial*spatial; nout=5
    wf=g.standard_normal((nout,nflat)).astype(np.float32); bf=(g.standard_normal(nout)*.1).astype(np.float32)
    def init(name,arr): return helper.make_tensor(name,TensorProto.FLOAT,arr.shape,arr.flatten())
    nodes=[
        helper.make_node('Conv',['X','w1','b1'],['c1'],kernel_shape=[k,k]),
        helper.make_node('Relu',['c1'],['r1']),
        helper.make_node('Conv',['r1','w2','b2'],['c2'],kernel_shape=[k,k]),
        helper.make_node('Relu',['c2'],['r2']),
        helper.make_node('Flatten',['r2'],['f'],axis=1),
        helper.make_node('Gemm',['f','wf','bf'],['Y'],transB=1),
    ]
    graph=helper.make_graph(nodes,'toy',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,Cin,H,W])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,nout])],
        [init('w1',w1),init('b1',b1),init('w2',w2),init('b2',b2),init('wf',wf),init('bf',bf)])
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])
    m.ir_version=8
    onnx.save(m,path); return Cin,H,W

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dt=torch.float64
Cin,H,W = build_onnx('/tmp/toy_rg.onnx')
graph=load_onnx('/tmp/toy_rg.onnx')
gg=graph.gpu_graph(dev,dt)
print("gg ops:", [(o['name'],o['type'],o.get('layer_idx')) for o in gg['ops']])
n_in=Cin*H*W
c0=np.random.default_rng(9).standard_normal(n_in)*0.3
eps=0.1
xl=torch.tensor(c0-eps,dtype=dt,device=dev); xh=torch.tensor(c0+eps,dtype=dt,device=dev)
bbr_t,_=_forward_zonotope_graph(xl,xh,gg,dev,dt)
bbr={L:(bbr_t[L][0].cpu().numpy(),bbr_t[L][1].cpu().numpy()) for L in bbr_t}
print("bbr layers:", sorted(bbr.keys()), "| sizes:", {L:bbr[L][0].size for L in bbr})
# random valid alpha in [0,1] for unstable neurons
alpha={}
for L,(lo,hi) in bbr.items():
    lo_n=np.asarray(lo); hi_n=np.asarray(hi)
    a=np.random.default_rng(L+1).uniform(0.05,0.95,size=lo_n.shape)
    alpha[L]=torch.tensor(a,dtype=dt,device=dev)
# unstable_per_layer
unst={}
for L,(lo,hi) in bbr.items():
    u=np.where((np.asarray(lo)<0)&(np.asarray(hi)>0))[0]
    unst[L]=torch.as_tensor(u,dtype=torch.long,device=dev)
print("unstable per layer:", {L:int(unst[L].numel()) for L in unst})
# forward ground truth
x_lo_64=xl.cpu().numpy(); x_hi_64=xh.cpu().numpy()
z,pre=ac.forward_zono_dir_adaptive(xl,xh,gg,alpha,bbr,dev,dt,settings=None,unstable_per_layer=unst)
st=verify_gen_lp.state_from_alpha_zono(z,pre,alpha,bbr,x_lo_64,x_hi_64,_serialize_gg_ops(gg),gg['input_name'],gg['ops'][-1]['name'],unstable_per_layer=unst)
print("STATE: n_gens=%d n_input=%d n_unstable=%d"%(st['n_gens'],st['n_input'],len(st['unstable_list'])))
u0=st['unstable_list'][0]
print("sample unstable[0]: keys=",list(u0.keys()))
print("  layer=%s neuron=%s lam=%.4f mu=%.4f c_in=%.4f rowlen=%d e_new_col=%s"%(
    u0['layer_idx'],u0['neuron_idx'],float(u0['lam']),float(u0['mu']),float(u0['c_in']),
    len(np.asarray(u0['row_indices'])),u0['e_new_col']))
import pickle; pickle.dump({'state':st,'bbr':{L:(np.asarray(bbr[L][0]),np.asarray(bbr[L][1])) for L in bbr},
    'alpha':{L:alpha[L].cpu().numpy() for L in alpha},'c0':c0,'eps':eps},open('/tmp/toy_rg_gt.pkl','wb'))
print("ground truth saved to /tmp/toy_rg_gt.pkl")
