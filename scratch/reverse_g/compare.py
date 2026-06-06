"""Compare reverse-mode state vs forward ground truth (saved pkl). Per unstable
neuron: lam, mu, c_in, e_new_col, and the DENSE row (scatter row_values into
row_indices over n_gens)."""
import numpy as np, torch, pickle
from onnx import helper, TensorProto
import onnx
from vibecheck.onnx_loader import load_onnx
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck import alpha_crown as ac
import sys; sys.path.insert(0, '/home/stan/repositories/vibecheck-nn/scratch/reverse_g')
from reverse_build import build_state_reverse

gt = pickle.load(open('/tmp/toy_rg_gt.pkl','rb'))
st_fwd = gt['state']; bbr = gt['bbr']; alpha_np = gt['alpha']; c0 = gt['c0']; eps = gt['eps']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dt = torch.float64
graph = load_onnx('/tmp/toy_rg.onnx'); gg = graph.gpu_graph(dev, dt)
n_in = c0.size
xl = torch.tensor(c0-eps, dtype=dt, device=dev); xh = torch.tensor(c0+eps, dtype=dt, device=dev)
alpha = {L: torch.tensor(alpha_np[L], dtype=dt, device=dev) for L in alpha_np}
st_rev = build_state_reverse(gg, xl, xh, bbr, alpha, dev, dt)

print(f"fwd: n_gens={st_fwd['n_gens']} n_input={st_fwd['n_input']} nu={len(st_fwd['unstable_list'])}")
print(f"rev: n_gens={st_rev['n_gens']} n_input={st_rev['n_input']} nu={len(st_rev['unstable_list'])}")
ng = max(st_fwd['n_gens'], st_rev['n_gens'])
# index both by (layer, neuron)
fwd = {(u['layer_idx'], u['neuron_idx']): u for u in st_fwd['unstable_list']}
rev = {(u['layer_idx'], u['neuron_idx']): u for u in st_rev['unstable_list']}
keys = sorted(set(fwd) & set(rev))
print(f"common neurons: {len(keys)} (fwd {len(fwd)}, rev {len(rev)})")
wlam=wmu=wcin=wenc=wrow=0.0; nbad_enc=0
for key in keys:
    a, b = fwd[key], rev[key]
    wlam = max(wlam, abs(float(a['lam'])-float(b['lam'])))
    wmu = max(wmu, abs(float(a['mu'])-float(b['mu'])))
    wcin = max(wcin, abs(float(a['c_in'])-float(b['c_in'])))
    if int(a['e_new_col']) != int(b['e_new_col']): nbad_enc += 1
    da = np.zeros(ng); db = np.zeros(ng)
    da[np.asarray(a['row_indices'],int)] = np.asarray(a['row_values'],float)
    db[np.asarray(b['row_indices'],int)] = np.asarray(b['row_values'],float)
    wrow = max(wrow, np.abs(da-db).max())
print(f"max|Δlam|={wlam:.2e} max|Δmu|={wmu:.2e} max|Δc_in|={wcin:.2e} e_new_col mismatches={nbad_enc} max|Δrow|={wrow:.2e}")
print("RESULT:", "PASS" if max(wlam,wmu,wcin,wrow)<1e-6 and nbad_enc==0 else "FAIL")
