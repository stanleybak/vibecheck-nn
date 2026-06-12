"""INVPROP-style output-constrained intermediate tightening (alpha_tighten).

`tighten_layer_alpha_crown(..., output_constraints=(W_spec, b_spec))` adds
gamma >= 0 multipliers on the assumed-SAT output rows (W_spec.y + b_spec <= 0)
to the per-target backward objective: for any gamma >= 0,
LB(z_k + gamma.(W y + b)) is a valid lower bound of the CONSTRAINED min of
z_k (Lagrangian duality), and gamma = 0 (the init) recovers the plain
alpha-CROWN bound, so the best-of over iterations can only tighten.

Pins:
  1. analytic toy: x in [-2,2], z = x, y = relu(z), constraint y <= 1.
     Constrained UB(z) = 1 (plain bound: 2). The gamma-ascent must recover
     (close to) 1.
  2. random 2-hidden-layer ReLU net + Gurobi EXACT MILP ground truth:
     constrained bounds must stay SOUND (lb <= exact min, ub >= exact max)
     and no looser than the plain alpha-CROWN bounds.
"""
import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto

from vibecheck.alpha_tighten import tighten_layer_alpha_crown
from vibecheck.onnx_loader import load_onnx
from vibecheck.settings import default_settings


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                              arr.flatten())


def _mk_graph(tmp_path, Ws, bs, name):
    nodes, inits, prev = [], [], 'X'
    for i, (W, b) in enumerate(zip(Ws, bs)):
        nodes.append(helper.make_node('MatMul', [prev, f'W{i}'], [f'm{i}']))
        nodes.append(helper.make_node('Add', [f'm{i}', f'b{i}'], [f'a{i}']))
        inits += [_init(f'W{i}', W), _init(f'b{i}', b)]
        if i < len(Ws) - 1:
            nodes.append(helper.make_node('Relu', [f'a{i}'], [f'r{i}']))
            prev = f'r{i}'
        else:
            nodes[-1] = helper.make_node('Add', [f'm{i}', f'b{i}'], ['Y'])
    g = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                       [1, Ws[0].shape[0]])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [1, Ws[-1].shape[1]])],
        inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / f'{name}.onnx')
    onnx.save(model, path)
    graph = load_onnx(path, dtype=np.float64)
    graph.optimize(default_settings())
    return graph.gpu_graph(torch.device('cpu'), torch.float64)


def _box_bbr(gg, xl, xh):
    """Interval-propagated pre-ReLU bounds per relu layer (loose but sound)."""
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    sb, _ = _forward_zonotope_graph(
        torch.tensor(xl, dtype=torch.float64),
        torch.tensor(xh, dtype=torch.float64),
        gg, torch.device('cpu'), torch.float64)
    return {L: (np.asarray(sb[L][0], np.float64).ravel(),
                np.asarray(sb[L][1], np.float64).ravel()) for L in sb}


def test_analytic_relu_cap():
    """y = relu(z), z = x in [-2,2], constraint y <= 1 => UB(z) -> 1."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        gg = _mk_graph(pathlib.Path(td),
                       [np.eye(1, dtype=np.float32),
                        np.eye(1, dtype=np.float32)],
                       [np.zeros(1, np.float32), np.zeros(1, np.float32)],
                       'cap')
        xl = np.array([-2.0]); xh = np.array([2.0])
        bbr = _box_bbr(gg, xl, xh)
        dev = torch.device('cpu'); dt = torch.float64
        lo_p, hi_p = tighten_layer_alpha_crown(
            gg, xl, xh, bbr, 0, device=dev, dtype=dt, n_iters=80, lr=0.2,
            target_indices=[0])
        assert hi_p[0] == pytest.approx(2.0, abs=1e-6)   # plain bound
        lo_c, hi_c = tighten_layer_alpha_crown(
            gg, xl, xh, bbr, 0, device=dev, dtype=dt, n_iters=200, lr=0.2,
            target_indices=[0],
            output_constraints=(np.array([[1.0]]), np.array([-1.0])))
        assert hi_c[0] <= 2.0 - 0.5, f'no tightening: {hi_c[0]}'
        assert hi_c[0] >= 1.0 - 1e-6, f'UNSOUND: {hi_c[0]} < true max 1.0'
        assert lo_c[0] <= -2.0 + 1e-6   # lb unaffected (constraint inactive)


def test_random_net_sound_vs_gurobi_milp():
    import gurobipy as grb
    import tempfile, pathlib
    rng = np.random.RandomState(3)
    Ws = [rng.randn(3, 6).astype(np.float32) * 0.8,
          rng.randn(6, 5).astype(np.float32) * 0.8,
          rng.randn(5, 2).astype(np.float32) * 0.8]
    bs = [rng.randn(6).astype(np.float32) * 0.3,
          rng.randn(5).astype(np.float32) * 0.3,
          rng.randn(2).astype(np.float32) * 0.3]
    with tempfile.TemporaryDirectory() as td:
        gg = _mk_graph(pathlib.Path(td), Ws, bs, 'rand')
        xl = -np.ones(3); xh = np.ones(3)
        bbr = _box_bbr(gg, xl, xh)
        # constraints: y0 <= y(center)0, y1 <= y(center)1 (feasible: the
        # center input satisfies both with equality)
        xc = np.zeros(3)
        h = np.maximum(xc @ Ws[0].astype(np.float64) + bs[0], 0)
        h = np.maximum(h @ Ws[1].astype(np.float64) + bs[1], 0)
        yc = h @ Ws[2].astype(np.float64) + bs[2]
        W_spec = np.eye(2)
        b_spec = -yc
        dev = torch.device('cpu'); dt = torch.float64
        target_layer = 1
        lo_p, hi_p = tighten_layer_alpha_crown(
            gg, xl, xh, bbr, target_layer, device=dev, dtype=dt,
            n_iters=60, lr=0.1)
        lo_c, hi_c = tighten_layer_alpha_crown(
            gg, xl, xh, bbr, target_layer, device=dev, dtype=dt,
            n_iters=150, lr=0.1,
            output_constraints=(W_spec, b_spec))

        # exact constrained min/max per L1 pre-relu neuron via MILP
        env = grb.Env(empty=True); env.setParam('OutputFlag', 0); env.start()

        def exact(j, sense):
            m = grb.Model(env=env)
            x = m.addMVar(3, lb=-1.0, ub=1.0)
            z1 = m.addMVar(6, lb=-grb.GRB.INFINITY)
            m.addConstr(z1 == Ws[0].astype(np.float64).T @ x
                        + bs[0].astype(np.float64))
            r1 = m.addMVar(6, lb=0.0)
            for k in range(6):
                m.addGenConstrMax(r1[k].item(), [z1[k].item()], 0.0)
            z2 = m.addMVar(5, lb=-grb.GRB.INFINITY)
            m.addConstr(z2 == Ws[1].astype(np.float64).T @ r1
                        + bs[1].astype(np.float64))
            r2 = m.addMVar(5, lb=0.0)
            for k in range(5):
                m.addGenConstrMax(r2[k].item(), [z2[k].item()], 0.0)
            y = m.addMVar(2, lb=-grb.GRB.INFINITY)
            m.addConstr(y == Ws[2].astype(np.float64).T @ r2
                        + bs[2].astype(np.float64))
            m.addConstr(W_spec @ y <= -b_spec)
            m.setObjective(z2[j].item(), sense)
            m.optimize()
            assert m.Status == 2, f'MILP status {m.Status}'
            v = m.ObjVal
            m.dispose()
            return v

        n_tighter = 0
        for j in range(5):
            emin = exact(j, grb.GRB.MINIMIZE)
            emax = exact(j, grb.GRB.MAXIMIZE)
            assert lo_c[j] <= emin + 1e-6, \
                f'UNSOUND lb neuron {j}: {lo_c[j]} > exact {emin}'
            assert hi_c[j] >= emax - 1e-6, \
                f'UNSOUND ub neuron {j}: {hi_c[j]} < exact {emax}'
            # never looser than plain (gamma=0 is in the search space)
            assert lo_c[j] >= lo_p[j] - 1e-7
            assert hi_c[j] <= hi_p[j] + 1e-7
            if lo_c[j] > lo_p[j] + 1e-9 or hi_c[j] < hi_p[j] - 1e-9:
                n_tighter += 1
        assert n_tighter >= 1, 'constraints never tightened anything'
