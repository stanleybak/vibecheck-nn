"""The spec-backward over a SQUARE (`mul_bilinear(a, a)`, same variable on both
sides) must accumulate the gradient from BOTH sides.

The backward used to fetch `ea = ew_at.get(ia)` / `eb = ew_at.get(ib)` up front
and then assign `ew_at[ia]=ea+ew_a; ew_at[ib]=eb+ew_b`. For a self-product
`ia == ib`, the second assignment overwrote the first with the *stale* `eb`,
dropping `ew_a` — so only half the McCormick gradient was applied. That made the
backward lower bound on `-X^2` come out ABOVE the true minimum (an UNSOUND bound
that false-verified SAT adaptive_cc nonlinear-augment cases). The fix re-fetches
`ew_at` inline so `ia == ib` accumulates both sides.
"""
import numpy as np
import onnx
import onnx.helper as oh
import torch
from onnx import TensorProto

from vibecheck.onnx_loader import load_onnx
from vibecheck.settings import default_settings
from vibecheck.forward_lirpa import forward_lirpa_compat_zono_batched as _FL
from vibecheck.verify_zono_bnb import _spec_backward_graph_batched


def _square_net(path):
    # X[1,1] -> a = 1*X (Gemm, so the bilinear input is an INTERMEDIATE node
    # whose box the forward stashes) -> Y = a*a  (mul_bilinear(a, a)).
    inp = oh.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])
    out = oh.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
    nodes = [oh.make_node('Gemm', ['X', 'W', 'b'], ['a'], transB=1),
             oh.make_node('Mul', ['a', 'a'], ['Y'])]
    inits = [oh.make_tensor('W', TensorProto.FLOAT, [1, 1], [1.0]),
             oh.make_tensor('b', TensorProto.FLOAT, [1], [0.0])]
    g = oh.make_graph(nodes, 'g', [inp], [out], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid('', 14)])
    m.ir_version = 7
    onnx.save(m, str(path))
    return str(path)


def test_square_backward_lower_bound_is_sound(tmp_path):
    """backward LB(-X^2) over [-3, 5] must be <= the true min (-25), not -20."""
    path = _square_net(tmp_path / 'sq.onnx')
    g = load_onnx(path)
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    dev, dt = torch.device('cpu'), torch.float64
    gg = g.gpu_graph(dev, dt)
    xl = torch.tensor([[-3.0]], dtype=dt)
    xh = torch.tensor([[5.0]], dtype=dt)
    if hasattr(_FL, 'last_bilinear_op_bounds'):
        _FL.last_bilinear_op_bounds = None
    sb, _ = _FL(xl, xh, gg, dev, dt)
    bob = _FL.last_bilinear_op_bounds
    # lower bound on -Y = -X^2
    w = torch.tensor([-1.0], dtype=dt)
    lb = float(_spec_backward_graph_batched(
        sb, xl, xh, gg, {0: (w, 0.0)}, dev, dt, bilinear_op_bounds=bob)[0, 0])
    xs = np.linspace(-3.0, 5.0, 100001)
    true_min = float((-xs * xs).min())          # -25 at x=5
    assert lb <= true_min + 1e-6, (
        f'unsound backward LB(-X^2)={lb} > true min {true_min} '
        '(same-variable gradient dropped one side)')
