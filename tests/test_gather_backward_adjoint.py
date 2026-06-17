"""Soundness regression for the slice/gather backward-CROWN adjoint.

The adjoint of a forward gather ``out[k] = in[flat_idx[k]]`` is scatter-ADD:
``in_grad[j] = sum_{k: idx[k]==j} out_grad[k]``. A FAN-OUT gather (duplicate
flat_idx — one input feeds several outputs, as AC power-flow nets do gathering a
bus voltage into many line equations) therefore SUMS the per-output backward
weights. Implementing it with ``index_copy_`` (overwrite) instead of
``index_add_`` silently drops all but the last contribution, producing an
unsound (too-small-magnitude) backward coefficient. This was a real bug (fixed
in `_spec_backward_graph` + the batched/alpha_crown variants); these tests pin
the fix at the lowest level — they FAIL with ``index_copy_`` and pass with
``index_add_``.
"""
import numpy as np
import torch

from vibecheck.verify_zono_bnb import _spec_backward_graph


def _input_coeff(flat_idx, n_in, w):
    """Backward-CROWN the linear functional ``w · gather(x)`` to input space and
    return its per-input coefficient vector (d/dx of ``w · gather(x)``)."""
    gg = {
        'ops': [{
            'name': 'g', 'type': 'gather', 'inputs': ['x'],
            'flat_idx': list(flat_idx),
            'in_shapes_nd': [(n_in,)],
            'out_shape_nd': (len(flat_idx),),
        }],
        'input_name': 'x',
    }
    xl = torch.zeros(n_in, dtype=torch.float64)
    xh = torch.ones(n_in, dtype=torch.float64)
    spec_ew = {0: (torch.tensor(w, dtype=torch.float64), 0.0)}
    _, _, input_linear = _spec_backward_graph(
        {}, xl, xh, gg, spec_ew, [0], 0, 'cpu', torch.float64,
        return_input_linear=True)
    return input_linear[0][0]   # ew_inp (n_in,)


def test_gather_fanout_backward_sums_contributions():
    # out = x[[0, 0, 1]]  -> out0=x0, out1=x0, out2=x1.
    # w = [1,1,1]  ->  w·out = x0 + x0 + x1 = 2*x0 + x1  ->  d/dx = [2, 1].
    # index_add_ (correct): [2, 1].  index_copy_ (bug): [1, 1] (x0 overwritten).
    coeff = _input_coeff([0, 0, 1], 2, [1.0, 1.0, 1.0])
    assert np.allclose(coeff, [2.0, 1.0]), coeff


def test_gather_fanout_weighted():
    # out = x[[2, 2, 2, 0]], w = [1, 1, 1, 4]:
    #   w·out = 3*x2 + 4*x0  ->  d/dx = [4, 0, 3].
    coeff = _input_coeff([2, 2, 2, 0], 3, [1.0, 1.0, 1.0, 4.0])
    assert np.allclose(coeff, [4.0, 0.0, 3.0]), coeff


def test_gather_unique_indices_unchanged():
    # No duplicate indices (a permutation / slice): index_add_ == index_copy_,
    # so this case is unaffected by the fix.
    #   out = x[[1, 0]], w = [3, 5]  ->  w·out = 3*x1 + 5*x0  ->  d/dx = [5, 3].
    coeff = _input_coeff([1, 0], 2, [3.0, 5.0])
    assert np.allclose(coeff, [5.0, 3.0]), coeff
