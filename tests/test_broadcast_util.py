"""Tests for the shared broadcast util — index/shape logic must match torch's
own numpy-style broadcasting exactly (it underpins sound bound expansion)."""
import numpy as np
import pytest
import torch

from vibecheck.broadcast_util import (reconstruct_live_shape,
                                      broadcast_row_index, broadcast_rows,
                                      broadcast_rows_backward)

_F64 = torch.float64


@pytest.mark.parametrize('live_sh,out_sh', [
    ((3, 1), (3, 4)),
    ((1, 4), (3, 4)),
    ((1, 1), (3, 4)),
    ((2, 1, 5), (2, 6, 5)),
    ((1,), (7,)),
    ((), (5,)),
])
def test_index_matches_torch_broadcast(live_sh, out_sh):
    n_live = int(np.prod(live_sh)) if live_sh else 1
    v = torch.arange(n_live, dtype=_F64) + 0.5
    # ground truth: torch broadcast of the value tensor
    gt = torch.broadcast_to(v.reshape(live_sh if live_sh else (1,)),
                            out_sh).reshape(-1)
    idx = broadcast_row_index(live_sh, out_sh)
    assert torch.equal(v.index_select(0, idx), gt)
    # broadcast_rows on a 2-D (rows, feat) tensor expands the row axis
    feat = torch.randn(n_live, 3, dtype=_F64)
    got = broadcast_rows(feat, live_sh, out_sh, 0)
    exp = torch.broadcast_to(
        feat.reshape((live_sh if live_sh else (1,)) + (3,)),
        tuple(out_sh) + (3,)).reshape(-1, 3)
    assert torch.equal(got, exp)


@pytest.mark.parametrize('live_sh,out_sh', [
    ((3, 1), (3, 4)), ((1, 4), (3, 4)), ((2, 1, 5), (2, 6, 5)), ((1,), (7,)),
])
def test_broadcast_backward_is_adjoint(live_sh, out_sh):
    # <broadcast_rows(a), b> == <a, broadcast_rows_backward(b)>  (adjoint pair)
    n_live = int(np.prod(live_sh))
    n_out = int(np.prod(out_sh))
    a = torch.randn(n_live, dtype=_F64)
    b = torch.randn(n_out, dtype=_F64)
    fwd = broadcast_rows(a, live_sh, out_sh, 0)            # (n_out,)
    bwd = broadcast_rows_backward(b, live_sh, out_sh, 0)   # (n_live,)
    assert abs(float(fwd @ b) - float(a @ bwd)) < 1e-9
    # backward of an all-ones output = the multiplicity of each live row
    mult = broadcast_rows_backward(torch.ones(n_out, dtype=_F64),
                                   live_sh, out_sh, 0)
    assert float(mult.sum()) == n_out


def test_reconstruct_const_driven():
    # const has a 1 where the live tensor varies.
    live, ca = reconstruct_live_shape((3, 4), (4,), (3, 4), n_live=3)
    assert live == (3, 1) and ca == (1, 4)


def test_reconstruct_fallback_to_in_shape():
    # const fully drives both dims (==out) -> live would be (1,1); falls back
    # to the recorded in_shape which preserves n_live.
    live, ca = reconstruct_live_shape((3, 4), (3, 4), (3, 4), n_live=12)
    assert int(np.prod(live)) == 12 and ca == (3, 4)


def test_reconstruct_raises_on_ambiguous():
    with pytest.raises(NotImplementedError):
        reconstruct_live_shape((3, 4), (3, 4), (5,), n_live=12)
