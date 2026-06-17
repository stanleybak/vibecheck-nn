"""Shared numpy/ONNX broadcast reconstruction for the verifier's bound reps.

A "live" tensor of ``n_live`` neurons (whose recorded N-D shape may have lost
trailing-1 dims to an aliased Unsqueeze) is broadcast against a constant / other
operand of shape ``const_sh`` into ``out_sh``. BOTH bound representations need
to expand their per-neuron rows from ``n_live`` to ``prod(out_sh)`` following the
same broadcast:

  * the zonotope — center (n_live,) and generator rows (n_live, K);
  * the LiRPA linear bound — A rows (B, n_live, n_in) and b rows (B, n_live).

This module is the SINGLE source of that (soundness-sensitive) shape + index
logic, so the three places that used to reconstruct it independently
(`verify_zono_bnb._zono_broadcast_const`, `_point_bcast_bias`, and the
`forward_lirpa` broadcast handlers) stay in lock-step. Validated in
tests/test_broadcast_util.py against torch's own broadcasting.
"""
import numpy as np
import torch


def assert_no_outer_broadcast(in_shapes_nd, out_shape_nd, op_type, where):
    """Fail loud if an Add/Sub backward operand is OUTER-broadcast (operand
    numel != output numel) on a backward pass that propagates the effective
    weight through UNCHANGED — i.e. without the broadcast reduce-sum adjoint.

    The adjoint of a forward broadcast (one operand element feeding many output
    elements) must SUM the effective weight over the broadcast dims (see
    ``broadcast_rows_backward`` / ``verify_zono_bnb._bcast_ew_back``, used by the
    scalar ``_spec_backward_graph``). The batched / matrix / per-neuron CROWN
    passes pass ew through unchanged, which is sound ONLY when no operand is
    outer-broadcast. Those passes can't currently build an outer-broadcast
    Add/Sub graph (they have no Sin/Cos handler, so the ml4acopf trig nets that
    produce such broadcasts never reach them), so this never fires today — it
    converts a LATENT silent-unsoundness into a loud error if op coverage ever
    extends to route such a graph here. Same bug class as the gather fan-out
    adjoint fix (index_copy_ -> index_add_).
    """
    if in_shapes_nd is None or out_shape_nd is None:
        return
    out_n = int(np.prod([int(d) for d in out_shape_nd]))
    for k, s in enumerate(in_shapes_nd):
        if s is None:
            continue
        in_n = int(np.prod([int(d) for d in s]))
        if in_n != out_n:
            raise NotImplementedError(
                f"{op_type} backward ({where}): operand {k} is outer-broadcast "
                f"(in numel {in_n} != out numel {out_n}); the broadcast "
                f"reduce-sum adjoint is not implemented on this pass — only "
                f"verify_zono_bnb._spec_backward_graph (via _bcast_ew_back) "
                f"handles it. Route such graphs there, or implement the "
                f"reduce here, rather than silently dropping the fan-out sum.")


def reconstruct_live_shape(out_sh, const_sh, in_sh, n_live):
    """Reconstruct the live tensor's broadcast shape inside ``out_sh``.

    The constant right-aligns into ``out_sh``; every out dim the constant does
    NOT drive (its dim == 1 there) belongs to the live tensor (= that out dim),
    the rest are live-1. If that doesn't preserve ``n_live`` (the live N-D shape
    was ambiguous), fall back to right-aligning the recorded ``in_sh``. If THAT
    also mismatches the broadcast layout is genuinely ambiguous → raise rather
    than emit an unsound (wrong-layout) expansion.

    Returns (live_sh, const_aligned), both full-rank tuples of length
    ``len(out_sh)``.
    """
    out_sh = tuple(int(d) for d in out_sh)
    const_sh = tuple(int(d) for d in const_sh)
    rank = len(out_sh)
    const_aligned = (1,) * (rank - len(const_sh)) + const_sh
    live_sh = tuple(out_sh[d] if const_aligned[d] == 1 else 1
                    for d in range(rank))
    if int(np.prod(live_sh)) != n_live:
        in_sh = tuple(int(d) for d in in_sh)
        live_sh = (1,) * (rank - len(in_sh)) + in_sh
        if int(np.prod(live_sh)) != n_live:
            raise NotImplementedError(
                f'broadcast: cannot place live numel {n_live} into out shape '
                f'{out_sh} given const shape {const_sh} (in_shape={in_sh})')
    return live_sh, const_aligned


def broadcast_row_index(live_sh, out_sh, device=None):
    """Index map ``out-flat-neuron -> live-flat-neuron`` for broadcasting a
    tensor of shape ``live_sh`` (full-rank, dims 1 or == out) to ``out_sh``.
    Use with ``tensor.index_select(neuron_dim, idx)`` on any representation."""
    live_sh = tuple(int(d) for d in live_sh)
    out_sh = tuple(int(d) for d in out_sh)
    n_live = int(np.prod(live_sh)) if live_sh else 1
    idx = torch.arange(n_live, device=device).reshape(
        live_sh if live_sh else (1,)).expand(out_sh).reshape(-1)
    return idx.contiguous()


def broadcast_rows(t, live_sh, out_sh, row_dim):
    """Expand ``t`` along ``row_dim`` from ``prod(live_sh)`` rows to
    ``prod(out_sh)`` rows via the broadcast index (a gather — works on any
    rep/axis; the zono uses ``.expand`` directly where a view suffices)."""
    idx = broadcast_row_index(live_sh, out_sh, device=t.device)
    return t.index_select(row_dim, idx)


def broadcast_rows_backward(ew, live_sh, out_sh, row_dim):
    """Adjoint of ``broadcast_rows``: reduce ``ew`` from ``prod(out_sh)`` rows
    back to ``prod(live_sh)`` rows along ``row_dim`` by summing every output
    row onto the live row it broadcast from (index_add). This is the CROWN
    backward through an ONNX broadcast — the effective weight at a broadcast
    output is the SUM over the copies it was expanded into."""
    idx = broadcast_row_index(live_sh, out_sh, device=ew.device)
    n_live = int(np.prod(live_sh)) if live_sh else 1
    shape = list(ew.shape)
    shape[row_dim] = n_live
    out = torch.zeros(shape, dtype=ew.dtype, device=ew.device)
    out.index_add_(row_dim, idx, ew)
    return out
