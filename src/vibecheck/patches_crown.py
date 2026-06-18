"""Patches-mode backward CROWN: the backward dual of ``PatchesZonotope``.

When computing intermediate-layer bounds for a deep conv net by backward
CROWN, the linear coefficient relation ``A`` of a single bounded neuron is
spatially LOCAL — nonzero only inside that neuron's receptive field. The
dense representation stores ``A`` as a full ``(B, C, H, W)`` feature-map tensor
(mostly zeros); this module stores it as a per-spec localized patch
``(B, C, ph, pw)`` with a per-spec offset ``(oy, ox)``, exactly mirroring
``PatchesZonotope`` but in the BACKWARD direction:

  * forward zono: a generator patch grows by ``(k-1)`` per conv (receptive
    field) and is placed by ``conv2d``;
  * backward CROWN: a coefficient patch grows by ``(k-1)`` per conv and is
    placed by ``conv_transpose2d`` (the conv adjoint).

On VGG16, while bounding a block-5 neuron the dense relation at block-4 is
``(B, 512, 28, 28)`` but only a ``~7x7`` window is nonzero — a ~16x waste that
compounds across the ~18-layer walk. Patches mode removes it, which is what
makes the per-layer intermediate retighten tractable inside the 1200 s budget.

This is SOUND iff it reproduces the dense ``_crown_backward_matrix`` lower
bound exactly (the patch is just a sparse view of the same linear relation).
``tests``/scratch validate that equivalence on the real VGG16 graph before
this path is relied on. Lower bound only (upper bound = negate the init and
negate the result), min-area ReLU relaxation (no per-neuron alpha), feed-
forward conv graphs with the maxpool->relu decomposition ops
(conv/relu/slice[channel-block]/sub_bilinear/add[merge|bias]).
"""

import numpy as np
import torch
import torch.nn.functional as F


def _gather_windows(maps_chw, offsets, ph, pw):
    """Gather, for each of B spec rows, the ``(C, ph, pw)`` window of
    ``maps_chw`` (a ``(C, H, W)`` tensor) at that row's top-left ``offset``.
    Positions outside ``[0,H) x [0,W)`` are zeroed (phantom mask). Returns
    ``(B, C, ph, pw)``."""
    C, H, W = maps_chw.shape
    B = offsets.shape[0]
    dev = maps_chw.device
    dy = torch.arange(ph, device=dev)
    dx = torch.arange(pw, device=dev)
    ty = offsets[:, 0].view(B, 1) + dy.view(1, ph)      # (B, ph)
    tx = offsets[:, 1].view(B, 1) + dx.view(1, pw)      # (B, pw)
    vy = (ty >= 0) & (ty < H)
    vx = (tx >= 0) & (tx < W)
    tyc = ty.clamp(0, H - 1)
    txc = tx.clamp(0, W - 1)
    spatial = (tyc.view(B, ph, 1) * W + txc.view(B, 1, pw)).reshape(-1)  # (B*ph*pw,)
    maps_flat = maps_chw.reshape(C, H * W)
    g = maps_flat[:, spatial].reshape(C, B, ph, pw).permute(1, 0, 2, 3)  # (B,C,ph,pw)
    valid = (vy.view(B, ph, 1) & vx.view(B, 1, pw)).view(B, 1, ph, pw)
    return g * valid.to(maps_chw.dtype)


class PatchesA:
    """Backward-CROWN linear relation as per-spec localized patches.

    Attributes:
        patches: ``(B, C, ph, pw)`` coefficient patch per spec row.
        offsets: ``(B, 2)`` long, top-left ``(oy, ox)`` in the current
            feature map ``(C, H, W)``. May be negative / out of range; those
            positions are phantom and kept zeroed.
        shape: ``(C, H, W)`` of the current feature map.
        acc: ``(B,)`` running constant (bias / relu-intercept) accumulator.
    """

    def __init__(self, patches, offsets, shape, acc):
        self.patches = patches
        self.offsets = offsets
        self.shape = shape
        self.acc = acc

    @property
    def B(self):
        return self.patches.shape[0]

    @classmethod
    def one_hot(cls, neuron_flat_idx, shape, sign, device, dtype):
        """Init relation: one spec row per selected neuron, a 1x1 patch of
        value ``sign`` (+1 lower / -1 for the upper-bound pass) at the
        neuron's ``(c, y, x)``."""
        C, H, W = shape
        HW = H * W
        idx = neuron_flat_idx.to(device=device, dtype=torch.long)
        B = idx.numel()
        c = idx // HW
        yx = idx % HW
        y = yx // W
        x = yx % W
        patches = torch.zeros(B, C, 1, 1, dtype=dtype, device=device)
        patches[torch.arange(B, device=device), c, 0, 0] = sign
        offsets = torch.stack([y, x], dim=1)
        acc = torch.zeros(B, dtype=dtype, device=device)
        return cls(patches, offsets, shape, acc)

    def _zero_phantoms(self):
        """Zero patch positions that fall outside the feature map, so a
        downstream conv/bias never treats phantom values as real."""
        C, H, W = self.shape
        B, _, ph, pw = self.patches.shape
        dev = self.patches.device
        ty = self.offsets[:, 0].view(B, 1) + torch.arange(ph, device=dev).view(1, ph)
        tx = self.offsets[:, 1].view(B, 1) + torch.arange(pw, device=dev).view(1, pw)
        vy = (ty >= 0) & (ty < H)
        vx = (tx >= 0) & (tx < W)
        valid = (vy.view(B, ph, 1) & vx.view(B, 1, pw)).view(B, 1, ph, pw)
        self.patches = self.patches * valid.to(self.patches.dtype)

    def backward_conv(self, op, device, dtype):
        """Adjoint of a forward conv: bias -> acc, then ``conv_transpose2d``
        moves the relation to the conv input; offset shifts by ``o*s - p`` and
        the window grows by ``k-1`` (stride 1)."""
        kernel = op['kernel'].to(device=device, dtype=dtype)
        bias = op['bias']
        C_out, C_in, kh, kw = kernel.shape
        sH, sW = op['stride']
        pH, pW = op['padding']
        in_shape = tuple(op['in_shapes_nd'][0])
        # Bias contributes per output position in the (phantom-zeroed) patch.
        if bias is not None:
            bias_t = bias.to(device=device, dtype=dtype)
            # patch: (B, C_out, ph, pw); sum over spatial, dot channels w/ bias
            self.acc = self.acc + (self.patches.sum(dim=(-1, -2)) * bias_t).sum(dim=-1)
        new_patches = F.conv_transpose2d(self.patches, kernel,
                                         stride=(sH, sW), padding=(0, 0))
        new_off_y = self.offsets[:, 0] * sH - pH
        new_off_x = self.offsets[:, 1] * sW - pW
        new_off = torch.stack([new_off_y, new_off_x], dim=1)
        out = PatchesA(new_patches, new_off, in_shape, self.acc)
        out._zero_phantoms()
        return out

    def backward_relu(self, lo_flat, hi_flat):
        """Lower-bound min-area ReLU relaxation, elementwise per neuron.
        ``ep*eff_slope + en*up_s`` with ``acc += sum(en*up_t)``; same
        offset/shape (relu is elementwise)."""
        C, H, W = self.shape
        lo = lo_flat.reshape(C, H, W)
        hi = hi_flat.reshape(C, H, W)
        ub_r = torch.clamp(hi, min=0)
        lb_r = torch.clamp(lo, max=0)
        ub_r = torch.maximum(ub_r, lb_r + 1e-8)
        up_s = ub_r / (ub_r - lb_r)
        up_t = -lb_r * up_s
        active = (lo >= 0).to(self.patches.dtype)
        unstable = ((lo < 0) & (hi > 0)).to(self.patches.dtype)
        eff_slope = active + unstable * (up_s > 0.5).to(self.patches.dtype)
        B, _, ph, pw = self.patches.shape
        # One stacked gather (3C channels) instead of three separate scattered
        # gathers -> 1/3 the kernel launches per relu.
        stk = torch.cat([eff_slope, up_s, up_t], dim=0)         # (3C, H, W)
        g = _gather_windows(stk, self.offsets, ph, pw)          # (B, 3C, ph, pw)
        sl = g[:, :C]; us = g[:, C:2 * C]; ut = g[:, 2 * C:3 * C]
        ep = self.patches.clamp(min=0)
        en = self.patches.clamp(max=0)
        self.acc = self.acc + (en * ut).sum(dim=(-1, -2, -3))
        self.patches = ep * sl + en * us
        return self

    def backward_slice(self, op, device):
        """Adjoint of a channel-block slice: scatter the C_out patch channels
        back into the input channel range ``[c0, c0+C_out)``; spatial window
        unchanged. Only the channel-block pattern emitted by the
        maxpool->relu decomposition is supported."""
        in_shape = tuple(op['in_shapes_nd'][0])
        C_in, H, W = in_shape
        out_shape = tuple(op['out_shape_nd'])
        C_out = out_shape[0]
        HW = H * W
        # Parse + verify the channel block ONCE per op, in NumPy (CPU, no GPU
        # transfer/sync), then cache `c0`. Previously this rebuilt an up-to-
        # 802k-element index tensor on GPU and ran `.item()` + `torch.equal`
        # (two GPU->CPU syncs) on EVERY backward call — ~32 calls per pass,
        # stalling the GPU pipeline. The scatter below needs only the scalar
        # `c0`, so after caching this op does pure-GPU work with no syncs.
        c0 = op.get('_patches_chanblock_c0')
        if c0 is None:
            fi = op['flat_idx']
            fi = (fi.cpu().numpy() if isinstance(fi, torch.Tensor)
                  else np.asarray(fi))
            first = int(fi[0])
            n = C_out * HW
            if (first % HW != 0 or fi.size != n
                    or not np.array_equal(
                        fi, np.arange(first, first + n))):
                raise NotImplementedError(
                    'patches-CROWN slice backward only supports a contiguous '
                    'channel block (maxpool->relu phase extraction); '
                    f'flat_idx[0]={first}')
            c0 = first // HW
            op['_patches_chanblock_c0'] = c0
        B, _, ph, pw = self.patches.shape
        new_patches = torch.zeros(B, C_in, ph, pw,
                                  dtype=self.patches.dtype, device=self.patches.device)
        new_patches[:, c0:c0 + C_out] = self.patches
        return PatchesA(new_patches, self.offsets, in_shape, self.acc)

    def backward_sub_bilinear(self):
        """Adjoint of ``y = a - b``: input0 gets +A, input1 gets -A; the
        constant ``acc`` flows entirely to input0 (input1 carries no constant
        so it isn't double counted at the join)."""
        a = PatchesA(self.patches, self.offsets, self.shape, self.acc)
        b = PatchesA(-self.patches, self.offsets, self.shape,
                     torch.zeros_like(self.acc))
        return a, b

    def backward_add_merge(self):
        """Adjoint of ``y = a + b`` (skip/merge): both inputs get +A; the
        constant flows to input0 only."""
        a = PatchesA(self.patches, self.offsets, self.shape, self.acc)
        b = PatchesA(self.patches.clone(), self.offsets, self.shape,
                     torch.zeros_like(self.acc))
        return a, b

    def backward_add_bias(self, op, device, dtype):
        """Adjoint of ``y = a + bias``: bias -> acc, relation passes through."""
        bias = op.get('bias')
        if bias is not None:
            bias_t = torch.as_tensor(bias, device=device, dtype=dtype)
            C, H, W = self.shape
            if bias_t.numel() == C:
                self.acc = self.acc + (self.patches.sum(dim=(-1, -2)) * bias_t).sum(dim=-1)
            elif bias_t.numel() == 1:
                self.acc = self.acc + self.patches.sum(dim=(-1, -2, -3)) * bias_t.flatten()
            else:
                raise NotImplementedError(
                    f'patches-CROWN add bias numel={bias_t.numel()} vs C={C}')
        return self

    def add_(self, other):
        """Accumulate ``other`` into this relation at a shared op. Fast path:
        identical offsets & window size -> direct add. General path: align
        both to the union bounding box per spec row, then add (sound; used
        only if a join has misaligned windows, which VGG's disjoint-channel
        maxpool joins never trigger)."""
        assert self.shape == other.shape
        self.acc = self.acc + other.acc
        if (self.patches.shape == other.patches.shape
                and torch.equal(self.offsets, other.offsets)):
            self.patches = self.patches + other.patches
            return self
        # General alignment.
        B = self.B
        oy = torch.minimum(self.offsets[:, 0], other.offsets[:, 0])
        ox = torch.minimum(self.offsets[:, 1], other.offsets[:, 1])
        end_y = torch.maximum(self.offsets[:, 0] + self.patches.shape[2],
                              other.offsets[:, 0] + other.patches.shape[2])
        end_x = torch.maximum(self.offsets[:, 1] + self.patches.shape[3],
                              other.offsets[:, 1] + other.patches.shape[3])
        ph = int((end_y - oy).max().item())
        pw = int((end_x - ox).max().item())
        C = self.shape[0]
        new = torch.zeros(B, C, ph, pw, dtype=self.patches.dtype,
                          device=self.patches.device)
        base = torch.stack([oy, ox], dim=1)
        for src, srcoff in ((self.patches, self.offsets), (other.patches, other.offsets)):
            dy = (srcoff[:, 0] - oy)
            dx = (srcoff[:, 1] - ox)
            sph, spw = src.shape[2], src.shape[3]
            for b in range(B):
                new[b, :, dy[b]:dy[b] + sph, dx[b]:dx[b] + spw] += src[b]
        self.patches = new
        self.offsets = base
        return self

    def concretize(self, xl, xh, in_shape):
        """At the network input: ``lb = acc + A+ . x_lo + A- . x_hi`` gathered
        over each spec row's input window."""
        C, H, W = in_shape
        xl_chw = xl.reshape(C, H, W).to(self.patches.dtype)
        xh_chw = xh.reshape(C, H, W).to(self.patches.dtype)
        B, _, ph, pw = self.patches.shape
        g = _gather_windows(torch.cat([xl_chw, xh_chw], dim=0),
                            self.offsets, ph, pw)              # (B, 2C, ph, pw)
        xl_w = g[:, :C]; xh_w = g[:, C:2 * C]
        ep = self.patches.clamp(min=0)
        en = self.patches.clamp(max=0)
        return self.acc + (ep * xl_w).sum(dim=(-1, -2, -3)) + (en * xh_w).sum(dim=(-1, -2, -3))


def crown_backward_patches(gg, xl, xh, sb, start_op_name, init_A, device, dtype):
    """Patches-mode backward CROWN from ``start_op_name``'s output to the
    network input. ``init_A`` is the initial ``PatchesA`` at that op's output.
    Returns ``lb`` per spec row. Mirrors ``_crown_backward_matrix`` op-by-op
    in patches form (feed-forward conv graphs + maxpool->relu ops)."""
    ops = gg['ops']
    start_idx = next(i for i, op in enumerate(ops) if op['name'] == start_op_name)
    input_name = gg['input_name']
    A_at = {start_op_name: init_A}
    in_shape = None
    for i in range(start_idx, -1, -1):
        op = ops[i]
        name = op['name']
        if name not in A_at:
            continue
        A = A_at.pop(name)
        t = op['type']
        if t == 'conv':
            in_shape = tuple(op['in_shapes_nd'][0])
            res = A.backward_conv(op, device, dtype)
            inp = op['inputs'][0]
            _accumulate(A_at, inp, res)
        elif t == 'relu':
            L = op['layer_idx']
            lo, hi = sb[L]
            res = A.backward_relu(lo, hi)
            inp = op['inputs'][0]
            _accumulate(A_at, inp, res)
        elif t == 'slice':
            in_shape = tuple(op['in_shapes_nd'][0])
            res = A.backward_slice(op, device)
            inp = op['inputs'][0]
            _accumulate(A_at, inp, res)
        elif t == 'sub_bilinear':
            a_res, b_res = A.backward_sub_bilinear()
            _accumulate(A_at, op['inputs'][0], a_res)
            _accumulate(A_at, op['inputs'][1], b_res)
        elif t == 'add':
            if op.get('is_merge'):
                a_res, b_res = A.backward_add_merge()
                _accumulate(A_at, op['inputs'][0], a_res)
                _accumulate(A_at, op['inputs'][1], b_res)
            else:
                res = A.backward_add_bias(op, device, dtype)
                _accumulate(A_at, op['inputs'][0], res)
        else:
            raise NotImplementedError(
                f'patches-CROWN backward does not handle op type {t!r} '
                f'(op {name}); add a handler or route to dense backward')
    # Concretize the surviving input adjoint.
    A_inp = A_at.get(input_name)
    if A_inp is None:
        raise RuntimeError('patches-CROWN: no relation reached the input node')
    if in_shape is None:
        in_shape = A_inp.shape
    return A_inp.concretize(xl, xh, A_inp.shape)


def _accumulate(A_at, name, res):
    if name in A_at:
        A_at[name].add_(res)
    else:
        A_at[name] = res


# Largest neuron-batch that fit on the last successful backward (warm start).
# Halving on OOM discovers the memory ceiling once; later layers reuse it
# instead of re-running the full->halve cascade (each failed attempt builds GB
# of patches before OOMing). Shrinks monotonically as layers deepen (bigger
# patches); a module global so it persists across layers within a forward.
# Cap the per-pass neuron batch. Measured sweet spot ~256 (L16): big batches
# (~2048 -> 20 GB) are ~2x SLOWER (memory-bound conv) AND OOM-thrash at depth;
# ~256 keeps conv tensors small/cache-friendly (ABC's small-chunk strategy).
# The half-split below still shrinks further on OOM at deeper layers.
_LAST_OK_BATCH = [256]


def _bounds_one(gg, xl, xh, sb, start_op_name, out_shape, idx, device, dtype):
    A_lo = PatchesA.one_hot(idx, out_shape, 1.0, device, dtype)
    lb = crown_backward_patches(gg, xl, xh, sb, start_op_name, A_lo,
                                device, dtype)
    A_hi = PatchesA.one_hot(idx, out_shape, -1.0, device, dtype)
    ub = -crown_backward_patches(gg, xl, xh, sb, start_op_name, A_hi,
                                 device, dtype)
    return lb, ub


def patches_bounds(gg, xl, xh, sb, start_op_name, out_shape, neuron_idx,
                   device, dtype):
    """Lower & upper bounds for ``neuron_idx`` (flat indices into
    ``start_op_name``'s ``out_shape`` feature map) via patches-CROWN. Processes
    the neurons in chunks sized by the warm-start memo; on CUDA OOM the chunk
    size is halved (no fixed magic chunk — the size is discovered and reused).
    Returns ``(lb, ub)``."""
    n = neuron_idx.numel()
    bs = min(n, _LAST_OK_BATCH[0])
    lbs, ubs = [], []
    s = 0
    while s < n:
        idx = neuron_idx[s:s + bs]
        oom = False
        try:
            lb, ub = _bounds_one(gg, xl, xh, sb, start_op_name, out_shape,
                                 idx, device, dtype)
            lbs.append(lb)
            ubs.append(ub)
            s += idx.numel()
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            oom = True
        if oom:
            # Leave the except block first (drop the traceback pinning the
            # failed frame's patches), then reclaim and shrink the chunk.
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            _LAST_OK_BATCH[0] = bs
    return torch.cat(lbs), torch.cat(ubs)
