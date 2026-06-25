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

# Fuse the maxpool->relu decomposition's backward data movement: share merge
# branches (no clone) and place disjoint phase-slice channel blocks in one tensor
# (no per-slice zero-fill + channel-dim add). Bit-identical to the naive path;
# a toggle only so the A/B harness can measure it. See backward_add_merge /
# backward_slice / _accumulate.
_FAST_MAXPOOL_BWD = [True]

# Backward the maxpool 1-hot phase-extraction conv as a pure strided SCATTER
# instead of a winograd conv_transpose2d (it computes a 1-hot map — running cuDNN
# winograd for it is wasted FLOPs AND adds non-deterministic dgrad atomics). The
# scatter is EXACT. Toggle for the A/B harness. See _detect_phase_conv /
# PatchesA._backward_phase_scatter.
_FAST_PHASE_CONV = [True]


def _detect_phase_conv(op):
    """Is ``op`` the 1-hot maxpool phase-extraction conv (C -> P*C, stride =
    kernel = P window, out_ch ``p*C+c`` reads in_ch ``c`` at window position
    ``(p//kW, p%kW)``, zero bias, zero pad)? Its CROWN adjoint is a strided
    scatter, not a real conv. Returns ``(kH, kW, sH, sW)`` if so, else ``False``.
    Verified ONCE per op (the caller caches the result on the op dict)."""
    kernel = op['kernel']
    k = (kernel.detach().cpu().numpy() if isinstance(kernel, torch.Tensor)
         else np.asarray(kernel))
    C_out, C_in, kH, kW = k.shape
    sH, sW = op['stride']
    if C_out != kH * kW * C_in or (sH, sW) != (kH, kW):
        return False
    if tuple(op.get('padding', (0, 0))) != (0, 0):
        return False
    bias = op.get('bias')
    if bias is not None:
        bt = (bias.detach().cpu().numpy() if isinstance(bias, torch.Tensor)
              else np.asarray(bias))
        if bt.size and np.any(bt != 0):
            return False
    expect = np.zeros_like(k)
    for p in range(kH * kW):
        for c in range(C_in):
            expect[p * C_in + c, c, p // kW, p % kW] = 1.0
    if not np.array_equal(k, expect):
        return False
    return (kH, kW, sH, sW)


def _window_index(offsets, shape, ph, pw):
    """Flat gather index ``(B*ph*pw,)`` for the ``(ph, pw)`` window at each of
    B rows' top-left ``offset`` into a ``(C, H, W)`` map. Out-of-range
    positions point at index ``H*W`` (a sentinel zero column appended by
    ``_gather_at``) so phantom positions gather 0 with NO separate mask
    multiply — saving a full ``(B, C, ph, pw)`` pass per gather (the masked
    multiply was a measured hot spot on VGG's 130x130 deep-neuron windows)."""
    C, H, W = shape
    B = offsets.shape[0]
    dev = offsets.device
    HW = H * W
    ty = offsets[:, 0].view(B, 1) + torch.arange(ph, device=dev).view(1, ph)
    tx = offsets[:, 1].view(B, 1) + torch.arange(pw, device=dev).view(1, pw)
    inb = (ty >= 0) & (ty < H)                          # (B, ph)
    inx = (tx >= 0) & (tx < W)                          # (B, pw)
    flat = (ty.clamp(0, H - 1).view(B, ph, 1) * W
            + tx.clamp(0, W - 1).view(B, 1, pw))        # (B, ph, pw)
    valid = inb.view(B, ph, 1) & inx.view(B, 1, pw)
    return torch.where(valid, flat, flat.new_full((), HW)).reshape(-1)


def _gather_at(map_chw, spatial, B, ph, pw):
    """Gather a single ``(C, H, W)`` map at the precomputed ``spatial`` index
    (from ``_window_index``) -> ``(B, C, ph, pw)``. A zero column is appended
    so sentinel ``H*W`` indices gather 0 (phantom)."""
    C, H, W = map_chw.shape
    mf = map_chw.reshape(C, H * W)
    mf = torch.cat([mf, mf.new_zeros(C, 1)], dim=1)     # (C, H*W+1); zero col
    return mf[:, spatial].reshape(C, B, ph, pw).permute(1, 0, 2, 3)


def _gather_windows(maps_chw, offsets, ph, pw):
    """Compat shim: gather the ``(C, ph, pw)`` window per row (phantom->0).
    Kept for callers/tests; prefer ``_window_index`` + ``_gather_at`` to gather
    one map at a time (lower peak memory)."""
    spatial = _window_index(offsets, maps_chw.shape, ph, pw)
    return _gather_at(maps_chw, spatial, offsets.shape[0], ph, pw)


# torch.compile the ReLU-backward hot core (gather + clamps + products). This is
# the single biggest cost of the deep-layer backward (~53% on VGG conv10) — a
# chain of small elementwise kernels over the (B,C,ph,pw) patch that the
# compiler fuses (gather epilogue + clamp + multiply + addcmul) into far fewer
# launches, which is exactly where AB-CROWN's single fused slope-multiply beats
# the eager chain. Compiles per (B,ph,pw) shape (cached); first call pays the
# compile. Numerically equivalent up to fp32 reassociation (validated against
# the dense backward by the bit-equivalence test). Toggle with _RELU_COMPILE.
_RELU_COMPILE = [True]


def _relu_bwd_eager(patches, up_s, up_t, eff_slope, spatial, B, ph, pw):
    C = up_s.shape[0]
    ep = patches.clamp(min=0)
    en = patches.clamp(max=0)

    def gat(m):
        mf = torch.cat([m.reshape(C, -1), m.new_zeros(C, 1)], dim=1)
        return mf[:, spatial].reshape(C, B, ph, pw).permute(1, 0, 2, 3)

    ut_w = gat(up_t)
    acc_c = (en * ut_w).sum(dim=(-1, -2, -3))
    us_w = gat(up_s)
    new = en * us_w
    sl_w = gat(eff_slope)
    new = new.addcmul(ep, sl_w)
    return new, acc_c


_relu_bwd_compiled = [None]


def _relu_bwd_core(patches, up_s, up_t, eff_slope, spatial, B, ph, pw):
    if not _RELU_COMPILE[0]:
        return _relu_bwd_eager(patches, up_s, up_t, eff_slope, spatial, B, ph, pw)
    if _relu_bwd_compiled[0] is None:
        _relu_bwd_compiled[0] = torch.compile(_relu_bwd_eager)
    return _relu_bwd_compiled[0](patches, up_s, up_t, eff_slope, spatial, B, ph, pw)


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
        # Set by backward_slice to (c0, C_out): the relation is a CHANNEL BLOCK
        # (C_out channels at offset c0 in the C_in input) not yet materialized to
        # the full width. _accumulate resolves it. None = a normal full relation.
        self._block = None

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

    def _backward_phase_scatter(self, pc_info, op):
        """Adjoint of the 1-hot phase-extraction conv: phase block ``p``
        (channels ``[p*C_in:(p+1)*C_in]``) lands on the sub-grid
        ``(p//kW :: sH, p%kW :: sW)`` of the ``(sH*ph, sW*pw)`` upsampled input
        patch — exactly what ``conv_transpose2d`` with the 1-hot kernel computes,
        but as pure copies (no winograd FLOPs, no non-deterministic dgrad). Bias
        is zero by construction (checked in detection); ``acc`` passes through."""
        kH, kW, sH, sW = pc_info
        in_shape = tuple(op['in_shapes_nd'][0])
        C_in = in_shape[0]
        B, _, ph, pw = self.patches.shape
        new = torch.zeros(B, C_in, sH * ph, sW * pw,
                          dtype=self.patches.dtype, device=self.patches.device)
        for p in range(kH * kW):
            py, px = p // kW, p % kW
            new[:, :, py::sH, px::sW] = self.patches[:, p * C_in:(p + 1) * C_in]
        new_off = torch.stack([self.offsets[:, 0] * sH,
                               self.offsets[:, 1] * sW], dim=1)
        out = PatchesA(new, new_off, in_shape, self.acc)
        out._zero_phantoms()
        return out

    def backward_conv(self, op, device, dtype):
        """Adjoint of a forward conv: bias -> acc, then ``conv_transpose2d``
        moves the relation to the conv input; offset shifts by ``o*s - p`` and
        the window grows by ``k-1`` (stride 1)."""
        pc_info = op.get('_phase_conv')
        if pc_info is None:
            pc_info = _detect_phase_conv(op)
            op['_phase_conv'] = pc_info
        if pc_info and _FAST_PHASE_CONV[0]:
            return self._backward_phase_scatter(pc_info, op)
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
        # Gather the three (C,H,W) maps into the patch frame and combine. The
        # old code cat'd them into a (B,3C,ph,pw) tensor (3x the patch) before
        # consuming — a peak-memory hog. The fused core gathers one map at a
        # time (low peak) AND torch.compile fuses the gather+clamp+multiply
        # chain into few kernels (the deep-layer hot spot). `new = ep*eff_slope
        # + en*up_s`, `acc += sum(en*up_t)` (en=patch∧≤0, ep=patch∧≥0).
        spatial = _window_index(self.offsets, (C, H, W), ph, pw)
        new, acc_c = _relu_bwd_core(self.patches, up_s, up_t, eff_slope,
                                    spatial, B, ph, pw)
        self.acc = self.acc + acc_c
        self.patches = new
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
        if _FAST_MAXPOOL_BWD[0]:
            # Return a CHANNEL-BLOCK relation (C_out channels living at offset c0
            # inside the C_in input) WITHOUT materializing the full (B,C_in,ph,pw)
            # zero tensor. ``_accumulate`` places the P disjoint phase blocks of a
            # maxpool into one tensor in place — saving P-1 full zero-fills and the
            # P-1 big channel-dim adds that the old per-slice zeros+merge incurred.
            res = PatchesA(self.patches, self.offsets, in_shape, self.acc)
            res._block = (c0, C_out)
            return res
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
        if _FAST_MAXPOOL_BWD[0]:
            # No clone: every backward op REASSIGNS ``patches`` (relu/conv/phantom
            # all build a fresh tensor) rather than mutating it in place, so the
            # two merge branches can safely SHARE the tensor — a join that later
            # re-adds them computes ``T + T`` (correct: both branches carry +A).
            # Saves a full (B,C,ph,pw) clone + DtoD copy per merge (≈2% on VGG16).
            b = PatchesA(self.patches, self.offsets, self.shape,
                         torch.zeros_like(self.acc))
        else:
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
        spatial = _window_index(self.offsets, (C, H, W), ph, pw)
        ep = self.patches.clamp(min=0)
        en = self.patches.clamp(max=0)
        xl_w = _gather_at(xl_chw, spatial, B, ph, pw)
        out = self.acc + (ep * xl_w).sum(dim=(-1, -2, -3))
        del xl_w
        xh_w = _gather_at(xh_chw, spatial, B, ph, pw)
        out = out + (en * xh_w).sum(dim=(-1, -2, -3))
        return out


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
    blk = res._block
    if blk is not None:
        c0, C_out = blk
        res._block = None
        tgt = A_at.get(name)
        C_in = res.shape[0]
        B, _, ph, pw = res.patches.shape
        if (tgt is not None and tgt._block is None
                and tgt.patches.shape[1] == C_in
                and tgt.patches.shape[2:] == res.patches.shape[2:]
                and torch.equal(tgt.offsets, res.offsets)):
            # The full channel tensor already exists (an earlier phase block of
            # the same maxpool) — drop this block into its slot in place. No new
            # zero tensor, no channel-dim add over the (mostly-zero) full width.
            tgt.patches[:, c0:c0 + C_out] += res.patches
            tgt.acc = tgt.acc + res.acc
            return
        # First block (or a shape/offset mismatch fallback): materialize the full
        # (B,C_in,ph,pw) tensor once, then accumulate normally below.
        full = torch.zeros(B, C_in, ph, pw, dtype=res.patches.dtype,
                           device=res.patches.device)
        full[:, c0:c0 + C_out] = res.patches
        res = PatchesA(full, res.offsets, res.shape, res.acc)
    if name in A_at:
        A_at[name].add_(res)
    else:
        A_at[name] = res


# Largest neuron-batch that fit on the last successful backward (warm start).
# Halving on OOM discovers the memory ceiling once; later layers reuse it
# instead of re-running the full->halve cascade (each failed attempt builds GB
# of patches before OOMing). Shrinks monotonically as layers deepen (bigger
# patches); a module global so it persists across layers within a forward.
# Per-pass neuron batch (each of the two lb/ub walks processes this many rows).
# 256 is the measured sweet spot on VGG16's deepest conv (peak ~14.6 G): bigger
# batches are memory-bound and OOM-thrash at depth, smaller waste GPU. The
# half-split below still shrinks further on OOM at deeper layers.
_LAST_OK_BATCH = [256]


def _bounds_one(gg, xl, xh, sb, start_op_name, out_shape, idx, device, dtype):
    # Two SEPARATE backward walks (lower then upper), one at a time. Combining
    # them into a single 2B-row walk was MEASURED slower at the deep layers:
    # carrying both directions at once ~doubles peak memory, and the deep VGG
    # layers are memory-bound (conv12: 512-row walk OOM-thrashes the 23 G A10G
    # -> chunk collapses). Sequential walks free the first before the second.
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
