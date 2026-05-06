"""Patches-based zonotope for memory-efficient forward propagation through convs.

Each generator's contribution to the current feature map is stored as a small
``(C, kH, kW)`` "patch" with an offset ``(oy, ox)`` indicating the patch's
top-left corner in the feature map. This exploits the spatial sparsity of
generators that originate from a single input pixel: through the first conv
the contribution is a single kernel-shaped patch, through subsequent convs
the patch grows by ``(kH_new - 1)`` per axis but stays much smaller than the
full feature map.

Memory: ``K * C * kH * kW`` instead of the dense form's ``K * C * H * W``.
On a CIFAR-100 ResNet's wide convs this is the difference between fitting on
a 10 GB GPU and OOMing.

Falls back to dense (``TorchZonotope``) when:
  - an FC layer is encountered (no spatial structure left);
  - the patch grows to feature-map size (no further savings);
  - the caller requests ``to_dense()``;
  - a stride > 1 conv is encountered (Phase 1 doesn't handle stride yet).

Same external API as ``TorchZonotope``: ``center``, ``generators`` (lazily
materialised dense view), ``bounds``, ``propagate_conv``, ``propagate_fc``,
``apply_relu``, ``copy``, ``to_``, ``add``.
"""

import torch
import torch.nn.functional as F

from .zonotope import TorchZonotope


def _flat_to_chw(flat_idx, in_shape):
    """Convert flat input indices to (c, y, x) given in_shape (C, H, W)."""
    C, H, W = in_shape
    c_idx = flat_idx // (H * W)
    yx = flat_idx % (H * W)
    y_idx = yx // W
    x_idx = yx % W
    return c_idx, y_idx, x_idx


class PatchesZonotope:
    """Forward-mode zonotope with per-generator localised patches."""

    def __init__(self, center, patches, offsets, out_shape,
                 _mode='patches', _dense=None):
        """
        Args:
            center: flat (n,) tensor — current feature-map center.
            patches: (K, C, kH, kW) tensor — uniform-shape per-gen patches,
                or None if _mode == 'dense'.
            offsets: (K, 2) long tensor — (oy, ox) top-left in feature map,
                or None if _mode == 'dense'. Offsets may be negative or
                outside [0, H) x [0, W); the corresponding patch positions
                are masked at bounds()/relu time.
            out_shape: (C, H, W) of the current feature map, or None after
                an FC layer (no spatial structure left).
            _mode: 'patches' or 'dense'.
            _dense: TorchZonotope instance when _mode == 'dense'.
        """
        self._center = center
        self._patches = patches
        self._offsets = offsets
        self.out_shape = out_shape
        self._mode = _mode
        self._dense = _dense

    @property
    def center(self):
        """Center vector. In dense mode delegates to the inner TorchZonotope
        so callers that read or write ``z.center`` keep both in sync."""
        if self._mode == 'dense':
            return self._dense.center
        return self._center

    @center.setter
    def center(self, value):
        if self._mode == 'dense':
            self._dense.center = value
        self._center = value

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_input_bounds(cls, x_lo, x_hi, in_shape, device, dtype):
        """Build initial zonotope. Each non-zero-radius input pixel becomes
        a 1x1 patch at its (c, y, x) position with magnitude = radius."""
        center = ((x_lo + x_hi) / 2).to(dtype=dtype, device=device)
        radii = ((x_hi - x_lo) / 2).to(dtype=dtype, device=device)
        nz = torch.nonzero(radii.flatten()).squeeze(1)
        K = nz.numel()
        C, H, W = in_shape
        patches = torch.zeros(K, C, 1, 1, dtype=dtype, device=device)
        offsets = torch.zeros(K, 2, dtype=torch.long, device=device)
        if K > 0:
            c_idx, y_idx, x_idx = _flat_to_chw(nz, in_shape)
            patches[torch.arange(K, device=device), c_idx, 0, 0] = \
                radii.flatten()[nz]
            offsets[:, 0] = y_idx
            offsets[:, 1] = x_idx
        return cls(center, patches, offsets, in_shape)

    # ------------------------------------------------------------------
    # External properties / dense materialisation
    # ------------------------------------------------------------------

    @property
    def n_gens(self):
        if self._mode == 'dense':
            return self._dense.n_gens
        return self._patches.shape[0]

    @property
    def generators(self):
        """External (n_flat, K) view. Materialises dense form on demand."""
        if self._mode == 'dense':
            return self._dense.generators
        return self._materialize_G()

    def to_dense(self):
        """Return a fresh ``TorchZonotope`` with the same set semantics.

        Does not modify ``self``. Use this when downstream code needs the
        explicit dense ``(n_flat, K)`` G matrix.
        """
        if self._mode == 'dense':
            return self._dense.copy()
        G = self._materialize_G()
        return TorchZonotope(self.center.clone(), G)

    def _materialize_G(self, chunk=512):
        """Build dense ``(n_flat, K)`` tensor by placing each patch at its
        offset.

        Chunked over K to avoid the 2× peak that would arise from materialising
        a full ``(K, C*H*W)`` intermediate plus its contiguous transpose. Each
        chunk writes directly into the final G via an in-place scatter on the
        column-slice view.
        """
        C, H, W = self.out_shape
        K, C_p, kH, kW = self._patches.shape
        assert C_p == C
        device = self._patches.device
        dtype = self._patches.dtype
        n_flat = C * H * W
        if K == 0:
            return torch.zeros(n_flat, 0, dtype=dtype, device=device)

        target_y, target_x, valid = self._target_grid()
        ty = target_y.clamp(0, H - 1)
        tx = target_x.clamp(0, W - 1)
        flat_idx = ty * W + tx  # (K, kH, kW)
        c_off = (torch.arange(C, device=device) * (H * W)).view(1, C, 1, 1)

        G = torch.zeros(n_flat, K, dtype=dtype, device=device)
        M = C * kH * kW
        for start in range(0, K, chunk):
            end = min(start + chunk, K)
            kc = end - start
            patches_c = self._patches[start:end]
            valid_c = valid[start:end]
            flat_c = flat_idx[start:end]
            masked_c = patches_c * valid_c[:, None, :, :].to(dtype)
            row_idx_c = c_off + flat_c.unsqueeze(1).expand(kc, C, kH, kW)
            # Reshape to (M, kc) so scatter_add_ on G[:, start:end] (shape
            # (n_flat, kc)) along dim=0 writes the right cells.
            src_t = masked_c.permute(1, 2, 3, 0).reshape(M, kc).contiguous()
            idx_t = row_idx_c.permute(1, 2, 3, 0).reshape(M, kc).contiguous()
            G[:, start:end].scatter_add_(0, idx_t, src_t)
        return G

    def _materialize_to_dense(self):
        """Convert in-place from patches to dense mode."""
        if self._mode == 'dense':
            return
        G = self._materialize_G()
        self._dense = TorchZonotope(self.center, G)
        self._patches = None
        self._offsets = None
        self._mode = 'dense'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_grid(self):
        """Return (target_y, target_x, valid), each shape (K, kH, kW).

        target_(y|x): position in current feature map for patch index (dy, dx).
        valid: bool mask of in-bounds positions.
        """
        K, C, kH, kW = self._patches.shape
        _C, H, W = self.out_shape
        device = self._patches.device
        dys = torch.arange(kH, device=device)
        dxs = torch.arange(kW, device=device)
        ty = (self._offsets[:, 0:1, None] + dys[None, :, None]).expand(
            K, kH, kW)
        tx = (self._offsets[:, 1:2, None] + dxs[None, None, :]).expand(
            K, kH, kW)
        valid = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        return ty, tx, valid

    def _zero_phantoms(self):
        """Zero patch values at positions outside the current feature map.

        After ``propagate_conv`` the patches grow in extent and may straddle
        the feature-map boundary; values at the out-of-bounds positions
        ("phantoms") would otherwise leak into the next conv via the patch-
        local convolution. ``apply_relu`` zeros them implicitly (lam masked
        to 0 outside the map); for raw conv→conv chains we have to do it
        explicitly.
        """
        K, C, kH, kW = self._patches.shape
        if K == 0:
            return
        _, _, valid = self._target_grid()
        self._patches.mul_(valid[:, None, :, :].to(self._patches.dtype))

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def bounds(self):
        """Compute element-wise (lo, hi) bounds."""
        if self._mode == 'dense':
            return self._dense.bounds()

        C, H, W = self.out_shape
        K, C_p, kH, kW = self._patches.shape
        device = self._patches.device
        dtype = self._patches.dtype
        if K == 0:
            return self.center.clone(), self.center.clone()

        target_y, target_x, valid = self._target_grid()
        ty = target_y.clamp(0, H - 1)
        tx = target_x.clamp(0, W - 1)
        flat_idx = (ty * W + tx).flatten()  # (K*kH*kW,)
        valid_flat = valid.flatten().to(dtype)

        # Per-channel scatter: avoids the (K, C, kH, kW) intermediate that
        # would dominate memory at K~8000, C~64, kH~13.
        out_flat = torch.zeros(C, H * W, dtype=dtype, device=device)
        abs_p = torch.abs(self._patches)
        for c in range(C):
            src = abs_p[:, c, :, :].flatten() * valid_flat
            out_flat[c].scatter_add_(0, flat_idx, src)
        abs_sum = out_flat.flatten()
        return self.center - abs_sum, self.center + abs_sum

    # ------------------------------------------------------------------
    # Conv / FC propagation
    # ------------------------------------------------------------------

    def propagate_conv(self, kernel, bias, in_shape, stride, padding):
        """Propagate through a Conv2d layer.

        Stride-1 keeps the patches representation. Stride > 1 falls back
        to dense (the alignment math gets fiddly and the memory benefit
        diminishes at strided / downsampled feature maps anyway).
        """
        sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
        pH, pW = padding if isinstance(padding, tuple) else (padding, padding)

        if self._mode == 'dense':
            self._dense.propagate_conv(
                kernel, bias, in_shape, (sH, sW), (pH, pW))
            self.center = self._dense.center
            C_out = kernel.shape[0]
            kH_new, kW_new = kernel.shape[2], kernel.shape[3]
            C_in, H_in, W_in = in_shape
            H_out = (H_in + 2 * pH - kH_new) // sH + 1
            W_out = (W_in + 2 * pW - kW_new) // sW + 1
            self.out_shape = (C_out, H_out, W_out)
            return

        C_out = kernel.shape[0]
        kH_new, kW_new = kernel.shape[2], kernel.shape[3]
        C_in, H_in, W_in = in_shape

        # Center: just a normal conv (uses the actual stride).
        self.center = F.conv2d(
            self.center.reshape(1, *in_shape), kernel, bias=bias,
            stride=(sH, sW), padding=(pH, pW)).flatten()
        H_out = (H_in + 2 * pH - kH_new) // sH + 1
        W_out = (W_in + 2 * pW - kW_new) // sW + 1
        new_out_shape = (C_out, H_out, W_out)

        K = self._patches.shape[0]
        if K == 0:
            self._patches = torch.zeros(
                0, C_out, 1, 1, dtype=self._patches.dtype,
                device=self._patches.device)
            self._offsets = torch.zeros(
                0, 2, dtype=torch.long, device=self._patches.device)
            self.out_shape = new_out_shape
            return

        # Stride-1 patches conv with padding=(kH_new-1, kW_new-1) so the
        # output covers every position where the new kernel touches the old
        # patch. Output shape: (K, C_out, kH_p + kH_new - 1, kW_p + kW_new - 1).
        pre = F.conv2d(
            self._patches, kernel, stride=(1, 1),
            padding=(kH_new - 1, kW_new - 1))
        kH_pre, kW_pre = pre.shape[2], pre.shape[3]

        # Stride-1 offset shift: the smallest position oy in the stride-1
        # virtual output is oy_k + pH - (kH_new - 1) (derived from
        # oy * 1 - pH + ky = oy_k with ky = kH_new - 1).
        s1_off_y = self._offsets[:, 0] + (pH - kH_new + 1)
        s1_off_x = self._offsets[:, 1] + (pW - kW_new + 1)

        if sH == 1 and sW == 1:
            # Stride-1: no subsampling needed.
            new_patches = pre
            new_offsets = torch.stack([s1_off_y, s1_off_x], dim=1)
        else:
            # Stride > 1: per-gen subsample to keep only stride-s aligned
            # output positions. Different gens have different alignment
            # (depending on s1_off mod s); we gather a uniform shape with
            # zero-padding for misaligned positions.
            device = pre.device
            dtype_pre = pre.dtype
            # First valid stride-s output position dy_min in stride-1 frame:
            # need (s1_off_y + dy_min) divisible by sH and >= 0.
            dy_min = (-s1_off_y) % sH  # (K,)
            dx_min = (-s1_off_x) % sW
            # Max number of stride-s positions across all alignments.
            max_num_y = (kH_pre + sH - 1) // sH
            max_num_x = (kW_pre + sW - 1) // sW
            js_y = torch.arange(max_num_y, device=device)
            js_x = torch.arange(max_num_x, device=device)
            y_idx = dy_min[:, None] + js_y[None, :] * sH  # (K, max_num_y)
            x_idx = dx_min[:, None] + js_x[None, :] * sW
            y_valid = y_idx < kH_pre
            x_valid = x_idx < kW_pre
            y_safe = y_idx.clamp(0, kH_pre - 1)
            x_safe = x_idx.clamp(0, kW_pre - 1)
            # Chunked gather over K: advanced indexing into the full pre
            # tensor would create a (K, C_out, max_num_y, max_num_x) +
            # broadcast intermediate that can exceed GPU memory on wide
            # nets. Chunking keeps the intermediate bounded while still
            # using torch advanced indexing for speed.
            new_patches = torch.empty(
                K, C_out, max_num_y, max_num_x,
                dtype=dtype_pre, device=device)
            chunk = 1024
            c_ar = torch.arange(C_out, device=device).view(1, C_out, 1, 1)
            for start in range(0, K, chunk):
                end = min(start + chunk, K)
                kc = end - start
                k_ar_c = torch.arange(
                    kc, device=device).view(kc, 1, 1, 1)
                # y_safe[start:end] shape (kc, max_num_y).
                y_idx_c = y_safe[start:end, None, :, None]
                x_idx_c = x_safe[start:end, None, None, :]
                new_patches[start:end] = pre[start:end][
                    k_ar_c.expand(kc, C_out, max_num_y, max_num_x),
                    c_ar.expand(kc, C_out, max_num_y, max_num_x),
                    y_idx_c.expand(kc, C_out, max_num_y, max_num_x),
                    x_idx_c.expand(kc, C_out, max_num_y, max_num_x),
                ]
            del pre
            mask = (y_valid[:, None, :, None] & x_valid[:, None, None, :])
            new_patches.mul_(mask.to(dtype_pre))
            # New offsets in the stride-s output frame.
            new_off_y = (s1_off_y + dy_min) // sH
            new_off_x = (s1_off_x + dx_min) // sW
            new_offsets = torch.stack([new_off_y, new_off_x], dim=1)

        self._patches = new_patches
        self._offsets = new_offsets
        self.out_shape = new_out_shape

        # Zero phantom positions so a raw conv→conv chain is correct (the
        # patch-local conv would otherwise treat phantom values as real
        # feature-map content).
        self._zero_phantoms()

        # Materialise once patches are no longer smaller than dense.
        kH_p, kW_p = self._patches.shape[2], self._patches.shape[3]
        if kH_p * kW_p >= H_out * W_out:
            self._materialize_to_dense()

    def propagate_fc(self, W, bias):
        """FC: materialise to dense and delegate."""
        self._materialize_to_dense()
        self._dense.propagate_fc(W, bias)
        self.center = self._dense.center
        self.out_shape = None

    # ------------------------------------------------------------------
    # ReLU
    # ------------------------------------------------------------------

    def apply_relu(self, tight_lo=None, tight_hi=None):
        """Apply min-area ReLU relaxation."""
        if self._mode == 'dense':
            lo, hi = self._dense.apply_relu(
                tight_lo=tight_lo, tight_hi=tight_hi)
            self.center = self._dense.center
            return lo, hi

        lo_int, hi_int = self.bounds()
        lo = lo_int if tight_lo is None else torch.maximum(lo_int, tight_lo)
        hi = hi_int if tight_hi is None else torch.minimum(hi_int, tight_hi)
        ust = (lo < 0) & (hi > 0)
        dead = hi <= 0
        lam = torch.where(
            ust, hi / (hi - lo),
            torch.where(dead, torch.zeros_like(hi), torch.ones_like(hi)))
        mu = torch.where(
            ust, -hi * lo / (2 * (hi - lo)), torch.zeros_like(hi))
        self.center = lam * self.center + mu

        C, H, W = self.out_shape
        lam_3d = lam.reshape(C, H, W)
        K, C_p, kH, kW = self._patches.shape
        device = self._patches.device
        dtype = self._patches.dtype

        if K > 0:
            target_y, target_x, valid = self._target_grid()
            ty = target_y.clamp(0, H - 1)
            tx = target_x.clamp(0, W - 1)
            flat_idx = ty * W + tx  # (K, kH, kW)

            # Gather lam at each (k, c, dy, dx) target position.
            # lam_flat[c, flat_idx[k, dy, dx]] -> (K, C, kH, kW) lam-at-target.
            lam_flat = lam_3d.reshape(C, H * W)  # (C, H*W)
            idx_expand = flat_idx.unsqueeze(1).expand(
                K, C, kH, kW).reshape(K, C, kH * kW)
            lam_at = lam_flat.unsqueeze(0).expand(K, C, H * W).gather(
                2, idx_expand).reshape(K, C, kH, kW)
            # Mask invalid positions (we're about to multiply patches by these).
            lam_at = lam_at * valid[:, None, :, :].to(dtype)
            self._patches.mul_(lam_at)

        # Append new gens (one per unstable neuron) for the μ noise symbols.
        ui = torch.where(ust)[0]
        nu = ui.numel()
        if nu > 0:
            kH_now = self._patches.shape[2]
            kW_now = self._patches.shape[3]
            new_patches = torch.zeros(
                nu, C, kH_now, kW_now, dtype=dtype, device=device)
            c_un, y_un, x_un = _flat_to_chw(ui, (C, H, W))
            new_patches[torch.arange(nu, device=device), c_un, 0, 0] = mu[ui]
            new_offsets = torch.zeros(
                nu, 2, dtype=torch.long, device=device)
            new_offsets[:, 0] = y_un
            new_offsets[:, 1] = x_un
            self._patches = torch.cat([self._patches, new_patches], dim=0)
            self._offsets = torch.cat([self._offsets, new_offsets], dim=0)

        return lo, hi

    def nonzero_rows(self, unstable_idx, chunk=256):
        """Sparse ``(row_ids, col_ids, values)`` of ``G[unstable_idx, :]``.

        Patches form: never materialises the full ``[n_flat, K]`` G. For
        each chunk of unstable neurons we form the (chunk, K) ``dy / dx``
        masks identifying which gens' patches cover each neuron, then
        single-shot gather their patch values. ``row_ids`` index into
        ``unstable_idx`` (NOT into flat neuron space).
        """
        if self._mode == 'dense':
            return self._dense.nonzero_rows(unstable_idx)

        if not isinstance(unstable_idx, torch.Tensor):
            unstable_idx = torch.as_tensor(
                unstable_idx, dtype=torch.long, device=self._patches.device)
        else:
            unstable_idx = unstable_idx.to(self._patches.device)

        K, C, kH, kW = self._patches.shape
        _C, H, W = self.out_shape
        device = self._patches.device
        dtype = self._patches.dtype

        n_u = unstable_idx.numel()
        if K == 0 or n_u == 0:
            empty_l = torch.empty(0, dtype=torch.long, device=device)
            empty_v = torch.empty(0, dtype=dtype, device=device)
            return empty_l, empty_l.clone(), empty_v

        c_u, y_u, x_u = _flat_to_chw(unstable_idx, (C, H, W))
        oy_K = self._offsets[:, 0]
        ox_K = self._offsets[:, 1]

        all_u, all_k, all_v = [], [], []
        for start in range(0, n_u, chunk):
            end = min(start + chunk, n_u)
            c_chunk = c_u[start:end]
            y_chunk = y_u[start:end]
            x_chunk = x_u[start:end]
            # (chunk, K) covering masks: dy = y_u - oy ∈ [0, kH).
            dy = y_chunk.unsqueeze(1) - oy_K.unsqueeze(0)
            dx = x_chunk.unsqueeze(1) - ox_K.unsqueeze(0)
            valid = (dy >= 0) & (dy < kH) & (dx >= 0) & (dx < kW)
            pairs = valid.nonzero(as_tuple=False)  # (M, 2)
            if pairs.numel() == 0:
                continue
            u_local = pairs[:, 0]
            k_local = pairs[:, 1]
            dy_local = dy[u_local, k_local]
            dx_local = dx[u_local, k_local]
            c_local = c_chunk[u_local]
            values = self._patches[k_local, c_local, dy_local, dx_local]
            nz_mask = values != 0
            if nz_mask.any():
                all_u.append(u_local[nz_mask] + start)
                all_k.append(k_local[nz_mask])
                all_v.append(values[nz_mask])

        if not all_u:
            empty_l = torch.empty(0, dtype=torch.long, device=device)
            empty_v = torch.empty(0, dtype=dtype, device=device)
            return empty_l, empty_l.clone(), empty_v
        return torch.cat(all_u), torch.cat(all_k), torch.cat(all_v)

    def apply_relu_custom(self, lam, mu, shift):
        """Apply caller-supplied (lam, mu, shift) — see TorchZonotope docs.

        Patches semantics mirror ``apply_relu``: patches are gather-multiplied
        by lam at each patch position, then new gens are appended for neurons
        with ``mu != 0``.
        """
        if self._mode == 'dense':
            self._dense.apply_relu_custom(lam, mu, shift)
            self.center = self._dense.center
            return

        self.center = lam * self.center + shift
        C, H, W = self.out_shape
        lam_3d = lam.reshape(C, H, W)
        K, C_p, kH, kW = self._patches.shape
        device = self._patches.device
        dtype = self._patches.dtype

        if K > 0:
            target_y, target_x, valid = self._target_grid()
            ty = target_y.clamp(0, H - 1)
            tx = target_x.clamp(0, W - 1)
            flat_idx = ty * W + tx
            lam_flat = lam_3d.reshape(C, H * W)
            idx_expand = flat_idx.unsqueeze(1).expand(
                K, C, kH, kW).reshape(K, C, kH * kW)
            lam_at = lam_flat.unsqueeze(0).expand(K, C, H * W).gather(
                2, idx_expand).reshape(K, C, kH, kW)
            lam_at = lam_at * valid[:, None, :, :].to(dtype)
            self._patches.mul_(lam_at)

        ui = torch.where(mu != 0)[0]
        nu = ui.numel()
        if nu > 0:
            kH_now = self._patches.shape[2]
            kW_now = self._patches.shape[3]
            new_patches = torch.zeros(
                nu, C, kH_now, kW_now, dtype=dtype, device=device)
            c_un, y_un, x_un = _flat_to_chw(ui, (C, H, W))
            new_patches[torch.arange(nu, device=device), c_un, 0, 0] = mu[ui]
            new_offsets = torch.zeros(
                nu, 2, dtype=torch.long, device=device)
            new_offsets[:, 0] = y_un
            new_offsets[:, 1] = x_un
            self._patches = torch.cat([self._patches, new_patches], dim=0)
            self._offsets = torch.cat([self._offsets, new_offsets], dim=0)

    # ------------------------------------------------------------------
    # Memory / device management
    # ------------------------------------------------------------------

    def copy(self):
        """Return an independent copy."""
        new_patches = (self._patches.clone()
                       if self._patches is not None else None)
        new_offsets = (self._offsets.clone()
                       if self._offsets is not None else None)
        new_dense = self._dense.copy() if self._dense is not None else None
        return PatchesZonotope(
            self.center.clone(), new_patches, new_offsets,
            self.out_shape, self._mode, new_dense)

    def to_(self, device, non_blocking=False):
        """Move tensors to device in place."""
        self.center = self.center.to(device, non_blocking=non_blocking)
        if self._patches is not None:
            self._patches = self._patches.to(
                device, non_blocking=non_blocking)
        if self._offsets is not None:
            self._offsets = self._offsets.to(
                device, non_blocking=non_blocking)
        if self._dense is not None:
            self._dense.to_(device, non_blocking=non_blocking)
        return self

    # ------------------------------------------------------------------
    # Skip-connection add
    # ------------------------------------------------------------------

    def add(self, other, shared_gens):
        """Element-wise addition for skip connections.

        Mirrors ``TorchZonotope.add``: the first ``shared_gens`` generator
        columns are shared noise symbols (from before the fork point) — they
        are added element-wise. Remaining columns are branch-specific and
        are concatenated.

        Patches form: shared gens are added in place if both branches'
        patches share the same offsets (typical for residual blocks where
        both branches use matched stride+padding). When kH or kW differ we
        zero-pad the smaller patches to the common shape. When offsets
        themselves disagree (e.g. stride-2 in only one branch) we fall back
        to the dense add — sound, just not memory-optimal.
        """
        if not isinstance(other, PatchesZonotope):
            # Other is a TorchZonotope; promote self to dense and add.
            return self.to_dense().add(other, shared_gens)
        if self._mode == 'dense' or other._mode == 'dense':
            return self.to_dense().add(other.to_dense(), shared_gens)

        a_patches, b_patches = self._patches, other._patches
        a_offsets, b_offsets = self._offsets, other._offsets
        K_a, C, kH_a, kW_a = a_patches.shape
        K_b, C_b, kH_b, kW_b = b_patches.shape
        assert C == C_b, f"channel mismatch: {C} vs {C_b}"
        assert self.out_shape == other.out_shape, (
            f"out_shape mismatch: {self.out_shape} vs {other.out_shape}")
        assert K_a >= shared_gens and K_b >= shared_gens, (
            f"shared_gens={shared_gens} > min(K_a={K_a}, K_b={K_b})")

        device = a_patches.device
        dtype = a_patches.dtype

        # Two paths:
        #   FAST: offset deltas (a_o - b_o) are uniform across shared gens.
        #         Use simple slicing; A goes at one anchor, B at another.
        #   SCATTER: per-gen delta varies (e.g. stride-2 with mismatched
        #         kernel sizes between branches — common in ResNet shortcuts).
        #         Each shared gen has its own (a_ry[k], b_ry[k]) anchor; we
        #         scatter A and B per-gen into the uniform combined patch.
        if shared_gens > 0:
            delta = a_offsets[:shared_gens] - b_offsets[:shared_gens]
            uniform = (shared_gens == 1
                       or bool(torch.all(delta == delta[0:1])))
        else:
            uniform = True

        if uniform:
            if shared_gens > 0:
                dy_int = int(delta[0, 0].item())
                dx_int = int(delta[0, 1].item())
            else:
                dy_int = dx_int = 0
            a_ry_s = max(0, dy_int); a_rx_s = max(0, dx_int)
            b_ry_s = max(0, -dy_int); b_rx_s = max(0, -dx_int)
            new_kH = max(a_ry_s + kH_a, b_ry_s + kH_b)
            new_kW = max(a_rx_s + kW_a, b_rx_s + kW_b)
            new_K = K_a + K_b - shared_gens
            new_patches = torch.zeros(
                new_K, C, new_kH, new_kW, dtype=dtype, device=device)
            if shared_gens > 0:
                new_patches[:shared_gens, :,
                            a_ry_s:a_ry_s + kH_a,
                            a_rx_s:a_rx_s + kW_a] = a_patches[:shared_gens]
                new_patches[:shared_gens, :,
                            b_ry_s:b_ry_s + kH_b,
                            b_rx_s:b_rx_s + kW_b] += b_patches[:shared_gens]
            if K_a > shared_gens:
                new_patches[shared_gens:K_a, :, :kH_a, :kW_a] = \
                    a_patches[shared_gens:]
            if K_b > shared_gens:
                new_patches[K_a:, :, :kH_b, :kW_b] = b_patches[shared_gens:]
            new_offsets = torch.empty(
                new_K, 2, dtype=torch.long, device=device)
            if shared_gens > 0:
                anchor = torch.tensor(
                    [a_ry_s, a_rx_s], dtype=torch.long, device=device)
                new_offsets[:shared_gens] = a_offsets[:shared_gens] - anchor
            if K_a > shared_gens:
                new_offsets[shared_gens:K_a] = a_offsets[shared_gens:]
            if K_b > shared_gens:
                new_offsets[K_a:] = b_offsets[shared_gens:]
            return PatchesZonotope(
                self.center + other.center, new_patches, new_offsets,
                self.out_shape)

        # Scatter path: per-gen anchors. The combined patch's top-left
        # is at min(a_offset[k], b_offset[k]) per shared gen.
        a_ry = torch.clamp(delta[:, 0], min=0)
        a_rx = torch.clamp(delta[:, 1], min=0)
        b_ry = torch.clamp(-delta[:, 0], min=0)
        b_rx = torch.clamp(-delta[:, 1], min=0)
        # Per-gen combined extent.
        ext_y_per = torch.maximum(a_ry + kH_a, b_ry + kH_b)
        ext_x_per = torch.maximum(a_rx + kW_a, b_rx + kW_b)
        new_kH = int(ext_y_per.max().item())
        new_kW = int(ext_x_per.max().item())
        new_K = K_a + K_b - shared_gens
        new_patches = torch.zeros(
            new_K, C, new_kH, new_kW, dtype=dtype, device=device)

        def _scatter_per_gen(dst_slice, src, ry, rx, kH_s, kW_s):
            """Scatter ``src`` (gens, C, kH_s, kW_s) into ``dst_slice``
            (gens, C, new_kH, new_kW) at per-gen offsets (ry, rx)."""
            n = src.shape[0]
            if n == 0:
                return
            dys = torch.arange(kH_s, device=device)
            dxs = torch.arange(kW_s, device=device)
            ty = (ry[:, None, None] + dys[None, :, None]
                  ).expand(n, kH_s, kW_s)
            tx = (rx[:, None, None] + dxs[None, None, :]
                  ).expand(n, kH_s, kW_s)
            flat_idx = (ty * new_kW + tx).reshape(n, kH_s * kW_s)
            idx = flat_idx.unsqueeze(1).expand(n, C, kH_s * kW_s)
            dst_flat = dst_slice.reshape(n, C, new_kH * new_kW)
            dst_flat.scatter_add_(2, idx, src.reshape(n, C, kH_s * kW_s))

        # Shared portion.
        if shared_gens > 0:
            _scatter_per_gen(
                new_patches[:shared_gens], a_patches[:shared_gens],
                a_ry, a_rx, kH_a, kW_a)
            _scatter_per_gen(
                new_patches[:shared_gens], b_patches[:shared_gens],
                b_ry, b_rx, kH_b, kW_b)
        # A's extras at top-left (their offsets target feature-map directly).
        if K_a > shared_gens:
            zero_a = torch.zeros(
                K_a - shared_gens, dtype=torch.long, device=device)
            _scatter_per_gen(
                new_patches[shared_gens:K_a], a_patches[shared_gens:],
                zero_a, zero_a, kH_a, kW_a)
        if K_b > shared_gens:
            zero_b = torch.zeros(
                K_b - shared_gens, dtype=torch.long, device=device)
            _scatter_per_gen(
                new_patches[K_a:], b_patches[shared_gens:],
                zero_b, zero_b, kH_b, kW_b)

        # New offsets: shared use min(a_o, b_o) per gen; extras keep theirs.
        new_offsets = torch.empty(
            new_K, 2, dtype=torch.long, device=device)
        if shared_gens > 0:
            new_offsets[:shared_gens, 0] = a_offsets[:shared_gens, 0] - a_ry
            new_offsets[:shared_gens, 1] = a_offsets[:shared_gens, 1] - a_rx
        if K_a > shared_gens:
            new_offsets[shared_gens:K_a] = a_offsets[shared_gens:]
        if K_b > shared_gens:
            new_offsets[K_a:] = b_offsets[shared_gens:]

        return PatchesZonotope(
            self.center + other.center, new_patches, new_offsets,
            self.out_shape)
