"""Zonotope forward propagation — dense numpy and torch implementations.

Generator column-order invariant (load-bearing for skip-connection add)
-----------------------------------------------------------------------
Each generator column in a zonotope's G matrix represents the sensitivity
to one noise symbol ε_i ∈ [-1, 1]. When two branches of a fork-and-merge
DAG meet at a skip-connection add, the first ``shared_gens`` columns in
both branches MUST correspond to the same noise symbols in the same
order — otherwise positional elementwise add produces UNSOUND bounds.

This invariant is maintained STATICALLY by:

  1. Every propagation op preserves column count and order:
       - ``propagate_conv`` applies the conv to each column in place
         (preserves count, no reorder).
       - ``propagate_fc`` does ``G_out = W @ G_in`` (preserves count,
         no reorder).
       - ``apply_relu`` / ``apply_relu_custom`` scale existing columns
         by λ in place, then APPEND new μ-columns at the end via
         ``torch.cat([old, new], dim=1)`` — never insert mid-matrix.
       - ``copy`` clones column-for-column.

  2. ``_find_shared_gens_count`` (in ``verify_zono_bnb.py`` and
     ``network.py``) returns the K at the deepest common fork ancestor
     — precisely the count of columns that both branches inherited in
     order from the fork.

Any future op that reorders, drops, or inserts generator columns must
also update ``_find_shared_gens_count`` and re-verify that skip merges
stay sound. No runtime tracking is done — the invariant is entirely
static.
"""

import numpy as np
import torch
import torch.nn.functional as F


def is_conv(layer):
    """Check if layer is Conv (3-tuple) vs FC (2-tuple)."""
    return len(layer) == 3


class TorchZonotope:
    """Zonotope with torch tensors on GPU/CPU for BnB verification.

    Representation: { center + G @ e | ||e||_inf <= 1 }

    The external generator view is the 2D `(n_flat, K)` tensor exposed
    via the `generators` property — shape contract unchanged. Internally
    we keep two alternate representations and pick whichever avoids
    reshapes:

      - `_gen_2d`: shape `(n_flat, K)` — canonical for FC, merge-add,
        external reads.
      - `_gen_4d`: shape `(K, C, H, W)` — canonical after a conv. Lets
        consecutive conv + relu steps avoid the non-contiguous transpose
        that the naive (n_flat, K) form forces through F.conv2d.

    Exactly one is populated at a time; the property getter transitions
    4D → 2D on demand. Setter writes store 2D. `propagate_conv` and
    `apply_relu` operate in whichever form is already populated, so a
    `conv → relu → conv → relu` sequence stays 4D end-to-end.
    """

    def __init__(self, center, generators):
        self.center = center
        self._gen_2d = generators
        self._gen_4d = None

    @property
    def generators(self):
        """External 2D view — (n_flat, K). Materializes from 4D on demand."""
        if self._gen_4d is not None:
            K = self._gen_4d.shape[0]
            self._gen_2d = self._gen_4d.reshape(K, -1).t().contiguous()
            self._gen_4d = None
        return self._gen_2d

    @generators.setter
    def generators(self, value):
        self._gen_2d = value
        self._gen_4d = None

    @property
    def n_gens(self):
        """Number of generator columns without materialising the 2D view."""
        if self._gen_4d is not None:
            return self._gen_4d.shape[0]
        return self._gen_2d.shape[1]

    @classmethod
    def from_input_bounds(cls, x_lo, x_hi, device, dtype):
        """Create zonotope from input bounds (torch tensors)."""
        center = (x_lo + x_hi) / 2
        radii = (x_hi - x_lo) / 2
        nz = torch.nonzero(radii).squeeze(1)
        n = len(center)
        K = len(nz)
        generators = torch.zeros(n, K, dtype=dtype, device=device)
        generators[nz, torch.arange(K, device=device)] = radii[nz]
        return cls(center, generators)

    def bounds(self):
        """Compute element-wise (lo, hi) bounds."""
        if self._gen_4d is not None:
            K = self._gen_4d.shape[0]
            abs_sum = torch.abs(self._gen_4d).reshape(K, -1).sum(dim=0)
        else:
            abs_sum = torch.abs(self._gen_2d).sum(dim=1)
        return self.center - abs_sum, self.center + abs_sum

    def get_gen_row(self, neuron_idx):
        """Read a single row of G at flat neuron_idx without forcing 4D→2D.

        Returns a 1-D tensor of shape `(K,)`. When _gen_4d is populated,
        this reshape-indexes the 4D tensor (cheap view; no materialization
        of the 2D cache). When _gen_2d is populated, returns a row slice.
        """
        if self._gen_4d is not None:
            K = self._gen_4d.shape[0]
            return self._gen_4d.reshape(K, -1)[:, neuron_idx]
        return self._gen_2d[neuron_idx, :]

    def propagate_conv(self, kernel, bias, input_shape, stride, padding):
        """Propagate through a Conv2d layer.

        Operates natively in 4D `(K, C, H, W)`. If incoming generators
        are in 2D form, converts once (single transpose+reshape); the
        output stays 4D so a subsequent conv/relu is reshape-free.

        For large G (K ≫ 1000), `F.conv2d` on the full generator batch
        triggers a cuDNN workspace + output-alloc that can reach 3-4 GB
        on resnet-sized nets. When K exceeds a threshold, we split G
        into chunks, convolve each, and write directly into a
        pre-allocated output — avoiding the cat-based double-alloc.
        """
        self.center = F.conv2d(
            self.center.reshape(1, *input_shape), kernel, bias=bias,
            stride=stride, padding=padding).flatten()
        # Empty-K early return must still resize gens to the post-conv
        # flat size — otherwise downstream `bounds()` mismatches the
        # (updated) center against (pre-conv) gens. Manifests on metaroom
        # specs where the input box has zero radii everywhere (point
        # input) so K==0 at the first Conv.
        n_out = self.center.numel()
        if self._gen_4d is None:
            if self._gen_2d is None or self._gen_2d.shape[1] == 0:
                self._gen_2d = torch.zeros(
                    n_out, 0, dtype=self.center.dtype,
                    device=self.center.device)
                return
            K = self._gen_2d.shape[1]
            g4d = self._gen_2d.t().contiguous().reshape(K, *input_shape)
        else:
            g4d = self._gen_4d
            if g4d.shape[0] == 0:
                # K==0 in 4D form: drop the stale 4D, write empty 2D
                # of the new flat output size.
                self._gen_4d = None
                self._gen_2d = torch.zeros(
                    n_out, 0, dtype=self.center.dtype,
                    device=self.center.device)
                return
            K = g4d.shape[0]
        # Chunked convolution: bounds cuDNN workspace to ~chunk_size gens.
        # Threshold 512 picked so chunks on 32×32 maps fit in ~300 MB.
        chunk_size = 512
        if K > chunk_size:
            # Probe output shape with a tiny slice, preallocate, stream.
            sample = F.conv2d(
                g4d[:1], kernel, stride=stride, padding=padding)
            out_shape = (K,) + tuple(sample.shape[1:])
            out = torch.empty(
                out_shape, dtype=g4d.dtype, device=g4d.device)
            out[:1] = sample
            del sample
            for start in range(1, K, chunk_size):
                end = min(start + chunk_size, K)
                out[start:end] = F.conv2d(
                    g4d[start:end], kernel, stride=stride, padding=padding)
            self._gen_4d = out
        else:
            self._gen_4d = F.conv2d(
                g4d, kernel, stride=stride, padding=padding)
        self._gen_2d = None

    def propagate_fc(self, W, bias):
        """Propagate through a fully-connected layer (forces 2D form)."""
        self.center = F.linear(self.center, W, bias)
        self.generators = W @ self.generators

    def apply_relu(self, tight_lo=None, tight_hi=None):
        """Standard min-area ReLU relaxation, appending new generators.

        Optional `tight_lo` / `tight_hi` are externally-computed sound
        over-approximations of the pre-activation range (e.g. from a
        per-neuron adaptive-zonotope backward pass or an LP probe).
        When provided, the relu relaxation uses the intersection of
        these with the zonotope's internal bounds. This is sound: the
        true reachable set is contained in both the zonotope and the
        external bounds, so the triangle relaxation using the
        intersection still covers all true post-relu values.

        Returns the (lo, hi) *actually used* for the relaxation, so
        the caller can record the tightest pre-activation bounds.

        Operates in-place on 4D when the generators are already in 4D;
        otherwise in 2D. Output shape matches the input shape.
        """
        lo_int, hi_int = self.bounds()
        lo = lo_int if tight_lo is None else torch.maximum(lo_int, tight_lo)
        hi = hi_int if tight_hi is None else torch.minimum(hi_int, tight_hi)
        ust = (lo < 0) & (hi > 0)
        dead = hi <= 0
        lam = torch.where(ust, hi / (hi - lo),
                          torch.where(dead, torch.zeros_like(hi),
                                      torch.ones_like(hi)))
        mu = torch.where(ust, -hi * lo / (2 * (hi - lo)),
                         torch.zeros_like(hi))
        self.center = lam * self.center + mu
        ui = torch.where(ust)[0]
        nu = len(ui)

        if self._gen_4d is not None:
            K, C, H, W = self._gen_4d.shape
            lam_4d = lam.reshape(1, C, H, W)
            # In-place scale: avoids doubling G transiently.
            self._gen_4d.mul_(lam_4d)
            if nu > 0:
                new_flat = torch.zeros(
                    nu, C * H * W,
                    dtype=self.center.dtype, device=self.center.device)
                new_flat[torch.arange(nu, device=new_flat.device), ui] = mu[ui]
                new_4d = new_flat.reshape(nu, C, H, W)
                self._gen_4d = torch.cat([self._gen_4d, new_4d], dim=0)
        else:
            # In-place scale.
            self._gen_2d.mul_(lam.unsqueeze(1))
            if nu > 0:
                n = len(self.center)
                ng = torch.zeros(n, nu, dtype=self.center.dtype,
                                 device=self.center.device)
                ng[ui, torch.arange(nu, device=self.center.device)] = mu[ui]
                self._gen_2d = torch.cat([self._gen_2d, ng], dim=1)
        return lo, hi

    def nonzero_rows(self, unstable_idx):
        """Return sparse triples ``(row_ids, col_ids, values)`` of all
        nonzero entries in ``G[unstable_idx, :]``.

        ``row_ids`` index into ``unstable_idx`` (0..len(unstable_idx)-1),
        not into the flat neuron space, so the caller can join back via
        ``unstable_idx[row_ids]``. ``col_ids`` are gen indices.

        Both ``TorchZonotope`` and ``PatchesZonotope`` implement this so
        ``_record_zono_pre_relu_rows`` can dispatch polymorphically.
        """
        if not isinstance(unstable_idx, torch.Tensor):
            unstable_idx = torch.as_tensor(unstable_idx, dtype=torch.long)
        if self._gen_4d is not None:
            K = self._gen_4d.shape[0]
            G2d = self._gen_4d.reshape(K, -1).t()  # (n_flat, K) view
            sub = G2d.index_select(0, unstable_idx.to(G2d.device))
        else:
            sub = self._gen_2d.index_select(
                0, unstable_idx.to(self._gen_2d.device))
        nz = sub.nonzero(as_tuple=False)
        if nz.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=sub.device),
                torch.empty(0, dtype=torch.long, device=sub.device),
                torch.empty(0, dtype=sub.dtype, device=sub.device))
        rid, cid = nz[:, 0], nz[:, 1]
        return rid, cid, sub[rid, cid]

    def apply_relu_custom(self, lam, mu, shift):
        """Apply a caller-supplied (lam, mu, shift) ReLU relaxation.

        Unlike ``apply_relu`` — which computes min-area (lam, mu) from
        its internal bounds — this takes them as arguments. Used by
        α-CROWN direction-adaptive forward, which picks the tight
        triangle edge (lower for `ep > 0`, upper otherwise) per neuron.

        Semantics:
            new_center = lam * center + shift
            generators (rows) are scaled by lam
            for neurons with ``mu != 0`` a new generator column of value
            ``mu`` at that neuron index is appended.

        Shapes:
            lam, mu, shift: 1-D tensors of length ``n_flat``.
        """
        self.center = lam * self.center + shift
        if self._gen_4d is not None:
            K, C, H, W = self._gen_4d.shape
            self._gen_4d.mul_(lam.reshape(1, C, H, W))
        else:
            self._gen_2d.mul_(lam.unsqueeze(1))
        ui = torch.where(mu != 0)[0]
        nu = ui.numel()
        if nu > 0:
            # Force 2D form for the concat (cheaper than appending in 4D).
            gens_2d = self.generators  # collapses 4D -> 2D if needed
            n = self.center.numel()
            new_g = torch.zeros(
                n, nu, dtype=gens_2d.dtype, device=gens_2d.device)
            new_g[ui, torch.arange(nu, device=new_g.device)] = mu[ui]
            self._gen_2d = torch.cat([gens_2d, new_g], dim=1)
            self._gen_4d = None

    def copy(self):
        """Return an independent copy."""
        out = TorchZonotope(self.center.clone(), None)
        if self._gen_4d is not None:
            out._gen_4d = self._gen_4d.clone()
        else:
            out._gen_2d = self._gen_2d.clone()
        return out

    def to_(self, device, non_blocking=False):
        """Move this zonotope's tensors to `device` in place."""
        self.center = self.center.to(device, non_blocking=non_blocking)
        if self._gen_2d is not None:
            self._gen_2d = self._gen_2d.to(device, non_blocking=non_blocking)
        if self._gen_4d is not None:
            self._gen_4d = self._gen_4d.to(device, non_blocking=non_blocking)
        return self

    def add(self, other, shared_gens):
        """Element-wise addition for skip connections (mirrors DenseZonotope.add).

        Relies on the column-order invariant documented at the top of this
        module: the first ``shared_gens`` columns of ``self.G`` and
        ``other.G`` correspond to the same noise symbols in the same order.
        That invariant is maintained statically by propagation ops (conv,
        fc, relu, copy) and by ``_find_shared_gens_count`` returning the
        fork's K value.

        Two implementation paths
        ------------------------
        FAST (in-place): triggered when ``K_b == shared_gens`` (the skip
          branch has no branch-specific extras). The merged G is then
          structurally ``[a.G[:, :s] + b.G[:, :s] | a.G[:, s:]]``, which
          equals ``a.G`` with its first ``s`` columns incremented by
          ``b.G[:, :s]``. Mutating ``self`` avoids the fresh
          ``torch.empty(n, K_out)`` allocation (~1.5 GB per ResBlock
          merge at 32×32×64 on resnet_large). This is the standard
          ResNet skip pattern: the skip fork exits immediately after a
          ReLU and hits the add unchanged, so its K equals ``shared_gens``.
          Returns ``self`` (mutated). Caller must not rely on ``self``
          keeping its pre-add state (current use pattern in
          ``forward_zono_dir_adaptive`` / ``_forward_zonotope_interleaved``
          always replaces the old ``zono_state[...]`` entry with the
          merged result, so this is safe).

        SLOW: allocates a fresh output tensor and returns a fresh
          ``TorchZonotope``. Used when the skip branch has its own
          branch-specific generators (K_b > shared_gens), which happens
          only if a ReLU (or other gen-appending op) sat on the skip
          path — rare in standard ResNets but correct in general.
        """
        # Force 2D form; cheap if already there.
        a = self.generators
        b = other.generators
        n = a.shape[0]
        K_a, K_b = a.shape[1], b.shape[1]

        # Invariant guards (cheap shape checks; no value check possible —
        # see docstring for why).
        assert a.shape[0] == b.shape[0], (
            f"row-dim mismatch in add: {a.shape[0]} vs {b.shape[0]}")
        assert 0 <= shared_gens <= K_a, (
            f"shared_gens={shared_gens} out of range [0, {K_a}] (K_a)")
        assert 0 <= shared_gens <= K_b, (
            f"shared_gens={shared_gens} out of range [0, {K_b}] (K_b)")

        # When any operand requires grad (α-CROWN / α-zono optimization), the
        # in-place / out= ops below break autograd — use functional ops then.
        _grad = (a.requires_grad or b.requires_grad
                 or self.center.requires_grad or other.center.requires_grad)

        if K_b == shared_gens:
            # Fast path: skip has no extras. Merged G = [a[:,:s]+b[:,:s] | a[:,s:]].
            # Column ordering invariant is maintained statically by the
            # propagation ops (see module docstring).
            if not _grad:
                # mutate self in place (no-grad forward; perf)
                if shared_gens > 0:
                    a[:, :shared_gens].add_(b[:, :shared_gens])
                self.center.add_(other.center)
                return self
            if shared_gens > 0:
                g_out = torch.cat([a[:, :shared_gens] + b[:, :shared_gens],
                                   a[:, shared_gens:]], dim=1)
            else:
                g_out = a
            return TorchZonotope(self.center + other.center, g_out)

        # Slow path: skip has its own branch-specific generators.
        parts = []
        if shared_gens > 0:
            parts.append(a[:, :shared_gens] + b[:, :shared_gens])
        if K_a > shared_gens:
            parts.append(a[:, shared_gens:])
        if K_b > shared_gens:
            parts.append(b[:, shared_gens:])
        out = (torch.cat(parts, dim=1) if parts
               else a.new_zeros(n, 0))
        return TorchZonotope(self.center + other.center, out)


def make_input_zonotope(settings, x_lo, x_hi, device, dtype, in_shape=None):
    """Build the initial input zonotope per ``settings.zono_impl``.

    Args:
        settings: DotMap-style settings (uses ``zono_impl``: 'dense' or
            'patches'; default 'dense').
        x_lo, x_hi: input bound tensors (1-D flat).
        device, dtype: torch placement.
        in_shape: (C, H, W) for image inputs. Required for 'patches' mode;
            ignored for 'dense'.

    Returns:
        A ``TorchZonotope`` (dense) or ``PatchesZonotope`` (patches) with
        the same external API.
    """
    impl = str(getattr(settings, 'zono_impl', 'dense'))
    if impl == 'patches':
        # Strip a leading batch dim if the network reports (1, C, H, W).
        if in_shape is not None and len(in_shape) == 4 and in_shape[0] == 1:
            in_shape = tuple(in_shape[1:])
        # Patches representation only makes sense for image-shaped inputs.
        # Sequential / FC-only networks (in_shape None or non-3D) get dense.
        if in_shape is None or len(in_shape) != 3:
            return TorchZonotope.from_input_bounds(x_lo, x_hi, device, dtype)
        # Local import to avoid circular dependency at module load.
        from .patches_zonotope import PatchesZonotope
        return PatchesZonotope.from_input_bounds(
            x_lo, x_hi, in_shape, device, dtype)
    assert impl == 'dense', f"unknown zono_impl={impl!r}"
    return TorchZonotope.from_input_bounds(x_lo, x_hi, device, dtype)


def conv_output_shape(input_shape, kernel, params):
    """Compute output spatial shape for a Conv layer."""
    C_in, H_in, W_in = input_shape
    C_out = kernel.shape[0]
    kH, kW = kernel.shape[2], kernel.shape[3]
    sH, sW = params['stride']
    pH, pW = params['padding']
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    return (C_out, H_out, W_out)


class DenseZonotope:
    """Zonotope with dense numpy center and generator matrix.

    Representation: { center + G @ e | ||e||_inf <= 1 }

    Attributes:
        center: (n,) array
        generators: (n, k) array — one column per noise symbol
    """

    def __init__(self, center: np.ndarray, generators: np.ndarray):
        self.center = center
        self.generators = generators

    @property
    def dtype(self):
        return self.center.dtype

    @classmethod
    def from_input_bounds(cls, x_lo: np.ndarray, x_hi: np.ndarray,
                          dtype=None) -> 'DenseZonotope':
        center = (x_lo + x_hi) / 2
        radii = (x_hi - x_lo) / 2
        if dtype is not None:
            assert center.dtype == dtype, (
                f"x_lo/x_hi dtype {x_lo.dtype} != expected {dtype}")
        # Only create generator columns for dimensions with nonzero radius
        nonzero = np.nonzero(radii)[0]
        n = len(center)
        generators = np.zeros((n, len(nonzero)), dtype=center.dtype)
        for i, j in enumerate(nonzero):
            generators[j, i] = radii[j]
        return cls(center, generators)

    def bounds(self):
        """Compute element-wise lower and upper bounds."""
        abs_sum = np.abs(self.generators).sum(axis=1)
        return self.center - abs_sum, self.center + abs_sum

    def propagate_linear(self, layer):
        """Propagate through a linear layer (FC or Conv)."""
        if is_conv(layer):
            self._propagate_conv(layer)
        else:
            W, b = layer
            assert W.dtype == self.dtype, f"W dtype {W.dtype} != zonotope dtype {self.dtype}"
            self.center = W @ self.center + b
            self.generators = W @ self.generators

    def _propagate_conv(self, layer):
        """Propagate through a Conv layer via torch conv2d.

        Uses pre-cached torch tensors from params if available (set during
        graph loading), otherwise creates them on the fly. Matches the
        zonotope's dtype (float32 or float64).
        """
        kernel, bias, params = layer
        input_shape = params['input_shape']
        stride, padding = params['stride'], params['padding']
        torch_dt = torch.float32 if self.dtype == np.float32 else torch.float64
        cache_key = '_torch_kernel_f32' if torch_dt == torch.float32 else '_torch_kernel'
        bias_key = '_torch_bias_f32' if torch_dt == torch.float32 else '_torch_bias'
        if cache_key not in params:
            params[cache_key] = torch.tensor(kernel, dtype=torch_dt)
            params[bias_key] = torch.tensor(bias, dtype=torch_dt)
        k = params[cache_key]
        b = params[bias_key]

        c_4d = torch.tensor(self.center, dtype=torch_dt).reshape(1, *input_shape)
        self.center = F.conv2d(c_4d, k, bias=b, stride=stride, padding=padding).flatten().numpy()

        n_gen = self.generators.shape[1]
        if n_gen == 0:
            out_shape = conv_output_shape(input_shape, kernel, params)
            self.generators = np.zeros((out_shape[0] * out_shape[1] * out_shape[2], 0),
                                       dtype=self.center.dtype)
        else:
            g_batch = torch.tensor(self.generators.T, dtype=torch_dt).reshape(n_gen, *input_shape)
            g_out = F.conv2d(g_batch, k, stride=stride, padding=padding)
            self.generators = g_out.reshape(n_gen, -1).numpy().T

    def propagate_conv_transpose(self, kernel, bias, input_shape,
                                    stride, padding, output_padding=(0, 0)):
        """Propagate through ConvTranspose2d. Kernel layout (C_in, C_out, kH, kW).

        Mirrors `_propagate_conv` — apply F.conv_transpose2d to the center
        (with bias) and to each generator (without bias).
        """
        torch_dt = torch.float32 if self.dtype == np.float32 else torch.float64
        k = torch.as_tensor(kernel, dtype=torch_dt)
        b = (torch.as_tensor(bias, dtype=torch_dt)
             if bias is not None else None)
        c_4d = torch.as_tensor(self.center,
                                 dtype=torch_dt).reshape(1, *input_shape)
        c_out = F.conv_transpose2d(
            c_4d, k, bias=b, stride=stride, padding=padding,
            output_padding=output_padding)
        self.center = c_out.flatten().numpy().astype(self.dtype)
        n_gen = self.generators.shape[1]
        if n_gen == 0:
            self.generators = np.zeros((self.center.shape[0], 0),
                                          dtype=self.dtype)
        else:
            g_batch = torch.as_tensor(
                self.generators.T, dtype=torch_dt).reshape(n_gen, *input_shape)
            g_out = F.conv_transpose2d(
                g_batch, k, stride=stride, padding=padding,
                output_padding=output_padding)
            self.generators = g_out.reshape(n_gen, -1).numpy().T.astype(
                self.dtype)

    def _propagate_conv_slow(self, layer):
        """Original Conv implementation without kernel caching (for testing)."""
        kernel, bias, params = layer
        input_shape = params['input_shape']
        stride, padding = params['stride'], params['padding']
        k = torch.tensor(kernel, dtype=torch.float64)
        b = torch.tensor(bias, dtype=torch.float64)

        c_4d = torch.tensor(self.center, dtype=torch.float64).reshape(1, *input_shape)
        self.center = F.conv2d(c_4d, k, bias=b, stride=stride, padding=padding).flatten().numpy()

        n_gen = self.generators.shape[1]
        if n_gen == 0:
            out_shape = conv_output_shape(input_shape, kernel, params)
            self.generators = np.zeros((out_shape[0] * out_shape[1] * out_shape[2], 0))
        else:
            g_batch = torch.tensor(self.generators.T, dtype=torch.float64).reshape(n_gen, *input_shape)
            g_out = F.conv2d(g_batch, k, stride=stride, padding=padding)
            self.generators = g_out.reshape(n_gen, -1).numpy().T

    def apply_relu(self, pre_lo: np.ndarray, pre_hi: np.ndarray, relu_type: str = 'std'):
        """Apply ReLU relaxation, appending new error generators for unstable neurons.

        Only touches dead and unstable neuron indices — active neurons (the
        common case) are left untouched.  When there are no unstable neurons
        (e.g. point propagation) the unstable block is skipped entirely.

        Args:
            pre_lo, pre_hi: pre-ReLU bounds (used to classify neurons)
            relu_type: 'std' | 'y_bloat' | 'box'
        """
        dead = np.where(pre_hi <= 0)[0]
        unstable = np.where((pre_lo < 0) & (pre_hi > 0))[0]

        if len(dead) > 0:
            self.center[dead] = 0.0
            self.generators[dead, :] = 0.0

        if len(unstable) > 0:
            u_lo = pre_lo[unstable]
            u_hi = pre_hi[unstable]
            if relu_type == 'std':
                lam = u_hi / (u_hi - u_lo)
                mu = -u_hi * u_lo / (2 * (u_hi - u_lo))
            elif relu_type == 'y_bloat':
                lam = np.ones(len(unstable), dtype=self.dtype)
                mu = -u_lo / 2
            elif relu_type == 'box':
                lam = np.zeros(len(unstable), dtype=self.dtype)
                mu = u_hi / 2
            else:
                assert False, f"Unknown relu_type: {relu_type}"

            self.center[unstable] = lam * self.center[unstable] + mu
            self.generators[unstable, :] = lam[:, None] * self.generators[unstable, :]

            n = len(self.center)
            new_g = np.zeros((n, len(unstable)), dtype=self.dtype)
            new_g[unstable, np.arange(len(unstable))] = mu
            self.generators = np.hstack([self.generators, new_g])

    def apply_relu_slow(self, pre_lo: np.ndarray, pre_hi: np.ndarray, relu_type: str = 'std'):
        """Original scalar-loop implementation of apply_relu (for testing)."""
        n = len(self.center)
        scale = np.ones(n)
        offsets = np.zeros(n)

        for j in range(n):
            lo, hi = pre_lo[j], pre_hi[j]
            if hi <= 0:
                scale[j] = 0.0
            elif lo < 0:
                if relu_type == 'std':
                    lam = hi / (hi - lo)
                    mu = -hi * lo / (2 * (hi - lo))
                elif relu_type == 'y_bloat':
                    lam = 1.0
                    mu = -lo / 2
                elif relu_type == 'box':
                    lam = 0.0
                    mu = hi / 2
                else:
                    assert False, f"Unknown relu_type: {relu_type}"
                scale[j] = lam
                offsets[j] = mu

        self.center = scale * self.center + offsets

        # Scale existing generators, append one new column per unstable neuron with mu > 0
        new_cols = np.where((pre_lo < 0) & (pre_hi > 0) & (offsets > 0))[0]
        new_g = np.zeros((n, self.generators.shape[1] + len(new_cols)))
        new_g[:, :self.generators.shape[1]] = scale[:, None] * self.generators
        for i, j in enumerate(new_cols):
            new_g[j, self.generators.shape[1] + i] = offsets[j]
        self.generators = new_g

    def copy(self):
        """Return an independent copy of this zonotope."""
        return DenseZonotope(self.center.copy(), self.generators.copy())

    def add(self, other, shared_gens):
        """Element-wise addition with another zonotope (for skip connections).

        The first `shared_gens` generator columns are shared noise symbols
        (from before the fork point) — these are added element-wise.
        Remaining columns are branch-specific and get concatenated.
        """
        g_shared = self.generators[:, :shared_gens] + other.generators[:, :shared_gens]
        g_self_extra = self.generators[:, shared_gens:]
        g_other_extra = other.generators[:, shared_gens:]
        return DenseZonotope(
            self.center + other.center,
            np.hstack([g_shared, g_self_extra, g_other_extra]),
        )


# ---------------------------------------------------------------------------
# Bilinear ops with one point side, and ReduceSum (linear).
#
# Used by nn4sys mscn_* models: masked mean encoded as
#   Div(ReduceSum(features * mask), ReduceSum(mask))
# where features vary with the input and mask is constant per-disjunct.
# The dispatch site in `_forward_zonotope_graph` calls these helpers
# after detecting whether the bilinear's other side is a point zonotope
# (zero-radius generators); both-varying raises NotImplementedError
# rather than silently propagating an unsound bound.
# ---------------------------------------------------------------------------


def _torch_zono_reduce_sum(center, gens, in_shape_nd, axes, keepdims):
    """Linear ReduceSum on a flat (n,) center + (n, K) gens, reshaped
    to `in_shape_nd` for axis indexing.

    Returns (center_out, gens_out) — both flat per the same convention
    as the rest of TorchZonotope's API (`(n_out,)` and `(n_out, K)`).

    ReduceSum is purely linear: summing across an axis sums centers and
    generators identically. The result's flat layout is the post-reduce
    nd-shape flattened in C order.
    """
    K = gens.shape[1] if gens.numel() > 0 else 0
    c_nd = center.reshape(*in_shape_nd)
    if K > 0:
        g_nd = gens.reshape(*in_shape_nd, K)
    for ax in sorted(axes, reverse=True):
        c_nd = c_nd.sum(dim=ax, keepdim=bool(keepdims))
        if K > 0:
            g_nd = g_nd.sum(dim=ax, keepdim=bool(keepdims))
    c_out = c_nd.flatten() if not keepdims else c_nd.reshape(-1)
    if K > 0:
        g_out = g_nd.reshape(-1, K)
    else:
        g_out = gens.new_zeros(c_out.numel(), 0)
    return c_out, g_out


def _torch_zono_mul_bilinear(c_a, g_a, c_b, g_b,
                              shape_a=None, shape_b=None, shape_out=None,
                              shared_gens=None):
    """Element-wise Mul(a, b). Supports ND broadcasting via optional shapes.

    Both sides varying = a bilinear (nonlinear) product, soundly
    over-approximated as a zonotope. CRITICAL noise bookkeeping: only the
    first ``shared_gens`` generator columns of a and b are the SAME noise
    symbols (the shared prefix from the common-ancestor fork). Columns beyond
    that are BRANCH-SPECIFIC and INDEPENDENT.

    Column layout of the returned generators (the forward handler assigns
    column IDs to match — see verify_zono_bnb mul_bilinear):
        [ shared (s) | a_extra (K_a-s) | b_extra (K_b-s) | box (fresh) ]
    The FIRST-ORDER terms are EXACT and kept as linear columns (no boxing):
        center:        ca⊙cb  + 0.5·Σ_{k<s} Ga[:,k]⊙Gb[:,k]   (diagonal shift)
        shared k<s:    (Ga[:,k]⊙cb + ca⊙Gb[:,k]) e_k          (same noise both)
        a_extra k≥s:   (Ga[:,k]⊙cb) e^a_k                     (a's own noise)
        b_extra j≥s:   (ca⊙Gb[:,j]) e^b_j                     (b's own noise)
    Keeping a_extra/b_extra linear (rather than boxing them) is sound because
    the forward handler now tracks explicit per-column IDs: a downstream merge
    aligns by ID (common-prefix), so a branch's own noise is never conflated
    with a different branch's noise at the same position. The SECOND-ORDER
    remainder is boxed into one fresh per-element column:
        box_i = radA_i·radB_i − 0.5·Σ_{k<s}|Ga[i,k]·Gb[i,k]|
    The subtraction is the shared-noise diagonal tightening: those terms are
    Σ_{k<s} Ga_k Gb_k e_k² with e_k² ∈ [0,1] (NOT [−1,1]), so they have a
    +0.5·Σ Ga_k Gb_k center shift and only ±0.5·Σ|Ga_k Gb_k| spread, vs the
    ±Σ|Ga_k Gb_k| a symmetric box would charge. Reductions: s==0 → decorrelated
    interval product (a/b extras = full gens, box = radA·radB); s==K_a==K_b →
    full linear correlation + diagonal-tightened quadratic. ``shared_gens=None``
    defaults to 0 (always sound; the merge handlers pass the real fork count).
    """
    a_is_point = (g_a.numel() == 0
                  or bool(g_a.abs().max() < 1e-12))
    b_is_point = (g_b.numel() == 0
                  or bool(g_b.abs().max() < 1e-12))
    if not (a_is_point or b_is_point):
        sa = shape_a if shape_a is not None else (c_a.numel(),)
        sb = shape_b if shape_b is not None else (c_b.numel(),)
        K_a = g_a.shape[1]
        K_b = g_b.shape[1]
        s = 0 if shared_gens is None else int(max(0, min(shared_gens, K_a, K_b)))
        dev = c_a.device; dt = c_a.dtype
        a_nd = c_a.reshape(*sa)
        b_nd = c_b.reshape(*sb)
        c_out_nd = a_nd * b_nd
        GA_nd = g_a.t().reshape(K_a, *sa)   # (K_a, *sa)
        GB_nd = g_b.t().reshape(K_b, *sb)   # (K_b, *sb)
        # Shared-noise diagonal: e_k² ∈ [0,1] → center shift + reduced spread.
        if s > 0:
            prod_s = GA_nd[:s] * GB_nd[:s]                  # (s, *out)
            diag_sum = prod_s.sum(dim=0)                    # Σ Ga_k Gb_k
            diag_abs = prod_s.abs().sum(dim=0)              # Σ|Ga_k Gb_k|
            c_out_nd = c_out_nd + 0.5 * diag_sum * torch.ones_like(c_out_nd)
        else:
            diag_abs = torch.zeros_like(c_out_nd)
        n_out = c_out_nd.numel()
        cols = []
        if s > 0:                                           # shared linear
            lin_s = (GA_nd[:s] * b_nd.unsqueeze(0)
                     + a_nd.unsqueeze(0) * GB_nd[:s]).expand(s, *c_out_nd.shape)
            cols.append(lin_s.reshape(s, n_out).t())
        if K_a > s:                                         # a's own noise
            lin_a = (GA_nd[s:] * b_nd.unsqueeze(0)).expand(
                K_a - s, *c_out_nd.shape)
            cols.append(lin_a.reshape(K_a - s, n_out).t())
        if K_b > s:                                         # b's own noise
            lin_b = (a_nd.unsqueeze(0) * GB_nd[s:]).expand(
                K_b - s, *c_out_nd.shape)
            cols.append(lin_b.reshape(K_b - s, n_out).t())
        G_lin = (torch.cat(cols, dim=1).contiguous() if cols
                 else torch.zeros(n_out, 0, dtype=dt, device=dev))
        radA = GA_nd.abs().sum(dim=0)
        radB = GB_nd.abs().sum(dim=0)
        box = ((radA * radB) * torch.ones_like(c_out_nd)
               - 0.5 * diag_abs * torch.ones_like(c_out_nd)).clamp_min(0.0)
        Rf = box.reshape(-1)
        nz = torch.nonzero(Rf).flatten()
        G_box = torch.zeros(n_out, nz.numel(), dtype=dt, device=dev)
        G_box[nz, torch.arange(nz.numel(), device=dev)] = Rf[nz]
        return c_out_nd.reshape(-1), torch.cat([G_lin, G_box], dim=1)

    if shape_a is None and shape_b is None:
        # Plain element-wise (same shape both sides).
        if a_is_point and b_is_point:
            return c_a * c_b, c_a.new_zeros(c_a.numel(), 0)
        if b_is_point:
            return c_a * c_b, g_a * c_b.unsqueeze(-1)
        return c_a * c_b, g_b * c_a.unsqueeze(-1)

    # Broadcasting path: reshape to nd, multiply.
    a_nd = c_a.reshape(*shape_a)
    b_nd = c_b.reshape(*shape_b)
    c_out_nd = a_nd * b_nd
    out_shape = shape_out if shape_out is not None else c_out_nd.shape
    c_out = c_out_nd.reshape(-1)
    if a_is_point and b_is_point:
        return c_out, c_out.new_zeros(c_out.numel(), 0)
    # For a point operand only the OTHER side's generators propagate (= the
    # constant scales each generator column), so the output column count is that
    # side's OWN count — NOT max(K_a, K_b). A point operand may carry more (zero)
    # columns than the varying one (mismatched K from different fork ancestries);
    # using max() then reshaped the varying gens to the wrong width and crashed.
    if b_is_point:
        if g_a.numel() == 0:
            return c_out, c_out.new_zeros(c_out.numel(), 0)
        Ka = g_a.shape[1]
        g_a_nd = g_a.reshape(*shape_a, Ka)
        g_out_nd = (g_a_nd * b_nd.unsqueeze(-1)).expand(*out_shape, Ka)
        return c_out, g_out_nd.contiguous().reshape(-1, Ka)
    # a is point
    if g_b.numel() == 0:
        return c_out, c_out.new_zeros(c_out.numel(), 0)
    Kb = g_b.shape[1]
    g_b_nd = g_b.reshape(*shape_b, Kb)
    g_out_nd = (g_b_nd * a_nd.unsqueeze(-1)).expand(*out_shape, Kb)
    return c_out, g_out_nd.contiguous().reshape(-1, Kb)


def _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b):
    """Div(a_vector, b_scalar) — preserves a-correlations + a↔b shared eps.

    Encoding (b sign-stable, scalar shape):
      1) Linearize v = 1/b on [b_lo, b_hi] via chord-tangent parallelogram:
           v = lam·b + mu + gamma·eps_v   (eps_v ∈ [-1, 1])
         For b>0: 1/b convex decreasing → chord (UB) parallel to tangent
         at b* = 1/sqrt(|lam|).
      2) Bilinear y_i = a_i · v as a sound zonotope:
           c_y_i = c_a_i · c_v
           g_y_i[k] = c_a_i · (lam·g_b[0,k]) + c_v · g_a_i[k]   (shared cols)
           g_y_i[K_v] = c_a_i · gamma                            (new eps_v col)
           g_y_i[K+1+i] = rad_a_i · (|lam|·rad_b + gamma)        (per-i remainder)

    Preserves a_i ↔ a_j correlation (via shared linear gens) AND a ↔ b
    correlation (via lam·g_b sharing columns with g_a). Strictly tighter
    than the 4-corner decorrelated `_torch_zono_div_bilinear` fallback when
    b is scalar.
    """
    import torch
    assert c_b.numel() == 1, (
        f'_torch_zono_div_scalar_b expects scalar b, got {c_b.shape}')
    K_a = g_a.shape[1] if g_a.numel() > 0 else 0
    K_b = g_b.shape[1] if g_b.numel() > 0 else 0
    K = max(K_a, K_b)
    n = c_a.numel()
    # Pad gens to common K (gens are eps-index-aligned by construction).
    if K_a < K:
        g_a = torch.cat([g_a, g_a.new_zeros(n, K - K_a)], dim=1)
    if K_b < K:
        g_b = torch.cat([g_b, g_b.new_zeros(1, K - K_b)], dim=1)
    rad_a = g_a.abs().sum(dim=1) if K > 0 else torch.zeros_like(c_a)
    rad_b_s = (g_b.abs().sum(dim=1) if K > 0
               else torch.zeros_like(c_b)).reshape(-1)[0]
    c_b_s = c_b.reshape(-1)[0]
    b_lo = c_b_s - rad_b_s
    b_hi = c_b_s + rad_b_s
    if not (bool((b_lo > 0).item()) or bool((b_hi < 0).item())):
        raise ZeroDivisionError(
            'div_scalar_b: denominator not sign-stable')
    # Chord-tangent for 1/b on [b_lo, b_hi].
    f_lo = 1.0 / b_lo
    f_hi = 1.0 / b_hi
    diff_b = (b_hi - b_lo).clamp(min=1e-30)
    lam = (f_hi - f_lo) / diff_b  # negative (1/b is decreasing on sign-stable)
    # b* with f'(b*) = lam.  f'(b) = -1/b^2  → b* = ±1/sqrt(|lam|).
    bstar_mag = 1.0 / lam.abs().clamp(min=1e-30).sqrt()
    bstar = torch.where(b_hi < 0, -bstar_mag, bstar_mag)
    bstar = torch.maximum(torch.minimum(bstar, b_hi), b_lo)
    f_star = 1.0 / bstar
    chord_intercept = f_lo - lam * b_lo
    tan_intercept = f_star - lam * bstar
    mu = (chord_intercept + tan_intercept) / 2
    gamma = (chord_intercept - tan_intercept).abs() / 2
    c_v_s = lam * c_b_s + mu  # scalar 1/b center
    # Linear gens for y:
    if K > 0:
        # g_v_row = lam · g_b, shape (1, K) → broadcast to (n, K) per i.
        g_y_lin = c_a.unsqueeze(-1) * (lam * g_b) + c_v_s * g_a  # (n, K)
    else:
        g_y_lin = g_a.new_zeros(n, 0)
    g_y_slack_v = (c_a * gamma).unsqueeze(-1)  # (n, 1)  shared eps_v col
    quad_mag = rad_a * (lam.abs() * rad_b_s + gamma)  # (n,)
    g_y_quad = torch.diag(quad_mag)  # (n, n)
    c_y = c_a * c_v_s
    g_y = torch.cat([g_y_lin, g_y_slack_v, g_y_quad], dim=1)
    # Sound structural tightening for SOFTMAX-LIKE pattern:
    #   b = sum(a) exactly (same gens, b's center = sum of a's centers).
    # AND a element-wise non-negative.
    # Then y_i = a_i / sum(a) satisfies y_i ∈ [0, 1] and sum_i y_i = 1.
    # We intersect the encoded y range with [0, 1] per element. For
    # pensieve-style Pow→ReduceSum→Div the Y range collapses from
    # O(magnitude of weights × box width) to O(1), often closing leaves
    # that otherwise stay below 0.
    #
    # Detection: check `c_b ≈ sum(c_a)` AND `g_b ≈ sum(g_a, axis=0)`
    # (b = ReduceSum(a) in zono-arithmetic terms). Soundness depends on
    # this exact identity; the check uses a tight tolerance.
    a_lo = c_a - rad_a
    g_a_sum = (g_a.sum(dim=0, keepdim=True) if K > 0
               else g_a.new_zeros(1, 0))
    is_softmax = (
        bool((a_lo >= -1e-9).all())
        and bool((c_b_s - c_a.sum()).abs() < 1e-6 * c_b_s.abs().clamp(min=1.0))
        and (K == 0 or bool((g_b - g_a_sum).abs().max()
                            < 1e-6 * g_b.abs().max().clamp(min=1.0))))
    if is_softmax:
        y_lo_enc = c_y - g_y.abs().sum(dim=1)
        y_hi_enc = c_y + g_y.abs().sum(dim=1)
        y_lo_c = torch.maximum(y_lo_enc, torch.zeros_like(c_y))
        y_hi_c = torch.minimum(y_hi_enc, torch.ones_like(c_y))
        # Only intersect when current encoding is LOOSER than [0, 1].
        needs_tighten = ((y_lo_enc < y_lo_c - 1e-12)
                         | (y_hi_enc > y_hi_c + 1e-12))
        if bool(needs_tighten.any()):
            # Per-element shrink: keep gens for elements not needing
            # tighten; for tight-needed elements, re-encode as
            # decorrelated box [y_lo_c, y_hi_c]. Mixed encoding:
            # `g_y_keep` retains existing gens for non-tight elements,
            # `g_y_box` adds n new diag cols for tight elements.
            # Drop old gens for tight elements (set their row to 0).
            tight_mask = needs_tighten  # (n,)
            keep_mask = ~tight_mask
            # Zero out existing gen rows for tight elements.
            g_y_keep = g_y.clone()
            g_y_keep[tight_mask] = 0
            # Tight elements get new diagonal gens of magnitude
            # (y_hi_c - y_lo_c) / 2.
            new_rad = (y_hi_c - y_lo_c) / 2
            new_center = (y_lo_c + y_hi_c) / 2
            n_y = c_y.numel()
            # Replace center for tight elements.
            c_y = torch.where(tight_mask, new_center, c_y)
            # Add diag gens for tight elements (zeros for keep).
            new_diag = torch.zeros(n_y, n_y, dtype=g_y.dtype,
                                    device=g_y.device)
            idx_t = torch.arange(n_y, device=g_y.device)
            new_diag[idx_t, idx_t] = torch.where(
                tight_mask, new_rad, torch.zeros_like(new_rad))
            g_y = torch.cat([g_y_keep, new_diag], dim=1)
    return c_y, g_y


def _torch_zono_div_bilinear(c_a, g_a, c_b, g_b, fallback='raise',
                              prefer_shared_when_scalar_b=True):
    """Element-wise Div(a, b). Four regimes:

    1) `b` is a point zonotope (zero radius): exact linear pass —
       `(c_a / c_b, g_a / c_b)`. Always applied when applicable.
    2) `b` is scalar (1 element) AND `prefer_shared_when_scalar_b`:
       use `_torch_zono_div_scalar_b` (preserves correlations).
    3) `b` is non-point but sign-stable AND `fallback='box'`: sound
       box-bound encoding using 4-corner enclosure of `a * (1/b)`,
       returning a decorrelated zonotope (new error gen per element).
       Looser than (2) when applicable; used for vector-b cases.
    4) Otherwise: `NotImplementedError`.
    """
    b_is_point = (g_b.numel() == 0
                  or bool(g_b.abs().max() < 1e-12))
    if b_is_point:
        if bool((c_b == 0).any()):
            raise ZeroDivisionError(
                'div_bilinear: denominator has a zero element')
        inv = c_b.reciprocal()
        if g_a.numel() == 0:
            return c_a * inv, g_a.new_zeros(c_a.numel(), 0)
        return c_a * inv, g_a * inv.unsqueeze(-1)
    if prefer_shared_when_scalar_b and c_b.numel() == 1:
        return _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b)
    if fallback != 'box':
        raise NotImplementedError(
            'div_bilinear: non-point denominator. 1/x is nonlinear; no '
            'sound zonotope encoding without explicit handling. '
            "Pass fallback='box' for a sound decorrelated bound when "
            'the denominator is sign-stable.')
    import torch
    K_a = g_a.shape[1] if g_a.numel() > 0 else 0
    rad_a = g_a.abs().sum(dim=1) if K_a > 0 else torch.zeros_like(c_a)
    rad_b = g_b.abs().sum(dim=1)
    a_lo = c_a - rad_a; a_hi = c_a + rad_a
    b_lo = c_b - rad_b; b_hi = c_b + rad_b
    if not (bool((b_lo > 0).all()) or bool((b_hi < 0).all())):
        raise ZeroDivisionError(
            'div_bilinear box-fallback: denominator not sign-stable '
            '(range crosses zero). 1/b unbounded.')
    # For all-positive AND all-negative b alike, 1/b on [b_lo, b_hi]
    # has range [1/b_hi, 1/b_lo].
    inv_lo = 1.0 / b_hi
    inv_hi = 1.0 / b_lo
    corners = torch.stack([
        a_lo * inv_lo, a_lo * inv_hi, a_hi * inv_lo, a_hi * inv_hi,
    ])
    out_lo = corners.min(dim=0).values
    out_hi = corners.max(dim=0).values
    c_out = (out_lo + out_hi) / 2
    rad_out = (out_hi - out_lo) / 2
    n = c_out.numel()
    new_eye = torch.diag(rad_out)
    return c_out, new_eye


def _torch_zono_pow_int(c_in, g_in, exponent, relaxation='chord',
                        tight_lo=None, tight_hi=None):
    """Element-wise x^p for integer p >= 2 as a sound zonotope.

    Returns (c_out, g_out). Two relaxation modes:

    - `relaxation='chord'` (default): chord-tangent parallelogram per
      element. For each i where [lo_i, hi_i] has a uniform-curvature
      regime (lo_i >= 0 always; or lo_i, hi_i same sign and odd p),
      use the chord slope λ_i = (hi_i^p - lo_i^p)/(hi_i - lo_i) and
      sandwich the curve between chord and tangent at the interior
      extremum. The output zonotope preserves x-y correlation through
      the λ slope; a single new gen per element carries the chord-tangent
      half-gap as slack. For mixed-curvature intervals (0 strictly
      inside [lo, hi] and even/odd cases that defy uniform chord),
      fall back to box-decorrelated for that element.

    - `relaxation='box'`: box-decorrelated encoding (no x-y correlation
      preserved). Sound, simpler, but loose.

    Soundness is the invariant in all paths.
    """
    import torch
    p = int(exponent)
    assert p == exponent, (
        f'_torch_zono_pow_int requires integer exponent, got {exponent}')
    assert p >= 2, f'_torch_zono_pow_int requires p >= 2, got {p}'
    K = g_in.shape[1] if g_in.numel() > 0 else 0
    rad_in = g_in.abs().sum(dim=1) if K > 0 else torch.zeros_like(c_in)
    lo = c_in - rad_in
    hi = c_in + rad_in
    # nonlinear-split clamp: relaxation built over the split sub-interval (the
    # chord/box bound over [tight_lo,tight_hi] is valid for the sub-domain where
    # x lies there; children of a split cover the parent — sound, same argument
    # as the affine-band op_clamps path).
    if tight_lo is not None:
        lo = torch.maximum(lo, torch.as_tensor(
            tight_lo, dtype=c_in.dtype, device=c_in.device))
    if tight_hi is not None:
        hi = torch.minimum(hi, torch.as_tensor(
            tight_hi, dtype=c_in.dtype, device=c_in.device))
        hi = torch.maximum(hi, lo)
    n = c_in.numel()
    if relaxation == 'box':
        return _pow_int_box(c_in, g_in, p, lo, hi, K)
    # Chord-tangent path. For each element decide:
    #   uniform_convex: convex on [lo, hi] AND f differentiable inside
    #   - lo >= 0 (any p >= 2)
    #   - hi <= 0 AND p even (convex for x < 0 too)
    #   uniform_concave: concave on [lo, hi]
    #   - hi <= 0 AND p odd (f = x^p concave on x < 0, since f'' = p(p-1)x^(p-2) < 0 when x < 0 and p odd)
    # else mixed → use box bound for that element.
    eps = torch.tensor(1e-12, dtype=c_in.dtype, device=c_in.device)
    diff = (hi - lo).clamp(min=1e-30)
    f_lo = lo ** p
    f_hi = hi ** p
    # Slope of chord through endpoints.
    lam = (f_hi - f_lo) / diff
    # x* where f'(x*) = lam, i.e. p * x*^(p-1) = lam → x* = (lam/p)^(1/(p-1))
    # Only valid when sign-stable (no zero crossing in derivative).
    sign_stable_pos = lo >= 0  # all x >= 0
    sign_stable_neg_oddp = (hi <= 0) & (p % 2 == 1)
    sign_stable_neg_evenp = (hi <= 0) & (p % 2 == 0)
    use_chord = sign_stable_pos | sign_stable_neg_oddp | sign_stable_neg_evenp
    # x* candidate (handle sign properly).
    # For lo >= 0 and lam >= 0: x* = (lam/p)^(1/(p-1)) >= 0.
    # For hi <= 0 (negative x): x* may be derived analogously but signs flip.
    # Use abs to compute magnitude, then attach the right sign.
    lam_abs = lam.abs()
    x_star_mag = (lam_abs / p).clamp(min=0).pow(1.0 / (p - 1))
    # For the positive branch x* >= 0; for negative branch x* <= 0.
    x_star = torch.where(hi <= 0, -x_star_mag, x_star_mag)
    # Clip x* into [lo, hi] for safety (rounding).
    x_star = torch.maximum(torch.minimum(x_star, hi), lo)
    f_star = x_star ** p
    # Chord values at x_star and tangent values: tangent has same slope
    # so y_tangent(x) = f(x_star) + lam*(x - x_star) → at x_star: f(x_star).
    # y_chord(x_star) = lam * (x_star - lo) + f(lo).
    chord_at_star = lam * (x_star - lo) + f_lo
    # Sign of curvature on the interval (1 = convex chord_above, -1 = concave chord_below).
    # For lo >= 0, p >= 2: convex (chord above curve).
    # For hi <= 0, p even: convex (chord above curve).
    # For hi <= 0, p odd: concave (chord below curve).
    # Map: convex → chord_at_star >= f_star; concave → chord_at_star <= f_star.
    # Half-gap γ = |chord_at_star - f_star| / 2 (clamp to 0).
    gap_at_star = (chord_at_star - f_star).abs()
    # Midline μ such that y(x) ∈ [λx + μ - γ, λx + μ + γ].
    # μ_chord_intercept = f_lo - λ*lo (intercept of chord line)
    # convex: chord above curve; midline = chord_intercept - γ
    # concave: chord below curve; midline = chord_intercept + γ
    # In both cases: midline_intercept = chord_intercept - sign * γ where
    #   sign = +1 convex, -1 concave. But because gap_at_star is already
    #   |chord - f|, midline always lies between the two lines:
    #   midline_intercept = (chord_intercept + tangent_intercept) / 2
    #   tangent_intercept = f_star - λ * x_star
    chord_intercept = f_lo - lam * lo
    tangent_intercept = f_star - lam * x_star
    mu = (chord_intercept + tangent_intercept) / 2
    gamma = (gap_at_star / 2)
    # Build chord-relaxed output:
    #   c_out_i = λ_i * c_in_i + μ_i
    #   g_out_i_old = λ_i * g_in_i  (preserve correlation through chord slope)
    #   g_out_i_new = γ_i  (new gen column for this element)
    c_chord = lam * c_in + mu
    if K > 0:
        g_chord_old = lam.unsqueeze(-1) * g_in
    else:
        g_chord_old = c_in.new_zeros(n, 0)
    new_eye = torch.diag(gamma)
    g_chord = torch.cat([g_chord_old, new_eye], dim=1)
    # Where chord regime isn't valid (mixed-sign interval), fall back to
    # box-decorrelated bound for that element.
    if bool((~use_chord).any()):
        # Box for those elements.
        if p % 2 == 1:
            box_lo = torch.minimum(f_lo, f_hi)
            box_hi = torch.maximum(f_lo, f_hi)
        else:
            zero_in = (lo <= 0) & (hi >= 0)
            box_lo = torch.where(zero_in, torch.zeros_like(lo),
                                   torch.minimum(f_lo, f_hi))
            box_hi = torch.maximum(f_lo, f_hi)
        c_box = (box_lo + box_hi) / 2
        rad_box = (box_hi - box_lo) / 2
        # Replace element-wise.
        c_out = torch.where(use_chord, c_chord, c_box)
        # For non-chord elements: zero out chord-derived old-gen rows and
        # set new-gen diagonal to rad_box; for chord elements: keep.
        not_chord = ~use_chord
        if K > 0:
            g_old = torch.where(use_chord.unsqueeze(-1), g_chord_old,
                                 torch.zeros_like(g_chord_old))
        else:
            g_old = g_chord_old
        # New eye column: chord -> gamma, non-chord -> rad_box.
        new_gen_diag = torch.where(use_chord, gamma, rad_box)
        new_eye_combined = torch.diag(new_gen_diag)
        g_out = torch.cat([g_old, new_eye_combined], dim=1)
        return c_out, g_out
    return c_chord, g_chord


def _pow_int_box(c_in, g_in, p, lo, hi, K):
    """Box-decorrelated zonotope encoding of x^p — sound but loses
    correlation between input and output noise."""
    import torch
    f_lo = lo ** p
    f_hi = hi ** p
    if p % 2 == 1:
        box_lo = torch.minimum(f_lo, f_hi)
        box_hi = torch.maximum(f_lo, f_hi)
    else:
        zero_in = (lo <= 0) & (hi >= 0)
        box_lo = torch.where(zero_in, torch.zeros_like(lo),
                              torch.minimum(f_lo, f_hi))
        box_hi = torch.maximum(f_lo, f_hi)
    c_out = (box_lo + box_hi) / 2
    rad_out = (box_hi - box_lo) / 2
    n = c_out.numel()
    new_eye = torch.diag(rad_out)
    if K > 0:
        zeros_pad = c_in.new_zeros(n, K)
        g_out = torch.cat([zeros_pad, new_eye], dim=1)
    else:
        g_out = new_eye
    return c_out, g_out
