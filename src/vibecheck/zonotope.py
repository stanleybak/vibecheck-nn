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
        if self._gen_4d is None:
            if self._gen_2d is None or self._gen_2d.shape[1] == 0:
                return
            K = self._gen_2d.shape[1]
            g4d = self._gen_2d.t().contiguous().reshape(K, *input_shape)
        else:
            g4d = self._gen_4d
            if g4d.shape[0] == 0:
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

        if K_b == shared_gens:
            # Fast path: skip has no extras. Mutate self in place.
            # Correctness: merged G = [a[:,:s] + b[:,:s] | a[:,s:]].
            # Since K_b == s, the slow path's `out[:, K_a:] = b[:, s:]`
            # branch would copy zero columns, so out == a with first s
            # columns updated. In-place mutation of a gives the same.
            # Column ordering invariant is maintained statically by the
            # propagation ops (see module docstring).
            if shared_gens > 0:
                a[:, :shared_gens].add_(b[:, :shared_gens])
            self.center.add_(other.center)
            return self

        # Slow path: skip has its own branch-specific generators.
        K_out = K_a + K_b - shared_gens
        out = torch.empty(n, K_out, dtype=a.dtype, device=a.device)
        # Shared prefix: element-wise sum of first `shared_gens` cols
        torch.add(a[:, :shared_gens], b[:, :shared_gens],
                   out=out[:, :shared_gens])
        # Branch extras: copy into the rest of `out`
        if K_a > shared_gens:
            out[:, shared_gens:K_a] = a[:, shared_gens:]
        if K_b > shared_gens:
            out[:, K_a:] = b[:, shared_gens:]
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
