"""Zonotope forward propagation — dense numpy and torch implementations."""

import numpy as np
import torch
import torch.nn.functional as F


def is_conv(layer):
    """Check if layer is Conv (3-tuple) vs FC (2-tuple)."""
    return len(layer) == 3


class TorchZonotope:
    """Zonotope with torch tensors on GPU/CPU for BnB verification.

    Representation: { center + G @ e | ||e||_inf <= 1 }
    """

    def __init__(self, center, generators):
        self.center = center
        self.generators = generators

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
        abs_sum = torch.abs(self.generators).sum(dim=1)
        return self.center - abs_sum, self.center + abs_sum

    def propagate_conv(self, kernel, bias, input_shape, stride, padding):
        """Propagate through a Conv2d layer."""
        self.center = F.conv2d(
            self.center.reshape(1, *input_shape), kernel, bias=bias,
            stride=stride, padding=padding).flatten()
        K = self.generators.shape[1]
        if K > 0:
            self.generators = F.conv2d(
                self.generators.T.reshape(K, *input_shape), kernel,
                stride=stride, padding=padding).reshape(K, -1).T

    def propagate_fc(self, W, bias):
        """Propagate through a fully-connected layer."""
        self.center = F.linear(self.center, W, bias)
        self.generators = W @ self.generators

    def apply_relu(self):
        """Standard min-area ReLU relaxation, appending new generators."""
        lo, hi = self.bounds()
        ust = (lo < 0) & (hi > 0)
        dead = hi <= 0
        lam = torch.where(ust, hi / (hi - lo),
                          torch.where(dead, torch.zeros_like(hi),
                                      torch.ones_like(hi)))
        mu = torch.where(ust, -hi * lo / (2 * (hi - lo)),
                         torch.zeros_like(hi))
        self.center = lam * self.center + mu
        self.generators = lam.unsqueeze(1) * self.generators
        ui = torch.where(ust)[0]
        nu = len(ui)
        if nu > 0:
            n = len(self.center)
            ng = torch.zeros(n, nu, dtype=self.center.dtype,
                             device=self.center.device)
            ng[ui, torch.arange(nu, device=self.center.device)] = mu[ui]
            self.generators = torch.cat([self.generators, ng], dim=1)
        return lo, hi

    def copy(self):
        """Return an independent copy."""
        return TorchZonotope(self.center.clone(), self.generators.clone())

    def add(self, other, shared_gens):
        """Element-wise addition for skip connections (mirrors DenseZonotope.add).

        First `shared_gens` generator columns are shared noise symbols
        (from before the fork point) — added element-wise.
        Remaining columns are branch-specific and get concatenated.
        """
        g_shared = (self.generators[:, :shared_gens]
                    + other.generators[:, :shared_gens])
        return TorchZonotope(
            self.center + other.center,
            torch.cat([g_shared,
                       self.generators[:, shared_gens:],
                       other.generators[:, shared_gens:]], dim=1))


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
