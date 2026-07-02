"""LinMap: the one linear-map abstraction of the core.

Every affine op in a network (Gemm, Conv, avg-pool, slice/transpose/gather
reindexing, per-element scale+shift) is a LinMap: y = lin(x) + bias. All
propagators are written against this interface only, so adding a physical
layout (dense, conv, index, diagonal, later: patches) never touches them.

The interface is four maps plus the bias:

  lin(X)      apply the linear part to a batch of row vectors, (B,n) -> (B,m)
  lin_t(Y)    apply the transpose,                             (B,m) -> (B,n)
  lin_abs(X)  apply the elementwise-absolute linear part to a NONNEGATIVE
              batch, (B,n) -> (B,m); used for interval radii and zonotope
              concretization without materializing |W|
  point(x)    lin(x) + bias

All tensors are torch, flat per design (a conv LinMap reshapes internally).
A LinMap is constructed with numpy params and lazily materializes torch
tensors per (device, dtype), cached, so the same Net serves cpu/gpu runs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class LinMap:
    """Base: y = lin(x) + bias, with transpose and abs variants."""

    n_in: int
    n_out: int

    def __init__(self):
        self._cache = {}

    def _p(self, arr, X):
        """Torch view of numpy param `arr` on X's device/dtype (cached)."""
        key = (id(arr), X.device, X.dtype)
        t = self._cache.get(key)
        if t is None:
            t = torch.as_tensor(arr, device=X.device, dtype=X.dtype)
            self._cache[key] = t
        return t

    def lin(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def lin_t(self, Y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def lin_abs(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bias_vec(self, X: torch.Tensor) -> torch.Tensor:
        """The bias as a flat (n_out,) tensor on X's device/dtype (0 if none)."""
        raise NotImplementedError

    def point(self, x: torch.Tensor) -> torch.Tensor:
        """(B,n) -> (B,m) exact affine evaluation."""
        return self.lin(x) + self.bias_vec(x)


class Dense(LinMap):
    """y = W x + b with a dense (m,n) weight."""

    def __init__(self, W: np.ndarray, b: np.ndarray | None):
        super().__init__()
        assert W.ndim == 2, W.shape
        self.W = W
        self.b = b
        self.n_in, self.n_out = W.shape[1], W.shape[0]
        self._absW = None

    def lin(self, X):
        return X @ self._p(self.W, X).T

    def lin_t(self, Y):
        return Y @ self._p(self.W, Y)

    def lin_abs(self, X):
        if self._absW is None:
            self._absW = np.abs(self.W)
        return X @ self._p(self._absW, X).T

    def bias_vec(self, X):
        if self.b is None:
            return torch.zeros(self.n_out, device=X.device, dtype=X.dtype)
        return self._p(self.b, X)


class Conv2d(LinMap):
    """2D convolution as a LinMap over flat vectors.

    kernel (C_out, C_in/groups, kH, kW); in_shape/out_shape are (C,H,W).
    1D convs are lifted to 2D at conversion (kH=1). lin_t is the exact
    adjoint via conv_transpose2d (correct for any stride/padding).
    """

    def __init__(self, kernel, bias, in_shape, out_shape, stride, padding,
                 groups=1):
        super().__init__()
        self.kernel, self.b = kernel, bias
        self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
        self.stride, self.padding, self.groups = tuple(stride), tuple(padding), groups
        self.n_in = int(np.prod(in_shape))
        self.n_out = int(np.prod(out_shape))
        self._absk = None

    def _conv(self, X, kernel):
        B = X.shape[0]
        x4 = X.reshape(B, *self.in_shape)
        y = F.conv2d(x4, self._p(kernel, X), stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return y.reshape(B, self.n_out)

    def lin(self, X):
        return self._conv(X, self.kernel)

    def lin_abs(self, X):
        if self._absk is None:
            self._absk = np.abs(self.kernel)
        return self._conv(X, self._absk)

    def lin_t(self, Y):
        # exact adjoint (autograd-verified): conv_transpose2d with
        # output_padding = (H + 2p - k) mod s recovering the rows a strided
        # forward conv floor-divided away
        B = Y.shape[0]
        y4 = Y.reshape(B, *self.out_shape)
        C, H, W = self.in_shape
        kH, kW = self.kernel.shape[2], self.kernel.shape[3]
        op = ((H + 2 * self.padding[0] - kH) % self.stride[0],
              (W + 2 * self.padding[1] - kW) % self.stride[1])
        x4 = F.conv_transpose2d(y4, self._p(self.kernel, Y),
                                stride=self.stride, padding=self.padding,
                                output_padding=op, groups=self.groups)
        return x4.reshape(B, self.n_in)

    def bias_vec(self, X):
        out = torch.zeros(self.out_shape, device=X.device, dtype=X.dtype)
        if self.b is not None:
            out += self._p(self.b, X).reshape(-1, 1, 1)
        return out.reshape(self.n_out)


class ConvT2d(LinMap):
    """2D transposed convolution (kernel (C_in, C_out/groups, kH, kW))."""

    def __init__(self, kernel, bias, in_shape, out_shape, stride, padding,
                 output_padding=(0, 0), groups=1):
        super().__init__()
        self.kernel, self.b = kernel, bias
        self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
        self.stride, self.padding = tuple(stride), tuple(padding)
        self.output_padding, self.groups = tuple(output_padding), groups
        self.n_in = int(np.prod(in_shape))
        self.n_out = int(np.prod(out_shape))
        self._absk = None

    def _convt(self, X, kernel):
        B = X.shape[0]
        x4 = X.reshape(B, *self.in_shape)
        y = F.conv_transpose2d(x4, self._p(kernel, X), stride=self.stride,
                               padding=self.padding,
                               output_padding=self.output_padding,
                               groups=self.groups)
        return y.reshape(B, self.n_out)

    def lin(self, X):
        return self._convt(X, self.kernel)

    def lin_abs(self, X):
        if self._absk is None:
            self._absk = np.abs(self.kernel)
        return self._convt(X, self._absk)

    def lin_t(self, Y):
        # adjoint of conv_transpose is conv with the same kernel; with
        # output_padding < stride (ONNX invariant) the size is exact
        B = Y.shape[0]
        y4 = Y.reshape(B, *self.out_shape)
        x4 = F.conv2d(y4, self._p(self.kernel, Y), stride=self.stride,
                      padding=self.padding, groups=self.groups)
        return x4.reshape(B, self.n_in)

    def bias_vec(self, X):
        out = torch.zeros(self.out_shape, device=X.device, dtype=X.dtype)
        if self.b is not None:
            out += self._p(self.b, X).reshape(-1, 1, 1)
        return out.reshape(self.n_out)


class AvgPool(LinMap):
    """Average pooling as a LinMap (depthwise uniform conv)."""

    def __init__(self, in_shape, out_shape, kernel_shape, stride, padding):
        super().__init__()
        self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
        self.kernel_shape, self.stride, self.padding = \
            tuple(kernel_shape), tuple(stride), tuple(padding)
        self.n_in = int(np.prod(in_shape))
        self.n_out = int(np.prod(out_shape))

    def lin(self, X):
        B = X.shape[0]
        x4 = X.reshape(B, *self.in_shape)
        y = F.avg_pool2d(x4, kernel_size=self.kernel_shape,
                         stride=self.stride, padding=self.padding)
        return y.reshape(B, self.n_out)

    lin_abs = lin      # all-positive weights: |W| == W

    def lin_t(self, Y):
        # adjoint of avg-pool: distribute each output equally over its window
        # (same output_padding adjoint identity as Conv2d.lin_t)
        B = Y.shape[0]
        C, H, W = self.in_shape
        kH, kW = self.kernel_shape
        y4 = Y.reshape(B, *self.out_shape) / float(kH * kW)
        kernel = torch.ones(C, 1, kH, kW, device=Y.device, dtype=Y.dtype)
        op = ((H + 2 * self.padding[0] - kH) % self.stride[0],
              (W + 2 * self.padding[1] - kW) % self.stride[1])
        x4 = F.conv_transpose2d(y4, kernel, stride=self.stride,
                                padding=self.padding, output_padding=op,
                                groups=C)
        return x4.reshape(B, self.n_in)

    def bias_vec(self, X):
        return torch.zeros(self.n_out, device=X.device, dtype=X.dtype)


class Select(LinMap):
    """y[i] = x[idx[i]]: transpose / slice / gather / split as one gather.

    idx is a precomputed int64 array of length n_out (conversion-time index
    math holds ALL the shape complexity; propagation is a gather).
    """

    def __init__(self, idx: np.ndarray, n_in: int):
        super().__init__()
        self.idx = np.asarray(idx, dtype=np.int64)
        self.n_in, self.n_out = int(n_in), int(len(self.idx))

    def _idx(self, X):
        key = ('idx', X.device)
        t = self._cache.get(key)
        if t is None:
            t = torch.as_tensor(self.idx, device=X.device)
            self._cache[key] = t
        return t

    def lin(self, X):
        return X[:, self._idx(X)]

    lin_abs = lin

    def lin_t(self, Y):
        out = torch.zeros(Y.shape[0], self.n_in, device=Y.device, dtype=Y.dtype)
        out.index_add_(1, self._idx(Y), Y)
        return out

    def bias_vec(self, X):
        return torch.zeros(self.n_out, device=X.device, dtype=X.dtype)


class SumAxis(LinMap):
    """ReduceSum/ReduceMean over one axis: (pre, k, post) -> (pre, post)."""

    def __init__(self, pre, k, post, mean=False):
        super().__init__()
        self.pre, self.k, self.post = int(pre), int(k), int(post)
        self.scale = (1.0 / k) if mean else 1.0
        self.n_in = self.pre * self.k * self.post
        self.n_out = self.pre * self.post

    def lin(self, X):
        B = X.shape[0]
        return X.reshape(B, self.pre, self.k, self.post).sum(dim=2) \
                .reshape(B, self.n_out) * self.scale

    lin_abs = lin      # all weights are +1/k

    def lin_t(self, Y):
        B = Y.shape[0]
        y = Y.reshape(B, self.pre, 1, self.post) * self.scale
        return y.expand(B, self.pre, self.k, self.post).reshape(B, self.n_in)

    def bias_vec(self, X):
        return torch.zeros(self.n_out, device=X.device, dtype=X.dtype)


class ScaleShift(LinMap):
    """y = a * x + b elementwise (const Mul/Div/Add/Sub, folded BN).

    a and b are full flat vectors (broadcasting resolved at conversion).
    """

    def __init__(self, a: np.ndarray | None, b: np.ndarray | None, n: int):
        super().__init__()
        self.a, self.b = a, b
        self.n_in = self.n_out = int(n)
        self._absa = None

    def lin(self, X):
        return X if self.a is None else X * self._p(self.a, X)

    def lin_t(self, Y):
        return self.lin(Y)

    def lin_abs(self, X):
        if self.a is None:
            return X
        if self._absa is None:
            self._absa = np.abs(self.a)
        return X * self._p(self._absa, X)

    def bias_vec(self, X):
        if self.b is None:
            return torch.zeros(self.n_out, device=X.device, dtype=X.dtype)
        return self._p(self.b, X)
