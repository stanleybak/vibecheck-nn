"""Dense generator-based LP/MILP for spec verification.

Separates the linear transformation (G matrix per op) from the domain
(generators e). Produces a much smaller LP model than per-neuron builders:
for CIFAR100 ResNet medium ~4K vars vs ~104K for _build_optimized, with
identical LP triangle bounds. Uses GPU batched conv for G propagation.

Formulation:
  input:   y_i = c_i + G_i @ e  where e_i ∈ [-1, 1]
  conv/fc: G_new = W @ G_prev; c_new = W @ c_prev + b
  relu stable-on:  passthrough (G row unchanged)
  relu dead:       zero row
  relu unstable j: introduce e_new_j ∈ [0, hi_j] with triangle constraints
                   (or big-M if in milp_set)
  merge-Add:       G_new = G_a + G_b (padded to same columns)
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import gurobipy as grb

from .gurobi_util import optimize_checked


def forward_point(gg_ops_ser, x, input_name, output_op_name):
    """Evaluate the network at a single input point x (numpy float64).

    Supports the op types that appear in our serialized graph: conv, fc,
    relu, add (with or without is_merge), reshape. Returns numpy array.
    """
    vals = {input_name: np.asarray(x, dtype=np.float64)}
    for op in gg_ops_ser:
        nm = op['name']
        t = op['type']
        if t == 'conv':
            a = vals[op['inputs'][0]]
            C_in, H_in, W_in = op['in_shape']
            at = torch.from_numpy(a.reshape(1, C_in, H_in, W_in))
            k = torch.from_numpy(op['kernel_np']).to(at.dtype)
            b = torch.from_numpy(op['bias_np']).to(at.dtype)
            sH, sW = op['stride']; pH, pW = op['padding']
            y = F.conv2d(at, k, bias=b, stride=(sH, sW),
                         padding=(pH, pW)).flatten()
            vals[nm] = y.numpy()
        elif t == 'fc':
            a = vals[op['inputs'][0]]
            W = op['W_np']; b = op['bias_np']
            vals[nm] = W @ a + b
        elif t == 'relu':
            a = vals[op['inputs'][0]]
            vals[nm] = np.maximum(a, 0.0)
        elif t == 'add':
            a = vals[op['inputs'][0]]
            if op.get('is_merge'):
                b = vals[op['inputs'][1]]
                vals[nm] = a + b
            else:
                b = op.get('bias')
                vals[nm] = (a + np.asarray(b, dtype=np.float64).flatten()
                            if b is not None else a)
        elif t == 'reshape':
            vals[nm] = vals[op['inputs'][0]]
        elif t == 'sub':
            a = vals[op['inputs'][0]]
            b = op.get('bias')
            vals[nm] = (a - np.asarray(b, dtype=np.float64).flatten()
                        if b is not None else a)
        elif t == 'mul':
            a = vals[op['inputs'][0]]
            scale = np.asarray(op.get('scale'), dtype=np.float64).flatten()
            vals[nm] = a * scale
        elif t == 'sigmoid':
            from scipy.special import expit
            vals[nm] = expit(vals[op['inputs'][0]])
        elif t == 'tanh':
            vals[nm] = np.tanh(vals[op['inputs'][0]])
        elif t == 'reshape':
            vals[nm] = vals[op['inputs'][0]]
        else:
            raise NotImplementedError(
                f'forward_point: unsupported op {t!r}')
    return vals[output_op_name]


class _StructuredSparseG:
    """Custom storage for precompute_gen_state's per-ReLU G output.

    Exploits the fact that G_out has a very specific structure:
      - `stable_idx` rows passthrough rows from G_in (dense in the first
        `n_gens_old` cols, zero in the new cols);
      - `unstable_idx` rows have a single 1.0 entry in a new col (identity);
      - optionally `stable_new_idx` rows have a single 1.0 entry in a later
        new col (sparse mode);
      - all other rows are zero.

    Rather than allocating an `(n × n_total)` dense tensor with ~92% zeros,
    we keep:
      - `stable_g`: a dense tensor of just the stable-passthrough rows,
        shape `(len(stable_idx), n_gens_old)`;
      - `stable_idx`, `unstable_idx`, optional `stable_new_idx` and their
        new-col starts;
      - method `materialize_cols(c_lo, c_hi)` reconstructs an `(n × chunk)`
        dense slice on the fly — for use by chunked conv/fc consumers.

    Memory: stable_g is `len(stable) × n_gens_old × 4 B`. For cifar100_
    resnet_large L1: stable=8117, n_gens_old=4050 → 132 MB vs the dense
    65536×6350 = 1.55 GB → ~12× smaller.
    """
    def __init__(self, n_rows, n_total_cols, n_gens_old,
                  stable_idx, stable_g, unstable_idx,
                  unstable_new_col_start,
                  stable_new_idx=None, stable_new_col_start=None,
                  device=None, dtype=None):
        self.n_rows = int(n_rows)
        self.n_total_cols = int(n_total_cols)
        self.n_gens_old = int(n_gens_old)
        self.stable_idx = stable_idx                   # 1-d long tensor (device)
        self.stable_g = stable_g                       # dense (n_stable, n_gens_old)
        self.unstable_idx = unstable_idx               # 1-d long tensor
        self.unstable_new_col_start = int(unstable_new_col_start)
        self.stable_new_idx = stable_new_idx
        self.stable_new_col_start = (int(stable_new_col_start)
                                       if stable_new_col_start is not None else None)
        self.device = device if device is not None else stable_g.device
        self.dtype = dtype if dtype is not None else stable_g.dtype

    @property
    def shape(self):
        return (self.n_rows, self.n_total_cols)

    def materialize(self):
        """Return the full dense `(n_rows, n_total_cols)` tensor."""
        out = torch.zeros(self.n_rows, self.n_total_cols,
                           dtype=self.dtype, device=self.device)
        if self.stable_g.shape[0] > 0 and self.n_gens_old > 0:
            out[self.stable_idx, :self.n_gens_old] = self.stable_g
        if self.unstable_idx is not None and self.unstable_idx.numel() > 0:
            cols = torch.arange(
                self.unstable_new_col_start,
                self.unstable_new_col_start + self.unstable_idx.numel(),
                device=self.device, dtype=torch.long)
            out[self.unstable_idx, cols] = 1.0
        if (self.stable_new_idx is not None
                and self.stable_new_idx.numel() > 0):
            cols = torch.arange(
                self.stable_new_col_start,
                self.stable_new_col_start + self.stable_new_idx.numel(),
                device=self.device, dtype=torch.long)
            out[self.stable_new_idx, cols] = 1.0
        return out

    def materialize_cols(self, c_lo, c_hi):
        """Return dense `(n_rows, c_hi-c_lo)` slice of columns."""
        chunk = c_hi - c_lo
        out = torch.zeros(self.n_rows, chunk,
                           dtype=self.dtype, device=self.device)
        # Stable passthrough rows occupy cols [0, n_gens_old)
        lo_p = max(c_lo, 0); hi_p = min(c_hi, self.n_gens_old)
        if hi_p > lo_p and self.stable_g.shape[0] > 0:
            out[self.stable_idx, : (hi_p - lo_p)] = (
                self.stable_g[:, lo_p:hi_p])
            # If chunk slice starts beyond col 0 in the chunk, shift dst:
            if lo_p > c_lo:
                # The passthrough section is at chunk offset (lo_p - c_lo)
                # but we wrote to indices [0, hi_p-lo_p) above. Need to
                # re-do correctly. Fall back to clearer logic:
                out[self.stable_idx, : (hi_p - lo_p)] = 0.0
                out[self.stable_idx,
                    (lo_p - c_lo):(lo_p - c_lo + hi_p - lo_p)] = (
                    self.stable_g[:, lo_p:hi_p])
        # Unstable identity entries at col [unstable_new_col_start, +n_un)
        if self.unstable_idx is not None and self.unstable_idx.numel() > 0:
            un_lo = self.unstable_new_col_start
            un_hi = un_lo + self.unstable_idx.numel()
            lo_u = max(c_lo, un_lo); hi_u = min(c_hi, un_hi)
            if hi_u > lo_u:
                # Rows for those cols
                un_offset_lo = lo_u - un_lo
                un_offset_hi = hi_u - un_lo
                rows = self.unstable_idx[un_offset_lo:un_offset_hi]
                cols = torch.arange(lo_u - c_lo, hi_u - c_lo,
                                      device=self.device, dtype=torch.long)
                out[rows, cols] = 1.0
        # Stable-new identity entries
        if (self.stable_new_idx is not None
                and self.stable_new_idx.numel() > 0):
            sn_lo = self.stable_new_col_start
            sn_hi = sn_lo + self.stable_new_idx.numel()
            lo_s = max(c_lo, sn_lo); hi_s = min(c_hi, sn_hi)
            if hi_s > lo_s:
                sn_offset_lo = lo_s - sn_lo
                sn_offset_hi = hi_s - sn_lo
                rows = self.stable_new_idx[sn_offset_lo:sn_offset_hi]
                cols = torch.arange(lo_s - c_lo, hi_s - c_lo,
                                      device=self.device, dtype=torch.long)
                out[rows, cols] = 1.0
        return out

    def __getitem__(self, key):
        """Support prev_G[idx] indexing (used in conv/fc passthrough)."""
        # Tolerate the materialize-then-index pattern as a fallback;
        # if someone wants raw indexed slicing we just materialize the
        # required rows. Cheap because we go row-wise.
        if isinstance(key, (tuple, list)) and len(key) == 2:
            row_idx, col_slice = key
            full = self.materialize()
            return full[row_idx, col_slice]
        if isinstance(key, torch.Tensor):
            # row indexing: materialize and select
            full = self.materialize()
            return full[key]
        return self.materialize()[key]


def _materialize_g(g):
    """Return a dense torch tensor regardless of storage type."""
    if isinstance(g, _StructuredSparseG):
        return g.materialize()
    return g


def _conv2d_chunked_with_oom_halving(prev_G, n_gens, C_in, H_in, W_in,
                                       kernel, stride, padding,
                                       chunk_size, min_chunk=1,
                                       oom_log=None):
    """Compute F.conv2d(prev_G.t().reshape(n_gens,C,H,W), kernel) chunked.

    `prev_G` is shape (C_in*H_in*W_in, n_gens) — the un-transposed gen
    matrix. We slice columns then transpose+contiguous PER CHUNK so we
    never materialize the full (n_gens, C*H*W) tensor (that's the 1.4 GB
    allocation that OOMs on resnet_large for un-chunked).

    Output is PRE-ALLOCATED and chunks are scatter-copied into it — avoids
    the torch.cat-induced peak (which would double-buffer the output).

    On per-chunk OOM, halve chunk_size and retry the SAME slice.

    Returns (g_out_4d, final_chunk_size, oom_count).
    """
    sH, sW = stride
    pH, pW = padding
    C_out, _, kH, kW = kernel.shape
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    g_out = torch.empty(n_gens, C_out, H_out, W_out,
                         dtype=kernel.dtype, device=kernel.device)
    i = 0
    oom_count = 0
    _is_sparse = isinstance(prev_G, _StructuredSparseG)
    while i < n_gens:
        actual = min(chunk_size, n_gens - i)
        try:
            if _is_sparse:
                # Materialize just this column slice (n_rows × chunk)
                slab = prev_G.materialize_cols(i, i + actual)  # (n_rows, chunk)
                g_chunk = slab.t().contiguous().reshape(
                    actual, C_in, H_in, W_in)
                del slab
            else:
                g_chunk = prev_G[:, i:i + actual].t().contiguous().reshape(
                    actual, C_in, H_in, W_in)
            out_chunk = F.conv2d(g_chunk, kernel, bias=None,
                                  stride=stride, padding=padding)
            g_out[i:i + actual].copy_(out_chunk)
            del g_chunk, out_chunk
            i += actual
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            if chunk_size <= min_chunk:
                raise
            chunk_size = max(min_chunk, chunk_size // 2)
            torch.cuda.empty_cache()
            if oom_log is not None:
                oom_log.append({'at_index': i,
                                 'new_chunk_size': chunk_size})
    return g_out, chunk_size, oom_count


def _matmul_chunked_with_oom_halving(W, prev_G, chunk_size, min_chunk=1,
                                       oom_log=None):
    """Compute W @ prev_G col-chunked along the n_gens dim.

    prev_G shape: (n_in_features, n_gens). Output: (n_out_features, n_gens).
    Pre-allocated output (no torch.cat doubling). OOM-halve-retry per chunk.
    """
    n_gens = prev_G.shape[1]
    n_out = W.shape[0]
    out = torch.empty(n_out, n_gens, dtype=W.dtype, device=W.device)
    _is_sparse = isinstance(prev_G, _StructuredSparseG)
    j = 0
    oom_count = 0
    while j < n_gens:
        actual = min(chunk_size, n_gens - j)
        try:
            if _is_sparse:
                slab = prev_G.materialize_cols(j, j + actual)
                chunk_out = W @ slab
                del slab
            else:
                chunk_out = W @ prev_G[:, j:j + actual]
            out[:, j:j + actual].copy_(chunk_out)
            del chunk_out
            j += actual
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            if chunk_size <= min_chunk:
                raise
            chunk_size = max(min_chunk, chunk_size // 2)
            torch.cuda.empty_cache()
            if oom_log is not None:
                oom_log.append({'at_index': j,
                                 'new_chunk_size': chunk_size})
    return out, chunk_size, oom_count


def precompute_gen_state(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                         output_op_name, *,
                         device='cuda', dtype=torch.float64,
                         formulation='sparse',
                         conv_chunk_size=None,
                         g_storage='dense',
                         _oom_log=None):
    """Compute G/center arrays on GPU, return numpy-only state.

    The returned state is query-independent (no qw/qb) and milp_set-independent.
    It can be reused for all Phase 7/8 solves on the same (network, bbr) —
    eliminating the GPU-nondeterminism that previously made each build_gen_lp
    call produce slightly different coefficients.

    formulation:
      'dense' — every stable-on ReLU neuron passes through via the raw G
          row. Subsequent conv W @ G_prev multiplies these dense rows, so
          G accumulates nonzeros through conv layers and the coefficient
          range compounds (can span 13 orders of magnitude on a 10-layer
          ResNet). Fewer Gurobi vars but numerically fragile — Gurobi
          can silently certify wrong bounds on this formulation.

      'sparse' (default) — only stable-on neurons at the LAST hidden
          ReLU layer get a new generator variable bound to
          `v == c + G_in[j,:]@e`. Earlier layers still use dense
          passthrough (they're cheap and stay in the conv-compounded G).
          This "cut once, at the spec boundary" structure keeps the
          simplex basis away from the tiny cascaded-G coefficients that
          appear only in equality rows, while adding minimal overhead
          (~tens of vars for a typical network).

    Returns state dict with:
      - formulation: 'dense' or 'sparse'
      - n_input: number of input generators
      - n_gens:  total number of generators (input + unstable new gens [+ stable new gens])
      - unstable_list: list of per-unstable-neuron dicts with:
          layer_idx, neuron_idx, c_in (float), lo, hi, e_new_col,
          row_indices (np.int32), row_values (np.float64)
      - stable_list: (sparse only) list of per-stable-on dicts with same keys
          as unstable_list, with lo >= 0; build adds `v == c + G@e` and
          var bounds [lo, hi].
      - obj_c_out: np.ndarray of output centers
      - obj_G_out_csr: scipy.sparse.csr_matrix output G
    """
    gpu = torch.device(device)
    last_use = {}
    last_relu_idx = -1
    for i, op in enumerate(gg_ops_ser):
        for inp in op['inputs']:
            last_use[inp] = i
        if op['type'] == 'relu' and 'layer_idx' in op:
            last_relu_idx = max(last_relu_idx, op['layer_idx'])

    n_in = len(x_lo)
    center = {input_name: torch.tensor((x_hi + x_lo) / 2, dtype=dtype, device=gpu)}
    half_width = torch.tensor((x_hi - x_lo) / 2, dtype=dtype, device=gpu)
    G_by_op = {input_name: torch.diag(half_width)}
    n_gens = n_in
    unstable_list = []
    stable_list = []  # used only if formulation == 'sparse'

    def pad_cols(G):
        # For sparse storage: bump the recorded total cols (cheap update;
        # additional cols are implicitly zero). Materialize will produce
        # the right shape downstream.
        if isinstance(G, _StructuredSparseG):
            if G.n_total_cols < n_gens:
                G.n_total_cols = int(n_gens)
            return G
        if G.shape[1] < n_gens:
            return torch.cat([G, torch.zeros(
                G.shape[0], n_gens - G.shape[1], dtype=dtype, device=gpu)],
                dim=1)
        return G

    for op_idx, op in enumerate(gg_ops_ser):
        nm = op['name']
        t = op['type']

        if t == 'conv':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            C_in, H_in, W_in = op['in_shape']
            kernel = torch.tensor(op['kernel_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            sH, sW = op['stride']
            pH, pW = op['padding']
            c_img = prev_c.reshape(1, C_in, H_in, W_in)
            c_out = F.conv2d(c_img, kernel, bias=bias,
                             stride=(sH, sW), padding=(pH, pW)).flatten()
            if conv_chunk_size is None:
                # Default un-chunked path — preserves legacy behavior.
                # For sparse storage, materialize first.
                _prev_G_dense = _materialize_g(prev_G)
                g_img = _prev_G_dense.t().contiguous().reshape(
                    n_gens, C_in, H_in, W_in)
                g_out = F.conv2d(g_img, kernel, bias=None,
                                  stride=(sH, sW), padding=(pH, pW))
            else:
                # Chunk over n_gens — slice prev_G columns then transpose
                # PER CHUNK to avoid materializing the full (n_gens,C*H*W)
                # tensor (1.4 GB at L3 resnet_large). With OOM-halve-retry.
                g_out, _, _ = _conv2d_chunked_with_oom_halving(
                    prev_G, n_gens, C_in, H_in, W_in, kernel,
                    (sH, sW), (pH, pW),
                    chunk_size=int(conv_chunk_size),
                    oom_log=_oom_log)
            g_out = g_out.reshape(n_gens, -1).t()
            center[nm] = c_out
            G_by_op[nm] = g_out

        elif t == 'fc':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            W = torch.tensor(op['W_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            center[nm] = W @ prev_c + bias
            if conv_chunk_size is None:
                G_by_op[nm] = W @ _materialize_g(prev_G)
            else:
                # Same OOM-halve pattern for fc.
                g_out, _, _ = _matmul_chunked_with_oom_halving(
                    W, prev_G, chunk_size=int(conv_chunk_size),
                    oom_log=_oom_log)
                G_by_op[nm] = g_out

        elif t == 'relu':
            if 'layer_idx' not in op:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]
                continue
            li = op['layer_idx']
            lo_r, hi_r = bounds_by_relu[li]
            c_in = center[op['inputs'][0]]
            G_in = pad_cols(G_by_op[op['inputs'][0]])
            n = len(c_in)
            dead_mask = hi_r <= 0
            stable_mask = (lo_r >= 0) & ~dead_mask
            unstable_mask = ~dead_mask & ~stable_mask
            unstable_idx = np.where(unstable_mask)[0]
            stable_idx = np.where(stable_mask)[0]

            # 'sparse' cuts only at the last hidden ReLU (numerical
            # conditioning of deep dense-passthrough rows still bad).
            # 'all_sparse' cuts at every ReLU — stable-on neurons at
            # every layer get a fresh v_k var + equality. Keeps each
            # row numerically short (just one layer's weights) at the
            # cost of a much larger LP.
            do_sparse = (
                (formulation == 'sparse' and li == last_relu_idx)
                or formulation == 'all_sparse')
            c_in_cpu = c_in.detach().cpu().numpy()
            # Materialize only the needed rows from sparse G if applicable.
            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(unstable_idx, device=gpu, dtype=torch.long)
                if isinstance(G_in, _StructuredSparseG):
                    # Materialize just unstable rows
                    _full = G_in.materialize()  # for now, materialize all
                    G_unstable = _full[uidx_t].detach().cpu().numpy()
                    del _full
                else:
                    G_unstable = G_in[uidx_t].detach().cpu().numpy()
            if do_sparse and len(stable_idx) > 0:
                sidx_t_cpu = torch.tensor(stable_idx, device=gpu, dtype=torch.long)
                if isinstance(G_in, _StructuredSparseG):
                    _full = G_in.materialize()
                    G_stable = _full[sidx_t_cpu].detach().cpu().numpy()
                    del _full
                else:
                    G_stable = G_in[sidx_t_cpu].detach().cpu().numpy()

            # Unstable neurons always get a new gen
            for local_idx, j in enumerate(unstable_idx):
                row = G_unstable[local_idx]
                nz = np.nonzero(row)[0]
                unstable_list.append({
                    'layer_idx': li,
                    'neuron_idx': int(j),
                    'c_in': float(c_in_cpu[j]),
                    'lo': float(lo_r[j]),
                    'hi': float(hi_r[j]),
                    'e_new_col': n_gens + local_idx,
                    'row_indices': nz.astype(np.int32),
                    'row_values': row[nz].astype(np.float64),
                    # ALPHA form: the e_new column has coefficient 1.0
                    # in this neuron's post-ReLU row and is paired with
                    # `a_k ∈ [0, hi_k]` in the LP/MILP. Builders MUST
                    # NOT interpret this as the zono parallelogram.
                    # See `_build_gen_cone_lp_phase1` for the zono
                    # variant (incompatible coordinate system).
                    'form': 'alpha',
                })
            n_unstable = len(unstable_idx)

            # Sparse: stable-on neurons also get a new gen + equality constraint
            n_stable_new = 0
            if do_sparse:
                for local_idx, j in enumerate(stable_idx):
                    row = G_stable[local_idx]
                    nz = np.nonzero(row)[0]
                    stable_list.append({
                        'layer_idx': li,
                        'neuron_idx': int(j),
                        'c_in': float(c_in_cpu[j]),
                        'lo': float(lo_r[j]),
                        'hi': float(hi_r[j]),
                        'e_new_col': n_gens + n_unstable + local_idx,
                        'row_indices': nz.astype(np.int32),
                        'row_values': row[nz].astype(np.float64),
                        'form': 'alpha',
                    })
                n_stable_new = len(stable_idx)

            n_new = n_unstable + n_stable_new

            # Build output G on GPU
            import os as _os
            if _os.environ.get('VC_TRACE_GEN_STATE'):
                _bytes = n * (n_gens + n_new) * (8 if dtype == torch.float64 else 4)
                print(f'[gen_state] li={li:2d} n_layer={n:6d} n_gens_in={n_gens:6d} '
                      f'n_unstable={n_unstable:5d} n_stable={len(stable_idx):6d} '
                      f'n_gens_out={n_gens + n_new:6d}  '
                      f'G_out=({n}×{n_gens + n_new})  '
                      f'alloc={_bytes/1024/1024:.0f} MB')
            sidx_t = (torch.tensor(stable_idx, device=gpu, dtype=torch.long)
                       if len(stable_idx) > 0 else None)
            uidx_t = (torch.tensor(unstable_idx, device=gpu, dtype=torch.long)
                       if len(unstable_idx) > 0 else None)
            # Compute c_out (independent of G storage choice)
            c_out = torch.zeros(n, dtype=dtype, device=gpu)
            if not do_sparse and len(stable_idx) > 0:
                c_out[sidx_t] = c_in[sidx_t]
            # Sparse: stable-on center is absorbed into the equality constraint
            # (v == c + G@e), so c_out[j] stays 0 and the var carries the full value
            center[nm] = c_out

            if g_storage == 'sparse':
                # Structured sparse storage. Stable-passthrough rows kept as
                # a thin (n_stable × n_gens_old) dense buffer; unstable +
                # stable_new identities stored as index lists.
                if not do_sparse and sidx_t is not None:
                    if isinstance(G_in, _StructuredSparseG):
                        _full = G_in.materialize()
                        stable_g = _full[sidx_t].contiguous()
                        del _full
                    else:
                        stable_g = G_in[sidx_t].contiguous()
                else:
                    # In do_sparse mode the stable rows become identity new
                    # cols, no passthrough dense block.
                    stable_g = torch.zeros(0, n_gens, dtype=dtype, device=gpu)
                G_out = _StructuredSparseG(
                    n_rows=n, n_total_cols=n_gens + n_new,
                    n_gens_old=(n_gens if not do_sparse else 0),
                    stable_idx=(sidx_t if not do_sparse else
                                 torch.tensor([], dtype=torch.long, device=gpu)),
                    stable_g=stable_g,
                    unstable_idx=(uidx_t if uidx_t is not None
                                    else torch.tensor([], dtype=torch.long, device=gpu)),
                    unstable_new_col_start=n_gens,
                    stable_new_idx=(sidx_t if do_sparse else None),
                    stable_new_col_start=(n_gens + n_unstable if do_sparse else None),
                    device=gpu, dtype=dtype)
                G_by_op[nm] = G_out
            else:
                G_out = torch.zeros(n, n_gens + n_new, dtype=dtype, device=gpu)
                if not do_sparse and sidx_t is not None:
                    G_out[sidx_t, :n_gens] = _materialize_g(G_in)[sidx_t]
                if uidx_t is not None:
                    new_col_idx = torch.arange(n_gens, n_gens + n_unstable,
                                                device=gpu, dtype=torch.long)
                    G_out[uidx_t, new_col_idx] = 1.0
                if do_sparse and sidx_t is not None:
                    stable_cols = torch.arange(
                        n_gens + n_unstable, n_gens + n_new,
                        device=gpu, dtype=torch.long)
                    G_out[sidx_t, stable_cols] = 1.0
                G_by_op[nm] = G_out
            n_gens += n_new

        elif t == 'add':
            if op.get('is_merge'):
                ca = center[op['inputs'][0]]
                cb = center[op['inputs'][1]]
                # Residual add: any sparse operand must be materialized
                # before the elementwise add (sparse + sparse fast path
                # not implemented).
                Ga = _materialize_g(pad_cols(G_by_op[op['inputs'][0]]))
                Gb = _materialize_g(pad_cols(G_by_op[op['inputs'][1]]))
                center[nm] = ca + cb
                G_by_op[nm] = Ga + Gb
            else:
                b = op.get('bias')
                if b is not None:
                    bt = torch.tensor(
                        np.asarray(b, dtype=np.float64).flatten(),
                        dtype=dtype, device=gpu)
                    center[nm] = center[op['inputs'][0]] + bt
                else:
                    center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'sub':
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            b = op.get('bias')
            if b is not None:
                bt = torch.tensor(b.flatten(), dtype=dtype, device=gpu)
                center[nm] = prev_c - bt
            else:
                center[nm] = prev_c
            G_by_op[nm] = prev_G

        elif t == 'reshape':
            center[nm] = center[op['inputs'][0]]
            G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'mul':
            # y = scale * x (constant scale, possibly scalar-broadcast).
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            scale = op.get('scale')
            if isinstance(scale, np.ndarray):
                st = torch.from_numpy(scale).to(device=gpu, dtype=dtype)
            elif not isinstance(scale, torch.Tensor):
                st = torch.tensor(scale, dtype=dtype, device=gpu)
            else:
                st = scale.to(device=gpu, dtype=dtype)
            sflat = st.flatten()
            n = prev_c.numel()
            if sflat.numel() == 1 or sflat.numel() == n:
                center[nm] = prev_c * sflat
                Gm = _materialize_g(pad_cols(prev_G))
                G_by_op[nm] = (Gm * sflat if sflat.numel() == 1
                               else Gm * sflat.unsqueeze(-1))
            else:
                in_shape = op.get('in_shapes_nd', [None])[0]
                C, H, W = in_shape
                assert sflat.numel() == C
                s4d = sflat.view(1, C, 1, 1).expand(1, C, H, W).reshape(-1)
                center[nm] = prev_c * s4d
                Gm = _materialize_g(pad_cols(prev_G))
                G_by_op[nm] = Gm * s4d.unsqueeze(-1)

        else:
            raise NotImplementedError(
                f'gen-LP state forward: unsupported op {t!r} (name={nm!r}). '
                'Silent skip would leave stale G_by_op → unsound encoding.')

        for inp in op['inputs']:
            if (last_use.get(inp) == op_idx and inp in G_by_op
                    and inp != nm):
                del G_by_op[inp]
                del center[inp]

    if device == 'cuda':
        torch.cuda.synchronize()

    c_out = center[output_op_name]
    G_out = G_by_op[output_op_name]
    # Materialize sparse storage before the final dense conversion below.
    G_out = _materialize_g(G_out)
    if G_out.shape[1] < n_gens:
        G_out = torch.cat([G_out, torch.zeros(
            G_out.shape[0], n_gens - G_out.shape[1],
            dtype=dtype, device=gpu)], dim=1)
    obj_c_out = c_out.detach().cpu().numpy().astype(np.float64)
    obj_G_out_dense = G_out.detach().cpu().numpy().astype(np.float64)
    obj_G_out_csr = sp.csr_matrix(obj_G_out_dense)

    G_by_op.clear()
    center.clear()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'n_input': n_in,
        'n_gens': n_gens,
        'formulation': formulation,
        'unstable_list': unstable_list,
        'stable_list': stable_list,
        'obj_c_out': obj_c_out,
        'obj_G_out_csr': obj_G_out_csr,
        # For witness forward-pass: cheap to carry, network-level not per-query.
        'gg_ops_ser': gg_ops_ser,
        'input_name': input_name,
        'output_op_name': output_op_name,
        'x_lo': np.asarray(x_lo, dtype=np.float64),
        'x_hi': np.asarray(x_hi, dtype=np.float64),
    }


def state_from_phase1(z_final, rec_zono, x_lo, x_hi, gg_ops_ser,
                       input_name, output_op_name):
    """Build a `solve_spec`-compatible state dict from Phase 1's already-
    propagated zonotope `z_final` and the per-layer pre-ReLU rows cached
    in `rec_zono`.

    Avoids the expensive second forward pass through
    `precompute_gen_state` — Phase 1's PatchesZonotope-driven forward
    already touched every conv/fc with patch sparsity, leaving the dense
    `(n_layer, n_gens)` G matrix avoided. We reuse those numbers as
    `obj_c_out, obj_G_out_csr` and the unstable G rows from `rec_zono`
    as `unstable_list`. No GPU memory needed beyond `z_final`'s output
    G (n_output × n_gens, small for spec-output dimensions).

    Triangle constraints (built later in `build_gen_lp_from_state`) use
    the same (lo, hi) Phase 1 used to compute `z_final`'s μ, λ — soundness
    requires this consistency. (Using Phase-2.5-tightened (lo, hi) in the
    constraints would create a NEW linear combination of (e_in, e_new)
    that doesn't equal y_k under Phase 1's parametrization — and so
    "y_NEW ≥ 0" is NOT generally a fact about real ReLU values; can cut
    off real (e_in, e_new) tuples corresponding to valid x. See the
    soundness note in `build_gen_lp_from_phase1` constraint emission.)

    Args:
        z_final: TorchZonotope or PatchesZonotope at the network's
            last op (returned by `_forward_zonotope_interleaved`).
        rec_zono: dict with keys 'gen_rows_by_layer', 'col_origin',
            'n_input' populated by `_record_zono_pre_relu_rows`.
        x_lo, x_hi: input bounds (np.ndarray, float64).
        gg_ops_ser, input_name, output_op_name: forwarded to the state
            for downstream witness check.

    Returns:
        State dict with formulation='phase1', compatible with
        `solve_spec(state=...)` and `build_gen_lp_from_state`.
    """
    n_input = int(rec_zono['n_input'])
    # Materialize z_final's G as dense numpy then sparse CSR. The output G
    # is (n_output, n_gens); for spec output dimensions (n_out ~= 10 for
    # CIFAR), this is small even on networks with thousands of generators.
    G_t = z_final.generators
    c_t = z_final.center
    # n_gens is the number of columns in z_final's G — equals
    # n_input + sum_li(|unstable at li|).
    n_gens = int(G_t.shape[1])
    obj_c_out = c_t.detach().cpu().numpy().astype(np.float64)
    obj_G_out_dense = G_t.detach().cpu().numpy().astype(np.float64)
    obj_G_out_csr = sp.csr_matrix(obj_G_out_dense)

    # Flatten gen_rows_by_layer into a single sorted-by-e_new_col list.
    # rec_zono entries are tagged with form='phase1' (zono parallelogram).
    # Carry that tag through so downstream builders dispatch correctly.
    unstable_list = []
    for li, layer_dict in rec_zono.get('gen_rows_by_layer', {}).items():
        for k, entry in layer_dict.items():
            # Each entry already has the right shape: layer_idx, neuron_idx,
            # c_in, lo, hi, e_new_col, row_indices, row_values, form.
            ul = {
                'layer_idx': int(entry['layer_idx']),
                'neuron_idx': int(entry['neuron_idx']),
                'c_in': float(entry['c_in']),
                'lo': float(entry['lo']),
                'hi': float(entry['hi']),
                'e_new_col': int(entry['e_new_col']),
                'row_indices': np.asarray(
                    entry['row_indices'], dtype=np.int32),
                'row_values': np.asarray(
                    entry['row_values'], dtype=np.float64),
                'form': entry.get('form', 'phase1'),
            }
            unstable_list.append(ul)
    unstable_list.sort(key=lambda e: e['e_new_col'])

    return {
        'n_input': n_input,
        'n_gens': n_gens,
        'formulation': 'phase1',
        'unstable_list': unstable_list,
        'stable_list': [],
        'obj_c_out': obj_c_out,
        'obj_G_out_csr': obj_G_out_csr,
        'gg_ops_ser': gg_ops_ser,
        'input_name': input_name,
        'output_op_name': output_op_name,
        'x_lo': np.asarray(x_lo, dtype=np.float64),
        'x_hi': np.asarray(x_hi, dtype=np.float64),
    }


def state_from_alpha_zono(z_alpha, pre_relu_gpu, alpha_per_layer, bbr,
                            x_lo, x_hi, gg_ops_ser, input_name,
                            output_op_name, unstable_per_layer=None):
    """Build a `solve_spec`-compatible state dict from an α-CROWN-tightened
    forward zonotope `(c_α, G_α)` (returned by `forward_zono_dir_adaptive`).

    The α-CROWN forward picks a per-neuron slope λ_α (direction-adaptive)
    and shift μ_α = max((1-λ_α)·hi/2, -λ_α·lo/2). Each unstable neuron k
    contributes a new generator column with value μ_α at row k:
        y_k = λ_α·z_k + μ_α·(1 + e_new_k),    e_new_k ∈ [-1, 1]
    where z_k = c_in[k] + G_pre[k]·e is the pre-activation.

    The α-zono LP/MILP produced from this state encodes the parallelogram
    relaxation y_k = λ_α·z_k + μ_α·(1+e_new) WITHOUT adding the standard
    triangle floor (`y ≥ 0`, `y ≥ z`) for non-binarized neurons; the
    bare parallelogram is the looser-but-sound enclosure that the α-CROWN
    spec lower bound is computed on. With `milp_set = ∅`, the LP min
    must equal α-CROWN's spec lower bound (the bin-0 sanity check).

    For binarized neurons (in `milp_set`), the full ReLU big-M encoding
    is added on top of the parallelogram (4 constraints + 1 binary).

    Args:
        z_alpha: output zonotope (TorchZonotope or PatchesZonotope) at the
            network's output op (returned by `forward_zono_dir_adaptive`).
        pre_relu_gpu: dict `{L: (c_pre, G_pre)}` of pre-ReLU snapshots.
            Either slim form (`(n_unstable_at_L, K_L)`) when
            `unstable_per_layer` was passed to the forward, or full form
            (`(n_at_L, K_L)`).
        alpha_per_layer: dict `{L: λ tensor}` of per-neuron slopes used
            by the forward (1 for active, 0 for dead, α for unstable).
        bbr: dict `{L: (lo_np, hi_np)}` pre-ReLU bounds (the same bounds
            passed to `forward_zono_dir_adaptive`).
        x_lo, x_hi: input bounds (np.ndarray, float64).
        gg_ops_ser, input_name, output_op_name: forwarded for the
            downstream witness check.
        unstable_per_layer: dict `{L: LongTensor of indices}`. When
            provided, `pre_relu_gpu[L]` is interpreted as slim form;
            else full form.

    Returns:
        State dict with formulation='alpha_zono', usable with
        `solve_spec(state=...)` and `build_gen_lp_from_state`.
    """
    n_input = int(len(x_lo))

    # Output zonotope → (obj_c_out, obj_G_out_csr) — same n_gens as the
    # full alpha forward (input gens + sum_L n_unstable_at_L).
    G_t = z_alpha.generators
    c_t = z_alpha.center
    n_gens = int(G_t.shape[1])
    obj_c_out = c_t.detach().cpu().numpy().astype(np.float64)
    obj_G_out_dense = G_t.detach().cpu().numpy().astype(np.float64)
    obj_G_out_csr = sp.csr_matrix(obj_G_out_dense)

    # Tag sigmoid/tanh layers so downstream consumers
    # (score_box_halfspace_delta_lb) can skip them — gen-LP encodes
    # ReLU triangles only, but z_alpha includes Sigmoid's noise vars
    # so we have to keep cur_n_gens aligned with z_alpha's column count.
    sigmoid_tanh_layer_ids = {op['layer_idx'] for op in gg_ops_ser
                              if op['type'] in ('sigmoid', 'tanh')
                              and 'layer_idx' in op}
    unstable_list = []
    cur_n_gens = n_input  # starts with input generators
    for L in sorted(bbr.keys()):
        # Sigmoid/tanh layers add one slack column per output cell
        # (box-relax with preserved column indexing — see
        # `alpha_crown.forward_zono_dir_adaptive` sigmoid branch).
        # We don't put them in unstable_list (gen-LP MILP can't binarise
        # them) but we MUST advance cur_n_gens past them so subsequent
        # ReLU e_new_col indices align with z_alpha's actual columns.
        if L in sigmoid_tanh_layer_ids:
            n_cells = int(np.asarray(bbr[L][0]).flatten().size)
            cur_n_gens += n_cells
            continue
        lo_arr = np.asarray(bbr[L][0], dtype=np.float64)
        hi_arr = np.asarray(bbr[L][1], dtype=np.float64)
        # Identify unstable neurons in ascending neuron-index order
        # (matches `apply_relu_custom`'s `torch.where(mu != 0)` allocation).
        unstable = np.where((lo_arr < 0) & (hi_arr > 0))[0]
        if len(unstable) == 0:
            continue
        if L not in pre_relu_gpu:
            cur_n_gens += len(unstable)
            continue

        c_pre_t, G_pre_t = pre_relu_gpu[L]
        c_pre = c_pre_t.detach().cpu().numpy().astype(np.float64)
        G_pre = G_pre_t.detach().cpu().numpy().astype(np.float64)

        # Slim → look up by row position in unstable_per_layer order.
        # Full → look up by neuron index directly.
        slim_map = None
        if (unstable_per_layer is not None and L in unstable_per_layer
                and unstable_per_layer[L].numel() > 0
                and c_pre.shape[0] == int(unstable_per_layer[L].numel())):
            up_arr = unstable_per_layer[L].detach().cpu().numpy().astype(
                np.int64)
            slim_map = {int(up_arr[i]): i for i in range(len(up_arr))}

        alpha_t = alpha_per_layer.get(L)
        if alpha_t is None:
            cur_n_gens += len(unstable)
            continue
        alpha_np = alpha_t.detach().cpu().numpy().astype(np.float64)

        for local_idx, j in enumerate(unstable):
            j = int(j)
            lo_j = float(lo_arr[j])
            hi_j = float(hi_arr[j])
            if slim_map is not None:
                row_pos = slim_map.get(j)
                if row_pos is None:
                    cur_n_gens += 0  # neuron not in slim cache; skip
                    continue
                c_in_j = float(c_pre[row_pos])
                row_full = G_pre[row_pos]
            else:
                if j >= c_pre.shape[0]:
                    continue
                c_in_j = float(c_pre[j])
                row_full = G_pre[j]
            # Defend against 1-elem upstream layers (e.g. dist_shift's
            # mnist_concat Sigmoid output) collapsing G_pre[j] to scalar.
            row_full = np.atleast_1d(row_full)
            nz = np.nonzero(row_full)[0]
            row_indices = nz.astype(np.int32)
            row_values = row_full[nz].astype(np.float64)

            # α-CROWN slopes used by the forward.
            lam_a = float(alpha_np[j])
            mu_a = max((1.0 - lam_a) * hi_j / 2.0,
                        -lam_a * lo_j / 2.0)

            # Defensive: skip if e_new_col would exceed z_alpha's actual
            # n_gens. Misalignment can happen on weird graphs (e.g.,
            # dist_shift's mnist_concat Sigmoid dead branch) where bbr
            # disagrees with z_alpha's apply_relu_custom on stability.
            if cur_n_gens + local_idx >= n_gens:
                continue
            unstable_list.append({
                'layer_idx': int(L),
                'neuron_idx': j,
                'c_in': c_in_j,
                'lo': lo_j,
                'hi': hi_j,
                'e_new_col': cur_n_gens + local_idx,
                'row_indices': row_indices,
                'row_values': row_values,
                'lam': lam_a,
                'mu': mu_a,
                # α-CROWN parallelogram form: same coordinate system as
                # 'phase1' (zono e_new ∈ [-1, 1] with μ-scaled column),
                # but with α-tightened (λ, μ). Distinct tag so the
                # `_build_alpha_zono_lp` path is selected.
                'form': 'alpha_zono',
            })

        cur_n_gens += len(unstable)

    # cur_n_gens should match n_gens; tolerate mismatch (extra cols are
    # padded as unused vars in the LP builder).

    return {
        'n_input': n_input,
        'n_gens': n_gens,
        'formulation': 'alpha_zono',
        'unstable_list': unstable_list,
        'stable_list': [],
        'obj_c_out': obj_c_out,
        'obj_G_out_csr': obj_G_out_csr,
        'gg_ops_ser': gg_ops_ser,
        'input_name': input_name,
        'output_op_name': output_op_name,
        'x_lo': np.asarray(x_lo, dtype=np.float64),
        'x_hi': np.asarray(x_hi, dtype=np.float64),
        # Tag layer indices whose nonlinearity is sigmoid/tanh — these
        # don't form ReLU triangles, so binarisation scoring should
        # skip them.
        'sigmoid_tanh_layer_ids': sigmoid_tanh_layer_ids,
    }


def _build_alpha_zono_lp(state, qw, qb, milp_set, n_threads,
                           unsafe_halfspace, triangle_set=None):
    """Build Gurobi LP/MILP for the α-zono form:

    y_k = λ_α·z_k + μ_α·(1 + e_new_k),    e_new_k ∈ [-1, 1]
        z_k = c_in_k + Σ row_values·e[row_indices]

    Three tiers of per-neuron encoding (most→least restrictive):

      A) Binarised (`(li, j) in milp_set`) — full ReLU big-M, exact.
            y ≥ 0                : λ·z + μ·(1+e_new)              ≥ 0
            y ≥ z                : (λ-1)·z + μ·(1+e_new)          ≥ 0
            y ≤ hi·s             : λ·z + μ·(1+e_new) - hi·s       ≤ 0
            y ≤ z - lo·(1-s)     : (λ-1)·z + μ·(1+e_new) - lo·s   ≤ -lo

      B) Triangulated (`(li, j) in triangle_set` and NOT binarised) —
         add the triangle floor (sound, no binary):
            y ≥ 0  and  y ≥ z   (the same two constraints as in A).
         Tighter than the bare parallelogram (which can dip below 0 in
         the lam·z line) without the cost of a binary. Useful when the
         next K-most-important neurons would push ObjBound > 0 if their
         lower-floor were enforced, but full big-M is too expensive.

      C) Parallelogram-only (everyone else) — no extra constraints, just
         e_new ∈ [-1, 1]. The bare zonotope upper bound on |Σ qw·G_α|.

    The objective `min qw·c_α + (qw·G_α)·e + qb` over `e ∈ [-1, 1]^n_gens`
    is the α-CROWN spec LB at `milp_set = ∅` AND `triangle_set = ∅`
    (closed-form `qw·c_α + qb - Σ |coef_k|`). Triangulating any neuron
    can only RAISE the LP min (relaxation gets tighter); binarising any
    neuron can only further raise it. Verifies when ObjBound > 0.
    """
    if milp_set is None:
        milp_set = set()
    if triangle_set is None:
        triangle_set = set()
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    n_input = state['n_input']
    n_gens = state['n_gens']
    combined = sorted(state.get('unstable_list', []),
                      key=lambda e: e['e_new_col'])

    # ---- Batched variable creation ----
    # Old loop did 1 addVar per e_in + 1 addVar + m.update() per unstable
    # (~10K addVar calls + ~1300 update() syncs ⇒ ~2.4s build on
    # tinyimagenet). addVars + a single update() drops it to ~0.3s.
    # Pre-build the FULL n_gens-length e_vars array. Slots that don't
    # correspond to a real e_in or e_new (gaps from skipped unstables)
    # get fixed to 0 via lb=ub=0; those slots have no objective weight.
    e_lb = np.zeros(n_gens, dtype=np.float64)
    e_ub = np.zeros(n_gens, dtype=np.float64)
    # Input noise symbols: e_in_i ∈ [-1, 1].
    e_lb[:n_input] = -1.0
    e_ub[:n_input] = 1.0
    # New noise symbols (one per unstable): e_new ∈ [-1, 1] at e_new_col.
    e_new_cols = np.array([int(ul['e_new_col']) for ul in combined],
                          dtype=np.int64)
    if len(e_new_cols) > 0:
        e_lb[e_new_cols] = -1.0
        e_ub[e_new_cols] = 1.0
    e_vars_arr = m.addMVar(n_gens, lb=e_lb, ub=e_ub, name='e')
    e_vars = list(e_vars_arr.tolist())
    # Binary vars in one batch (one per milp_set member).
    milp_keys_in_unstable = [(ul['layer_idx'], ul['neuron_idx'])
                              for ul in combined
                              if (ul['layer_idx'], ul['neuron_idx']) in milp_set]
    if milp_keys_in_unstable:
        s_arr = m.addMVar(len(milp_keys_in_unstable),
                          vtype=grb.GRB.BINARY, name='s')
        s_by_key = {k: s_arr[i].tolist()
                    for i, k in enumerate(milp_keys_in_unstable)}
    else:
        s_by_key = {}
    m.update()

    unstable_info = []
    for ul in combined:
        li = ul['layer_idx']
        j = ul['neuron_idx']
        lo_j = ul['lo']
        hi_j = ul['hi']
        c_j = ul['c_in']
        e_new_col = ul['e_new_col']
        row_idx = ul['row_indices']
        row_val = ul['row_values']
        lam = ul['lam']
        mu = ul['mu']

        e_new = e_vars[int(e_new_col)]
        expr_vars = [e_vars[int(k)] for k in row_idx]
        expr_coefs = [float(v) for v in row_val]

        key = (li, j)
        is_milp = key in milp_set
        is_triangle = (key in triangle_set) and not is_milp

        # Triangle floor (tier A and tier B): y ≥ 0 and y ≥ z.
        if is_milp or is_triangle:
            # y ≥ 0:  λ·z + μ·(1+e_new) ≥ 0
            #         Σ (λ·row)·e + μ·e_new ≥ -λ·c_in - μ
            lin_y0 = grb.LinExpr(
                [lam * w for w in expr_coefs], expr_vars)
            lin_y0.add(e_new, mu)
            m.addLConstr(lin_y0 >= -lam * c_j - mu,
                          name=f'tri_lo_L{li}_{j}')

            # y ≥ z:  (λ-1)·z + μ·(1+e_new) ≥ 0
            #         Σ ((λ-1)·row)·e + μ·e_new ≥ -(λ-1)·c_in - μ
            lin_yz = grb.LinExpr(
                [(lam - 1.0) * w for w in expr_coefs], expr_vars)
            lin_yz.add(e_new, mu)
            m.addLConstr(lin_yz >= -(lam - 1.0) * c_j - mu,
                          name=f'tri_up_L{li}_{j}')

        # Big-M binary (tier A only): adds the upper edge constraints.
        if is_milp:
            s = s_by_key[(li, j)]

            # y ≤ hi·s:  Σ (λ·row)·e + μ·e_new - hi·s ≤ -μ - λ·c_in
            lin_hi = grb.LinExpr(
                [lam * w for w in expr_coefs], expr_vars)
            lin_hi.add(e_new, mu)
            lin_hi.add(s, -hi_j)
            m.addLConstr(lin_hi <= -mu - lam * c_j,
                          name=f'bigM_hi_L{li}_{j}')

            # y ≤ z - lo·(1-s):  Σ ((λ-1)·row)·e + μ·e_new - lo·s
            #                                ≤ -lo - μ - (λ-1)·c_in
            lin_lo = grb.LinExpr(
                [(lam - 1.0) * w for w in expr_coefs], expr_vars)
            lin_lo.add(e_new, mu)
            lin_lo.add(s, -lo_j)
            m.addLConstr(lin_lo <= -lo_j - mu - (lam - 1.0) * c_j,
                          name=f'bigM_z_L{li}_{j}')

        unstable_info.append({
            'layer_idx': li,
            'neuron_idx': j,
            'e_new_var_name': f'a_L{li}_{j}',
            'row_coefs_names': [
                (float(row_val[k]),
                 e_vars[int(row_idx[k])].VarName)
                for k in range(len(row_idx))],
            'c_in': c_j,
            'lo': lo_j,
            'hi': hi_j,
            'e_new_col': e_new_col,
            'lam': lam,
            'mu': mu,
            'formulation': 'alpha_zono',
        })

    # (e_vars already padded to n_gens by the batched addMVar above —
    # gap slots are fixed at 0 via lb=ub=0 and contribute nothing.)

    # Objective: qw·c_α + (qw·G_α)·e + qb. The α-zono output zonotope
    # ALREADY encodes the full network with the α relaxation, so this
    # min over e ∈ [-1, 1]^n_gens at `milp_set = ∅` matches α-CROWN's
    # spec LB exactly (closed-form `qw·c_α + qb - Σ |obj_coef_k|`).
    obj_coef = state['obj_G_out_csr'].T @ qw
    obj_const = float(state['obj_c_out'] @ qw) + qb
    obj = grb.LinExpr()
    for k in range(n_gens):
        c = float(obj_coef[k])
        if c != 0:
            obj.add(e_vars[k], c)
    m.setObjective(obj + obj_const, grb.GRB.MINIMIZE)

    if unsafe_halfspace and unsafe_halfspace != 'none':
        hs = grb.LinExpr()
        for k in range(n_gens):
            c = float(obj_coef[k])
            if c != 0:
                hs.add(e_vars[k], c)
        rhs = -float(obj_const)
        if unsafe_halfspace == 'inequality':
            m.addLConstr(hs <= rhs, name='halfspace_unsafe')
        elif unsafe_halfspace == 'equality':
            m.addLConstr(hs == rhs, name='halfspace_unsafe')
        else:
            raise ValueError(
                f'unknown unsafe_halfspace {unsafe_halfspace!r}')

    if not milp_set:
        m.setParam('Method', 1)
    m.update()
    return m, env, unstable_info, np.asarray(obj_coef).flatten()


def _build_phase1_lp(state, qw, qb, milp_set, n_threads, unsafe_halfspace):
    """Build Gurobi LP/MILP for the Phase-1 form:

    Variables:
      e_in[i] ∈ [-1, 1]      for i in [0, n_input)
      e_new_k ∈ [-1, 1]      for each unstable k (column e_new_col)

    Phase 1 already encoded post-ReLU as
        y_k = λ_k · z_k + μ_k · (1 + e_new_k),    e_new_k ∈ [-1, 1]
    where (λ_k, μ_k) come from Phase 1's bounds (lo_k, hi_k):
        λ_k = hi_k / (hi_k - lo_k)
        μ_k = -hi_k · lo_k / (2 (hi_k - lo_k))
    The objective `qw · y_OUT + qb` is c_out + G_out · e — already a
    sound over-approximation. The parallelogram allows y values BELOW the
    standard triangle's `max(0, z)` floor (e.g. at e_new = -1, y = λ·z
    which can be negative for unstable z). Adding two LP constraints per
    unstable recovers the triangle:

        y_k ≥ 0:   μ·e_new + λ·z + μ ≥ 0
        y_k ≥ z_k: μ·e_new + (λ-1)·z + μ ≥ 0

    where z_k = c_in_k + Σ row_values[i]·e[row_indices[i]].

    The third triangle edge `y_k ≤ λ·(z_k - lo_k)` is automatic since at
    e_new_k = 1, y_k = λ·z + 2μ = λ·(z - lo). So `e_new_k ≤ 1` already
    enforces it.

    For neurons in `milp_set`, add big-M binaries on top:
        y_k ≤ hi · s_k:   μ·e_new + λ·z - hi·s ≤ -μ
        y_k ≤ z - lo·(1-s_k):  μ·e_new + (λ-1)·z - lo·s ≤ -lo - μ

    Constraint names match the existing builder ('tri_lo_L{li}_{j}',
    'tri_up_L{li}_{j}', 'bigM_hi/z_L{li}_{j}') so `compute_scores`'
    `lp_dual` method works unchanged.
    """
    if milp_set is None:
        milp_set = set()
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    n_input = state['n_input']
    n_gens = state['n_gens']
    e_vars = [m.addVar(lb=-1.0, ub=1.0, name=f'e_in_{i}')
              for i in range(n_input)]
    m.update()

    unstable_info = []
    combined = sorted(state.get('unstable_list', []),
                      key=lambda e: e['e_new_col'])

    for ul in combined:
        li = ul['layer_idx']
        j = ul['neuron_idx']
        lo_j = ul['lo']
        hi_j = ul['hi']
        c_j = ul['c_in']
        e_new_col = ul['e_new_col']
        row_idx = ul['row_indices']
        row_val = ul['row_values']

        while len(e_vars) < e_new_col:
            e_vars.append(m.addVar(lb=0.0, ub=0.0,
                                    name=f'_unused_{len(e_vars)}'))

        # Phase 1 form: e_new ∈ [-1, 1], NOT [0, hi_j]
        e_new = m.addVar(lb=-1.0, ub=1.0, name=f'a_L{li}_{j}')
        e_vars.append(e_new)
        m.update()

        gap = hi_j - lo_j
        # Phase 1 must have classified this neuron as unstable, so gap > 0.
        lam = hi_j / gap
        mu = -hi_j * lo_j / (2.0 * gap)
        # μ > 0 for unstable (lo<0, hi>0). Defensive: skip if μ ~= 0.

        expr_vars = [e_vars[int(k)] for k in row_idx]
        expr_coefs = [float(v) for v in row_val]

        # Constraint y_k ≥ 0 in Phase 1 form:
        #   μ·e_new + λ·(c_in + Σ row·e) + μ ≥ 0
        #   Σ (λ·row)·e + μ·e_new ≥ -λ·c_in - μ
        # Gurobi convention: store as `expr ≥ rhs`.
        lin_y0 = grb.LinExpr([lam * w for w in expr_coefs], expr_vars)
        lin_y0.add(e_new, mu)
        m.addLConstr(lin_y0 >= -lam * c_j - mu,
                      name=f'tri_lo_L{li}_{j}')

        # Constraint y_k ≥ z_k:
        #   μ·e_new + (λ-1)·z + μ ≥ 0
        #   Σ ((λ-1)·row)·e + μ·e_new ≥ -(λ-1)·c_in - μ
        lin_yz = grb.LinExpr(
            [(lam - 1.0) * w for w in expr_coefs], expr_vars)
        lin_yz.add(e_new, mu)
        m.addLConstr(lin_yz >= -(lam - 1.0) * c_j - mu,
                      name=f'tri_up_L{li}_{j}')

        key = (li, j)
        if key in milp_set:
            s = m.addVar(vtype=grb.GRB.BINARY, name=f's_L{li}_{j}')
            m.update()
            # y ≤ hi·s:  μ·e_new + λ·z + μ - hi·s ≤ 0
            #            Σ (λ·row)·e + μ·e_new - hi·s ≤ -μ - λ·c_in
            lin_hi = grb.LinExpr([lam * w for w in expr_coefs], expr_vars)
            lin_hi.add(e_new, mu)
            lin_hi.add(s, -hi_j)
            m.addLConstr(lin_hi <= -mu - lam * c_j,
                          name=f'bigM_hi_L{li}_{j}')
            # y ≤ z - lo·(1-s):  μ·e_new + (λ-1)·z + μ - lo·s ≤ -lo
            #                    Σ ((λ-1)·row)·e + μ·e_new - lo·s
            #                                       ≤ -lo - μ - (λ-1)·c_in
            lin_lo = grb.LinExpr(
                [(lam - 1.0) * w for w in expr_coefs], expr_vars)
            lin_lo.add(e_new, mu)
            lin_lo.add(s, -lo_j)
            m.addLConstr(lin_lo <= -lo_j - mu - (lam - 1.0) * c_j,
                          name=f'bigM_z_L{li}_{j}')

        unstable_info.append({
            'layer_idx': li,
            'neuron_idx': j,
            'e_new_var_name': f'a_L{li}_{j}',
            'row_coefs_names': [
                (float(row_val[k]),
                 e_vars[int(row_idx[k])].VarName)
                for k in range(len(row_idx))],
            'c_in': c_j,
            'lo': lo_j,
            'hi': hi_j,
            'e_new_col': e_new_col,
            # Phase 1 form: y is computed from (lam, mu, z, e_new), not
            # directly from e_new. Stash so `compute_scores` can reconstruct.
            'lam': lam,
            'mu': mu,
            'formulation': 'phase1',
        })

    # Pad e_vars to n_gens if needed (some layers may have produced no
    # unstable neurons but their column slots still need placeholder vars).
    while len(e_vars) < n_gens:
        e_vars.append(m.addVar(lb=0.0, ub=0.0,
                                name=f'_unused_{len(e_vars)}'))

    # Objective: same form as the existing builder.
    obj_coef = state['obj_G_out_csr'].T @ qw  # (n_gens,)
    obj_const = float(state['obj_c_out'] @ qw) + qb
    obj = grb.LinExpr()
    for k in range(n_gens):
        c = float(obj_coef[k])
        if c != 0:
            obj.add(e_vars[k], c)
    m.setObjective(obj + obj_const, grb.GRB.MINIMIZE)

    if unsafe_halfspace and unsafe_halfspace != 'none':
        hs = grb.LinExpr()
        for k in range(n_gens):
            c = float(obj_coef[k])
            if c != 0:
                hs.add(e_vars[k], c)
        rhs = -float(obj_const)
        if unsafe_halfspace == 'inequality':
            m.addLConstr(hs <= rhs, name='halfspace_unsafe')
        elif unsafe_halfspace == 'equality':
            m.addLConstr(hs == rhs, name='halfspace_unsafe')
        else:
            raise ValueError(
                f'unknown unsafe_halfspace {unsafe_halfspace!r}')

    if not milp_set:
        m.setParam('Method', 1)
    m.update()
    return m, env, unstable_info, np.asarray(obj_coef).flatten()


def build_gen_lp_from_state(state, qw, qb, *, milp_set=None, n_threads=1,
                              unsafe_halfspace='none', triangle_set=None):
    """Build Gurobi LP/MILP from a precomputed state (numpy-only).

    Same return as build_gen_lp:  (model, env, unstable_info, obj_coef)

    Dispatches on `state['formulation']`:
      - 'phase1' (from `state_from_phase1`): use Phase 1's parametrization
        (e_new ∈ [-1, 1], two triangle-lower constraints per unstable)
      - 'dense' / 'sparse' / 'all_sparse' (from `precompute_gen_state`):
        existing direct y_k = e_new ∈ [0, hi_j] formulation
      - 'alpha_zono' (from `state_from_alpha_zono`): per-query α-CROWN
        zonotope; bare parallelogram for non-binarized neurons unless
        also in `triangle_set`, in which case the triangle floor
        (y ≥ 0, y ≥ z) is added.

    triangle_set: only honored for `formulation='alpha_zono'`. Set of
        (li, neuron_idx) keys for which the triangle floor is added on
        top of the parallelogram (without binarising). Used to inject
        the top-K most-important neurons into the α-zono LP. Other
        formulations ignore this parameter (they already encode the
        triangle floor unconditionally).

    unsafe_halfspace: how to constrain the unsafe region `qw·y + qb ≤ 0`:
      'none' (default) : no extra constraint
      'inequality'     : add `qw·y + qb ≤ 0` as a linear inequality
      'equality'       : add `qw·y + qb == 0` as a linear equality
    The constraint name is 'halfspace_unsafe' when added. With the
    constraint, the LP/MILP optimum is trivially ≤ 0 (or = 0); the
    verification signal becomes `Gurobi.INFEASIBLE` — an empty polytope
    under the ReLU relaxation proves safety.
    """
    if state.get('formulation') == 'phase1':
        return _build_phase1_lp(state, qw, qb, milp_set, n_threads,
                                 unsafe_halfspace)
    if state.get('formulation') == 'alpha_zono':
        return _build_alpha_zono_lp(state, qw, qb, milp_set, n_threads,
                                      unsafe_halfspace,
                                      triangle_set=triangle_set)
    if milp_set is None:
        milp_set = set()
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    n_input = state['n_input']
    n_gens = state['n_gens']
    e_vars = [m.addVar(lb=-1.0, ub=1.0, name=f'e_in_{i}') for i in range(n_input)]
    m.update()

    unstable_info = []
    # Interleave unstable + stable (sparse only) by e_new_col order
    combined = list(state.get('unstable_list', []))
    combined.extend([dict(s, _kind='stable') for s in state.get('stable_list', [])])
    # Mark unstable entries
    for e in combined:
        if '_kind' not in e:
            e['_kind'] = 'unstable'
    combined.sort(key=lambda e: e['e_new_col'])

    for ul in combined:
        li = ul['layer_idx']
        j = ul['neuron_idx']
        lo_j = ul['lo']
        hi_j = ul['hi']
        c_j = ul['c_in']
        e_new_col = ul['e_new_col']
        row_idx = ul['row_indices']
        row_val = ul['row_values']
        kind = ul['_kind']

        while len(e_vars) < e_new_col:
            e_vars.append(m.addVar(lb=0.0, ub=0.0, name=f'_unused_{len(e_vars)}'))

        if kind == 'stable':
            # Sparse stable-on: new gen ∈ [lo, hi], equality v == c + G@e
            e_new = m.addVar(lb=lo_j, ub=hi_j, name=f'as_L{li}_{j}')
            e_vars.append(e_new)
            m.update()
            expr_vars = [e_vars[int(k)] for k in row_idx]
            expr_coefs = [float(v) for v in row_val]
            m.addLConstr(e_new - grb.LinExpr(expr_coefs, expr_vars) == c_j,
                          name=f'eq_L{li}_{j}')
            continue

        # Unstable path (same as before)
        e_new = m.addVar(lb=0.0, ub=hi_j, name=f'a_L{li}_{j}')
        e_vars.append(e_new)
        m.update()
        expr_vars = [e_vars[int(k)] for k in row_idx]
        expr_coefs = [float(v) for v in row_val]
        m.addLConstr(grb.LinExpr(expr_coefs, expr_vars) - e_new <= -c_j,
                      name=f'tri_lo_L{li}_{j}')
        key = (li, j)
        if key in milp_set:
            s = m.addVar(vtype=grb.GRB.BINARY, name=f's_L{li}_{j}')
            m.update()
            m.addLConstr(e_new - hi_j * s <= 0,
                          name=f'bigM_hi_L{li}_{j}')
            m.addLConstr(e_new - grb.LinExpr(expr_coefs, expr_vars)
                          - lo_j * s <= c_j - lo_j,
                          name=f'bigM_z_L{li}_{j}')
        else:
            slope = hi_j / (hi_j - lo_j)
            m.addLConstr(e_new - grb.LinExpr(
                [slope * w for w in expr_coefs], expr_vars)
                <= slope * (c_j - lo_j),
                name=f'tri_up_L{li}_{j}')
        unstable_info.append({
            'layer_idx': li,
            'neuron_idx': j,
            'e_new_var_name': f'a_L{li}_{j}',
            'row_coefs_names': [(float(row_val[k]), e_vars[int(row_idx[k])].VarName)
                                for k in range(len(row_idx))],
            'c_in': c_j,
            'lo': lo_j,
            'hi': hi_j,
            'e_new_col': e_new_col,
        })

    # Pad e_vars to n_gens if needed
    while len(e_vars) < n_gens:
        e_vars.append(m.addVar(lb=0.0, ub=0.0, name=f'_unused_{len(e_vars)}'))

    # Objective: qw @ y_out + qb where y_out = c + G_out @ e
    # obj_coef = qw @ G_out_csr, obj_const = qw @ c_out + qb
    obj_coef = state['obj_G_out_csr'].T @ qw  # (n_gens,) vector
    obj_const = float(state['obj_c_out'] @ qw) + qb

    obj = grb.LinExpr()
    for k in range(n_gens):
        c = float(obj_coef[k])
        if c != 0:
            obj.add(e_vars[k], c)
    m.setObjective(obj + obj_const, grb.GRB.MINIMIZE)

    # Optional unsafe-halfspace constraint on the spec output.
    if unsafe_halfspace and unsafe_halfspace != 'none':
        hs = grb.LinExpr()
        for k in range(n_gens):
            c = float(obj_coef[k])
            if c != 0:
                hs.add(e_vars[k], c)
        rhs = -float(obj_const)
        if unsafe_halfspace == 'inequality':
            m.addLConstr(hs <= rhs, name='halfspace_unsafe')
        elif unsafe_halfspace == 'equality':
            m.addLConstr(hs == rhs, name='halfspace_unsafe')
        else:
            raise ValueError(f'unknown unsafe_halfspace {unsafe_halfspace!r}')

    if not milp_set:
        m.setParam('Method', 1)
    m.update()

    return m, env, unstable_info, np.asarray(obj_coef).flatten()


def build_gen_lp(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
                 output_op_name, qw, qb, *, milp_set=None,
                 n_threads=1, device='cuda', dtype=torch.float64):
    """Convenience wrapper: precompute state then build Gurobi.

    Same signature and return as before.
    """
    state = precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op_name, device=device, dtype=dtype)
    return build_gen_lp_from_state(state, qw, qb,
                                    milp_set=milp_set, n_threads=n_threads)


def score_box_halfspace_delta_lb(state, qw, qb, ew_per_relu):
    """Per-neuron BaB worst-child delta_LB via closed-form box+halfspace LP.

    For each unstable k in the alpha_zono state, computes:
        delta_LB(k) = min(LB_off(k), LB_on(k)) - LB_baseline
    where LB_off forces y_k=0 (off-side binarisation) and LB_on forces
    y_k=z_k (on-side). Both sub-LPs are box + 1 halfspace + linear
    objective — solved exactly by `box_halfspace.lagrangian_min`.

    The trick: forcing y_k to its binarised value rewrites the spec
    objective by removing y_k's parallelogram contribution (substitute
    y_k=0 or y_k=z_k into qw·y_out). The resulting objective is still
    linear in `e`, just with adjusted coefficients on row_k and on the
    e_new_k column. Then add the corresponding halfspace `z_k ≤ 0`
    (off) or `z_k ≥ 0` (on) and solve.

    Args:
        state: alpha_zono state dict (state_from_alpha_zono output).
        qw, qb: spec direction (np.float64 array, scalar).
        ew_per_relu: dict {layer_idx: np.ndarray(n_layer,)} — the spec's
            effective weight at each ReLU layer (from
            `alpha_crown.capture_ew_per_relu`). ew[L][k] = ∂(qw·y_out)/∂y_k.

    Returns:
        scores: dict {(layer_idx, neuron_idx): delta_LB} where higher
            score = better candidate for binarisation. Comparable to
            kfsb in selection quality, ~10ms/1000 neurons via O(n log n)
            Lagrangian.
    """
    from .box_halfspace import lagrangian_min
    obj_c_out = np.asarray(state['obj_c_out'], dtype=np.float64)
    obj_G_out = state['obj_G_out_csr'].toarray().astype(np.float64)
    n_gens = state['n_gens']
    qw = np.asarray(qw, dtype=np.float64)
    qb = float(qb)
    d_obj_base = qw @ obj_G_out
    c0_obj_base = float(qw @ obj_c_out + qb)
    baseline_lb = c0_obj_base - float(np.sum(np.abs(d_obj_base)))

    scores = {}
    _skip_layers = state.get('sigmoid_tanh_layer_ids', set())
    for u in state.get('unstable_list', []):
        li = u['layer_idx']
        if li in _skip_layers:
            continue
        j = u['neuron_idx']
        c_in_k = float(u['c_in'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)
        row_val = np.asarray(u['row_values'], dtype=np.float64)
        e_new_col = int(u['e_new_col'])
        # Some state forms (phase1) don't have lam/mu — recompute from (lo,hi)
        lo_k = float(u['lo']); hi_k = float(u['hi'])
        if 'lam' in u and 'mu' in u:
            lam = float(u['lam'])
            mu = float(u['mu'])
        else:
            if hi_k <= lo_k or hi_k <= 0 or lo_k >= 0:
                scores[(li, j)] = 0.0
                continue
            lam = hi_k / (hi_k - lo_k)
            mu = -hi_k * lo_k / (2.0 * (hi_k - lo_k))
        # ew_k = ∂(qw·y_out)/∂y_k. Two equivalent forms:
        #   ew_α  := ew_per_relu[li][j]      (from α-CROWN backward)
        #   ew_st := d_obj_base[e_new_col]/μ (from state's e_new_col coefficient)
        # In theory ew_α == ew_st (same gradient). In practice they differ
        # by ~1e-8 (FP32 noise in α-CROWN backward vs forward state build).
        # For neurons with lam=0 (degenerate parallelogram), the off-side
        # substitution's only nonzero d_off adjustment is `d[e_new_col] -= ew*mu`.
        # Using ew_st makes this EXACTLY zero out d[e_new_col] regardless
        # of small bbr drift; using ew_α leaves a tiny residual that gets
        # amplified by the LP solve into wildly different ranking. Use ew_st
        # for stability (deterministic given state, no α-CROWN dep).
        if mu > 1e-12:
            ew_k = float(d_obj_base[e_new_col]) / mu
        else:
            ew_l = ew_per_relu.get(li)
            if ew_l is None or j >= len(ew_l):
                scores[(li, j)] = 0.0
                continue
            ew_k = float(ew_l[j])

        row_full = np.zeros(n_gens, dtype=np.float64)
        row_full[row_idx] = row_val

        # Off-side: y_k = 0 → subtract ew_k·y_k from objective; halfspace z_k ≤ 0
        d_off = d_obj_base - ew_k * lam * row_full
        d_off[e_new_col] -= ew_k * mu
        c0_off = c0_obj_base - ew_k * (lam * c_in_k + mu)
        lb_off = lagrangian_min(d_off, c0_off, row_full, -c_in_k)

        # On-side: y_k = z_k → add ew_k·(z_k - y_k); halfspace z_k ≥ 0
        d_on = d_obj_base + ew_k * (1.0 - lam) * row_full
        d_on[e_new_col] -= ew_k * mu
        c0_on = c0_obj_base + ew_k * ((1.0 - lam) * c_in_k - mu)
        lb_on = lagrangian_min(d_on, c0_on, -row_full, c_in_k)

        # BaB takes the worse child. Higher delta = harder to verify
        # without binarising = better candidate.
        scores[(li, j)] = float(min(lb_off, lb_on)) - baseline_lb
    return scores


def compute_scores(m, unstable_info, obj_coef, method='lp_ew_frac'):
    """Compute neuron scores from a solved LP.

    method='lp_ew_frac' : |obj_coef[col]| * h*|l|/(h-l)
    method='lp_fractional': |a_val - max(0, z_val)| at LP solution
    method='lp_dual': |dual of tri_lo| + |dual of tri_up| — identifies
        which triangle constraints are binding at the LP optimum. Tests
        on CIFAR100_resnet_medium_prop_idx_2477 showed the dual mass
        concentrates in L5 for the killer query, whereas lp_ew_frac
        promotes L9 (large gradient) — so lp_dual is the correct
        branching heuristic for picking binaries in Phase 8.
    """
    var_by_name = {v.VarName: v for v in m.getVars()}
    scores = {}
    if method == 'lp_dual':
        # Pre-collect all constraint duals keyed by (li, j)
        dual_by_key = {}
        for c in m.getConstrs():
            name = c.ConstrName
            if name.startswith('tri_lo_L') or name.startswith('tri_up_L'):
                parts = name.split('_')
                li = int(parts[2][1:])
                j = int(parts[3])
                try:
                    pi = float(c.Pi)
                except grb.GurobiError:
                    pi = 0.0
                key = (li, j)
                dual_by_key.setdefault(key, {'lo': 0.0, 'up': 0.0})
                if name.startswith('tri_lo'):
                    dual_by_key[key]['lo'] = pi
                else:
                    dual_by_key[key]['up'] = pi
        for info in unstable_info:
            key = (info['layer_idx'], info['neuron_idx'])
            dd = dual_by_key.get(key, {'lo': 0.0, 'up': 0.0})
            scores[key] = abs(dd['lo']) + abs(dd['up'])
        return scores
    for info in unstable_info:
        key = (info['layer_idx'], info['neuron_idx'])
        if method == 'lp_ew_frac':
            ew = abs(float(obj_coef[info['e_new_col']]))
            lo_j = info['lo']
            hi_j = info['hi']
            frac = hi_j * abs(lo_j) / (hi_j - lo_j)
            scores[key] = ew * frac
        elif method == 'lp_fractional':
            try:
                a_val = var_by_name[info['e_new_var_name']].X
                z_val = info['c_in'] + sum(
                    c * var_by_name[vn].X
                    for c, vn in info['row_coefs_names'])
                if info.get('formulation') == 'phase1':
                    # Phase-1 form: y_k = λ·z + μ·(1 + e_new), not e_new.
                    y_val = info['lam'] * z_val + info['mu'] * (1.0 + a_val)
                else:
                    y_val = a_val
                scores[key] = abs(y_val - max(0.0, z_val))
            except grb.GurobiError:
                scores[key] = 0.0
        else:
            raise ValueError(f'unknown scoring method {method}')
    return scores


def _resolve_standard_lb(state, qw, qb, milp_set, n_threads,
                          time_limit, gg_ops_ser, x_lo, x_hi,
                          bounds_by_relu, input_name, output_op_name,
                          device, dtype):
    """Soundness cross-check: solve the standard LP (no halfspace) on the
    SAME relaxation and return ObjBound. Used to verify that a halfspace
    INFEASIBLE result is consistent with standard `min > 0` (LP feasibility
    theorem). Returns float ObjBound, or None on numeric/API failure
    (caller treats None as 'cannot confirm').

    Cost: one extra LP solve. Half the parent `time_limit` so a hung
    cross-check doesn't double total runtime; BestBdStop=0 lets us bail
    early as soon as ObjBound crosses 0.
    """
    from .gurobi_util import GurobiNumericTrouble
    if state is not None:
        m_x, env_x, _info, _coef = build_gen_lp_from_state(
            state, qw, qb, milp_set=milp_set, n_threads=n_threads,
            unsafe_halfspace='none')
    else:
        m_x, env_x, _info, _coef = build_gen_lp(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            output_op_name, qw, qb, milp_set=milp_set,
            n_threads=n_threads, device=device, dtype=dtype)
    m_x.setParam('TimeLimit', float(time_limit) * 0.5)
    m_x.setParam('BestBdStop', 0.0)
    if milp_set:
        m_x.setParam('Cuts', 0)
        m_x.setParam('Heuristics', 0.0)
        m_x.setParam('Presolve', 0)
        m_x.setParam('MIPFocus', 1)
    try:
        optimize_checked(m_x)
        cross_lb = float(m_x.ObjBound)
    except (GurobiNumericTrouble, grb.GurobiError, AttributeError):
        # Numeric trouble or API failure during cross-solve: cannot
        # confirm soundness. Return None; caller downgrades to
        # INCONCLUSIVE.
        cross_lb = None
    m_x.dispose()
    env_x.dispose()
    return cross_lb


def solve_spec(gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
               output_op_name, qw, qb, *,
               milp_set=None, time_limit=60.0, best_bd_stop=None,
               n_threads=1, device='cuda', dtype=torch.float64,
               score_method='lp_ew_frac', state=None,
               unsafe_halfspace='none', triangle_set=None):
    """Build + solve gen LP or MILP for spec minimization.

    If state is provided (precomputed from precompute_gen_state), reuse it
    to build the Gurobi model — skips the GPU rebuild and ensures bit-identical
    coefficients across calls.

    unsafe_halfspace: see `build_gen_lp_from_state`. When set to
    'inequality' or 'equality' the extra `qw·y + qb ≤/= 0` constraint is
    added. The verification signal in this mode is `Gurobi.INFEASIBLE` —
    empty polytope proves safety. Reported as result='UNSAT' in that case.

    triangle_set: set of (li, neuron_idx) keys to receive the triangle
    floor (`y ≥ 0`, `y ≥ z`) on top of the parallelogram in α-zono
    formulation. Only honored for `state['formulation']=='alpha_zono'`.
    Ignored for triangle-LP formulations (which already encode the
    triangle floor unconditionally).

    Returns dict with: result ('UNSAT'/'INCONCLUSIVE'/'TIMEOUT'/'UNKNOWN'),
    lb (ObjBound), status, solve_time, build_time, scores (if LP).
    UNSAT = Gurobi proved ObjBound > 0 (standard) OR returned INFEASIBLE
    under an unsafe-halfspace constraint; INCONCLUSIVE = solved to
    optimum with optimum ≤ 0 and not infeasible; TIMEOUT = hit TimeLimit
    before reaching optimum. Real-witness check (true SAT) is the
    caller's job — this label is purely Gurobi's classification.
    """
    t_build = time.perf_counter()
    if state is not None:
        m, env, unstable_info, obj_coef = build_gen_lp_from_state(
            state, qw, qb, milp_set=milp_set, n_threads=n_threads,
            unsafe_halfspace=unsafe_halfspace,
            triangle_set=triangle_set)
    else:
        m, env, unstable_info, obj_coef = build_gen_lp(
            gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
            output_op_name, qw, qb, milp_set=milp_set,
            n_threads=n_threads, device=device, dtype=dtype)
        if unsafe_halfspace and unsafe_halfspace != 'none':
            raise NotImplementedError(
                'unsafe_halfspace requires precomputed state path')
    dt_build = time.perf_counter() - t_build

    m.setParam('TimeLimit', float(time_limit))
    if best_bd_stop is not None:
        m.setParam('BestBdStop', float(best_bd_stop))
    if milp_set:
        m.setParam('Cuts', 0)
        m.setParam('Heuristics', 0.0)
        m.setParam('Presolve', 0)
        m.setParam('MIPFocus', 1)

    import os as _os
    _dbg = _os.environ.get('GEN_LP_DEBUG_WRITE')
    if _dbg and milp_set:
        _path = f'{_dbg}_bins{len(milp_set)}.lp'
        m.write(_path)
    t_solve = time.perf_counter()
    optimize_checked(m)
    dt_solve = time.perf_counter() - t_solve

    status = m.Status
    try:
        lb = float(m.ObjBound)
    except (grb.GurobiError, AttributeError):
        lb = None
    n_sol = m.SolCount

    scores = None
    if (not milp_set) and status == grb.GRB.OPTIMAL:
        try:
            scores = compute_scores(m, unstable_info, obj_coef,
                                    method=score_method)
        except (grb.GurobiError, AttributeError):
            scores = None

    # Classify. Distinguish:
    #   UNSAT       — Gurobi proved ObjBound > 0 (true safety certificate)
    #   INCONCLUSIVE — Gurobi solved to OPTIMAL with optimum ≤ 0; the
    #                  formulation can't prove safety with these binaries
    #                  (might still be a true SAT; caller checks witness)
    #   TIMEOUT     — Gurobi hit TimeLimit before reaching optimum; ObjBound
    #                  is the partial proof so far. NOT a SAT verdict — only
    #                  a real witness from the integer solution can be SAT.
    #   UNKNOWN     — solver returned an unexpected status (INF_OR_UNBD etc.)
    # The caller's witness check (forward `e_in` through real network) is
    # the sole source of truth for the 'sat' verdict; this label only
    # describes Gurobi's own classification.
    #
    # SOUNDNESS NOTE — equality vs. inequality halfspace:
    # The user-facing `unsafe_halfspace` parameter has THREE legal values:
    #   'none'       — no extra constraint on the spec output.
    #   'inequality' — add `qw·y + qb ≤ 0`. This carves the relaxation
    #                  polytope down to its intersection with the unsafe
    #                  halfspace. INFEASIBLE here is a SOUND UNSAT signal:
    #                  if no relaxation point has qw·y + qb ≤ 0, then the
    #                  relaxation min > 0, which (because the relaxation
    #                  contains the real network image) proves the spec.
    #   'equality'   — add `qw·y + qb == 0`. This restricts the relaxation
    #                  to the BOUNDARY of unsafe. INFEASIBLE here means the
    #                  relaxation does NOT touch the boundary — but a
    #                  convex relaxation could be entirely BELOW the
    #                  hyperplane (relaxation max < 0, real-SAT case) OR
    #                  entirely ABOVE (relaxation min > 0, spec verified).
    #                  We CANNOT distinguish these two without a second
    #                  solve. Therefore equality + INFEASIBLE is treated
    #                  as INCONCLUSIVE — caller must use inequality
    #                  halfspace, the standard ObjBound>0 proof, or a
    #                  tighter relaxation.
    #                  (This was a soundness bug found on resnet_large
    #                  5162: the parallelogram α-zono relaxation was
    #                  tight enough to lie entirely below the spec
    #                  hyperplane on this real-SAT case, and equality+
    #                  INFEASIBLE → 'UNSAT' incorrectly contradicted the
    #                  α,β-CROWN-confirmed counterexample.)
    if unsafe_halfspace and unsafe_halfspace != 'none':
        if status == grb.GRB.INFEASIBLE:
            if unsafe_halfspace == 'inequality':
                # SOUNDNESS CROSS-CHECK: by LP feasibility theorem
                # inequality halfspace LP infeasible ⟺ standard LP min > 0.
                # Empirically observed (CIFAR100 resnet_large 5162) that
                # Gurobi can report INFEASIBLE on a barely-feasible
                # polytope due to floating-point precision on networks
                # with wide LP coefficient range — leading to a
                # false-verified result. Verify by re-solving the SAME
                # LP (same coefficients, same bounds) WITHOUT the
                # halfspace and confirming ObjBound > 0. If they
                # disagree, treat as INCONCLUSIVE.
                cross_lb = _resolve_standard_lb(
                    state, qw, qb, milp_set, n_threads,
                    time_limit, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                    input_name, output_op_name, device, dtype)
                if cross_lb is not None and cross_lb > 1e-7:
                    result = 'UNSAT'
                else:
                    # Numerical inconsistency: halfspace says infeasible
                    # but standard LP says optimum ≤ 0. NOT sound to claim
                    # verified — downgrade. (cross_lb=None can also fall
                    # here if the cross-solve itself failed.)
                    result = 'INCONCLUSIVE'
            elif unsafe_halfspace == 'equality':
                # Cannot disambiguate "relaxation entirely above
                # hyperplane" (spec verified) from "relaxation entirely
                # below" (real SAT). Treat as inconclusive — caller
                # should switch to 'inequality' for sound verification.
                result = 'INCONCLUSIVE'
            else:
                raise ValueError(
                    f'unknown unsafe_halfspace {unsafe_halfspace!r}')
        elif status == grb.GRB.TIME_LIMIT:
            result = 'TIMEOUT'
        else:
            # OPTIMAL / USER_OBJ_LIMIT / etc. — boundary or feasible point
            # found. Witness must be checked by caller. Never treat these
            # as UNSAT when the objective is bounded by the halfspace.
            result = 'INCONCLUSIVE'
    elif milp_set:
        if status == grb.GRB.OPTIMAL:
            result = 'UNSAT' if lb is not None and lb > 0 else 'INCONCLUSIVE'
        elif status == grb.GRB.USER_OBJ_LIMIT:
            # BestBdStop=0 fires when ObjBound crosses 0 → UNSAT.
            result = 'UNSAT' if lb is not None and lb > 0 else 'INCONCLUSIVE'
        elif status == grb.GRB.TIME_LIMIT:
            result = 'TIMEOUT'
        elif status == grb.GRB.INFEASIBLE:
            # No halfspace constraint but MILP infeasible — shouldn't happen
            # in our setup but pass through as INCONCLUSIVE.
            result = 'INCONCLUSIVE'
        else:
            result = 'UNKNOWN'
    else:
        if status == grb.GRB.OPTIMAL:
            result = 'UNSAT' if lb is not None and lb > 0 else 'INCONCLUSIVE'
        else:
            result = 'UNKNOWN'

    # Extract input witness (e_in values) if a feasible solution exists.
    e_in = None
    if n_sol > 0 and state is not None:
        try:
            n_input = state['n_input']
            e_in = np.empty(n_input, dtype=np.float64)
            for i in range(n_input):
                e_in[i] = m.getVarByName(f'e_in_{i}').X
        except (grb.GurobiError, AttributeError):
            e_in = None

    info = {
        'n_vars': m.NumVars,
        'n_constrs': m.NumConstrs,
        'n_bins': m.NumBinVars,
        'status': status,
        'lb': lb,
        'build_time': dt_build,
        'solve_time': dt_solve,
        'scores': scores,
        'e_in': e_in,
    }

    m.dispose()
    env.dispose()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return result, dt_build + dt_solve, info


def tighten_bounds(gg_ops_ser, x_lo, x_hi, initial_bounds, input_name, *,
                    sample_timeout=5.0, time_left_fn=None, n_threads=4,
                    device='cuda', dtype=torch.float64, print_progress=False):
    """Layer-by-layer per-neuron tightening via gen LP.

    Walks ops, maintaining (center, G) state. At each ReLU, solves
    min/max LPs to tighten each unstable neuron's pre-activation bounds
    before classifying stable/dead/unstable.

    Returns tightened bounds_by_relu (new dict).
    """
    gpu = torch.device(device)
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)
    m.setParam('Method', 1)  # dual simplex (fast re-solve after obj change)

    n_in = len(x_lo)
    e_vars = [m.addVar(lb=-1.0, ub=1.0) for _ in range(n_in)]
    m.update()

    center = {input_name: torch.tensor(
        (x_hi + x_lo) / 2, dtype=dtype, device=gpu)}
    half_width = torch.tensor((x_hi - x_lo) / 2, dtype=dtype, device=gpu)
    G_by_op = {input_name: torch.diag(half_width)}
    n_gens = n_in
    bounds_by_relu = {}

    last_use = {}
    for i, op in enumerate(gg_ops_ser):
        for inp in op['inputs']:
            last_use[inp] = i

    def pad_cols(G):
        if G.shape[1] < n_gens:
            return torch.cat([G, torch.zeros(
                G.shape[0], n_gens - G.shape[1], dtype=dtype,
                device=gpu)], dim=1)
        return G

    def _tighten_neuron(row, c_j, orig_lo, orig_hi):
        """Solve min/max LP for a neuron's pre-activation."""
        nz_coefs = [(float(row[k]), e_vars[k])
                    for k in range(n_gens) if row[k] != 0]
        if not nz_coefs:
            return orig_lo, orig_hi
        obj = grb.LinExpr(nz_coefs) + float(c_j)
        # Min
        m.setObjective(obj, grb.GRB.MINIMIZE)
        optimize_checked(m)
        if m.Status == grb.GRB.OPTIMAL:
            new_lo = max(orig_lo, float(m.ObjVal))
        else:
            new_lo = orig_lo
        # Max
        m.setObjective(obj, grb.GRB.MAXIMIZE)
        optimize_checked(m)
        if m.Status == grb.GRB.OPTIMAL:
            new_hi = min(orig_hi, float(m.ObjVal))
        else:
            new_hi = orig_hi
        return new_lo, new_hi

    for op_idx, op in enumerate(gg_ops_ser):
        nm = op['name']
        t = op['type']

        if t == 'conv':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            C_in, H_in, W_in = op['in_shape']
            kernel = torch.tensor(op['kernel_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            sH, sW = op['stride']
            pH, pW = op['padding']
            c_img = prev_c.reshape(1, C_in, H_in, W_in)
            c_out = F.conv2d(c_img, kernel, bias=bias,
                             stride=(sH, sW), padding=(pH, pW)).flatten()
            g_img = prev_G.t().contiguous().reshape(
                n_gens, C_in, H_in, W_in)
            g_out = F.conv2d(g_img, kernel, bias=None,
                             stride=(sH, sW), padding=(pH, pW))
            g_out = g_out.reshape(n_gens, -1).t()
            center[nm] = c_out
            G_by_op[nm] = g_out

        elif t == 'fc':
            prev_c = center[op['inputs'][0]]
            prev_G = pad_cols(G_by_op[op['inputs'][0]])
            W = torch.tensor(op['W_np'], dtype=dtype, device=gpu)
            bias = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            center[nm] = W @ prev_c + bias
            G_by_op[nm] = W @ prev_G

        elif t == 'relu':
            if 'layer_idx' not in op:
                center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]
                continue
            li = op['layer_idx']
            orig_lo_arr, orig_hi_arr = initial_bounds[li]
            new_lo = orig_lo_arr.copy()
            new_hi = orig_hi_arr.copy()
            c_in = center[op['inputs'][0]]
            G_in = pad_cols(G_by_op[op['inputs'][0]])
            n = len(c_in)

            # Tighten unstable neurons
            unstable = np.where((orig_lo_arr < 0) & (orig_hi_arr > 0))[0]
            c_in_cpu = c_in.detach().cpu().numpy()
            if len(unstable) > 0:
                uidx_t = torch.tensor(unstable, device=gpu, dtype=torch.long)
                G_unstable = G_in[uidx_t].detach().cpu().numpy()
            t_tighten_start = time.perf_counter()
            tightened_count = 0
            for local_idx, j in enumerate(unstable):
                if time_left_fn is not None and time_left_fn() <= 0:
                    break
                row = G_unstable[local_idx]
                nlo, nhi = _tighten_neuron(
                    row, float(c_in_cpu[j]),
                    float(orig_lo_arr[j]), float(orig_hi_arr[j]))
                new_lo[j] = nlo
                new_hi[j] = nhi
                if nlo >= 0 or nhi <= 0:
                    tightened_count += 1
            dt_tighten = time.perf_counter() - t_tighten_start
            bounds_by_relu[li] = (new_lo, new_hi)
            if print_progress:
                n_after_unstable = int(((new_lo < 0) & (new_hi > 0)).sum())
                print(f'  li={li}: {len(unstable)} → {n_after_unstable} '
                      f'unstable (tightened {tightened_count} in {dt_tighten:.2f}s)')

            # Apply relu using tightened bounds, add gen vars & constraints
            dead_mask = new_hi <= 0
            stable_mask = (new_lo >= 0) & ~dead_mask
            unstable_mask = ~dead_mask & ~stable_mask
            unstable_idx = np.where(unstable_mask)[0]
            stable_idx = np.where(stable_mask)[0]
            n_new = int(unstable_mask.sum())

            new_gen_vars = [m.addVar(lb=0.0, ub=float(new_hi[j]))
                            for j in unstable_idx]
            m.update()

            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(
                    unstable_idx, device=gpu, dtype=torch.long)
                G_new_unstable = G_in[uidx_t].detach().cpu().numpy()
            for local_idx, j in enumerate(unstable_idx):
                lo_j = float(new_lo[j])
                hi_j = float(new_hi[j])
                slope = hi_j / (hi_j - lo_j)
                row = G_new_unstable[local_idx]
                expr_coefs = [(float(row[k]), e_vars[k])
                              for k in range(n_gens) if row[k] != 0]
                m.addLConstr(grb.LinExpr(expr_coefs)
                              - new_gen_vars[local_idx]
                              <= -float(c_in_cpu[j]))
                m.addLConstr(new_gen_vars[local_idx]
                              - grb.LinExpr([(slope * w, v)
                                             for w, v in expr_coefs])
                              <= slope * (float(c_in_cpu[j]) - lo_j))
            m.update()

            # Build output G
            G_out = torch.zeros(n, n_gens + n_new, dtype=dtype, device=gpu)
            if len(stable_idx) > 0:
                sidx_t = torch.tensor(
                    stable_idx, device=gpu, dtype=torch.long)
                G_out[sidx_t, :n_gens] = G_in[sidx_t]
            if len(unstable_idx) > 0:
                uidx_t = torch.tensor(
                    unstable_idx, device=gpu, dtype=torch.long)
                new_col_idx = torch.arange(
                    n_gens, n_gens + n_new, device=gpu, dtype=torch.long)
                G_out[uidx_t, new_col_idx] = 1.0
            c_out = torch.zeros(n, dtype=dtype, device=gpu)
            if len(stable_idx) > 0:
                c_out[sidx_t] = c_in[sidx_t]
            center[nm] = c_out
            G_by_op[nm] = G_out
            e_vars.extend(new_gen_vars)
            n_gens += n_new

        elif t == 'add':
            if op.get('is_merge'):
                ca = center[op['inputs'][0]]
                cb = center[op['inputs'][1]]
                # Residual add: any sparse operand must be materialized
                # before the elementwise add (sparse + sparse fast path
                # not implemented).
                Ga = _materialize_g(pad_cols(G_by_op[op['inputs'][0]]))
                Gb = _materialize_g(pad_cols(G_by_op[op['inputs'][1]]))
                center[nm] = ca + cb
                G_by_op[nm] = Ga + Gb
            else:
                b = op.get('bias')
                if b is not None:
                    bt = torch.tensor(
                        np.asarray(b, dtype=np.float64).flatten(),
                        dtype=dtype, device=gpu)
                    center[nm] = center[op['inputs'][0]] + bt
                else:
                    center[nm] = center[op['inputs'][0]]
                G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'sub':
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            b = op.get('bias')
            if b is not None:
                bt = torch.tensor(b.flatten(), dtype=dtype, device=gpu)
                center[nm] = prev_c - bt
            else:
                center[nm] = prev_c
            G_by_op[nm] = prev_G

        elif t == 'reshape':
            center[nm] = center[op['inputs'][0]]
            G_by_op[nm] = G_by_op[op['inputs'][0]]

        elif t == 'mul':
            # y = scale * x (constant scale, possibly scalar-broadcast).
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            scale = op.get('scale')
            if isinstance(scale, np.ndarray):
                st = torch.from_numpy(scale).to(device=gpu, dtype=dtype)
            elif not isinstance(scale, torch.Tensor):
                st = torch.tensor(scale, dtype=dtype, device=gpu)
            else:
                st = scale.to(device=gpu, dtype=dtype)
            sflat = st.flatten()
            n = prev_c.numel()
            if sflat.numel() == 1 or sflat.numel() == n:
                center[nm] = prev_c * sflat
                Gm = _materialize_g(pad_cols(prev_G))
                G_by_op[nm] = (Gm * sflat if sflat.numel() == 1
                               else Gm * sflat.unsqueeze(-1))
            else:
                in_shape = op.get('in_shapes_nd', [None])[0]
                C, H, W = in_shape
                assert sflat.numel() == C
                s4d = sflat.view(1, C, 1, 1).expand(1, C, H, W).reshape(-1)
                center[nm] = prev_c * s4d
                Gm = _materialize_g(pad_cols(prev_G))
                G_by_op[nm] = Gm * s4d.unsqueeze(-1)

        else:
            raise NotImplementedError(
                f'gen-LP state forward: unsupported op {t!r} (name={nm!r}). '
                'Silent skip would leave stale G_by_op → unsound encoding.')

        for inp in op['inputs']:
            if (last_use.get(inp) == op_idx and inp in G_by_op
                    and inp != nm):
                del G_by_op[inp]
                del center[inp]

    if device == 'cuda':
        torch.cuda.synchronize()
    G_by_op.clear()
    center.clear()
    m.dispose()
    env.dispose()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return bounds_by_relu


def gen_lp_worker(args):
    """Multiprocessing-friendly worker wrapping solve_spec."""
    (gg_ops_ser, x_lo, x_hi, bbr, input_name, output_op_name,
     qw, qb, milp_set, time_limit, best_bd_stop, n_threads,
     device, score_method) = args
    return solve_spec(
        gg_ops_ser, x_lo, x_hi, bbr, input_name, output_op_name,
        qw, qb, milp_set=milp_set, time_limit=time_limit,
        best_bd_stop=best_bd_stop, n_threads=n_threads,
        device=device, score_method=score_method)


def racing_escalation(state, qw, qb, scored_keys, *,
                      time_left_fn, n_threads=1, print_progress=False):
    """Serial MILP racing over a doubling bin schedule with early stop.

    After each MILP solve, extracts the input witness from the integer
    solution and forward-propagates it through the real network; if
    `qw @ y + qb < 0` at that x, this is a true counterexample (early
    SAT exit). Otherwise escalates bins=2,4,8,... stopping at UNSAT or
    timeout.

    Returns (verdict, levels_info, witness_x) where verdict is
    'unsat' | 'sat' | 'unknown' and witness_x is the counterexample
    input (np.ndarray) on SAT, else None.
    """
    if state is None:
        raise ValueError('racing_escalation requires precomputed state')
    bin_schedule = []
    b = 2
    while b <= len(scored_keys):
        bin_schedule.append(b)
        b *= 2
    if scored_keys and (not bin_schedule
                        or bin_schedule[-1] < len(scored_keys)):
        bin_schedule.append(len(scored_keys))

    x_lo = state['x_lo']
    x_hi = state['x_hi']
    c = (x_hi + x_lo) / 2.0
    half_w = (x_hi - x_lo) / 2.0

    levels = []
    for n_bins in bin_schedule:
        tl = time_left_fn()
        if tl <= 0:
            break
        milp_set = set(scored_keys[:n_bins])
        result, dt, info = solve_spec(
            None, None, None, None, None, None, qw, qb,
            milp_set=milp_set, time_limit=tl, best_bd_stop=0.0,
            n_threads=n_threads, state=state)
        lb = info.get('lb')
        lb_s = f'{lb:+.4f}' if isinstance(lb, float) else 'n/a'
        levels.append({
            'n_bins': n_bins, 'result': result, 'time': dt,
            'lb': lb, 'info': info,
        })

        # Witness check on the integer solution's input assignment.
        e_in = info.get('e_in')
        witness_x = None
        if e_in is not None:
            x = c + half_w * e_in
            y = forward_point(state['gg_ops_ser'], x,
                              state['input_name'], state['output_op_name'])
            spec_val = float(np.dot(qw.astype(np.float64), y) + qb)
            if spec_val < 0:
                witness_x = x
                if print_progress:
                    print(f'    Racing bins={n_bins}: SAT (witness, '
                          f'spec={spec_val:+.6f}) lb={lb_s} ({dt:.1f}s)')
                return 'sat', levels, witness_x

        if print_progress:
            print(f'    Racing bins={n_bins}: {result} '
                  f'lb={lb_s} ({dt:.1f}s)')
        if result == 'UNSAT':
            return 'unsat', levels, None
    return 'unknown', levels, None


# Module-level globals for per-query worker (fork-shared on Linux)
_Q_STATE = None
_Q_STATE_BY_QI = None
_Q_N_THREADS = None
_Q_TIME_LEFT_DEADLINE = None
_Q_UNSAFE_HALFSPACE = 'none'
_Q_TRIANGLE_TOP_K = 0


def _query_race_init(state, n_threads, deadline, unsafe_halfspace='none',
                      triangle_top_k=0):
    global _Q_STATE, _Q_STATE_BY_QI, _Q_N_THREADS
    global _Q_TIME_LEFT_DEADLINE, _Q_UNSAFE_HALFSPACE, _Q_TRIANGLE_TOP_K
    # state is either a raw state dict (legacy) or a (shared_state, state_by_qi)
    # tuple. Workers fall back to shared_state when state_by_qi is None or
    # when a specific qi is absent from the per-query map.
    if isinstance(state, tuple) and len(state) == 2:
        _Q_STATE, _Q_STATE_BY_QI = state
    else:
        _Q_STATE, _Q_STATE_BY_QI = state, None
    _Q_N_THREADS = n_threads
    _Q_TIME_LEFT_DEADLINE = deadline
    _Q_UNSAFE_HALFSPACE = unsafe_halfspace
    _Q_TRIANGLE_TOP_K = int(triangle_top_k)


def _query_race_solve(args):
    qi, qw, qb, scored_keys = args
    import time as _t
    def _tl():
        return _Q_TIME_LEFT_DEADLINE - _t.perf_counter()
    verdict, levels, witness = racing_escalation(
        _Q_STATE, qw, qb, scored_keys,
        time_left_fn=_tl, n_threads=_Q_N_THREADS, print_progress=False)
    return qi, verdict, levels, witness


def _query_race_one_bin(args):
    """Solve one (spec, n_bins) task. Returns level info + witness."""
    qi, qw, qb, scored_keys, n_bins = args
    import time as _t
    tl = _Q_TIME_LEFT_DEADLINE - _t.perf_counter()
    if tl <= 0:
        return (qi, n_bins, 'unknown',
                {'n_bins': n_bins, 'result': 'timeout',
                 'time': 0.0, 'lb': None}, None)
    milp_set = set(scored_keys[:n_bins])
    # Triangle-top-K (alpha_zono only): the next K most-important neurons
    # beyond the binary set get triangle floor constraints (y ≥ 0,
    # y ≥ z) but no binary. Wider-than-binary, tighter-than-parallelogram.
    # Note: to keep the racing escalation monotone (more bins → tighter
    # MILP), the triangle set is always the top-K from `scored_keys`,
    # which IS a superset of milp_set when K ≥ n_bins.
    triangle_set = (set(scored_keys[:_Q_TRIANGLE_TOP_K])
                    if _Q_TRIANGLE_TOP_K > 0 else None)
    # Prefer per-query tightened state if available.
    state_for_qi = _Q_STATE
    if _Q_STATE_BY_QI is not None and qi in _Q_STATE_BY_QI:
        state_for_qi = _Q_STATE_BY_QI[qi]
    # BestBdStop=0 is a no-op when the halfspace is in the MILP (the
    # halfspace constraint forces the objective ≤ 0 so ObjBound > 0 is
    # impossible in exact arithmetic; any trigger is pure numerical
    # overshoot and was the source of the earlier false-UNSAT bug).
    # Only use BestBdStop in the standard Phase 8 path.
    _bds = (None if (_Q_UNSAFE_HALFSPACE
                      and _Q_UNSAFE_HALFSPACE != 'none')
            else 0.0)
    result, dt, info = solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=milp_set, time_limit=tl, best_bd_stop=_bds,
        n_threads=_Q_N_THREADS, state=state_for_qi,
        unsafe_halfspace=_Q_UNSAFE_HALFSPACE,
        triangle_set=triangle_set)
    level_info = {'n_bins': n_bins, 'result': result, 'time': dt,
                  'lb': info.get('lb'), 'info': info}

    # Witness check
    witness = None
    e_in = info.get('e_in')
    if e_in is not None:
        c = (state_for_qi['x_hi'] + state_for_qi['x_lo']) / 2.0
        half_w = (state_for_qi['x_hi'] - state_for_qi['x_lo']) / 2.0
        x = c + half_w * e_in
        y = forward_point(state_for_qi['gg_ops_ser'], x,
                          state_for_qi['input_name'],
                          state_for_qi['output_op_name'])
        if float(np.dot(qw.astype(np.float64), y) + qb) < 0:
            witness = x
            level_info['witness_source'] = 'milp_direct'

    if result == 'UNSAT':
        verdict = 'unsat'
    elif witness is not None:
        verdict = 'sat'
    else:
        verdict = 'unknown'
    return qi, n_bins, verdict, level_info, witness


def sequential_query_racing(state, query_specs, *, time_left_fn,
                             n_threads=1, print_progress=False):
    """Run MILP racing across open queries sequentially.

    For each open spec, runs `racing_escalation` (serial bins with early
    termination on UNSAT or witness). Uses `n_threads` Gurobi threads
    per MILP solve.

    Returns the same (qi, verdict, levels, witness) tuples as
    parallel_query_racing, in submission order.
    """
    results = []
    for qi, qw, qb, scored_keys in query_specs:
        if time_left_fn() <= 0:
            results.append((qi, 'unknown', [], None))
            continue
        verdict, levels, witness = racing_escalation(
            state, qw, qb, scored_keys,
            time_left_fn=time_left_fn, n_threads=n_threads,
            print_progress=print_progress)
        results.append((qi, verdict, levels, witness))
    return results


def parallel_query_racing(state, query_specs, *, time_left_fn,
                           n_threads_total=4, print_progress=False,
                           gurobi_threads=1, state_by_qi=None,
                           unsafe_halfspace='none',
                           triangle_top_k=0,
                           witness_refine_fn=None,
                           bin_schedule_override=None):
    """Run MILP racing across open queries with idle-core-filling.

    Submits one task per (spec, bin_level). Tasks are ordered so every
    spec's first bin runs before any spec's second bin, etc. Pool size is
    `n_threads_total`, each worker uses `gurobi_threads` threads. When a spec
    resolves (UNSAT via BestBdStop or SAT witness), later tasks for that
    spec still execute but their results are ignored; the pool exits on
    first SAT witness (global) or when every spec is resolved.

    Bin schedule is the additive 'octaves' schedule [8, 16, 24, ..., 8k]
    with 8k ≤ n_threads_total — all levels launch concurrently and
    BestBdStop=0 on each worker produces first-to-prove racing. Low-bin
    workers finish fast but may not prove tightness; high-bin workers
    are stronger but slower. Saturating the pool with distinct bin
    counts maximises the chance some level proves UNSAT.

    query_specs: list of (qi, qw, qb, scored_keys) tuples.
    Returns: list of (qi, verdict, levels, witness) in submission order,
    where verdict ∈ {'unsat', 'sat', 'unknown'}.
    """
    import multiprocessing as _mp
    import time as _t
    if not query_specs:
        return []
    tl = time_left_fn()
    if tl <= 0:
        return [(qi, 'unknown', [], None) for qi, _, _, _ in query_specs]
    deadline = _t.perf_counter() + tl

    def _build_schedule(n_scored):
        """Bin counts for one spec.

        Default: [8, 16, 24, ..., 8*n_threads_total]. ``bin_schedule_override``
        (per-call) replaces this with a fixed list — useful when you've
        empirically measured that intermediate K's catch real cases the
        stride-8 schedule skips (e.g., on TinyImageNet K=18, 22, 30 close
        cases that K=16/24/32 miss with our box+halfspace scoring).
        """
        if bin_schedule_override is not None:
            out = []
            for b in bin_schedule_override:
                b = min(int(b), n_scored)
                if b > 0 and b not in out:
                    out.append(b)
            return out
        k_max = max(1, n_threads_total)
        sched = [min(8 * k, n_scored) for k in range(1, k_max + 1)]
        out = []
        for b in sched:
            if b > 0 and b not in out:
                out.append(b)
        return out

    per_spec_schedule = []
    for qi, qw, qb, scored_keys in query_specs:
        schedule = _build_schedule(len(scored_keys))
        per_spec_schedule.append((qi, qw, qb, scored_keys, schedule))
    max_levels = max((len(s[-1]) for s in per_spec_schedule), default=0)
    tasks = []
    for lvl_idx in range(max_levels):
        for qi, qw, qb, scored_keys, schedule in per_spec_schedule:
            if lvl_idx < len(schedule):
                tasks.append((qi, qw, qb, scored_keys, schedule[lvl_idx]))

    results = {qi: {'verdict': 'unknown', 'levels': [], 'witness': None}
               for qi, _, _, _ in query_specs}
    done_specs = set()
    found_sat = False

    n_workers = max(1, n_threads_total)
    ctx = _mp.get_context('fork')
    # Per-query state dispatch: if state_by_qi provided, pass the full map to
    # workers and let each task pick its state by qi. Shared `state` is the
    # fallback for queries without a per-query override.
    init_state = (state, state_by_qi) if state_by_qi else (state, None)
    pool = ctx.Pool(n_workers, initializer=_query_race_init,
                     initargs=(init_state, gurobi_threads, deadline,
                                unsafe_halfspace,
                                int(triangle_top_k)))
    try:
        for out in pool.imap_unordered(_query_race_one_bin, tasks):
            # Hard wall-clock check: if past deadline, terminate the pool
            # immediately and stop consuming results. Without this, the
            # pool's context-manager exit waits for in-flight workers
            # whose Gurobi solves were started with a long per-MILP
            # `time_limit` (computed when the task was scheduled, not
            # the residual time at exit), causing total wall time to
            # overshoot `total_timeout` by many tens of seconds and
            # eventually getting killed by the OS-level outer timeout.
            if _t.perf_counter() >= deadline:
                break
            qi, n_bins, verdict, level_info, witness = out
            r = results[qi]
            if qi in done_specs:
                continue
            r['levels'].append(level_info)
            # Main-thread MILP-seeded PGD refinement: the worker already
            # did a one-point forward check on e_in; if that failed to
            # find a real counterexample but Gurobi DID find a feasible
            # integer point, the MILP's e_in is a near-boundary candidate
            # — PGD from there may walk to a real violation. Run in main
            # thread (no GPU contention with workers, no pool.terminate
            # races). Skip if the worker already produced a real witness.
            if (witness is None and witness_refine_fn is not None
                    and verdict != 'unsat'):
                info_in = level_info.get('info', {}) or {}
                e_in = info_in.get('e_in')
                if e_in is not None:
                    try:
                        refined = witness_refine_fn(qi, e_in)
                    except (RuntimeError, ValueError, KeyError):
                        # PGD refinement failures: RuntimeError from torch
                        # autograd on degenerate inputs, ValueError from
                        # bounds mismatch, KeyError if a layer is missing
                        # from the cached state. All fall back to no
                        # refinement (the unrefined witness is still sound).
                        refined = None
                    if refined is not None:
                        witness = refined
                        verdict = 'sat'
                        level_info['refined_by_pgd'] = True
                        level_info['witness_source'] = 'pgd_seeded'
            if verdict == 'unsat':
                r['verdict'] = 'unsat'
                done_specs.add(qi)
            elif verdict == 'sat':
                r['verdict'] = 'sat'
                r['witness'] = witness
                done_specs.add(qi)
                found_sat = True
            if found_sat or len(done_specs) == len(query_specs):
                break
    finally:
        # Force-terminate workers regardless of how we exit. Pool
        # context-manager `__exit__` calls terminate() too, but doing
        # it explicitly here ensures the SIGTERM is sent before the
        # `finally` continues, and any blocked Gurobi solves are
        # interrupted as fast as the OS allows.
        pool.terminate()
        pool.join()

    return [(qi, results[qi]['verdict'], results[qi]['levels'],
             results[qi]['witness'])
            for qi, _, _, _ in query_specs]
