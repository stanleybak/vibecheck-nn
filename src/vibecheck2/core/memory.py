"""The one memory-budget service (design 2.1).

Every batched kernel in the core sizes its work through `chunked`. Predictive
sizing first (declared bytes/item vs free memory with a safety factor), then
ONE narrow OOM fallback: catch CUDA OOM here only, halve, log, retry, and
re-raise loudly at the floor. Nothing else in the core may catch OOM (CLAUDE.md).
"""
from __future__ import annotations

import sys

import torch

SAFETY = 0.5          # use at most this fraction of free memory per chunk
_MIN_CHUNK = 1        # below this, the OOM is real: re-raise


def free_bytes(device) -> int:
    dev = torch.device(device)
    if dev.type == 'cuda':
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free)
    # CPU: keep chunks modest rather than probing the OS; 4 GB nominal.
    return 4 << 30


def chunk_size(n_items: int, bytes_per_item: float, device) -> int:
    """Predicted #items per chunk. Always in [1, n_items]."""
    if bytes_per_item <= 0:
        return n_items
    fit = int(free_bytes(device) * SAFETY / bytes_per_item)
    return max(1, min(n_items, fit))


def chunked(fn, X: torch.Tensor, bytes_per_item: float):
    """Apply `fn` over the leading dim of X in memory-budgeted chunks.

    fn maps (b, ...) -> (b, ...); results are concatenated on dim 0.
    The ONLY sanctioned CUDA-OOM catch in the core lives here.
    """
    n = X.shape[0]
    cs = chunk_size(n, bytes_per_item, X.device)
    outs = []
    i = 0
    while i < n:
        try:
            outs.append(fn(X[i:i + cs]))
            i += cs
        except torch.cuda.OutOfMemoryError:
            if cs <= _MIN_CHUNK:
                raise
            torch.cuda.empty_cache()
            cs = max(_MIN_CHUNK, cs // 2)
            print(f'[memory] CUDA OOM at chunk={2*cs}; retrying with {cs}',
                  file=sys.stderr, flush=True)
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
