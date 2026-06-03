"""VibeCheck — Vibe-Coded Neural Network Verification Tool."""

# Force single-threaded BLAS — multi-threaded OpenBLAS causes massive
# overhead on the small matrices typical in verification workloads.
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
# CUDA allocator: expandable segments drastically reduce fragmentation for
# our zonotope workflow, where G matrices grow by concatenation each ReLU
# layer. Without this, a 2.3 GB working set fragments so badly that
# allocating the next 1 GB G-cat fails even with 5+ GB formally free.
# α,β-CROWN on the same resnet_large uses ~2.3 GB and completes without
# an OOM only because (a) it uses Patches representation (no concat) and
# (b) we inherit this flag by default here.
_os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
del _os

# Disable TF32 on Ampere+ GPUs. By default cuDNN runs convolutions in TF32
# (10-bit mantissa), which introduces ~4e-3 forward error on conv nets —
# measured on the A10G TinyImageNet ResNet: gpu_graph forward off by 0.004
# vs the true fp32 model with TF32 on, vs 1e-5 with it off. That error
# swamps knife-edge counterexample margins (PGD then chases spurious
# witnesses and misses real CEXes) AND makes the verification BOUNDS
# unsound at tight margins (a `verified` with certified margin < ~4e-3 may
# not hold for the true model). α,β-CROWN disables both for the same reason
# (abcrown.py:76-77). matmul TF32 already defaults off in recent torch;
# cudnn does not, so set both explicitly.
import torch as _torch
_torch.backends.cuda.matmul.allow_tf32 = False
_torch.backends.cudnn.allow_tf32 = False
del _torch

from .network import ComputeGraph, GraphNode
from .zonotope import DenseZonotope, TorchZonotope
from .verify import zonotope_verify
from .verify_zono_bnb import zonotope_bnb_verify
from .settings import default_settings
from .vnnlib_loader import load_vnnlib, parse_vnnlib_text
from .spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint
