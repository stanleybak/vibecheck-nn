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

from .network import ComputeGraph, GraphNode
from .zonotope import DenseZonotope
from .verify import zonotope_verify
from .vnnlib_loader import load_vnnlib, parse_vnnlib_text
from .spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint
