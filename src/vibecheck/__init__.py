"""VibeCheck — Vibe-Coded Neural Network Verification Tool."""

# Force single-threaded BLAS — multi-threaded OpenBLAS causes massive
# overhead on the small matrices typical in verification workloads.
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
del _os

from .network import ComputeGraph, GraphNode
from .zonotope import DenseZonotope
from .verify import zonotope_verify
from .vnnlib_loader import load_vnnlib, parse_vnnlib_text
from .spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint
