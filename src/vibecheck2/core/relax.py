"""RelaxLib: one relaxation object per elementwise nonlinearity (design 3.4).

Each entry owns everything the rest of the core needs to know about its op:

  point(x)                exact evaluation (torch)
  planes(lo, hi)          sound elementwise linear bounds on [lo, hi]:
                          (al, bl, au, bu) with al*x+bl <= f(x) <= au*x+bu.
                          Closed-form / provably bracketing ONLY, never
                          sampled (CLAUDE.md). Where a plane is optimizable,
                          the default is the sound midpoint choice; alpha
                          parameterization arrives with the backward pass.

Adversarial sampling VALIDATES planes in tests; it never defines them.
Ops are registered in REL by name; an unknown fn raises KeyError loudly.
"""
from __future__ import annotations

import torch


class Relu:
    def point(self, x, params=None):
        return torch.relu(x)

    def planes(self, lo, hi):
        """DeepZ/CROWN triangle: exact on stable neurons, slope=hi/(hi-lo)
        chord above, adaptive (0 or 1) tangent below on unstable ones."""
        unstable = (lo < 0) & (hi > 0)
        pos = lo >= 0
        # upper: chord through (lo, relu(lo)), (hi, relu(hi))
        denom = (hi - lo).clamp_min(1e-30)
        au = torch.where(unstable, hi / denom, (pos).to(lo.dtype))
        bu = torch.where(unstable, -hi * lo / denom, torch.zeros_like(lo))
        # lower: adaptive tangent y=0 or y=x, whichever is tighter (|lo| vs hi)
        al = torch.where(unstable, (hi >= -lo).to(lo.dtype), pos.to(lo.dtype))
        bl = torch.zeros_like(lo)
        return al, bl, au, bu


class LeakyRelu:
    def point(self, x, params=None):
        alpha = (params or {}).get('alpha', 0.01)
        return torch.nn.functional.leaky_relu(x, alpha)


class Sigmoid:
    def point(self, x, params=None):
        return torch.sigmoid(x)


class Tanh:
    def point(self, x, params=None):
        return torch.tanh(x)


class Sin:
    def point(self, x, params=None):
        return torch.sin(x)


class Cos:
    def point(self, x, params=None):
        return torch.cos(x)


class Exp:
    def point(self, x, params=None):
        return torch.exp(x)


class Pow:
    def point(self, x, params=None):
        return x ** (params or {})['exponent']


class SignFn:
    def point(self, x, params=None):
        return torch.sign(x)


class Floor:
    def point(self, x, params=None):
        return torch.floor(x)


REL = {'relu': Relu(), 'leaky_relu': LeakyRelu(), 'sigmoid': Sigmoid(),
       'tanh': Tanh(), 'sin': Sin(), 'cos': Cos(), 'exp': Exp(),
       'pow': Pow(), 'sign': SignFn(), 'floor': Floor()}
