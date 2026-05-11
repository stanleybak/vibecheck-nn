"""Per-instance default-settings selection.

`default_settings_for(graph, spec)` chooses one of four profiles based on
a small fingerprint of the network and spec, then returns a fully-formed
DotMap settings object. Profiles encode what the cleanup plan's Phase 3
called the "auto-detection map":

    fingerprint                       profile           overrides
    --------------------------------  ---------------   -------------------
    input_dim ≤ 20                    input_split_small input-split fast leaf
    has_conv ∧ no forks ∧ n_relu ≥ 5  conv_deep         oval21-style routing
    no conv ∧ n_relu ≤ 6              fc_shallow        bab_refine cascade
    else                              default           current production

The fingerprint is read from the existing `ComputeGraph` API
(`input_shape`, `relu_nodes()`, `fork_points()`) plus the spec's
`x_lo.shape`. No new graph traversals are added.

Goal: per-benchmark scripts that previously set 5–7 knobs reduce to ≤ 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .settings import default_settings


@dataclass(frozen=True)
class GraphFingerprint:
    """Compact summary used by `default_settings_for` to pick a profile."""
    input_dim: int
    has_conv: bool
    n_relu: int
    fork_count: int

    @classmethod
    def from_graph_and_spec(cls, graph, spec) -> 'GraphFingerprint':
        if hasattr(spec, 'x_lo'):
            input_dim = int(np.prod(spec.x_lo.shape))
        elif graph.input_shape:
            input_dim = int(np.prod(graph.input_shape))
        else:
            input_dim = 10**9
        has_conv = any(getattr(n, 'op_type', '') == 'Conv'
                        for n in graph.nodes.values())
        n_relu = len(graph.relu_nodes())
        fork_count = len(graph.fork_points())
        return cls(input_dim=input_dim, has_conv=has_conv,
                   n_relu=n_relu, fork_count=fork_count)


def _profile_input_split_small(s) -> None:
    """Small-input nets (input_dim ≤ 20). cifar_biasfield is the type case
    — 16 input dims, ResNet-style backbone. Input-split BaB with the fast
    leaf path verifies these without the full Phase 1 pipeline. The
    defaults already enable input_split_enabled + input_split_fast_leaf,
    so this profile mostly tightens the leaf budget."""
    s.input_split_enabled = True
    s.input_split_fast_leaf = True
    s.input_split_alpha_iters = 3


def _profile_conv_deep(s) -> None:
    """Deep conv ResNets (oval21 cifar_*_kw): the historical milp_verify
    pipeline closes medium-eps cases that bab_refine times out on. The
    auto_route in verify_graph already dispatches to milp_verify for any
    conv graph with no forks; this profile keeps that on and tunes the
    bab_refine knobs to match."""
    s.auto_route_milp_for_conv = True
    s.milp_alpha_tighten = True


def _profile_fc_shallow(s) -> None:
    """Shallow FC nets (mnist_fc 256x4 / 256x6): the production default
    is already `phase1_method='bab_refine'` (recovered +6 mnist_fc cases
    over legacy on the relusplitter benchmark). The fc_shallow profile
    is currently a no-op overlay because bab_refine is the new global
    default; left in place so future per-family tuning has a slot."""
    s.phase1_method = 'bab_refine'


def _profile_default(s) -> None:
    """No-op: keep current production defaults."""
    pass


_PROFILES = {
    'input_split_small': _profile_input_split_small,
    'conv_deep':         _profile_conv_deep,
    'fc_shallow':        _profile_fc_shallow,
    'default':           _profile_default,
}


def select_profile(fp: GraphFingerprint) -> str:
    """Map a fingerprint to a profile name. Pure function for testability."""
    if fp.input_dim <= 20:
        return 'input_split_small'
    if fp.has_conv and fp.fork_count == 0 and fp.n_relu >= 5:
        return 'conv_deep'
    if not fp.has_conv and fp.n_relu <= 6:
        return 'fc_shallow'
    return 'default'


def default_settings_for(graph, spec, **overrides: Any):
    """Profile-aware settings constructor.

    Returns a settings DotMap with the picked profile's overrides
    applied on top of `default_settings()`, then user overrides on top
    of that. Stores the picked profile name in `settings._profile` for
    diagnostics.
    """
    fp = GraphFingerprint.from_graph_and_spec(graph, spec)
    name = select_profile(fp)
    s = default_settings()
    _PROFILES[name](s)
    for k, v in overrides.items():
        s[k] = v
    s._profile = name
    s._fingerprint = fp
    return s
