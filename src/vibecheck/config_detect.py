"""Auto-config detection: interpretable structural rules -> configs/*.yaml.

When no ``--config`` is given, ``detect_config`` picks the closest existing category
config from a small, interpretable fingerprint of the network + spec (input dim, params,
conv/transformer/nonlinear op presence, network-pair kind) -- NOT the specific network.
First matching rule wins; the rule string is logged so a run says which yaml it chose and
why. Design + expected-routing table: scratch/autotune_research/DESIGN.md.
"""
from __future__ import annotations

import gzip
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import onnx

# transcendental ops beyond ReLU that mark a "smooth nonlinear" family (ml4acopf).
# Sigmoid/Tanh are deliberately EXCLUDED -- they appear in otherwise-standard ReLU nets
# (dist_shift, cgan) and are handled by forward-LiRPA internally, not by a special config.
_SMOOTH_NONLINEAR = frozenset({'Sin', 'Cos', 'Pow', 'Exp', 'Log', 'Erf'})

# A multiply whose BOTH operands are variable refs, e.g. `(* X[0,1] X[0,1])`, marks a
# genuinely nonlinear CONSTRAINT (adaptive_cruise). Classification/reachability specs are
# linear in X,Y (0 matches), so this doesn't false-fire on the ReLU-net families.
_SPEC_NONLINEAR_RE = re.compile(r'\(\s*\*\s+[XY][\w\[\],.]*\s+[XY]')


def _load_onnx(path):
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        path = path + '.gz'
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return onnx.load_from_string(f.read())
    return onnx.load(path)


@dataclass(frozen=True)
class DetectFingerprint:
    """Interpretable features driving ``detect_config`` (see module docstring)."""
    is_pair: bool
    pair_kind: Optional[str]      # 'iso' | 'mono' | None
    has_conv: bool
    is_transformer: bool
    smooth_nonlinear: bool        # nonlinear OPS in the net (sin/cos/pow/...)
    spec_nonlinear: bool          # nonlinear CONSTRAINT in the spec (var*var product)
    params: int
    in_dim: int
    n_relu: int

    @classmethod
    def from_onnx_and_spec(cls, onnx_path, spec_nonlinear=False,
                           is_pair=False, pair_kind=None):
        m = _load_onnx(onnx_path)
        g = m.graph
        ops = Counter(n.op_type for n in g.node)
        params = 0
        for init in g.initializer:
            p = 1
            for d in init.dims:
                p *= d
            params += p
        inits = {i.name for i in g.initializer}
        ins = [vi for vi in g.input if vi.name not in inits]
        in_dim = 0
        if ins:
            in_dim = 1
            for d in ins[0].type.tensor_type.shape.dim:
                v = d.dim_value
                in_dim *= v if v and v > 0 else 1
        has_conv = ('Conv' in ops) or ('ConvTranspose' in ops)
        is_transformer = ('Attention' in ops or 'LayerNormalization' in ops
                          or ('Softmax' in ops and ops.get('MatMul', 0) >= 2))
        smooth = bool(set(ops) & _SMOOTH_NONLINEAR)
        return cls(is_pair=is_pair, pair_kind=pair_kind, has_conv=has_conv,
                   is_transformer=is_transformer, smooth_nonlinear=smooth,
                   spec_nonlinear=spec_nonlinear,
                   params=int(params), in_dim=int(in_dim), n_relu=ops.get('Relu', 0))


def detect_config(fp: DetectFingerprint):
    """Map a fingerprint to ``(yaml_basename, rule_description)``. Pure + ordered.

    5 decision features (pair, nonlinear, in_dim, transformer, conv); `params`/`n_relu`
    are diagnostics only. Size splits key off `in_dim` alone (a net that needs the "huge"
    leaf is always huge by input; params adds no separation). The low-input-dim test is a
    single node BEFORE the transformer check, so a tiny-input attention net (the cGAN
    transformer decoder, in=5) routes to its input-split family, not a transformer leaf.
    """
    if fp.is_pair and fp.pair_kind == 'mono':
        return 'monotonic_acasxu_2026.yaml', '1: network-pair (equal-to / monotone)'
    if fp.is_pair and fp.pair_kind == 'iso':
        return 'isomorphic_acasxu_2026.yaml', '2: network-pair (isomorphic)'
    if fp.spec_nonlinear:
        # nonlinear CONSTRAINT (var*var in the spec) -> needs the augment path
        # (nonlinear_v2_augment); ml4acopf's config does NOT handle it.
        return ('adaptive_cruise_control_non_linear_2026.yaml',
                '3: nonlinear spec constraint (var*var) -> nonlinear-v2 augment')
    if fp.smooth_nonlinear and not fp.has_conv:
        # nonlinear network OPS (sin/cos/pow/exp/log/erf) baked into the net.
        return ('ml4acopf_2024.yaml',
                '4: nonlinear net ops (sin/cos/pow/exp/log/erf), no conv')
    if fp.in_dim <= 20:
        if fp.has_conv:
            return 'cgan2026.yaml', '5: low input-dim (<=20) conv (tiny-latent generator)'
        return 'acasxu_2023.yaml', '5: low input-dim (<=20) FC (input-split)'
    if fp.is_transformer:
        if fp.in_dim > 2e4:
            return 'smart_turn_multimodal_2026.yaml', '6: transformer, huge input (in_dim>2e4)'
        return 'vit_2023.yaml', '6: transformer (attention/softmax)'
    if fp.has_conv:
        if fp.in_dim > 1e5:
            return 'vggnet16_2022.yaml', '7: very large conv (in_dim>1e5)'
        return 'cifar100_2024.yaml', '7: conv image classification'
    return 'cora_2024.yaml', '8: FC, mid/high input-dim (input-split)'


def detect_from_field(net_field, spec_path, base_dir=None):
    """End-to-end: resolve the net + spec, build a fingerprint, return
    ``(fingerprint, yaml_basename, rule_description)``.

    Handles the network-pair ``--net`` list form (short-circuits on the spec's
    isomorphic-to/equal-to keyword without loading onnx) and single-net paths.
    """
    from . import network_pair as npair
    spec_text = npair._read_vnnlib_text(spec_path)
    spec_nl = bool(_SPEC_NONLINEAR_RE.search(spec_text))
    if npair.is_network_pair_net_field(net_field):
        kind = npair.detect_kind(spec_text)
        fp = DetectFingerprint(is_pair=True, pair_kind=kind, has_conv=False,
                               is_transformer=False, smooth_nonlinear=False,
                               spec_nonlinear=spec_nl, params=0, in_dim=0, n_relu=0)
        name, rule = detect_config(fp)
        return fp, name, rule
    # single net: net_field is a path (possibly relative to base_dir / CWD)
    onnx_path = net_field
    if base_dir and not os.path.isabs(onnx_path) and not os.path.exists(onnx_path):
        onnx_path = os.path.join(base_dir, net_field)
    fp = DetectFingerprint.from_onnx_and_spec(onnx_path, spec_nonlinear=spec_nl)
    name, rule = detect_config(fp)
    return fp, name, rule
