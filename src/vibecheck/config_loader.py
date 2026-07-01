"""YAML config loading for per-benchmark settings overrides.

Schema: keys map 1:1 to `Settings` attrs. The YAML contains ONLY the
overrides on top of `default_settings()` (which itself is dumped to
`configs/default.yaml` for reference). Validation: every key must exist
in `default_settings()` — typos surface as KeyError at load time, not
silently ignored.
"""
import os
import yaml
from pathlib import Path

from .settings import default_settings

# Reserved meta keys allowed in a config YAML that are NOT `Settings` attrs: they
# document/annotate the config rather than override a knob, so they're stripped before
# key-validation and never reach `default_settings(**overrides)`.
#   description: one-sentence summary of the config's strategy, printed when it's used.
_RESERVED_META = frozenset({'description'})

# configs/ lives at the repo root (this file is src/vibecheck/config_loader.py).
_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs')


def config_path(name):
    """Absolute path to a bundled config by basename (e.g. 'acasxu_2023.yaml')."""
    return os.path.join(_CONFIGS_DIR, name)


def config_description(path):
    """The config's one-line `description:` meta field, or None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get('description') if isinstance(data, dict) else None


def load_config(path):
    """Load a YAML config file → dict suitable for `default_settings(**dict)`.

    Validates that every non-meta key exists in `default_settings()` so a typo
    (e.g. `pgd_resarts: 100`) raises immediately instead of being a silent extra
    DotMap key with no effect. Reserved meta keys (`_RESERVED_META`, e.g.
    `description`) are stripped and not treated as overrides.
    """
    p = Path(path)
    assert p.exists(), f'config not found: {path}'
    with open(p, encoding='utf-8') as f:
        overrides = yaml.safe_load(f) or {}
    assert isinstance(overrides, dict), (
        f'config must be a YAML mapping, got {type(overrides).__name__}')
    overrides = {k: v for k, v in overrides.items() if k not in _RESERVED_META}
    known = set(default_settings().keys())
    unknown = sorted(k for k in overrides if k not in known)
    assert not unknown, (
        f'unknown setting keys in {path}: {unknown}\n'
        f'(known keys: see configs/default.yaml)')
    return overrides


def parse_set_overrides(pairs):
    """Parse repeated ``--set KEY=VALUE`` CLI strings into a validated overrides dict.

    VALUE is YAML-coerced (so ``K=2`` -> int 2, ``ls=subgrad`` -> str, ``flag=true`` ->
    bool), consistent with how ``--config`` YAML values are parsed. Every KEY must exist
    in `default_settings()`, so a typo raises immediately instead of silently doing
    nothing. Returns {} for an empty/None list.
    """
    out = {}
    if not pairs:
        return out
    known = set(default_settings().keys())
    for item in pairs:
        assert '=' in item, f'--set expects KEY=VALUE, got {item!r}'
        key, raw = item.split('=', 1)
        key = key.strip()
        assert key in known, (
            f'unknown --set key {key!r} (known keys: see configs/default.yaml)')
        out[key] = yaml.safe_load(raw)
    return out
