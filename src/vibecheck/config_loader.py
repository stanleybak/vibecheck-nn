"""YAML config loading for per-benchmark settings overrides.

Schema: keys map 1:1 to `Settings` attrs. The YAML contains ONLY the
overrides on top of `default_settings()` (which itself is dumped to
`configs/default.yaml` for reference). Validation: every key must exist
in `default_settings()` — typos surface as KeyError at load time, not
silently ignored.
"""
import yaml
from pathlib import Path

from .settings import default_settings


def load_config(path):
    """Load a YAML config file → dict suitable for `default_settings(**dict)`.

    Validates that every key exists in `default_settings()` so a typo
    (e.g. `pgd_resarts: 100`) raises immediately instead of being a
    silent extra DotMap key with no effect.
    """
    p = Path(path)
    assert p.exists(), f'config not found: {path}'
    with open(p) as f:
        overrides = yaml.safe_load(f) or {}
    assert isinstance(overrides, dict), (
        f'config must be a YAML mapping, got {type(overrides).__name__}')
    known = set(default_settings().keys())
    unknown = sorted(k for k in overrides if k not in known)
    assert not unknown, (
        f'unknown setting keys in {path}: {unknown}\n'
        f'(known keys: see configs/default.yaml)')
    return overrides
