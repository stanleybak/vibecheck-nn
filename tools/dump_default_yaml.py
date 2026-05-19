"""Regenerate `configs/default.yaml` from `default_settings()`.

Run: `.venv/bin/python tools/dump_default_yaml.py > configs/default.yaml`
"""
import yaml
from vibecheck.settings import default_settings


def to_yaml(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, 'items') and not isinstance(v, (str, list, tuple)):
            out[k] = to_yaml(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif v is None or isinstance(v, (str, int, float, bool, list)):
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def main():
    s = default_settings()
    print('# Generated from default_settings() in src/vibecheck/settings.py')
    print('# Per-benchmark YAMLs in this dir only need to list OVERRIDES.')
    print('# Regenerate with: '
          '.venv/bin/python tools/dump_default_yaml.py > configs/default.yaml')
    print(yaml.safe_dump(to_yaml(s), sort_keys=True,
                         default_flow_style=False), end='')


if __name__ == '__main__':
    main()
