"""Unit coverage for the `--set KEY=VALUE` override parser (config_loader.parse_set_overrides)
and the dump-dir settings it can target."""
import pytest

from vibecheck.config_loader import parse_set_overrides
from vibecheck.settings import default_settings


def test_empty_and_none_return_empty():
    assert parse_set_overrides([]) == {}
    assert parse_set_overrides(None) == {}


def test_yaml_coercion_int_str_bool():
    # value is YAML-coerced, like --config: int, str, bool
    out = parse_set_overrides([
        'phase8_fast_dual_ascent_K=2',
        'phase8_fast_dual_ascent_ls=subgrad',
        'phase8_fast_dual_ascent=false',
    ])
    assert out['phase8_fast_dual_ascent_K'] == 2 and isinstance(out['phase8_fast_dual_ascent_K'], int)
    assert out['phase8_fast_dual_ascent_ls'] == 'subgrad'
    assert out['phase8_fast_dual_ascent'] is False


def test_value_may_contain_equals_and_whitespace_key():
    # split on the FIRST '=' only (paths with no '=' are the common case; be safe anyway)
    out = parse_set_overrides([' dump_bnb_dir =/tmp/d=ump'])
    assert out['dump_bnb_dir'] == '/tmp/d=ump'


def test_unknown_key_raises():
    with pytest.raises(AssertionError, match='unknown --set key'):
        parse_set_overrides(['pgd_resarts=10'])   # typo


def test_missing_equals_raises():
    with pytest.raises(AssertionError, match='KEY=VALUE'):
        parse_set_overrides(['phase8_fast_dual_ascent_K'])


def test_dump_dir_settings_exist_and_default_empty():
    s = default_settings()
    assert s.dump_bnb_dir == '' and s.dump_da_bab_dir == ''
    # and they are real, --set-able keys
    out = parse_set_overrides(['dump_da_bab_dir=/tmp/da'])
    assert out['dump_da_bab_dir'] == '/tmp/da'
