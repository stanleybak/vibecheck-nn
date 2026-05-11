"""Unit tests for src/vibecheck/config_profiles.py."""
from vibecheck.config_profiles import (
    GraphFingerprint, select_profile, default_settings_for,
)


def test_select_profile_small_input_picks_split():
    fp = GraphFingerprint(input_dim=16, has_conv=True, n_relu=8,
                           fork_count=0)
    assert select_profile(fp) == 'input_split_small'


def test_select_profile_deep_conv_no_forks_picks_conv_deep():
    fp = GraphFingerprint(input_dim=3072, has_conv=True, n_relu=12,
                           fork_count=0)
    assert select_profile(fp) == 'conv_deep'


def test_select_profile_fc_shallow_picks_fc_shallow():
    fp = GraphFingerprint(input_dim=784, has_conv=False, n_relu=4,
                           fork_count=0)
    assert select_profile(fp) == 'fc_shallow'


def test_select_profile_falls_through_to_default():
    # FC net with many ReLUs (not "shallow")
    fp = GraphFingerprint(input_dim=784, has_conv=False, n_relu=10,
                           fork_count=0)
    assert select_profile(fp) == 'default'
    # Conv net with forks (not "deep no-fork")
    fp2 = GraphFingerprint(input_dim=3072, has_conv=True, n_relu=12,
                            fork_count=2)
    assert select_profile(fp2) == 'default'


def test_default_settings_for_fc_shallow_picks_bab_refine():
    """The fc_shallow profile fixes phase1_method to bab_refine. Since
    bab_refine is now the global default, the profile's effective
    behavior matches the default — this test guards against a future
    silent default change leaving fc_shallow misaligned."""
    class _G:
        nodes = {}

        def relu_nodes(self):
            return list(range(4))

        def fork_points(self):
            return []

        input_shape = (784,)

    class _S:
        x_lo = type('A', (), {'shape': (784,)})()

    s = default_settings_for(_G(), _S())
    assert s._profile == 'fc_shallow'
    assert s.phase1_method == 'bab_refine'


def test_default_settings_for_overrides_win():
    """User-supplied overrides take precedence over the profile's."""
    class _G:
        nodes = {}

        def relu_nodes(self):
            return list(range(4))

        def fork_points(self):
            return []

        input_shape = (784,)

    class _S:
        x_lo = type('A', (), {'shape': (784,)})()

    s = default_settings_for(_G(), _S(),
                              phase1_method='legacy',
                              bab_refine_passes=5)
    assert s.phase1_method == 'legacy'  # user override beats fc_shallow profile
    assert s.bab_refine_passes == 5


def test_fingerprint_from_graph_and_spec_uses_spec_xlo():
    """Fingerprint's input_dim comes from spec.x_lo.shape, not graph."""
    import numpy as np

    class _G:
        nodes = {}

        def relu_nodes(self):
            return [1]

        def fork_points(self):
            return []

        input_shape = (3072,)

    class _S:
        x_lo = np.zeros((1, 16))

    fp = GraphFingerprint.from_graph_and_spec(_G(), _S())
    assert fp.input_dim == 16
