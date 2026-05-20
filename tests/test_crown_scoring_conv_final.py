"""Unit test for the Conv-final-layer branch of `_compute_crown_layer_weights`.

This is the path that lost 15 cifar_biasfield cases to
`NotImplementedError` until 2026-05-10. The test builds a tiny
Conv-final network (one hidden Conv → ReLU → Conv (out_spatial=1)) and
checks the `ew_at_layer` returned matches what an equivalent FC-final
network would produce (since a Conv with 1×1 output spatial is exactly
an FC layer when its kernel covers the full input spatial).
"""
import numpy as np
import torch

from vibecheck.verify_milp import _compute_crown_layer_weights


def test_compute_crown_layer_weights_conv_final_matches_fc_equivalent():
    """Build (Conv → ReLU → Conv-with-1x1-output) and check the spec-direction
    backward through the Conv-final layer matches the FC equivalent's."""
    # Hidden Conv: in (4, 3, 3) → out (8, 3, 3) (kernel 1×1, padding 0)
    # Final Conv: in (8, 3, 3) → out (10, 1, 1) (kernel 3×3, no padding)
    #             — kernel 3×3 over 3×3 input with 0 padding → output spatial 1×1
    in_shape_h = (4, 3, 3)
    out_shape_h = (8, 3, 3)
    in_shape_f = (8, 3, 3)

    rng = np.random.default_rng(0)
    k_hidden = rng.standard_normal((8, 4, 1, 1))  # shape (out_c, in_c, kH, kW)
    b_hidden = rng.standard_normal(8)
    k_final = rng.standard_normal((10, 8, 3, 3))
    b_final = rng.standard_normal(10)

    layers_conv = [
        {'type': 'conv', 'kernel': k_hidden, 'bias': b_hidden,
         'in_shape': in_shape_h, 'stride': (1, 1), 'padding': (0, 0)},
        {'type': 'conv', 'kernel': k_final, 'bias': b_final,
         'in_shape': in_shape_f, 'stride': (1, 1), 'padding': (0, 0)},
    ]
    nh = 1
    n_hidden_neurons = 8 * 3 * 3  # 72
    bounds_np = {
        0: (np.full(n_hidden_neurons, -1.0), np.full(n_hidden_neurons, 1.0)),
    }

    pred, comp = 3, 7
    out = _compute_crown_layer_weights(bounds_np, layers_conv, None,
                                        pred, comp, nh)

    # Equivalent FC: flatten the final Conv's kernel into a (10, 72) matrix
    # via the same channel-major flattening _compute_crown_layer_weights uses.
    # For each output class c, output[c] = sum over (in_c, kh, kw) of
    # k[c, in_c, kh, kw] * input[in_c, kh, kw]; with input flat ordering
    # in_c*9 + kh*3 + kw, the row vector is k[c].reshape(8, 9).flatten()
    # after a transpose-free reshape.
    W_fc = k_final.reshape(10, 8 * 3 * 3)
    layers_fc = [
        {'type': 'conv', 'kernel': k_hidden, 'bias': b_hidden,
         'in_shape': in_shape_h, 'stride': (1, 1), 'padding': (0, 0)},
        {'type': 'fc', 'W': W_fc, 'bias': b_final},
    ]
    out_fc = _compute_crown_layer_weights(bounds_np, layers_fc, None,
                                          pred, comp, nh)

    # The hidden-layer ew before slope application must match.
    assert 0 in out and 0 in out_fc
    assert out[0].shape == out_fc[0].shape
    assert np.allclose(out[0], out_fc[0], atol=1e-10), (
        f'Conv-final ew mismatch vs FC-final: '
        f'max |Δ| = {np.max(np.abs(out[0] - out_fc[0])):.3e}')


def test_compute_crown_layer_weights_conv_final_returns_finite_array():
    """Smoke: the Conv-final branch returns a finite numpy array of the
    right size for the previous (hidden) layer's flat dimension."""
    in_shape_h = (2, 4, 4)
    in_shape_f = (3, 4, 4)
    k_hidden = np.ones((3, 2, 1, 1)) * 0.1
    b_hidden = np.zeros(3)
    k_final = np.ones((5, 3, 4, 4)) * 0.05
    b_final = np.array([0.1, -0.2, 0.3, -0.4, 0.5])

    layers = [
        {'type': 'conv', 'kernel': k_hidden, 'bias': b_hidden,
         'in_shape': in_shape_h, 'stride': (1, 1), 'padding': (0, 0)},
        {'type': 'conv', 'kernel': k_final, 'bias': b_final,
         'in_shape': in_shape_f, 'stride': (1, 1), 'padding': (0, 0)},
    ]
    nh = 1
    n_hidden = 3 * 4 * 4
    bounds = {0: (np.full(n_hidden, -1.0), np.full(n_hidden, 1.0))}

    out = _compute_crown_layer_weights(bounds, layers, None, 0, 1, nh)
    assert 0 in out
    assert out[0].shape == (n_hidden,)
    assert np.all(np.isfinite(out[0]))
