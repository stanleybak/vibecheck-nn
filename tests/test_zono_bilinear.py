"""Soundness of the both-vary zonotope bilinear product (_torch_zono_mul_bilinear).

Two zonotopes that fork from a common ancestor each APPEND their own error
generators at the same column positions, so beyond the shared prefix their
columns are DIFFERENT noise symbols. The product must treat those extras as
independent (not align them positionally, which conflates two noises into one
column and can UNDER-approximate -> unsound). This reproduces the ml4acopf
full-model wide-box unsoundness and pins the fix.
"""
import torch

from vibecheck.zonotope import _torch_zono_mul_bilinear


def _true_product_range(c_a, G_a, c_b, G_b, shared, n=200000):
    """Sample the TRUE elementwise product where the first `shared` noise
    symbols are shared between a and b and the rest are independent."""
    ka, kb = G_a.shape[1], G_b.shape[1]
    ea_ind, eb_ind = ka - shared, kb - shared
    es = 2 * torch.rand(n, shared, dtype=torch.float64) - 1
    ea = 2 * torch.rand(n, ea_ind, dtype=torch.float64) - 1
    eb = 2 * torch.rand(n, eb_ind, dtype=torch.float64) - 1
    e_a = torch.cat([es, ea], 1)          # a uses shared + its own extras
    e_b = torch.cat([es, eb], 1)          # b uses shared + its OWN (distinct) extras
    a = c_a.unsqueeze(0) + e_a @ G_a.t()
    b = c_b.unsqueeze(0) + e_b @ G_b.t()
    return (a * b)


def test_mul_bilinear_compact_misaligned_is_sound():
    torch.manual_seed(0)
    # shared prefix of 2 cols; a has 1 extra, b has 1 extra (at the SAME
    # column position 2 -> different noise). Coeffs chosen so positional
    # alignment cancels (a_extra*c_b vs c_a*b_extra opposite sign).
    c_a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    c_b = torch.tensor([3.0, -1.0], dtype=torch.float64)
    G_a = torch.tensor([[0.5, 0.1, 0.7], [0.2, 0.3, -0.4]], dtype=torch.float64)
    G_b = torch.tensor([[0.4, 0.2, -0.6], [0.1, 0.5, 0.3]], dtype=torch.float64)
    shared = 2

    c_out, G_out = _torch_zono_mul_bilinear(
        c_a, G_a, c_b, G_b, shared_gens=shared)
    olo = c_out - G_out.abs().sum(1)
    ohi = c_out + G_out.abs().sum(1)

    prod = _true_product_range(c_a, G_a, c_b, G_b, shared)
    worst = float(torch.maximum(olo.unsqueeze(0) - prod,
                                prod - ohi.unsqueeze(0)).max())
    assert worst <= 1e-9, f'bilinear product UNSOUND: overshoot {worst:.3e}'


def test_mul_bilinear_linear_extras_sound():
    # Branch-specific extras are kept LINEAR (not boxed) in the new layout;
    # this must stay sound (col-ID tracking keeps them independent downstream).
    torch.manual_seed(3)
    c_a = torch.randn(4, dtype=torch.float64)
    c_b = torch.randn(4, dtype=torch.float64)
    G_a = 0.3 * torch.randn(4, 6, dtype=torch.float64)   # 2 shared + 4 extra
    G_b = 0.3 * torch.randn(4, 5, dtype=torch.float64)   # 2 shared + 3 extra
    shared = 2
    c_out, G_out = _torch_zono_mul_bilinear(c_a, G_a, c_b, G_b,
                                            shared_gens=shared)
    olo = c_out - G_out.abs().sum(1)
    ohi = c_out + G_out.abs().sum(1)
    prod = _true_product_range(c_a, G_a, c_b, G_b, shared)
    worst = float(torch.maximum(olo.unsqueeze(0) - prod,
                                prod - ohi.unsqueeze(0)).max())
    assert worst <= 1e-9, f'linear-extras bilinear UNSOUND: {worst:.3e}'


def test_mul_bilinear_diagonal_tighter_than_box():
    # Fully-shared product: the e_k^2 in [0,1] diagonal tightening must give a
    # strictly NARROWER (or equal) width than the symmetric radA*radB box,
    # while staying sound.
    torch.manual_seed(4)
    c_a = torch.randn(3, dtype=torch.float64)
    c_b = torch.randn(3, dtype=torch.float64)
    G_a = 0.5 * torch.randn(3, 4, dtype=torch.float64)
    G_b = 0.5 * torch.randn(3, 4, dtype=torch.float64)
    c_out, G_out = _torch_zono_mul_bilinear(c_a, G_a, c_b, G_b, shared_gens=4)
    width = G_out.abs().sum(1)
    # symmetric-box width: first-order linear gens + radA*radB box
    radA = G_a.abs().sum(1); radB = G_b.abs().sum(1)
    lin = (G_a * c_b.unsqueeze(1) + G_b * c_a.unsqueeze(1)).abs().sum(1)
    box_width = lin + radA * radB
    assert bool((width <= box_width + 1e-9).all()), 'diagonal not tighter'
    # and still sound
    olo = c_out - width; ohi = c_out + width
    prod = _true_product_range(c_a, G_a, c_b, G_b, 4)
    worst = float(torch.maximum(olo.unsqueeze(0) - prod,
                                prod - ohi.unsqueeze(0)).max())
    assert worst <= 1e-9, f'diagonal-tightened bilinear UNSOUND: {worst:.3e}'


def test_mul_bilinear_fully_shared_still_sound():
    # All columns shared (aligned) -> the classic case must remain sound.
    torch.manual_seed(1)
    c_a = torch.randn(3, dtype=torch.float64)
    c_b = torch.randn(3, dtype=torch.float64)
    G_a = 0.3 * torch.randn(3, 4, dtype=torch.float64)
    G_b = 0.3 * torch.randn(3, 4, dtype=torch.float64)
    c_out, G_out = _torch_zono_mul_bilinear(c_a, G_a, c_b, G_b, shared_gens=4)
    olo = c_out - G_out.abs().sum(1)
    ohi = c_out + G_out.abs().sum(1)
    prod = _true_product_range(c_a, G_a, c_b, G_b, 4)
    worst = float(torch.maximum(olo.unsqueeze(0) - prod,
                                prod - ohi.unsqueeze(0)).max())
    assert worst <= 1e-9, f'fully-shared bilinear UNSOUND: {worst:.3e}'
