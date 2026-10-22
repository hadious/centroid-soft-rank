"""Independent numerical checks for rank-only objectives; CPU tensors only."""
from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from losses.rank_only import (
    centroid_soft_rank,
    gram_defect,
    pair_rank,
    participation_rank,
    rank_only_loss,
)


METHODS = ("centroid_soft", "mmcr", "pair_rank", "soft_residual", "gram_defect")
NORMALIZED_METHODS = ("centroid_soft", "mmcr", "pair_rank")


def random_pair(batch=5, dimension=3, dtype=torch.float64, seed=1729):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(batch, dimension, dtype=dtype, generator=rng)
    zp = torch.randn(batch, dimension, dtype=dtype, generator=rng)
    return z, zp


def assert_finite_backward(z, zp, method, **kwargs):
    z = z.clone().detach().requires_grad_(True)
    zp = zp.clone().detach().requires_grad_(True)
    loss, components = rank_only_loss(z, zp, method, **kwargs)
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))
    assert components and all(value.ndim == 0 and bool(torch.isfinite(value)) for value in components.values())
    gradients = torch.autograd.grad(loss, (z, zp))
    for original, gradient in zip((z, zp), gradients):
        assert gradient.shape == original.shape
        assert bool(torch.isfinite(gradient).all()), f"Nonfinite {method} gradient"
    return loss, components, gradients


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_random_inputs_have_finite_loss_components_and_gradients(method, dtype):
    z, zp = random_pair(dtype=dtype)
    _, _, gradients = assert_finite_backward(z, zp, method)
    assert sum(float(gradient.square().sum()) for gradient in gradients) > 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ("zero", "constant", "near_zero"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_constant_and_nearzero_inputs_remain_finite(method, kind, dtype):
    if kind == "zero":
        z = zp = torch.zeros(4, 3, dtype=dtype)
    elif kind == "constant":
        # Separate constant views exercise both covariance/rank degeneracy and
        # a nonzero residual without assuming identical augmentation outputs.
        z = torch.tensor([0.2, -0.7, 1.1], dtype=dtype).expand(4, -1).clone()
        zp = torch.tensor([-0.4, 0.1, 0.8], dtype=dtype).expand(4, -1).clone()
    else:
        z, zp = random_pair(4, 3, dtype=dtype)
        z, zp = z * 1e-20, zp * 1e-20
    assert_finite_backward(z, zp, method)


@pytest.mark.parametrize("method", METHODS)
def test_antipodal_views_have_finite_backward_at_zero_centroid(method):
    z, _ = random_pair()
    assert_finite_backward(z, -z, method)


@pytest.mark.parametrize("method", ("centroid_soft", "pair_rank", "soft_residual", "gram_defect"))
def test_double_precision_gradcheck(method):
    z, zp = random_pair(4, 2, seed=19)
    z.requires_grad_(True)
    zp.requires_grad_(True)
    kwargs = {"kappa": 0.7, "soft_rank_tau": 0.8, "coverage_target": 0.9}
    assert torch.autograd.gradcheck(
        lambda a, b: rank_only_loss(a, b, method, **kwargs)[0],
        (z, zp), eps=1e-6, atol=1e-5, rtol=1e-3,
        check_undefined_grad=True, fast_mode=False,
    )


@pytest.mark.parametrize("shape", ((3, 7), (7, 3)))
def test_csr_gradcheck_below_and_above_feature_dimension(shape):
    z, zp = random_pair(*shape, seed=104)
    z.requires_grad_(True)
    zp.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b: centroid_soft_rank(a, b, kappa=1.3)[0],
        (z, zp), eps=1e-6, atol=1e-5, rtol=1e-3,
    )


@pytest.mark.parametrize("shape", ((2, 5), (5, 2), (4, 4)))
@pytest.mark.parametrize("kappa", (0.01, 0.1, 1.0, 10.0))
def test_csr_score_is_bounded_by_one(shape, kappa):
    for seed in range(5):
        z, zp = random_pair(*shape, seed=seed)
        loss, parts = centroid_soft_rank(z, zp, kappa=kappa)
        score = float(parts["centroid_rank_score"])
        assert -1e-12 <= score <= 1 + 1e-12
        assert -1e-12 <= float(loss) <= 1 + 1e-12
        assert float(parts["centroid_energy"]) <= 1 + 1e-12


def unit_tight_frame(batch, dimension):
    if batch <= dimension:
        return torch.eye(dimension, dtype=torch.float64)[:batch]
    assert batch % dimension == 0
    return torch.eye(dimension, dtype=torch.float64).repeat(batch // dimension, 1)


@pytest.mark.parametrize("shape", ((3, 7), (4, 4), (8, 4)))
@pytest.mark.parametrize("kappa", (0.01, 1.0, 10.0))
def test_csr_upper_bound_attained_by_aligned_unit_tight_frames(shape, kappa):
    u = unit_tight_frame(*shape)
    # Verify the fixture independently: unit rows and equal nonzero singular
    # values; this covers undercomplete and overcomplete frames.
    torch.testing.assert_close(u.square().sum(1), torch.ones(shape[0], dtype=u.dtype), atol=1e-12, rtol=1e-12)
    singular_values = torch.linalg.svdvals(u)
    target = math.sqrt(shape[0] / min(shape))
    torch.testing.assert_close(singular_values, torch.full_like(singular_values, target), atol=1e-12, rtol=1e-12)
    u.requires_grad_(True)
    loss, components = centroid_soft_rank(u, u, kappa=kappa)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-12, rtol=0)
    torch.testing.assert_close(components["centroid_rank_score"], torch.ones_like(loss), atol=1e-12, rtol=0)
    gradient, = torch.autograd.grad(loss, u)
    torch.testing.assert_close(gradient, torch.zeros_like(u), atol=1e-12, rtol=0)


@pytest.mark.parametrize("method,score_name", (("centroid_soft", "centroid_rank_score"), ("mmcr", "centroid_nuclear_score")))
def test_shuffled_and_antipodal_pairs_score_below_correct_alignment(method, score_name):
    u = torch.eye(5, dtype=torch.float64)
    correct = rank_only_loss(u, u, method)[1][score_name]
    shuffled = rank_only_loss(u, u.roll(1, dims=0), method)[1][score_name]
    antipodal = rank_only_loss(u, -u, method)[1][score_name]
    assert float(correct) > float(shuffled) > float(antipodal)
    torch.testing.assert_close(antipodal, torch.zeros_like(antipodal), atol=0, rtol=0)


def test_anchored_pair_rank_distinguishes_shuffled_and_antipodal_views():
    u = torch.eye(5, dtype=torch.float64)
    aligned = pair_rank(u, u)[1]["pair_rank_excess"]
    shuffled = pair_rank(u, u.roll(1, dims=0))[1]["pair_rank_excess"]
    antipodal = pair_rank(u, -u)[1]["pair_rank_excess"]
    torch.testing.assert_close(aligned, torch.tensor(0.0, dtype=u.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(shuffled, torch.tensor(0.6, dtype=u.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(antipodal, torch.tensor(1.0, dtype=u.dtype), atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", NORMALIZED_METHODS)
def test_positive_raw_row_scaling_is_invariant_above_normalization_floor(method):
    z, zp = random_pair(5, 3)
    scales_z = torch.tensor([1e-3, 0.1, 1.0, 10.0, 1e3], dtype=z.dtype)[:, None]
    scales_zp = scales_z.flip(0) * 2
    assert float((z * scales_z).norm(dim=1).min()) > 1e-6
    assert float((zp * scales_zp).norm(dim=1).min()) > 1e-6
    loss, parts = rank_only_loss(z, zp, method)
    scaled_loss, scaled_parts = rank_only_loss(z * scales_z, zp * scales_zp, method)
    torch.testing.assert_close(scaled_loss, loss, atol=1e-10, rtol=1e-10)
    for name in parts:
        torch.testing.assert_close(scaled_parts[name], parts[name], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("kind", ("random", "zero", "one_zero_view", "below_normalization_floor"))
def test_anchor_pair_formula_matches_actual_two_row_participation_rank(kind):
    z, zp = random_pair(5, 3)
    if kind == "zero":
        z.zero_()
        zp.zero_()
    elif kind == "one_zero_view":
        z.zero_()
    elif kind == "below_normalization_floor":
        z *= 1e-9
        zp *= 3e-9
    u, v = F.normalize(z, dim=1, eps=1e-6), F.normalize(zp, dim=1, eps=1e-6)
    anchor = torch.ones(len(z), 1, dtype=z.dtype)
    matrices = torch.stack((torch.cat((u, anchor), dim=1), torch.cat((v, anchor), dim=1)), dim=1) / math.sqrt(2)
    # Independent spectral definition, without using the implementation's
    # Gram formula or participation_rank helper.
    energies = torch.linalg.svdvals(matrices).square()
    actual_ranks = energies.sum(1).square() / energies.square().sum(1)
    formula_mean = pair_rank(z, zp)[1]["pair_rank_excess"] + 1
    torch.testing.assert_close(formula_mean, actual_ranks.mean(), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", ((3, 3), (6, 3)))
def test_gram_defect_has_zero_global_minimum_at_aligned_identity_grams(shape):
    batch, dimension = shape
    rng = torch.Generator().manual_seed(71)
    q, _ = torch.linalg.qr(torch.randn(batch, dimension, dtype=torch.float64, generator=rng), mode="reduced")
    z = math.sqrt(batch) * q
    torch.testing.assert_close(z.T @ z / batch, torch.eye(dimension, dtype=z.dtype), atol=1e-12, rtol=1e-12)
    loss, components, gradients = assert_finite_backward(z, z, "gram_defect")
    assert float(loss.detach()) >= 0
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-24, rtol=0)
    for value in components.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-24, rtol=0)
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    # Identity marginal Grams alone are insufficient: antipodal views violate
    # the cross-Gram identity while leaving the two marginal Grams unchanged.
    antipodal_loss, antipodal_parts = gram_defect(z, -z)
    assert float(antipodal_loss) > 0.1
    torch.testing.assert_close(antipodal_parts["view_gram_defect_rank"], torch.zeros_like(loss), atol=1e-24, rtol=0)


def test_gram_identity_rejects_infeasible_dimension():
    z, zp = random_pair(3, 7)
    with pytest.raises(ValueError, match="infeasible"):
        rank_only_loss(z, zp, "gram_defect")


@pytest.mark.parametrize("shape", ((3, 7), (7, 3)))
def test_stabilized_participation_rank_zero_backward(shape):
    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    value = participation_rank(x)
    gradient, = torch.autograd.grad(value, x)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_inputs_raise_explicitly(method):
    z, zp = random_pair()
    z[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        rank_only_loss(z, zp, method)
