"""Rank-based SSL candidates; normalization/anchors are explicit conventions.

CSR is a scale-sensitive soft numerical rank, not algebraic/entropy rank.
Pair-PR is an actual two-row participation rank but is also a monotone function
of positive cosine; the rank wording does not imply a new alignment mechanism.
"""
from __future__ import annotations
import math
import warnings
import torch
from torch import Tensor
from torch.nn import functional as F

SVD_FALLBACK_COUNT = 0


def _svdvals(x: Tensor) -> Tensor:
    """Retry only solver nonconvergence on the identical matrix in CPU FP64.

    No detached values, jitter, or changed regularization. Retain recovered
    precision through the caller's spectral reduction. A second failure is
    propagated instead of silently changing the mathematical objective.
    """
    global SVD_FALLBACK_COUNT
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite SVD input")
    try:
        return torch.linalg.svdvals(x)
    except torch.linalg.LinAlgError:
        SVD_FALLBACK_COUNT += 1
        warnings.warn(f"SVD failed for {tuple(x.shape)} {x.dtype}; retrying unchanged matrix in CPU float64",
                      RuntimeWarning, stacklevel=2)
        return torch.linalg.svdvals(x.to(device="cpu", dtype=torch.float64)).to(device=x.device)


def validate_pair(z: Tensor, zp: Tensor):
    if z.ndim != 2 or min(z.shape) < 1 or z.shape[0] < 2:
        raise ValueError("Expected [B,d], B>=2, d>=1")
    if z.shape != zp.shape or z.device != zp.device or z.dtype != zp.dtype:
        raise ValueError("Both views must have matching shape/device/dtype")
    if not z.is_floating_point():
        raise TypeError("Floating representations required")
    if not torch.isfinite(z).all() or not torch.isfinite(zp).all():
        raise FloatingPointError("Nonfinite representation")
    return (z, zp) if z.dtype in (torch.float32, torch.float64) else (z.float(), zp.float())


def soft_rank(x: Tensor, tau: float) -> Tensor:
    """sum s²/(s²+tau), with tau in squared-singular-value units."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    s2 = _svdvals(x).square()
    return (s2 / (s2 + tau)).sum().to(dtype=x.dtype)


def participation_rank(x: Tensor) -> Tensor:
    """Stabilized energy participation rank; 1e-12 protects zero backward."""
    g = x.T @ x if x.shape[1] <= x.shape[0] else x @ x.T
    energy = x.square().sum()
    return energy.square() / (g.square().sum() + 1e-12)


def centroid_soft_rank(z: Tensor, zp: Tensor, kappa=1., normalize_epsilon=1e-6, normalize_centroid=False, shuffle_pairs=False):
    u, v = F.normalize(z, dim=1, eps=normalize_epsilon), F.normalize(zp, dim=1, eps=normalize_epsilon)
    if shuffle_pairs:
        # Random cycle: every second view is paired with a different image.
        # Use a local generator so the control's augmentation RNG is unchanged.
        generator = torch.Generator(device=v.device)
        generator.manual_seed(torch.initial_seed() + centroid_soft_rank.shuffle_step)
        centroid_soft_rank.shuffle_step += 1
        order = torch.randperm(v.shape[0], device=v.device, generator=generator)
        permutation = torch.empty_like(order)
        permutation[order] = order.roll(1)
        v = v[permutation]
    c = (u + v) / 2
    if normalize_centroid:
        c = F.normalize(c, dim=1, eps=normalize_epsilon)
    r = min(c.shape)
    rank = soft_rank(c / math.sqrt(c.shape[0]), kappa / r)
    score = (1 + kappa) * rank / r
    return 1 - score, {"centroid_soft_rank": rank, "centroid_rank_score": score,
                       "centroid_energy": c.square().sum() / c.shape[0]}


centroid_soft_rank.shuffle_step = 0

def mmcr(z: Tensor, zp: Tensor, normalize_epsilon=1e-6):
    """Normalized two-view MMCR baseline; affine rescaling of nuclear objective."""
    u, v = F.normalize(z, dim=1, eps=normalize_epsilon), F.normalize(zp, dim=1, eps=normalize_epsilon)
    c = (u + v) / 2
    nuclear = _svdvals(c).sum().to(dtype=c.dtype)
    # Select zero subgradient at exact zero; no ambiguity affects nonzero runs.
    nuclear = torch.where(c.square().sum() > 0, nuclear, nuclear * 0)
    score = nuclear / math.sqrt(c.shape[0] * min(c.shape))
    return 1 - score, {"centroid_nuclear_score": score,
                       "centroid_energy": c.square().sum() / c.shape[0]}


def pair_rank(z: Tensor, zp: Tensor, pair_weight=8., diversity_weight=1., normalize_epsilon=1e-6):
    u, v = F.normalize(z, dim=1, eps=normalize_epsilon), F.normalize(zp, dim=1, eps=normalize_epsilon)
    # Actual rank of [[u,1],[v,1]]/sqrt(2). The generic formula also covers
    # normalization-floor rows with norm<1, including exact zero inputs.
    a = (u.square().sum(1) + 1) / 2
    b = (v.square().sum(1) + 1) / 2
    c = ((u*v).sum(1) + 1) / 2
    pair_pr = (a+b).square() / (a.square()+b.square()+2*c.square())
    pair = (pair_pr - 1).mean()
    r = min(u.shape)
    rank_u, rank_v = participation_rank(u), participation_rank(v)
    # Uncentered: centering followed by scale-invariant rank would hide
    # arbitrarily small angular noise around a common unit vector.
    diversity = -.5 * ((rank_u/r).clamp_min(1e-12).log() + (rank_v/r).clamp_min(1e-12).log())
    return pair_weight*pair + diversity_weight*diversity, {
        "pair_rank_excess": pair, "global_rank_penalty": diversity,
        "view_rank_mean": (rank_u+rank_v)/2}


def residual_coverage(z: Tensor, zp: Tensor, soft_rank_tau=1., coverage_target=.5, coverage_weight=4.):
    b, d = z.shape
    x = (z-z.mean(0)) / math.sqrt(b-1)
    y = (zp-zp.mean(0)) / math.sqrt(b-1)
    r = min(b-1, d)
    sx, sy = soft_rank(x, soft_rank_tau)/r, soft_rank(y, soft_rank_tau)/r
    residual = soft_rank((z-zp)/math.sqrt(b), soft_rank_tau) / min(b,d)
    coverage = F.relu(coverage_target-sx) + F.relu(coverage_target-sy)
    return residual + coverage_weight*coverage, {"residual_soft_rank": residual,
        "coverage_hinge": coverage, "view_coverage_mean": (sx+sy)/2}


def gram_defect(z: Tensor, zp: Tensor, soft_rank_tau=1.):
    b, d = z.shape
    if d > b:
        raise ValueError("Uncentered Gram identity is infeasible for d>B; use d<=B")
    x, y = z/math.sqrt(b), zp/math.sqrt(b)
    identity = torch.eye(d, device=z.device, dtype=z.dtype)
    gx, gy, k = x.T@x, y.T@y, x.T@y
    a = soft_rank(identity-gx, soft_rank_tau**2)/d
    ap = soft_rank(identity-gy, soft_rank_tau**2)/d
    cross = soft_rank(identity-k, soft_rank_tau**2)/d
    return .5*(a+ap)+cross, {"view_gram_defect_rank": .5*(a+ap),
                           "cross_gram_defect_rank": cross}


def rank_only_loss(z, zp, method, *, kappa=1., pair_weight=8., diversity_weight=1.,
                   soft_rank_tau=1., coverage_target=.5, coverage_weight=4., normalize_epsilon=1e-6):
    z, zp = validate_pair(z, zp)
    fallbacks_before = SVD_FALLBACK_COUNT
    if any(not math.isfinite(x) or x <= 0 for x in (kappa, soft_rank_tau, normalize_epsilon)):
        raise ValueError("Spectral thresholds and normalization epsilon must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (pair_weight, diversity_weight, coverage_weight)):
        raise ValueError("Weights must be finite and nonnegative")
    if not 0 <= coverage_target <= 1:
        raise ValueError("coverage_target must be in [0,1]")
    with torch.autocast(device_type=z.device.type, enabled=False):
        if method in ("centroid_soft", "centroid_normalized", "centroid_shuffled"):
            result = centroid_soft_rank(z,zp,kappa,normalize_epsilon,
                normalize_centroid=method == "centroid_normalized",
                shuffle_pairs=method == "centroid_shuffled")
        elif method == "mmcr":
            result = mmcr(z,zp,normalize_epsilon)
        elif method == "pair_rank":
            result = pair_rank(z,zp,pair_weight,diversity_weight,normalize_epsilon)
        elif method == "soft_residual":
            result = residual_coverage(z,zp,soft_rank_tau,coverage_target,coverage_weight)
        elif method == "gram_defect":
            result = gram_defect(z,zp,soft_rank_tau)
        else:
            raise ValueError(f"Unknown rank-only method: {method}")
    loss, parts = result
    parts = parts | {"svd_fallbacks": loss.new_tensor(SVD_FALLBACK_COUNT-fallbacks_before)}
    if not torch.isfinite(torch.stack([loss,*parts.values()])).all():
        raise FloatingPointError(f"Nonfinite {method} objective")
    return loss, parts
