"""Differentiable spectral objectives from the Spectral RankSSL proposal.

Numerical conventions:
* SVD and norms use float32 or float64, even when called under autocast.
* Residuals are NOT centered; representation matrices ARE batch centered.
* ``epsilon`` is added after the residual Frobenius norm, and after the
  squared representation Frobenius norm, exactly as in the proposal.
* At an exactly zero residual the residual objective and its gradient are
  defined to be zero. Nuclear norm has no unique gradient there; masking
  selects the zero subgradient and avoids an arbitrary SVD basis update.
* No singular values are dropped or jittered. Such changes would alter the
  objective, especially its behavior near dimensional collapse.

The normalized nuclear/Frobenius ratios are spectral diversity surrogates,
not entropy effective rank. Epsilon makes both ratios scale dependent close
to zero. The residual ratio can reach zero by exact alignment as well as
decrease by concentrating spectral energy.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _matrix(z: Tensor, *, centered: bool = False) -> Tensor:
    if z.ndim != 2 or z.shape[0] < (2 if centered else 1) or z.shape[1] < 1:
        raise ValueError("Expected nonempty [batch, features] matrix; centered rank requires batch >= 2")
    if not z.is_floating_point():
        raise TypeError("Representations must have floating-point dtype")
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("Nonfinite representation supplied to spectral loss")
    return z if z.dtype in (torch.float32, torch.float64) else z.float()


def _epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")


def _residual_rank(d: Tensor, epsilon: float) -> Tensor:
    singular_values = torch.linalg.svdvals(d)
    nuclear = singular_values.sum()
    frobenius = torch.linalg.vector_norm(d)
    ratio = nuclear / (math.sqrt(min(d.shape)) * (frobenius + epsilon))
    # At D=0 the nuclear norm subgradient is nonunique. Select zero.
    return torch.where(frobenius > 0, ratio, torch.zeros_like(ratio))


def _representation_rank(z: Tensor, epsilon: float) -> Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    nuclear = singular_values.sum()
    rank_bound = min(z.shape[0] - 1, z.shape[1])
    return nuclear.square() / (rank_bound * (centered.square().sum() + epsilon))


def residual_rank_loss(d: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Return ||D||_* / (sqrt(min(B,d)) * (||D||_F + epsilon))."""
    _epsilon(epsilon)
    d = _matrix(d)
    with torch.autocast(device_type=d.device.type, enabled=False):
        value = _residual_rank(d, epsilon)
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite residual-rank objective")
    return value


def representation_rank_score(z: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Normalized squared nuclear/Frobenius rank of a centered batch."""
    _epsilon(epsilon)
    z = _matrix(z, centered=True)
    with torch.autocast(device_type=z.device.type, enabled=False):
        value = _representation_rank(z, epsilon)
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite representation-rank score")
    return value


def representation_rank_loss(
    z1: Tensor, z2: Tensor, rho: float = 0.25, epsilon: float = 1e-6
) -> Tensor:
    """Sum of the two high-rank hinges, with no pressure above rho."""
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if z1.shape != z2.shape or z1.device != z2.device:
        raise ValueError("Both views must have identical shapes and devices")
    return torch.relu(rho - representation_rank_score(z1, epsilon)) + torch.relu(
        rho - representation_rank_score(z2, epsilon)
    )
