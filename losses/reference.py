"""Reusable, architecture-independent SSL losses and their ablations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .spectral_rankssl import (
    _matrix,
    _representation_rank,
    _residual_rank,
    representation_rank_loss,
    representation_rank_score,
    residual_rank_loss,
)
from .vicreg import covariance_loss, variance_loss

METHODS = ("vicreg", "av", "arv", "ahv", "full", "no_variance")


def ssl_loss(
    z1: Tensor,
    z2: Tensor,
    method: str = "full",
    lambda_align: float = 25.0,
    lambda_resrank: float = 5.0,
    lambda_reprrank: float = 10.0,
    lambda_variance: float = 25.0,
    lambda_covariance: float = 1.0,
    rho: float = 0.25,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    variance_epsilon: float = 1e-4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one of six controlled objectives and raw scalar components.

    Keys are ``alignment``, ``residual_rank``, ``representation_rank``,
    ``variance``, and ``covariance``. Disabled components are zero and are
    not computed. This avoids charging VICReg for unused SVD operations.
    All methods use the same MSE normalization and variance convention.
    Nonfinite inputs or resulting components raise immediately.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown SSL method {method!r}; expected one of {METHODS}")
    if z1.shape != z2.shape or z1.device != z2.device or z1.dtype != z2.dtype:
        raise ValueError("Both views must have identical shapes, devices, and dtypes")
    weights = (lambda_align, lambda_resrank, lambda_reprrank, lambda_variance, lambda_covariance)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Loss weights must be finite and nonnegative")
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if any(not math.isfinite(e) or e <= 0 for e in (epsilon, variance_epsilon)):
        raise ValueError("Numerical stabilizers must be finite and positive")
    z1, z2 = _matrix(z1, centered=True), _matrix(z2, centered=True)
    with torch.autocast(device_type=z1.device.type, enabled=False):
        residual = z1 - z2
        components = dict.fromkeys(
            ("alignment", "residual_rank", "representation_rank", "variance", "covariance"),
            z1.new_zeros(()),
        )
        components["alignment"] = residual.square().mean()
        if method in ("arv", "full", "no_variance"):
            components["residual_rank"] = _residual_rank(residual, epsilon)
        if method in ("ahv", "full", "no_variance"):
            components["representation_rank"] = torch.relu(rho - _representation_rank(z1, epsilon)) + torch.relu(
                rho - _representation_rank(z2, epsilon)
            )
        if method != "no_variance":
            components["variance"] = variance_loss(z1, z2, gamma, variance_epsilon)
        if method == "vicreg":
            components["covariance"] = covariance_loss(z1, z2)
        total = sum(weight * value for weight, value in zip(weights, components.values()))
        if not bool(torch.isfinite(torch.stack([total, *components.values()])).all()):
            raise FloatingPointError(f"Nonfinite {method} loss; inspect representation scale and inputs")
    return total, components


__all__ = [
    "METHODS",
    "ssl_loss",
    "residual_rank_loss",
    "representation_rank_score",
    "representation_rank_loss",
    "variance_loss",
    "covariance_loss",
]
