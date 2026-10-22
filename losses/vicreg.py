"""Standard VICReg variance and covariance components, shared by ablations."""

from __future__ import annotations

import torch
from torch import Tensor


def variance_loss(z1: Tensor, z2: Tensor, gamma: float = 1.0, epsilon: float = 1e-4) -> Tensor:
    """VICReg's average two-view variance hinge.

    Sample variance uses correction=1. Its 1e-4 stabilizer is deliberately
    separate from the spectral objectives' 1e-6 stabilizer. This convention
    is shared by every method, including the baseline.
    """
    std1 = torch.sqrt(z1.var(dim=0, correction=1) + epsilon)
    std2 = torch.sqrt(z2.var(dim=0, correction=1) + epsilon)
    return (torch.relu(gamma - std1).mean() + torch.relu(gamma - std2).mean()) / 2


def covariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    """Sum of squared off-diagonal sample covariances, divided by d.

    VICReg sums the two view contributions; it does not average them.
    """
    batch, features = z1.shape
    x1 = z1 - z1.mean(dim=0, keepdim=True)
    x2 = z2 - z2.mean(dim=0, keepdim=True)
    cov1 = x1.T @ x1 / (batch - 1)
    cov2 = x2.T @ x2 / (batch - 1)
    mask = ~torch.eye(features, device=z1.device, dtype=torch.bool)
    return (cov1[mask].square().sum() + cov2[mask].square().sum()) / features
