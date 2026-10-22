"""Numerically stable, explicitly defined spectral diagnostics.

Entropy effective rank uses probabilities proportional to singular values, not
eigenvalues: exp(-sum(p * log(p))). The nuclear participation rank is
(sum(s))**2 / sum(s**2). Its normalized version divides by the maximum
available rank, min(B - 1, d) for centered matrices and min(B, d) otherwise.
Both diagnostics are scale invariant for nonzero matrices. A zero matrix has
rank zero by convention. Diagnostics have no additive epsilon that can turn
arbitrarily tiny isotropic noise into an apparent rank-zero matrix; scale
collapse is assessed separately from rank collapse.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _checked_matrix(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x).detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(f"Expected a nonempty [samples, dimensions] matrix, got {x.shape}")
    if not torch.isfinite(x).all():
        raise ValueError("Representation contains NaN or infinity")
    return x


def spectrum_metrics(x: torch.Tensor, center: bool = True) -> dict[str, Any]:
    """Return JSON-safe rank statistics and L1/L2-normalized spectra."""
    x = _checked_matrix(x)
    n, d = x.shape
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    max_rank = min(n - int(center), d)
    singular = torch.linalg.svdvals(x)
    largest = float(singular[0]) if singular.numel() else 0.0
    if largest == 0.0 or max_rank == 0:
        p = torch.zeros_like(singular)
        energy = torch.zeros_like(singular)
        entropy_rank = nuclear_rank = stable_rank = 0.0
        numerical_rank = 0
    else:
        # Dividing by the largest singular value prevents underflow for tiny
        # feature scales without modifying a scale-invariant rank definition.
        scaled = singular / largest
        p = scaled / scaled.sum()
        energy = scaled.square() / scaled.square().sum()
        positive = p[p > 0]
        entropy_rank = math.exp(float(-(positive * positive.log()).sum()))
        nuclear_rank = float(scaled.sum().square() / scaled.square().sum())
        stable_rank = float(scaled.square().sum())
        tolerance = max(n, d) * torch.finfo(torch.float64).eps
        numerical_rank = int((scaled > tolerance).sum())
    return {
        "effective_rank": entropy_rank,
        "entropy_effective_rank": entropy_rank,
        "nuclear_participation_rank": nuclear_rank,
        "normalized_nuclear_rank": min(1.0, nuclear_rank / max_rank) if max_rank else 0.0,
        "normalized_entropy_rank": min(1.0, entropy_rank / max_rank) if max_rank else 0.0,
        "stable_rank": stable_rank,
        "numerical_rank": min(numerical_rank, max_rank),
        "max_rank": max_rank,
        "centered": center,
        "singular_values": singular.tolist(),
        "normalized_singular_values": p.tolist(),
        "normalized_spectral_energy": energy.tolist(),
    }


def effective_rank(x: torch.Tensor, center: bool = True) -> float:
    """Entropy effective rank (singular-value convention), zero for zero data."""
    return float(spectrum_metrics(x, center=center)["effective_rank"])
