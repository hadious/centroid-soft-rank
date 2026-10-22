"""Frozen-feature geometry, including separate scale/rank collapse flags."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .effective_rank import _checked_matrix, spectrum_metrics


def feature_metrics(x: torch.Tensor, center_spectrum: bool = True,
                    near_zero_threshold: float = 1e-3) -> dict[str, Any]:
    x = _checked_matrix(x)
    n, d = x.shape
    centered = x - x.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(n - 1, 1)
    variance = covariance.diagonal().clamp_min(0)
    std = variance.sqrt()
    norm = torch.linalg.vector_norm(x, dim=1)
    offdiag = covariance.clone()
    offdiag.fill_diagonal_(0)
    offdiag_count = max(d * (d - 1), 1)
    total_energy = float(covariance.square().sum())
    result = spectrum_metrics(x, center=center_spectrum)
    result.update({
        "samples": n,
        "dimensions": d,
        "mean_feature_std": float(std.mean()),
        "min_feature_std": float(std.min()),
        "median_feature_std": float(std.median()),
        "max_feature_std": float(std.max()),
        "near_zero_threshold": near_zero_threshold,
        "near_zero_dimensions_pct": float((std < near_zero_threshold).double().mean() * 100),
        "covariance_trace": float(variance.sum()),
        "covariance_diagonal_mean": float(variance.mean()),
        "covariance_diagonal_std": float(variance.std(unbiased=False)),
        "covariance_offdiagonal_mean_abs": float(offdiag.abs().sum() / offdiag_count),
        "covariance_offdiagonal_rms": float((offdiag.square().sum() / offdiag_count).sqrt()),
        "covariance_offdiagonal_energy_fraction": float(offdiag.square().sum()) / total_energy if total_energy else 0.0,
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std(unbiased=False)),
        "norm_min": float(norm.min()),
        "norm_max": float(norm.max()),
    })
    return result


def detect_collapse(metrics: dict[str, Any], linear_accuracy: float | None = None,
                    num_classes: int | None = None) -> dict[str, Any]:
    """Predeclared flags; accuracy is a fraction in [0, 1].

    Low scale means average per-coordinate sample standard deviation < 1e-3.
    Low rank means entropy rank <= max(1.5, 1% of available rank). Constant
    features mean at least 99% of coordinates have std < 1e-3. Covariance
    degeneracy means its mean diagonal is < 1e-6. Chance-level probes use an
    absolute tolerance of two percentage points. These flags identify a
    diagnostic concern and do not prove that a low-accuracy model is constant.
    """
    reasons = []
    # A full geometry dict or a single feature metrics dict is accepted.
    blocks = {k: metrics[k] for k in ("h", "z", "z_prime") if k in metrics}
    if not blocks:
        blocks = {"features": metrics}
    for name, block in blocks.items():
        if block["mean_feature_std"] < 1e-3:
            reasons.append(f"{name}: near-zero average feature std")
        if block["max_rank"] >= 2 and block["effective_rank"] <= max(1.5, 0.01 * block["max_rank"]):
            reasons.append(f"{name}: extremely low entropy effective rank")
        if block["near_zero_dimensions_pct"] >= 99.0:
            reasons.append(f"{name}: almost constant across samples")
        if block["covariance_diagonal_mean"] < 1e-6:
            reasons.append(f"{name}: degenerate covariance scale")
    if linear_accuracy is not None and num_classes is not None:
        if num_classes < 2:
            raise ValueError("Chance-level evaluation needs at least two classes")
        if linear_accuracy <= 1.0 / num_classes + 0.02:
            reasons.append("linear probe approximately chance")
    return {"collapsed": bool(reasons), "collapse_reasons": reasons}


def geometry_metrics(h: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor,
                     near_zero_threshold: float = 1e-3) -> dict[str, Any]:
    """JSON-safe diagnostics on a fixed paired-view batch.

    h and Z spectra are centered. D=Z-Z' is uncentered, matching the residual
    term in the training objective. A centered residual rank is also provided
    to reveal constant view offsets. Input rows must describe matched views;
    h may be deterministic-view encoder features from the same examples.
    """
    h, z1, z2 = map(_checked_matrix, (h, z1, z2))
    if z1.shape != z2.shape or h.shape[0] != z1.shape[0]:
        raise ValueError("Paired projections must match; encoder/projector sample counts must match")
    residual = z1 - z2
    result = {
        "h": feature_metrics(h, True, near_zero_threshold),
        "z": feature_metrics(z1, True, near_zero_threshold),
        "z_prime": feature_metrics(z2, True, near_zero_threshold),
        "residual": feature_metrics(residual, False, near_zero_threshold),
        "residual_centered": spectrum_metrics(residual, center=True),
        "positive_pair_cosine_mean": float(F.cosine_similarity(z1, z2, dim=1, eps=1e-12).mean()),
        "positive_pair_cosine_std": float(F.cosine_similarity(z1, z2, dim=1, eps=1e-12).std(unbiased=False)),
        "alignment_mse": float(residual.square().mean()),
        "diagnostic_protocol": "h/Z/Zprime centered; D uncentered; entropy over singular values; CPU float64 SVD",
    }
    result["residual_to_representation_effective_rank"] = (
        result["residual"]["effective_rank"] / result["z"]["effective_rank"]
        if result["z"]["effective_rank"] else None
    )
    result.update(detect_collapse(result))
    return result
