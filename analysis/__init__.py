"""Representation geometry and plotting utilities for Spectral RankSSL."""

from .effective_rank import effective_rank, spectrum_metrics
from .collapse_metrics import geometry_metrics, detect_collapse

__all__ = ["effective_rank", "spectrum_metrics", "geometry_metrics", "detect_collapse"]
