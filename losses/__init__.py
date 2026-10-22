"""Rank-only candidate dispatch, with separately labeled reference losses."""
from .rank_only import rank_only_loss
from .reference import ssl_loss as reference_loss

METHODS = ("centroid_soft", "centroid_normalized", "centroid_shuffled", "mmcr", "pair_rank", "soft_residual", "gram_defect", "vicreg", "ahv")
RANK_KEYS = {"kappa", "pair_weight", "diversity_weight", "soft_rank_tau", "coverage_target", "coverage_weight", "normalize_epsilon"}


def ssl_loss(z, zp, method, **kwargs):
    if method in ("vicreg", "ahv"):
        return reference_loss(z,zp,method,**{k:v for k,v in kwargs.items() if k not in RANK_KEYS})
    return rank_only_loss(z,zp,method,**{k:v for k,v in kwargs.items() if k in RANK_KEYS})


__all__ = ["ssl_loss", "rank_only_loss", "METHODS"]
