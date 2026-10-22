"""Shared encoder/projector architectures for every SSL objective."""

from .siamese import MNISTEncoder, SiameseEncoder, build_model

__all__ = ["MNISTEncoder", "SiameseEncoder", "build_model"]
