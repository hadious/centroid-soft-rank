"""VICReg-style Siamese network: both views use one encoder instance."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet18


class MNISTEncoder(nn.Module):
    """Small grayscale CNN with an unrestricted final representation layer."""

    def __init__(self, representation_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 28 -> 14 -> 7 -> 4. Unlike non-global adaptive pooling, this
            # fixed pool has a deterministic CUDA backward implementation.
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, representation_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SiameseEncoder(nn.Module):
    """Process one view; call this SAME module again for the second view."""

    def __init__(
        self,
        encoder: nn.Module,
        representation_dim: int,
        hidden_dims: Sequence[int],
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.representation_dim = representation_dim
        self.projection_dim = projection_dim
        self.projector_hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = []
        previous = representation_dim
        for width in hidden_dims:
            layers.extend(
                [nn.Linear(previous, width, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]
            )
            previous = width
        layers.append(nn.Linear(previous, projection_dim))
        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.projector(h)


def build_model(
    dataset: str,
    representation_dim: int | None = None,
    projector_hidden_dim: int | None = None,
    projection_dim: int | None = None,
    projector_hidden_dims: Sequence[int] | None = None,
) -> SiameseEncoder:
    """Build the identical model for baseline, ablations, and full RankSSL.

    Defaults: MNIST 128 -> 256 -> 256 -> 128; CIFAR 512 -> 1024 ->
    1024 -> 256. The method/loss is deliberately not an argument.
    """
    name = dataset.lower().replace("-", "").replace("_", "")
    if name not in {"mnist", "cifar10", "cifar100", "tinyimagenet", "imagenette"}:
        raise ValueError(f"Unknown dataset {dataset!r}; choose mnist, cifar10, cifar100")
    is_mnist = name == "mnist"
    rep_dim = representation_dim if representation_dim is not None else (128 if is_mnist else 512)
    proj_dim = projection_dim if projection_dim is not None else (128 if is_mnist else 256)
    hidden = projector_hidden_dim if projector_hidden_dim is not None else (256 if is_mnist else 1024)
    hidden_dims = tuple(projector_hidden_dims) if projector_hidden_dims is not None else (hidden, hidden)
    if any(width <= 0 for width in (rep_dim, proj_dim, *hidden_dims)):
        raise ValueError("All representation and projector dimensions must be positive")
    if is_mnist:
        encoder: nn.Module = MNISTEncoder(rep_dim)
    else:
        encoder = resnet18(weights=None)
        if name != "imagenette":
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity() if rep_dim == 512 else nn.Linear(512, rep_dim)
    return SiameseEncoder(encoder, rep_dim, hidden_dims, proj_dim)
