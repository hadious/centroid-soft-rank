"""Data splitting never uses official test data for training or validation.

``seed`` is a fixed partition seed, independent of model-training seeds. Test
data are only constructed when the caller explicitly requests include_test.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from .folder_benchmarks import FolderBenchmark


NORMALIZATION = {
    "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "imagenette": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
DATASET_CLASSES = {
    "mnist": tv_datasets.MNIST,
    "cifar10": tv_datasets.CIFAR10,
    "cifar100": tv_datasets.CIFAR100,
}


def _dataset_name(dataset: str) -> str:
    name = dataset.lower().replace("-", "").replace("_", "")
    if name not in DATASET_CLASSES and name not in {"tinyimagenet", "imagenette"}:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return name


class RandomGaussianNoise:
    def __init__(self, std: float = 0.02, probability: float = 0.5) -> None:
        self.std = std
        self.probability = probability

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std > 0 and torch.rand(()) < self.probability:
            return (x + torch.randn_like(x) * self.std).clamp(0, 1)
        return x

    def __repr__(self) -> str:
        return f"RandomGaussianNoise(std={self.std}, probability={self.probability})"


def build_transforms(dataset: str, augmentation: dict[str, Any] | None = None):
    """Return (SSL transform, deterministic evaluation transform).

    All stochastic operations use worker-seeded torch RNG. Small affine changes
    preserve MNIST identity; CIFAR follows a standard crop/color SSL recipe.
    """
    name = _dataset_name(dataset)
    config = augmentation or {}
    mean, std = NORMALIZATION[name]
    normalize = T.Normalize(mean, std)
    if name == "mnist":
        augment = T.Compose(
            [
                T.RandomAffine(
                    degrees=config.get("degrees", 12),
                    translate=tuple(config.get("translate", (0.08, 0.08))),
                    scale=tuple(config.get("scale", (0.92, 1.08))),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                T.ColorJitter(
                    brightness=config.get("brightness", 0.15),
                    contrast=config.get("contrast", 0.15),
                ),
                T.ToTensor(),
                RandomGaussianNoise(
                    config.get("noise_std", 0.02), config.get("noise_probability", 0.5)
                ),
                normalize,
            ]
        )
    else:
        augment = T.Compose(
            [
                T.RandomResizedCrop(
                    {"tinyimagenet": 64, "imagenette": 128}.get(name, 32),
                    scale=tuple(config.get("crop_scale", (0.2, 1.0))),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=config.get("flip_probability", 0.5)),
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=config.get("color_probability", 0.8),
                ),
                T.RandomGrayscale(p=config.get("grayscale_probability", 0.2)),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=config.get("blur_probability", 0.1),
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    evaluation_prefix = [T.Resize(144, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(128)] if name == "imagenette" else []
    evaluate = T.Compose(evaluation_prefix + [T.ToTensor(), normalize])
    return augment, evaluate


class TransformSubset(Dataset):
    """Apply an independent transform without mutating the shared base data."""

    def __init__(self, base: Dataset, indices: np.ndarray, transform, two_views: bool = False):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[int(self.indices[index])]
        if self.two_views:
            # Two separate calls draw independent augmentation parameters.
            return self.transform(image), self.transform(image)
        return self.transform(image), int(label)


def stratified_partition(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
    available: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select an exact-size proportional stratified subset without replacement.

    Largest-remainder allocation avoids dropping small classes, while preserving
    the source class proportions. Both returned partitions are shuffled.
    """
    labels = np.asarray(labels)
    indices = np.arange(len(labels), dtype=np.int64) if available is None else np.asarray(available, dtype=np.int64)
    if not 0 <= count <= len(indices):
        raise ValueError(f"Requested {count} examples from pool of {len(indices)}")
    if count == 0:
        return np.empty(0, dtype=np.int64), rng.permutation(indices)
    classes, frequencies = np.unique(labels[indices], return_counts=True)
    exact = frequencies * (count / len(indices))
    quotas = np.floor(exact).astype(int)
    remaining = count - int(quotas.sum())
    # Random tie-breaking, deterministic from the partition RNG.
    order = np.lexsort((rng.random(len(classes)), -(exact - quotas)))
    quotas[order[:remaining]] += 1
    chosen, rest = [], []
    for class_id, quota in zip(classes, quotas):
        members = rng.permutation(indices[labels[indices] == class_id])
        chosen.append(members[:quota])
        rest.append(members[quota:])
    return rng.permutation(np.concatenate(chosen)), rng.permutation(np.concatenate(rest))


def _index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _split_description(indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    classes, counts = np.unique(labels[indices], return_counts=True)
    return {
        "size": len(indices),
        "sha256_ordered_indices": _index_digest(indices),
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }


def build_datasets(
    dataset: str,
    root: str | Path = "./data",
    train_size: int | None = None,
    val_size: int = 5000,
    test_size: int | None = None,
    seed: int = 1729,
    probe_train_size: int | None = None,
    download: bool = True,
    include_test: bool = False,
    augmentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build disjoint SSL/probe-training and validation partitions.

    ``train_size=None`` uses all official training examples after validation is
    held out. ``probe_train_size`` can restrict probe labels to a smaller fixed
    subset of SSL training data. All methods and training seeds should use the
    same ``seed`` here. Test data are opt-in and never influence the split.

    Returned ``ssl_train`` and ``val_ssl`` yield ``(view1, view2)``; eval datasets
    yield ``(image, label)``. ``test_eval`` is None unless include_test=True.
    Actual index arrays are available as ``dataset.indices``; metadata contains
    their SHA256 digests, generation seed, class counts, and augmentation recipe.
    """
    name = _dataset_name(dataset)
    root = Path(root).expanduser()
    folder_dataset = name in {"tinyimagenet", "imagenette"}
    base = FolderBenchmark(root, name, train=True) if folder_dataset else DATASET_CLASSES[name](root=str(root), train=True, download=download)
    labels = np.asarray(base.targets, dtype=np.int64)
    if val_size <= 0 or val_size >= len(labels):
        raise ValueError("val_size must leave nonempty training and validation partitions")
    rng = np.random.default_rng(seed)
    val_indices, available = stratified_partition(labels, val_size, rng)
    n_train = len(available) if train_size is None else train_size
    if n_train < 2:
        raise ValueError("At least two SSL training examples are required")
    train_indices, _ = stratified_partition(labels, n_train, rng, available)
    if probe_train_size is None or probe_train_size == n_train:
        probe_indices = train_indices.copy()
    else:
        if probe_train_size < 1:
            raise ValueError("probe_train_size must be positive")
        probe_indices, _ = stratified_partition(labels, probe_train_size, rng, train_indices)
    augment, evaluate = build_transforms(name, augmentation)
    metadata: dict[str, Any] = {
        "dataset": name,
        "root": str(root.resolve()),
        "split_seed": seed,
        "split_algorithm": "proportional-largest-remainder-numpy-default_rng-v1",
        "official_train_size": len(labels),
        "num_classes": len(base.classes),
        "train": _split_description(train_indices, labels),
        "probe_train": _split_description(probe_indices, labels),
        "validation": _split_description(val_indices, labels),
        "train_validation_disjoint": bool(not np.intersect1d(train_indices, val_indices).size),
        "test_accessed": include_test,
        "augmentation": repr(augment),
        "evaluation_transform": repr(evaluate),
        "test": None,
    }
    if folder_dataset:
        metadata.update(class_to_idx=base.class_to_idx, train_sample_manifest_sha256=base.sample_manifest_sha256,
                        tuning_source="stratified subset of official training split",
                        heldout_source="official validation split; internal test_* fields refer to this split",
                        input_resolution=64 if name == "tinyimagenet" else 128)
    bundle: dict[str, Any] = {
        "ssl_train": TransformSubset(base, train_indices, augment, two_views=True),
        "train_eval": TransformSubset(base, probe_indices, evaluate),
        "val_eval": TransformSubset(base, val_indices, evaluate),
        "val_ssl": TransformSubset(base, val_indices, augment, two_views=True),
        "test_eval": None,
        "metadata": metadata,
    }
    if include_test:
        test = FolderBenchmark(root, name, train=False) if folder_dataset else DATASET_CLASSES[name](root=str(root), train=False, download=download)
        if folder_dataset:
            metadata["heldout_sample_manifest_sha256"] = test.sample_manifest_sha256
        test_labels = np.asarray(test.targets, dtype=np.int64)
        # Independent RNG means test/probe subset choices cannot perturb train/val.
        test_rng = np.random.default_rng(seed + 1)
        n_test = len(test_labels) if test_size is None else test_size
        if n_test <= 0:
            raise ValueError("test_size must be positive")
        test_indices, _ = stratified_partition(test_labels, n_test, test_rng)
        bundle["test_eval"] = TransformSubset(test, test_indices, evaluate)
        metadata["test"] = _split_description(test_indices, test_labels)
    return bundle


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn, used with a caller-seeded torch.Generator."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
