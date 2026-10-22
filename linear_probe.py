"""Deterministic softmax linear evaluation on cached, frozen encoder features.

Only training features determine centering/scaling and classifier updates.
Validation accuracy selects regularization and the training epoch, with
validation cross entropy breaking ties. Held-out test labels are read only
after model selection, for the final accuracy. Accuracies are fractions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _features(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(x).detach().to(device=device, dtype=torch.float32)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("Features must be a nonempty [samples, dimensions] matrix")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Non-finite frozen features")
    return x


def _labels(y: torch.Tensor, classes: torch.Tensor, n: int,
            device: torch.device) -> torch.Tensor:
    y = torch.as_tensor(y, device=device, dtype=torch.long).reshape(-1)
    if len(y) != n:
        raise ValueError("Feature and label sample counts differ")
    mapped = torch.searchsorted(classes, y)
    if bool((mapped >= len(classes)).any()) or not torch.equal(classes[mapped], y):
        raise ValueError("Evaluation contains a class absent from the classifier training split")
    return mapped


def _standardize(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Float64 subtraction avoids overflow for opposite-sign, finite float32
    # feature values and improves centering when a feature has a large offset.
    normalized = ((x.to(torch.float64) - mean) / scale).to(torch.float32)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("Non-finite standardized features")
    return normalized


@torch.no_grad()
def _score(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
           bias: torch.Tensor, batch_size: int) -> tuple[float, float]:
    correct = total_loss = 0.0
    for start in range(0, len(x), batch_size):
        target = y[start:start + batch_size]
        logits = F.linear(x[start:start + batch_size], weight, bias)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite linear-probe predictions")
        correct += float((logits.argmax(1) == target).sum())
        total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
    return correct / len(x), total_loss / len(x)


def train_linear_probe(
    train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor,
    test_x: torch.Tensor | None = None, test_y: torch.Tensor | None = None,
    seed: int = 0, epochs: int = 120, lr: float = 0.05,
    weight_decays: Sequence[float] = (0.0, 1e-4, 1e-2),
    batch_size: int = 4096, device: str | torch.device | None = None,
    eval_every: int = 5, output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit softmax heads with AdamW; select using validation labels only.

    The encoder is never updated: all supplied features are detached.
    train_x.device is used unless device is supplied. Zero head initialization
    avoids touching the application's RNG; a local generator controls shuffle.
    All candidates receive the same initialization and minibatch order.
    Per-coordinate scaling is fitted on the training features, with scale
    floored at 1e-6. Collapse is diagnosed on original, unscaled features.
    """
    if epochs < 1 or batch_size < 1 or eval_every < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, eval_every and lr must be positive")
    if not weight_decays or any(w < 0 or not math.isfinite(w) for w in weight_decays):
        raise ValueError("Provide finite, nonnegative regularization candidates")
    if (test_x is None) != (test_y is None):
        raise ValueError("Supply both test_x and test_y, or neither")
    dev = torch.device(device) if device is not None else torch.as_tensor(train_x).device
    train_x, val_x = _features(train_x, dev), _features(val_x, dev)
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Training and validation feature dimensions differ")
    raw_y = torch.as_tensor(train_y, device=dev, dtype=torch.long).reshape(-1)
    classes = raw_y.unique(sorted=True)
    if len(classes) < 2:
        raise ValueError("Linear evaluation needs at least two training classes")
    train_y = _labels(raw_y, classes, len(train_x), dev)
    val_y = _labels(val_y, classes, len(val_x), dev)
    mean = train_x.mean(0, dtype=torch.float64)
    scale = train_x.to(torch.float64).std(0, unbiased=False).clamp_min(1e-6)
    train_x = _standardize(train_x, mean, scale)
    val_x = _standardize(val_x, mean, scale)
    candidates = []
    winning = None
    for penalty in weight_decays:
        weight = torch.zeros((len(classes), train_x.shape[1]), device=dev, requires_grad=True)
        bias = torch.zeros(len(classes), device=dev, requires_grad=True)
        optimizer = torch.optim.AdamW([
            {"params": [weight], "weight_decay": float(penalty)},
            {"params": [bias], "weight_decay": 0.0},
        ], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best = None
        trace = []
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(train_x), generator=generator).to(dev)
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = F.linear(train_x[indices], weight, bias)
                loss = F.cross_entropy(logits, train_y[indices])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite linear-probe loss")
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % eval_every == 0 or epoch == epochs or epoch == 1:
                accuracy, val_loss = _score(val_x, val_y, weight, bias, batch_size)
                trace.append({"epoch": epoch, "val_accuracy": accuracy, "val_cross_entropy": val_loss})
                key = (accuracy, -val_loss)
                if best is None or key > best["key"]:
                    best = {"key": key, "epoch": epoch, "weight": weight.detach().clone(), "bias": bias.detach().clone()}
        candidate = {
            "weight_decay": float(penalty), "selected_epoch": best["epoch"],
            "val_accuracy": best["key"][0], "val_cross_entropy": -best["key"][1],
            "validation_trace": trace,
        }
        candidates.append(candidate)
        if winning is None or best["key"] > winning["key"]:
            winning = dict(best, penalty=float(penalty))
    train_accuracy, _ = _score(train_x, train_y, winning["weight"], winning["bias"], batch_size)
    result = {
        "val_accuracy": winning["key"][0],
        "val_cross_entropy": -winning["key"][1],
        "train_accuracy": train_accuracy,
        "test_accuracy": None,
        "selected_weight_decay": winning["penalty"],
        "selected_epoch": winning["epoch"],
        "seed": seed,
        "num_classes": len(classes),
        "train_samples": len(train_x),
        "val_samples": len(val_x),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "candidates": candidates,
        "protocol": "frozen encoder; train-only feature z-score; softmax AdamW; validation selects regularization and epoch; test evaluated once",
    }
    if test_x is not None:
        test_x = _features(test_x, dev)
        if test_x.shape[1] != train_x.shape[1]:
            raise ValueError("Test feature dimensions differ")
        test_y = _labels(test_y, classes, len(test_x), dev)
        test_x = _standardize(test_x, mean, scale)
        test_accuracy, test_loss = _score(test_x, test_y, winning["weight"], winning["bias"], batch_size)
        result.update(test_accuracy=test_accuracy, test_cross_entropy=test_loss, test_samples=len(test_x))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "weight": winning["weight"].detach().cpu(),
            "bias": winning["bias"].detach().cpu(),
            "feature_mean": mean.detach().cpu(),
            "feature_scale": scale.detach().cpu(),
            "classes": classes.detach().cpu(),
            "metadata": result,
        }, output_path)
        result["checkpoint_path"] = str(output_path)
    return result
