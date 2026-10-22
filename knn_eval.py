"""Cosine, temperature-weighted kNN evaluation with validation-selected k."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from linear_probe import _features, _labels


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # Float64 norm accumulation avoids overflow for finite float32 features.
    double = x.to(torch.float64)
    return (double / torch.linalg.vector_norm(double, dim=1, keepdim=True).clamp_min(1e-12)).float()


@torch.no_grad()
def _knn_scores(bank: torch.Tensor, bank_y: torch.Tensor, query: torch.Tensor,
                query_y: torch.Tensor, ks: Sequence[int], temperature: float,
                query_batch_size: int, num_classes: int) -> dict[int, float]:
    correct = {k: 0 for k in ks}
    for start in range(0, len(query), query_batch_size):
        samples = query[start:start + query_batch_size]
        similarity, indices = (samples @ bank.T).topk(max(ks), dim=1, sorted=True)
        neighbors = bank_y[indices]
        # Subtracting the largest logit protects overflow and preserves votes.
        votes = ((similarity - similarity[:, :1]) / temperature).exp()
        target = query_y[start:start + query_batch_size]
        for k in ks:
            scores = torch.zeros((len(samples), num_classes), device=bank.device)
            scores.scatter_add_(1, neighbors[:, :k], votes[:, :k])
            correct[k] += int((scores.argmax(1) == target).sum())
    return {k: count / len(query) for k, count in correct.items()}


@torch.no_grad()
def knn_evaluate(
    train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor,
    test_x: torch.Tensor | None = None, test_y: torch.Tensor | None = None,
    k_values: Sequence[int] = (5, 20, 50), temperature: float = 0.1,
    query_batch_size: int = 256, device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Select k on validation accuracy; evaluate selected k on held-out test.

    Cosine similarities use L2-normalized raw encoder features. No feature
    centering, projector, labels from validation in the bank, or test tuning is
    used. Votes are exp(cosine / temperature), with temperature fixed before
    evaluation. All accuracies are fractions rather than percentages.
    """
    if temperature <= 0 or query_batch_size < 1 or not k_values or min(k_values) < 1:
        raise ValueError("temperature, query_batch_size and all k values must be positive")
    if (test_x is None) != (test_y is None):
        raise ValueError("Supply both test_x and test_y, or neither")
    dev = torch.device(device) if device is not None else torch.as_tensor(train_x).device
    train_x, val_x = _features(train_x, dev), _features(val_x, dev)
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Feature dimensions differ")
    raw_y = torch.as_tensor(train_y, device=dev, dtype=torch.long).reshape(-1)
    classes = raw_y.unique(sorted=True)
    train_y = _labels(raw_y, classes, len(train_x), dev)
    val_y = _labels(val_y, classes, len(val_x), dev)
    train_x = _normalize(train_x)
    val_x = _normalize(val_x)
    ks = sorted(set(min(int(k), len(train_x)) for k in k_values))
    scores = _knn_scores(train_x, train_y, val_x, val_y, ks, temperature, query_batch_size, len(classes))
    best_k = max(ks, key=lambda k: (scores[k], -k))
    result = {
        "val_accuracy": scores[best_k], "test_accuracy": None,
        "selected_k": best_k, "temperature": temperature,
        "validation_accuracies": {str(k): acc for k, acc in scores.items()},
        "train_samples": len(train_x), "val_samples": len(val_x),
        "protocol": "raw frozen encoder features; cosine weighted votes; validation selects k; test evaluated once",
    }
    if test_x is not None:
        test_x = _features(test_x, dev)
        test_y = _labels(test_y, classes, len(test_x), dev)
        test_x = _normalize(test_x)
        result["test_accuracy"] = _knn_scores(train_x, train_y, test_x, test_y, [best_k], temperature, query_batch_size, len(classes))[best_k]
        result["test_samples"] = len(test_x)
    return result
