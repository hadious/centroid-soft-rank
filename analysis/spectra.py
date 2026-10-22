"""Headless plots from saved JSON diagnostic dictionaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectra(geometries: Mapping[str, dict[str, Any]], output_path: str | Path,
                 title: str = "Representation and residual spectra") -> None:
    """Compare methods with equal axes; spectra sum to one separately."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for axis, (key, label) in zip(axes, (("z", "Z (centered)"), ("z_prime", "Z' (centered)"), ("residual", "D = Z − Z' (uncentered)"))):
        for method, geometry in geometries.items():
            values = geometry[key]["normalized_singular_values"]
            axis.plot(range(1, len(values) + 1), values, label=method, linewidth=1.6)
        axis.set(title=label, xlabel="Singular-value index", ylabel="Singular value / nuclear norm")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_geometry_history(histories: Mapping[str, Sequence[dict[str, Any]]],
                          output_path: str | Path) -> None:
    """Accept nested geometry, top-level geometry, or train_ssl flat history."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fields = (("z", "effective_rank", "Z entropy effective rank"),
              ("residual", "effective_rank", "D entropy effective rank"),
              ("z", "mean_feature_std", "Z mean feature std"),
              ("z", "normalized_nuclear_rank", "Z normalized nuclear rank"))
    for axis, (block, field, label) in zip(axes.flat, fields):
        for method, records in histories.items():
            selected = []
            for row in records:
                geo = row.get("geometry", row)
                flat_key = f"geometry/{block}/{field}"
                if block in geo and field in geo[block]:
                    selected.append((row["epoch"], geo[block][field]))
                elif flat_key in row:
                    selected.append((row["epoch"], row[flat_key]))
            axis.plot([epoch for epoch, _ in selected], [value for _, value in selected], label=method)
        axis.set(xlabel="SSL epoch", ylabel=label)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
