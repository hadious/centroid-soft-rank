"""Independent v2 regression tests: provenance refusal and second-view safety.

All fixtures are CPU-only, use temporary directories, and never train a model
or load a real dataset/checkpoint.
"""
from __future__ import annotations

import itertools
import json

import pytest
import torch
from torch.utils.data import TensorDataset

import train_ssl
from analysis import detect_collapse, geometry_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def artifact_bytes(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("fault", ("config_mismatch", "source_mismatch", "config_missing", "source_missing"))
def test_checkpoint_refusal_precedes_dataset_model_load_and_provenance_writes(tmp_path, monkeypatch, fault):
    cfg = train_ssl.DEFAULTS | dict(device="cpu", amp=False, workers=0, output=str(tmp_path))
    # Deliberately invalid checkpoint bytes prove that early refusal also
    # precedes loading the checkpoint's large tensor payload.
    (tmp_path / "last.pt").write_bytes(b"checkpoint payload must not be loaded")
    write_json(tmp_path / "config.json", cfg)
    write_json(tmp_path / "source_hashes.json", {"source.py": "expected-source-hash"})
    (tmp_path / "split.json").write_bytes(b"original split metadata\n")
    (tmp_path / "history.json").write_bytes(b"original training history\n")
    if fault == "config_mismatch":
        write_json(tmp_path / "config.json", cfg | {"epochs": cfg["epochs"] + 1})
    elif fault == "source_mismatch":
        write_json(tmp_path / "source_hashes.json", {"source.py": "incompatible-source-hash"})
    elif fault == "config_missing":
        (tmp_path / "config.json").unlink()
    else:
        (tmp_path / "source_hashes.json").unlink()
    before = artifact_bytes(tmp_path)
    monkeypatch.setattr(train_ssl, "source_hashes", lambda: {"source.py": "expected-source-hash"})

    def prohibited(*args, **kwargs):
        pytest.fail("Refusal must happen before dataset or model construction")

    monkeypatch.setattr(train_ssl, "build_datasets", prohibited)
    monkeypatch.setattr(train_ssl, "build_model", prohibited)
    with pytest.raises(ValueError, match="provenance|configuration|Source changed"):
        train_ssl.run_experiment(cfg)
    assert artifact_bytes(tmp_path) == before


def test_internal_checkpoint_config_mismatch_preserves_saved_provenance(tmp_path, monkeypatch):
    cfg = train_ssl.DEFAULTS | dict(device="cpu", amp=False, workers=0, output=str(tmp_path), threads=2)
    torch.save({"config": cfg | {"epochs": cfg["epochs"] + 1}}, tmp_path / "last.pt")
    write_json(tmp_path / "config.json", cfg)
    write_json(tmp_path / "source_hashes.json", {"source.py": "expected-source-hash"})
    (tmp_path / "split.json").write_bytes(b"original split metadata\n")
    before = artifact_bytes(tmp_path)
    monkeypatch.setattr(train_ssl, "source_hashes", lambda: {"source.py": "expected-source-hash"})
    monkeypatch.setattr(train_ssl, "seed_all", lambda seed: None)
    monkeypatch.setattr(train_ssl, "build_datasets", lambda *a, **k: {"metadata": {"replacement": True}})
    monkeypatch.setattr(train_ssl, "build_model", lambda *a, **k: torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="Resume requires exactly the same configuration"):
        train_ssl.run_experiment(cfg)
    assert artifact_bytes(tmp_path) == before


def corners():
    return torch.tensor(list(itertools.product((-1., 1.), repeat=3)))


def test_second_view_only_collapse_is_detected_without_first_view_false_positive():
    healthy = corners()
    assert not geometry_metrics(healthy, healthy, healthy)["collapsed"]
    geometry = geometry_metrics(healthy, healthy, torch.zeros_like(healthy))
    assert geometry["collapsed"]
    assert all(reason.startswith("z_prime:") for reason in geometry["collapse_reasons"])
    assert not detect_collapse(geometry["h"])["collapsed"]
    assert not detect_collapse(geometry["z"])["collapsed"]
    assert detect_collapse(geometry["z_prime"])["collapsed"]


def test_diagnose_checks_both_raw_unit_second_view_and_both_normalization_floors(tmp_path, monkeypatch):
    class IdentityProjection(torch.nn.Module):
        def forward(self, x):
            features = x.flatten(1)
            return features, features

    x = corners().reshape(8, 3, 1, 1)
    pairs = TensorDataset(x, torch.zeros_like(x))
    cfg = train_ssl.DEFAULTS | dict(device="cpu", amp=False, workers=0, batch_size=4,
                                   diagnostics_size=8, output=str(tmp_path))
    monkeypatch.setattr(train_ssl, "seed_all", lambda seed: torch.random.default_generator.manual_seed(seed))
    result = train_ssl.diagnose(IdentityProjection(), pairs, cfg)
    assert result["normalization_floor_fraction_z"] == 0
    assert result["normalization_floor_fraction_z_prime"] == 1
    assert result["normalization_floor_fraction"] == 1
    assert result["collapsed"] and result["unit_projector"]["collapsed"]
    assert all(reason.startswith("z_prime:") for reason in result["collapse_reasons"])
    assert all(reason.startswith("z_prime:") for reason in result["unit_projector"]["collapse_reasons"])
    assert (tmp_path / "diagnostic_pairs.pt").is_file()
