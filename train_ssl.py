"""Deterministic, checkpointed training and held-out representation evaluation."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from datasets import build_datasets
from models import build_model
from losses import ssl_loss
from linear_probe import train_linear_probe
from knn_eval import knn_evaluate
from analysis import geometry_metrics, detect_collapse


DEFAULTS = dict(dataset="mnist", method="centroid_soft", seed=0, split_seed=1729,
    train_size=10000, val_size=5000, test_size=None, batch_size=256, workers=4,
    epochs=10, lr=0.001, weight_decay=0.0, warmup_epochs=1, min_lr_ratio=0.01,
    lambda_align=25., lambda_resrank=5., lambda_reprrank=10., lambda_variance=25.,
    lambda_covariance=1., rho=0.25, gamma=1., epsilon=1e-6, variance_epsilon=1e-4,
    grad_clip=10., amp=True, device="cuda", threads=4, diagnostics_size=1024,
    diagnostics_every=1, probe_epochs=120, probe_device="cpu", include_test=False,
    root="./data", output="results/run", deterministic=True, save_features=True,
    representation_dim=None, projector_hidden_dim=None, projection_dim=None,
    cuda_memory_fraction=0.28, precision_protocol="bfloat16 encoder; float32 projector/loss with exceptional same-matrix CPU float64 SVD retry; float32 evaluation",
    kappa=1.0, pair_weight=8.0, diversity_weight=1.0, soft_rank_tau=1.0, coverage_target=0.5, coverage_weight=4.0, normalize_epsilon=1e-6)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def loader(dataset, cfg, shuffle=False, drop_last=False, seed=0, batch_size=None, workers=None):
    return DataLoader(dataset, batch_size=batch_size or cfg["batch_size"],
        shuffle=shuffle, drop_last=drop_last,
        num_workers=cfg["workers"] if workers is None else workers,
        pin_memory=cfg["device"].startswith("cuda"), worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed), persistent_workers=False)


@torch.no_grad()
def extract_features(model, dataset, cfg):
    if dataset is None:
        return None, None
    xs, ys = [], []
    model.eval()
    for x, y in loader(dataset, cfg, batch_size=cfg["batch_size"]):
        x = x.to(cfg["device"], non_blocking=True, memory_format=torch.channels_last)
        # Evaluation is always FP32, including the frozen encoder.
        h = model.encoder(x)
        if not torch.isfinite(h).all():
            raise FloatingPointError("Nonfinite frozen encoder features")
        xs.append(h.cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def diagnose(model, dataset, cfg):
    """Same held-out images and fixed augmentations at every checkpoint and seed."""
    model.eval()
    hs, zs, zps = [], [], []
    dataset = Subset(dataset, range(min(len(dataset), cfg["diagnostics_size"])))
    devices = [torch.cuda.current_device()] if cfg["device"].startswith("cuda") else []
    # Training is reseeded per epoch too, so diagnostics cannot alter training RNG.
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        seed_all(93485)
        for a, b in loader(dataset, cfg, seed=93485, workers=0):
            a = a.to(cfg["device"], memory_format=torch.channels_last)
            b = b.to(cfg["device"], memory_format=torch.channels_last)
            h, z = model(a)
            _, zp = model(b)
            hs.append(h.cpu()); zs.append(z.cpu()); zps.append(zp.cpu())
    random.setstate(py_state); np.random.set_state(np_state)
    h, z, zp = torch.cat(hs), torch.cat(zs), torch.cat(zps)
    result = geometry_metrics(h, z, zp)
    u = torch.nn.functional.normalize(z, dim=1, eps=cfg["normalize_epsilon"])
    v = torch.nn.functional.normalize(zp, dim=1, eps=cfg["normalize_epsilon"])
    result["unit_projector"] = geometry_metrics(h, u, v)
    floor_z = float((z.norm(dim=1) < cfg["normalize_epsilon"]).float().mean())
    floor_zp = float((zp.norm(dim=1) < cfg["normalize_epsilon"]).float().mean())
    result["normalization_floor_fraction_z"] = floor_z
    result["normalization_floor_fraction_z_prime"] = floor_zp
    result["normalization_floor_fraction"] = max(floor_z, floor_zp)
    result["unit_positive_cosine"] = float((u*v).sum(1).mean())
    result["unit_alignment_mse"] = float((u-v).square().mean())
    atomic_checkpoint(Path(cfg["output"]) / "diagnostic_pairs.pt", dict(h=h, z=z, zp=zp))
    return result


def flatten_scalars(obj, prefix=""):
    result = {}
    for key, val in obj.items():
        name = prefix + key
        if isinstance(val, dict):
            result.update(flatten_scalars(val, name + "/"))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            result[name] = val
    return result


def scheduled_lr(cfg, epoch, step, steps):
    current = epoch * steps + step
    total = max(1, cfg["epochs"] * steps)
    warmup = min(cfg["warmup_epochs"] * steps, max(0, total - 1))
    if current < warmup:
        return cfg["lr"] * (current + 1) / max(1, warmup)
    progress = (current - warmup) / max(1, total - warmup - 1)
    ratio = cfg["min_lr_ratio"] + (1-cfg["min_lr_ratio"]) * 0.5 * (1+math.cos(math.pi*progress))
    return cfg["lr"] * ratio


def source_hashes():
    sources = {}
    for directory in ("models", "datasets", "losses", "analysis"):
        for file in sorted(Path(directory).glob("*.py")):
            sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    for file in [Path("train_ssl.py"), Path("linear_probe.py"), Path("knn_eval.py")]:
        sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    return sources


def run_experiment(config, resume=True):
    cfg = DEFAULTS | config
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check GPU access; use device='cpu' explicitly for CPU diagnostics.")
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and resume:
        old_cfg = json.loads((out / "config.json").read_text())
        if old_cfg != cfg:
            raise ValueError(f"Refusing mismatched completed configuration: {out}")
        if json.loads((out / "source_hashes.json").read_text()) != source_hashes():
            raise ValueError(f"Source code changed since completed run: {out}")
        return json.loads((out / "result.json").read_text())
    # Validate all saved provenance before split/config files can be written.
    if resume and (out / "last.pt").exists():
        if not (out / "config.json").exists() or not (out / "source_hashes.json").exists():
            raise ValueError("Checkpoint is missing saved configuration/source provenance")
        if json.loads((out / "config.json").read_text()) != cfg:
            raise ValueError("Refusing mismatched checkpoint configuration before writing provenance")
        if json.loads((out / "source_hashes.json").read_text()) != source_hashes():
            raise ValueError("Source changed since checkpoint; restart in a new run directory")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = False
    if cfg["device"].startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(cfg["cuda_memory_fraction"])
    seed_all(cfg["seed"])
    bundle = build_datasets(cfg["dataset"], root=cfg["root"], train_size=cfg["train_size"],
        val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["split_seed"],
        include_test=cfg["include_test"])
    model = build_model(cfg["dataset"], representation_dim=cfg["representation_dim"],
        projector_hidden_dim=cfg["projector_hidden_dim"], projection_dim=cfg["projection_dim"])
    model = model.to(cfg["device"], memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    start_epoch, history = 0, []
    checkpoint = out / "last.pt"
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=cfg["device"], weights_only=False)
        if state["config"] != cfg:
            raise ValueError("Resume requires exactly the same configuration, including epoch budget")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, history = state["epoch"], state["history"]
    atomic_json(out / "split.json", bundle["metadata"])
    atomic_json(out / "config.json", cfg)
    sources = source_hashes()
    hash_path = out / "source_hashes.json"
    if start_epoch and hash_path.exists():
        old_hashes = json.loads(hash_path.read_text())
        if old_hashes != sources:
            raise ValueError("Source code changed since checkpoint; restart in a new run directory")
    atomic_json(hash_path, sources)
    writer = SummaryWriter(str(out / "tensorboard"), purge_step=start_epoch if start_epoch else None)
    beginning = time.perf_counter()
    loss_keys = ("lambda_align", "lambda_resrank", "lambda_reprrank", "lambda_variance",
                 "lambda_covariance", "rho", "gamma", "epsilon", "variance_epsilon",
                 "kappa", "pair_weight", "diversity_weight", "soft_rank_tau", "coverage_target", "coverage_weight", "normalize_epsilon")
    loss_kwargs = {k: cfg[k] for k in loss_keys}
    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.perf_counter()
        epoch_seed = cfg["seed"] + 1009 * epoch
        seed_all(epoch_seed)
        from losses.rank_only import centroid_soft_rank
        centroid_soft_rank.shuffle_step = 0
        train_loader = loader(bundle["ssl_train"], cfg, shuffle=True, drop_last=True, seed=epoch_seed)
        if len(train_loader) == 0:
            raise ValueError("Training subset must contain at least one complete batch")
        model.train()
        totals, batches, clipped = {}, 0, 0
        for step, (a, b) in enumerate(train_loader):
            lr = scheduled_lr(cfg, epoch, step, len(train_loader))
            for group in optimizer.param_groups:
                group["lr"] = lr
            a = a.to(cfg["device"], non_blocking=True, memory_format=torch.channels_last)
            b = b.to(cfg["device"], non_blocking=True, memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(cfg["device"]).type,
                                dtype=torch.bfloat16, enabled=cfg["amp"]):
                h = model.encoder(a)
                hp = model.encoder(b)
            # Keep projected spectra free of final-layer bfloat16 quantization.
            z = model.projector(h.float())
            zp = model.projector(hp.float())
            # Blocking differentiable copies optionally offload small dense
            # losses to CPU. Backward returns gradients to the CUDA projector.
            # This is configured identically for every method in a comparison.
            loss_device = cfg.get("loss_device", cfg["device"])
            loss, parts = ssl_loss(z.float().to(loss_device), zp.float().to(loss_device),
                                   cfg["method"], **loss_kwargs)
            if not torch.isfinite(loss):
                atomic_json(out / "failure.json", dict(epoch=epoch, step=step, reason="nonfinite loss"))
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            loss.backward()
            # None records the gradient norm without clipping (shared across methods).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if cfg["grad_clip"] is None else cfg["grad_clip"],
                                                       error_if_nonfinite=True)
            optimizer.step()
            values = {"loss": loss.detach(), "grad_norm": grad_norm.detach().to(loss.device)} | {k: v.detach() for k,v in parts.items()}
            keys = list(values)
            numbers = torch.stack([values[k] for k in keys]).cpu().tolist()
            for k, value in zip(keys, numbers):
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite {k} at {epoch}/{step}")
                totals[k] = totals.get(k, 0.) + value
            clipped += cfg["grad_clip"] is not None and numbers[keys.index("grad_norm")] > cfg["grad_clip"]
            batches += 1
        row = {"epoch": epoch+1, "lr": lr, "steps": batches,
               "train_seconds": time.perf_counter()-epoch_start,
               "clipped_fraction": clipped/batches} | {k:v/batches for k,v in totals.items()}
        if (epoch+1) % cfg["diagnostics_every"] == 0 or epoch+1 == cfg["epochs"]:
            geom = diagnose(model, bundle["val_ssl"], cfg)
            atomic_json(out / f"geometry_epoch_{epoch+1:04d}.json", geom)
            row.update({"geometry/"+k:v for k,v in flatten_scalars(geom).items()})
        history.append(row)
        # Whole-epoch replay is deterministic; all randomness is derived from epoch_seed.
        atomic_checkpoint(checkpoint, dict(epoch=epoch+1, config=cfg, model=model.state_dict(),
                                          optimizer=optimizer.state_dict(), history=history))
        atomic_json(out / "history.json", history)
        fields = sorted({k for r in history for k in r})
        with (out / "history.csv").open("w") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader(); csv_writer.writerows(history)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch+1)
        writer.flush()
        print(json.dumps({"run": str(out), "epoch": epoch+1, "epochs": cfg["epochs"],
                          "loss": row["loss"], "seconds": round(row["train_seconds"], 2)}), flush=True)
    training_seconds = sum(r["train_seconds"] for r in history)
    evaluation_start = time.perf_counter()
    feature_data = {}
    for split in ("train", "val", "test"):
        x, y = extract_features(model, bundle[split+"_eval"], cfg)
        feature_data[split+"_x"], feature_data[split+"_y"] = x, y
    if cfg["save_features"]:
        atomic_checkpoint(out / "features.pt", feature_data)
    args = [feature_data[k] for k in ("train_x","train_y","val_x","val_y","test_x","test_y")]
    linear = train_linear_probe(*args, seed=cfg["seed"], epochs=cfg["probe_epochs"], device=cfg["probe_device"],
                                output_path=out / "linear_probe.pt")
    knn = knn_evaluate(*args, device=cfg["probe_device"])
    geometry = diagnose(model, bundle["val_ssl"], cfg)
    geometry.update(detect_collapse(geometry, linear["val_accuracy"], linear["num_classes"]))
    atomic_json(out / "geometry_final.json", geometry)
    result = dict(config=cfg, linear_probe=linear, knn=knn, geometry=geometry,
                  training_seconds=training_seconds, evaluation_seconds=time.perf_counter()-evaluation_start,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  elapsed_this_invocation=time.perf_counter()-beginning,
                  maximum_cuda_memory_mb=torch.cuda.max_memory_allocated()/2**20 if cfg["device"].startswith("cuda") else 0.)
    atomic_json(out / "result.json", result)
    writer.close()
    print(json.dumps({"completed": str(out), "val_linear": linear["val_accuracy"],
                      "test_linear": linear["test_accuracy"], "val_knn": knn["val_accuracy"],
                      "test_knn": knn["test_accuracy"], "collapsed": geometry["collapsed"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        run_experiment(config, resume=not args.no_resume)
    except torch.cuda.OutOfMemoryError:
        # The controller reruns every member of the affected comparison at this batch.
        out = Path(config["output"])
        out.mkdir(parents=True, exist_ok=True)
        batch = config.get("batch_size", DEFAULTS["batch_size"])
        atomic_json(out / "oom.json", {"batch_size": batch, "retry_batch_size": batch//2})
        print(f"CUDA OOM: controller must retry matched comparison at batch {batch//2}", flush=True)
        raise SystemExit(42)


if __name__ == "__main__":
    main()
