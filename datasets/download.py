"""Accelerate a slow CIFAR download from torchvision's official archive URL.

The archive is verified against torchvision's published checksum; torchvision
then handles extraction and official training-set loading. No test labels are
loaded. Normal ``build_datasets(download=True)`` remains the default path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
from pathlib import Path
import shutil
import time
import urllib.request

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.utils import check_integrity


def download_cifar(dataset: str, root: str = "data", workers: int = 8) -> None:
    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[dataset]
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / cls.filename
    if check_integrity(str(archive), cls.tgz_md5):
        cls(root=str(destination), train=True, download=True)
        return
    request = urllib.request.Request(cls.url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        if response.headers.get("Accept-Ranges") != "bytes":
            raise RuntimeError("Official server does not advertise byte-range support")
    # Preserve an interrupted torchvision download as a prefix. Its correctness
    # is checked with the final whole-archive MD5 before extraction.
    prefix_size = archive.stat().st_size if archive.exists() else 0
    if prefix_size >= size:
        prefix_size = 0
    parts_dir = destination / f".{cls.filename}.parts"
    parts_dir.mkdir(exist_ok=True)
    block_size = math.ceil((size - prefix_size) / workers)
    ranges = [(start, min(size - 1, start + block_size - 1)) for start in range(prefix_size, size, block_size)]
    print(f"Official URL: {cls.url}; preserve {prefix_size} bytes; {len(ranges)} ranges", flush=True)

    def fetch(bounds: tuple[int, int]) -> Path:
        start, stop = bounds
        path = parts_dir / f"{start}-{stop}"
        expected = stop - start + 1
        if path.exists() and path.stat().st_size == expected:
            return path
        for attempt in range(3):
            try:
                request = urllib.request.Request(cls.url, headers={"Range": f"bytes={start}-{stop}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}":
                        raise RuntimeError("Unexpected server response to byte-range request")
                    with path.open("wb") as target:
                        shutil.copyfileobj(response, target, length=256 * 1024)
                if path.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete range {start}-{stop}")
                print(f"Downloaded range {start}-{stop}", flush=True)
                return path
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(fetch, ranges))
    assembled = destination / f".{cls.filename}.assembled"
    with assembled.open("wb") as target:
        if prefix_size:
            with archive.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    if not check_integrity(str(assembled), cls.tgz_md5):
        raise RuntimeError("Official torchvision MD5 mismatch; archive was not extracted")
    assembled.replace(archive)
    # Torchvision verifies archive and extracted batches before loading train.
    cls(root=str(destination), train=True, download=True)
    for path in paths:
        path.unlink()
    if not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    print(f"{dataset}: torchvision checksum, extraction, and train loading passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    download_cifar(args.dataset, args.root, args.workers)
