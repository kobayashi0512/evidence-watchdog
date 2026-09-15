#!/usr/bin/env python3
"""Fetch public upstream assets without redistributing them in this repository.

The script downloads archives only from the public official routes documented in
docs/DATA_AVAILABILITY.md and uses Hugging Face's official client for models.
It prints SHA-256 values after download so users can record their local state.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


DATASETS = {
    "qasper": {
        "url": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "archive": "qasper-train-dev-v0.3.tgz",
    },
    "scifact": {
        "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "archive": "scifact-data.tar.gz",
    },
}
MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        root = destination.resolve()
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"refusing archive member outside destination: {member.name}")
        tar.extractall(destination, members=members)


def download_dataset(name: str, data_dir: Path) -> None:
    spec = DATASETS[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / spec["archive"]
    if not archive.exists():
        print(f"Downloading {name} from {spec['url']}")
        with urllib.request.urlopen(spec["url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    print(f"{archive}: sha256={sha256(archive)}")
    destination = data_dir / name
    if not destination.exists():
        destination.mkdir()
        safe_extract(archive, destination)
    print(f"Extracted {name} under {destination}")


def snapshot_model(name: str, model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc
    destination = model_dir / name
    print(f"Downloading {MODELS[name]} to {destination}")
    snapshot_download(repo_id=MODELS[name], local_dir=str(destination))
    weights = destination / "model.safetensors"
    if weights.exists():
        print(f"{weights}: sha256={sha256(weights)}")
    else:
        print(f"Downloaded {name}; no model.safetensors file was found at the expected top-level path.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all"], default=[])
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=[])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    datasets = list(DATASETS) if "all" in args.datasets else args.datasets
    models = list(MODELS) if "all" in args.models else args.models
    if not datasets and not models:
        parser.error("select at least one asset with --datasets or --models")
    for name in datasets:
        download_dataset(name, args.data_dir)
    for name in models:
        snapshot_model(name, args.model_dir)


if __name__ == "__main__":
    main()
