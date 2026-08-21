#!/usr/bin/env python3
"""Average selector tensors from compatible DFlash2 serving checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SELECTOR_KEYS = (
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "candidate_selector.hidden_projection.weight",
)


class AverageError(RuntimeError):
    """Raised when checkpoint averaging would violate serving compatibility."""


def _load_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    with safe_open(path, framework="pt", device="cpu") as source:
        return ({key: source.get_tensor(key) for key in source.keys()}, source.metadata())


def _normalized_weights(count: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise AverageError(f"expected {count} averaging weights, found {len(weights)}")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise AverageError("averaging weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise AverageError("at least one averaging weight must be positive")
    return [weight / total for weight in weights]


def average(
    base_dir: Path,
    source_dirs: list[Path],
    output_dir: Path,
    *,
    weights: list[float] | None,
    overwrite: bool,
) -> None:
    if len(source_dirs) < 2:
        raise AverageError("at least two source checkpoints are required")
    base_weights = base_dir / "model.safetensors"
    base_config = base_dir / "config.json"
    if not base_weights.is_file() or not base_config.is_file():
        raise AverageError(f"base directory is not a serving checkpoint: {base_dir}")
    if output_dir.exists() and not overwrite:
        raise AverageError(f"output already exists (pass --overwrite): {output_dir}")

    normalized = _normalized_weights(len(source_dirs), weights)
    base, metadata = _load_weights(base_weights)
    selector_accumulators = {
        key: torch.zeros_like(base[key], dtype=torch.float32) for key in SELECTOR_KEYS
    }
    for source_dir, weight in zip(source_dirs, normalized, strict=True):
        source_config = source_dir / "config.json"
        source_weights = source_dir / "model.safetensors"
        if not source_weights.is_file() or not source_config.is_file():
            raise AverageError(f"source directory is not a serving checkpoint: {source_dir}")
        if source_config.read_bytes() != base_config.read_bytes():
            raise AverageError(f"source config differs from base: {source_dir}")
        source, _ = _load_weights(source_weights)
        if source.keys() != base.keys():
            raise AverageError(f"source tensor keys differ from base: {source_dir}")
        for key in SELECTOR_KEYS:
            if source[key].shape != base[key].shape or source[key].dtype != base[key].dtype:
                raise AverageError(f"source selector tensor is incompatible: {source_dir}:{key}")
            selector_accumulators[key].add_(source[key].float(), alpha=weight)

    for key, averaged in selector_accumulators.items():
        base[key] = averaged.to(base[key].dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_weights = output_dir / ".model.safetensors.tmp"
    save_file(base, temporary_weights, metadata=metadata)
    os.replace(temporary_weights, output_dir / "model.safetensors")
    shutil.copy2(base_config, output_dir / "config.json")
    manifest = {
        "schema_version": 1,
        "kind": "dflash2-selector-serving-checkpoint-average",
        "base_serving_checkpoint": str(base_dir.resolve()),
        "source_serving_checkpoints": [str(path.resolve()) for path in source_dirs],
        "normalized_weights": normalized,
        "averaged_tensors": list(SELECTOR_KEYS),
        "accumulation_dtype": "float32",
    }
    (output_dir / "selector_average.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    average(
        args.base_dir,
        args.source_dir,
        args.output_dir,
        weights=args.weight,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
