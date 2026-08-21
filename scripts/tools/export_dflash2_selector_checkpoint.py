#!/usr/bin/env python3
"""Export a selector-only DFlash2 DCP checkpoint as a serving directory.

Selector-only training freezes every non-selector tensor. Reusing those bytes
from the declared base serving checkpoint avoids treating an optimizer/FSDP
checkpoint as a deployable model while still producing an ordinary Hugging
Face-style ``model.safetensors`` artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.filesystem import FileSystemReader
from typing_extensions import override

SELECTOR_KEYS = (
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "candidate_selector.hidden_projection.weight",
)
CHECKPOINT_PREFIX = "model_state.model.draft_model."


class ExportError(RuntimeError):
    """Raised when checkpoint provenance or tensor structure is invalid."""


class _WrappedStorageReader(FileSystemReader):
    def __init__(self, path: str):
        model_dir = Path(path) / "model"
        super().__init__(str(model_dir if model_dir.is_dir() else path))


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for key, value in metadata.state_dict_metadata.items():
            if "optimizer" in key:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_selector_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(str(checkpoint_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    replacements: dict[str, torch.Tensor] = {}
    for key in SELECTOR_KEYS:
        checkpoint_key = f"{CHECKPOINT_PREFIX}{key}"
        tensor = state_dict.get(checkpoint_key)
        if tensor is None:
            raise ExportError(f"checkpoint is missing selector tensor {checkpoint_key}")
        replacements[key] = tensor.contiguous().cpu()
    return replacements


def _merge_selector_tensors(
    base: dict[str, torch.Tensor], replacements: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(replacements) != set(SELECTOR_KEYS):
        raise ExportError(
            f"expected selector keys {sorted(SELECTOR_KEYS)}, found {sorted(replacements)}"
        )

    for key, replacement in replacements.items():
        original = base.get(key)
        if original is None:
            raise ExportError(f"base serving checkpoint is missing selector tensor {key}")
        if replacement.shape != original.shape:
            raise ExportError(
                f"shape mismatch for {key}: base={tuple(original.shape)}, "
                f"checkpoint={tuple(replacement.shape)}"
            )
        if replacement.dtype != original.dtype:
            raise ExportError(
                f"dtype mismatch for {key}: base={original.dtype}, checkpoint={replacement.dtype}"
            )
        base[key] = replacement
    return base


def export(checkpoint_dir: Path, base_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    base_weights = base_dir / "model.safetensors"
    base_config = base_dir / "config.json"
    if not base_weights.is_file() or not base_config.is_file():
        raise ExportError(f"base directory is not a serving checkpoint: {base_dir}")
    if not (checkpoint_dir / "model").is_dir():
        raise ExportError(f"distributed checkpoint has no model directory: {checkpoint_dir}")
    if output_dir.exists() and not overwrite:
        raise ExportError(f"output already exists (pass --overwrite): {output_dir}")

    replacements = _load_selector_tensors(checkpoint_dir)
    with safe_open(base_weights, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        base = {key: source.get_tensor(key) for key in source.keys()}
    merged = _merge_selector_tensors(base, replacements)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_weights = output_dir / ".model.safetensors.tmp"
    save_file(merged, temporary_weights, metadata=metadata)
    os.replace(temporary_weights, output_dir / "model.safetensors")
    shutil.copy2(base_config, output_dir / "config.json")

    manifest = {
        "schema_version": 1,
        "kind": "dflash2-selector-only-serving-export",
        "base_serving_checkpoint": str(base_dir.resolve()),
        "source_distributed_checkpoint": str(checkpoint_dir.resolve()),
        "replaced_tensors": list(SELECTOR_KEYS),
    }
    (output_dir / "selector_export.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export(args.checkpoint_dir, args.base_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
