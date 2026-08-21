from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "export_dflash2_selector_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("export_dflash2_selector_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _selector_tensors() -> dict[str, torch.Tensor]:
    return {
        "candidate_selector.predecessor_codebook": torch.zeros(4, 2, dtype=torch.bfloat16),
        "candidate_selector.successor_codebook": torch.zeros(4, 2, dtype=torch.bfloat16),
        "candidate_selector.hidden_projection.weight": torch.zeros(2, 3, dtype=torch.bfloat16),
    }


def test_merge_replaces_exact_selector_surface_only() -> None:
    base = _selector_tensors()
    base["layers.0.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    replacements = {key: value + 1 for key, value in _selector_tensors().items()}

    merged = MODULE._merge_selector_tensors(base, replacements)

    assert torch.equal(merged["layers.0.weight"], torch.ones(2, 2, dtype=torch.bfloat16))
    for key, replacement in replacements.items():
        assert torch.equal(merged[key], replacement)


@pytest.mark.parametrize("field", ["shape", "dtype"])
def test_merge_fails_closed_on_incompatible_selector(field: str) -> None:
    base = _selector_tensors()
    replacements = _selector_tensors()
    key = "candidate_selector.hidden_projection.weight"
    if field == "shape":
        replacements[key] = torch.zeros(3, 3, dtype=torch.bfloat16)
    else:
        replacements[key] = replacements[key].float()

    with pytest.raises(MODULE.ExportError, match=f"{field} mismatch"):
        MODULE._merge_selector_tensors(base, replacements)


def test_merge_rejects_incomplete_selector_surface() -> None:
    base = _selector_tensors()
    replacements = _selector_tensors()
    replacements.pop("candidate_selector.successor_codebook")

    with pytest.raises(MODULE.ExportError, match="expected selector keys"):
        MODULE._merge_selector_tensors(base, replacements)
