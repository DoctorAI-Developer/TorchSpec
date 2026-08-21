import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "average_dflash2_selector_checkpoints.py"
)
SPEC = importlib.util.spec_from_file_location("average_dflash2_selector_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _checkpoint(path: Path, selector_value: float, *, config: str = "{}\n") -> Path:
    path.mkdir()
    tensors = {
        "candidate_selector.predecessor_codebook": torch.full((2, 2), selector_value),
        "candidate_selector.successor_codebook": torch.full((2, 2), selector_value + 1),
        "candidate_selector.hidden_projection.weight": torch.full((2, 2), selector_value + 2),
        "draft.weight": torch.tensor([7.0]),
    }
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(config)
    return path


def test_average_changes_only_selector_tensors(tmp_path):
    base = _checkpoint(tmp_path / "base", 0.0)
    first = _checkpoint(tmp_path / "first", 2.0)
    second = _checkpoint(tmp_path / "second", 6.0)
    output = tmp_path / "output"

    MODULE.average(base, [first, second], output, weights=[1.0, 3.0], overwrite=False)

    averaged = load_file(output / "model.safetensors")
    assert torch.equal(averaged["draft.weight"], torch.tensor([7.0]))
    assert torch.equal(
        averaged["candidate_selector.predecessor_codebook"],
        torch.full((2, 2), 5.0),
    )
    manifest = json.loads((output / "selector_average.json").read_text())
    assert manifest["normalized_weights"] == [0.25, 0.75]
    assert manifest["accumulation_dtype"] == "float32"


@pytest.mark.parametrize("weights", ([1.0], [-1.0, 2.0], [0.0, 0.0]))
def test_average_rejects_invalid_weights(tmp_path, weights):
    base = _checkpoint(tmp_path / "base", 0.0)
    first = _checkpoint(tmp_path / "first", 2.0)
    second = _checkpoint(tmp_path / "second", 6.0)

    with pytest.raises(MODULE.AverageError):
        MODULE.average(
            base,
            [first, second],
            tmp_path / "output",
            weights=weights,
            overwrite=False,
        )


def test_average_rejects_config_mismatch(tmp_path):
    base = _checkpoint(tmp_path / "base", 0.0)
    first = _checkpoint(tmp_path / "first", 2.0)
    second = _checkpoint(tmp_path / "second", 6.0, config='{"different": true}\n')

    with pytest.raises(MODULE.AverageError, match="config differs"):
        MODULE.average(
            base,
            [first, second],
            tmp_path / "output",
            weights=None,
            overwrite=False,
        )
