from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchspec.models.draft.cutlass_cache import require_cutlass_module_hash


def test_preserves_hash_supplied_by_cutlass() -> None:
    dsl = SimpleNamespace(get_module_hash=Mock())

    result = require_cutlass_module_hash(dsl, object(), "already-computed", "kernel")

    assert result == "already-computed"
    dsl.get_module_hash.assert_not_called()


def test_computes_hash_when_cached_wrapper_changes_no_cache_call() -> None:
    module = object()
    dsl = SimpleNamespace(get_module_hash=Mock(return_value="computed"))

    result = require_cutlass_module_hash(dsl, module, None, "flash_attn_fwd")

    assert result == "computed"
    dsl.get_module_hash.assert_called_once_with(module, "flash_attn_fwd")


@pytest.mark.parametrize("value", [None, ""])
def test_rejects_invalid_computed_hash(value: str | None) -> None:
    dsl = SimpleNamespace(get_module_hash=Mock(return_value=value))

    with pytest.raises(RuntimeError, match="non-empty string"):
        require_cutlass_module_hash(dsl, object(), None, "kernel")


def test_rejects_cutlass_without_hash_api() -> None:
    with pytest.raises(RuntimeError, match="BaseDSL.get_module_hash"):
        require_cutlass_module_hash(SimpleNamespace(), object(), None, "kernel")
