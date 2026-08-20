from unittest import mock

from torchspec.utils.env import get_torchspec_env_vars


def test_get_torchspec_env_vars_forwards_training_runtime_controls():
    values = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
    }
    with mock.patch.dict("os.environ", values, clear=True):
        actor_env = get_torchspec_env_vars()

    for key, value in values.items():
        assert actor_env[key] == value


def test_get_torchspec_env_vars_does_not_forward_unlisted_values():
    with mock.patch.dict("os.environ", {"UNRELATED_SECRET": "do-not-forward"}, clear=True):
        actor_env = get_torchspec_env_vars()

    assert "UNRELATED_SECRET" not in actor_env
