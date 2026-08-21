# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from torchspec.training.optimizer import BF16Optimizer, split_named_params_for_muon

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused AdamW found_inf handling requires CUDA",
)


def _make_optimizer() -> BF16Optimizer:
    model = torch.nn.Linear(
        2,
        1,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]], device="cuda"))

    return BF16Optimizer(
        model,
        lr=0.1,
        weight_decay=0.1,
        max_grad_norm=10.0,
        total_steps=10,
        warmup_ratio=0.0,
        decay_style="constant",
    )


def _set_grad(optimizer: BF16Optimizer, first_value: float) -> None:
    optimizer.model_params[0].grad = torch.tensor(
        [[first_value, 0.25]],
        device="cuda",
        dtype=torch.bfloat16,
    )


def _initialize_adam_state(optimizer: BF16Optimizer) -> None:
    _set_grad(optimizer, 0.5)
    grad_norm = optimizer.step()
    assert torch.isfinite(grad_norm).item()


def _snapshot(optimizer: BF16Optimizer):
    model_params = [p.detach().clone() for p in optimizer.model_params]
    master_params = [p.detach().clone() for p in optimizer.fp32_params]
    states = []
    for master_param in optimizer.fp32_params:
        states.append(
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.optimizer.state[master_param].items()
            }
        )
    return model_params, master_params, states


def _assert_snapshot_equal(optimizer: BF16Optimizer, snapshot) -> None:
    model_params, master_params, states = snapshot
    for actual, expected in zip(optimizer.model_params, model_params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(optimizer.fp32_params, master_params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for master_param, expected_state in zip(optimizer.fp32_params, states):
        actual_state = optimizer.optimizer.state[master_param]
        assert actual_state.keys() == expected_state.keys()
        for key, expected in expected_state.items():
            actual = actual_state[key]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected


def test_finite_gradient_updates_parameters() -> None:
    optimizer = _make_optimizer()
    master_before = optimizer.fp32_params[0].detach().clone()

    _set_grad(optimizer, 0.5)
    grad_norm = optimizer.step()

    assert torch.isfinite(grad_norm).item()
    assert optimizer.optimizer.found_inf.item() == 0.0
    assert not torch.equal(optimizer.fp32_params[0], master_before)


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_nonfinite_gradient_skips_entire_adamw_update(bad_value: float) -> None:
    optimizer = _make_optimizer()
    _initialize_adam_state(optimizer)
    before = _snapshot(optimizer)

    _set_grad(optimizer, bad_value)
    grad_norm = optimizer.step()

    assert not torch.isfinite(grad_norm).item()
    assert optimizer.optimizer.found_inf.item() == 1.0
    _assert_snapshot_equal(optimizer, before)
    assert all(param.grad is None for param in optimizer.model_params)


def test_remote_nonfinite_signal_skips_local_finite_update() -> None:
    optimizer = _make_optimizer()
    _initialize_adam_state(optimizer)
    before = _snapshot(optimizer)
    _set_grad(optimizer, 0.5)

    def mark_remote_nonfinite(found_inf: torch.Tensor, op) -> None:
        assert op == dist.ReduceOp.MAX
        found_inf.fill_(1.0)

    with (
        mock.patch.object(dist, "is_initialized", return_value=True),
        mock.patch.object(dist, "get_world_size", return_value=2),
        mock.patch.object(dist, "all_reduce", side_effect=mark_remote_nonfinite) as all_reduce,
    ):
        grad_norm = optimizer.step()

    assert torch.isfinite(grad_norm).item()
    assert optimizer.optimizer.found_inf.item() == 1.0
    all_reduce.assert_called_once()
    _assert_snapshot_equal(optimizer, before)


class _MuonFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)
        self.embed_tokens = torch.nn.Embedding(8, 4)


def _make_muon_optimizer() -> BF16Optimizer:
    model = _MuonFixture().to(device="cuda", dtype=torch.bfloat16)
    return BF16Optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        max_grad_norm=10.0,
        total_steps=10,
        warmup_ratio=0.0,
        decay_style="constant",
        optimizer_type="muon",
        muon_lr=3e-3,
        muon_momentum=0.95,
        muon_weight_decay=0.1,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    )


def _set_all_grads(optimizer: BF16Optimizer, value: float) -> None:
    for parameter in optimizer.model_params:
        parameter.grad = torch.full_like(parameter, value)


def test_muon_partition_excludes_embeddings_and_vectors() -> None:
    model = _MuonFixture()
    muon, adamw = split_named_params_for_muon(list(model.named_parameters()))

    assert [name for name, _ in muon] == ["projection.weight"]
    assert [name for name, _ in adamw] == [
        "norm.weight",
        "norm.bias",
        "embed_tokens.weight",
    ]


def test_muon_and_adamw_update_as_one_step() -> None:
    optimizer = _make_muon_optimizer()
    before = [parameter.detach().clone() for parameter in optimizer.fp32_params]
    _set_all_grads(optimizer, 0.25)

    grad_norm = optimizer.step()

    assert torch.isfinite(grad_norm).item()
    assert optimizer.found_inf.item() == 0.0
    assert set(optimizer.optimizers) == {"muon", "adamw"}
    assert optimizer.muon_parameter_names == ["projection.weight"]
    assert optimizer.adamw_parameter_names == [
        "norm.weight",
        "norm.bias",
        "embed_tokens.weight",
    ]
    assert optimizer.muon_parameter_count == 16
    assert optimizer.adamw_parameter_count == 40
    assert all(
        not torch.equal(actual, original) for actual, original in zip(optimizer.fp32_params, before)
    )
    assert optimizer.get_optimizer_learning_rates() == {
        "muon": pytest.approx(3e-3),
        "adamw": pytest.approx(1e-3),
    }


def test_nonfinite_muon_step_skips_both_optimizers() -> None:
    optimizer = _make_muon_optimizer()
    _set_all_grads(optimizer, 0.25)
    optimizer.step()
    before = [parameter.detach().clone() for parameter in optimizer.fp32_params]
    _set_all_grads(optimizer, 0.25)
    optimizer.model_params[0].grad.flatten()[0] = float("nan")

    grad_norm = optimizer.step()

    assert not torch.isfinite(grad_norm).item()
    assert optimizer.found_inf.item() == 1.0
    for actual, original in zip(optimizer.fp32_params, before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in optimizer.model_params)


def test_muon_optimizer_and_scheduler_state_round_trip() -> None:
    first = _make_muon_optimizer()
    _set_all_grads(first, 0.25)
    first.step()
    optimizer_state = first.state_dict()
    scheduler_state = first.lr_scheduler.state_dict()

    second = _make_muon_optimizer()
    with mock.patch("torchspec.training.optimizer.print_on_rank0"):
        second.load_state_dict(optimizer_state)
    second.lr_scheduler.load_state_dict(scheduler_state)

    assert second.state_dict()["optimizers"].keys() == optimizer_state["optimizers"].keys()
    assert second.lr_scheduler.state_dict() == scheduler_state
    assert len(second.state) == len(first.state)
