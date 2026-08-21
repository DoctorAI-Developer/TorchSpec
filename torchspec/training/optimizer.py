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

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from torchspec.training.lr_scheduler import LRSchedulerWithWarmup
from torchspec.utils.logging import print_on_rank0

_ADAMW_NAME_HINTS = ("embed_tokens", "lm_head")


def split_named_params_for_muon(
    named_parameters: list[tuple[str, Tensor]],
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Route eligible matrices to Muon and all other parameters to AdamW."""
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, parameter in named_parameters:
        if (
            parameter.ndim == 2
            and min(parameter.shape) > 1
            and not any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            muon_params.append((name, parameter))
        else:
            adamw_params.append((name, parameter))
    return muon_params, adamw_params


class _OptimizerCollection:
    """Checkpoint-compatible view over independently configured optimizers."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("optimizer collection cannot be empty")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    @property
    def state(self) -> dict:
        return {
            parameter: value
            for optimizer in self.optimizers.values()
            for parameter, value in optimizer.state.items()
        }

    def clear_state(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.state.clear()

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        states = state_dict.get("optimizers")
        if not isinstance(states, dict) or set(states) != set(self.optimizers):
            raise ValueError(
                "optimizer collection checkpoint keys do not match the configured "
                f"optimizers: {sorted(states or {})} != {sorted(self.optimizers)}"
            )
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(states[name])

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)


class _SchedulerCollection:
    """Stateful scheduler facade used by the distributed checkpoint layer."""

    def __init__(self, schedulers: dict[str, LRSchedulerWithWarmup]) -> None:
        if not schedulers:
            raise ValueError("scheduler collection cannot be empty")
        self.schedulers = schedulers

    def step(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "schedulers": {
                name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        states = state_dict.get("schedulers")
        if not isinstance(states, dict) or set(states) != set(self.schedulers):
            raise ValueError(
                "scheduler collection checkpoint keys do not match the configured "
                f"schedulers: {sorted(states or {})} != {sorted(self.schedulers)}"
            )
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(states[name])


class BF16Optimizer:
    def __init__(
        self,
        model: Module,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        min_lr=0.0,
        wsd_decay_steps=None,
        wsd_decay_style=None,
        optimizer_type="adamw",
        muon_lr=None,
        muon_momentum=0.95,
        muon_weight_decay=0.1,
        muon_ns_steps=5,
        muon_adjust_lr_fn="match_rms_adamw",
    ):
        self.model = model
        named_model_params = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.model_param_names = [name for name, _ in named_model_params]
        self.model_params = [parameter for _, parameter in named_model_params]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer_type = optimizer_type
        self.optimizers: dict[str, torch.optim.Optimizer]
        self.schedulers: dict[str, LRSchedulerWithWarmup]
        scheduler_kwargs = {
            "total_steps": total_steps,
            "warmup_steps": int(warmup_ratio * total_steps),
            "decay_style": decay_style,
            "wsd_decay_steps": wsd_decay_steps,
            "wsd_decay_style": wsd_decay_style,
        }
        if optimizer_type == "adamw":
            adamw = torch.optim.AdamW(
                self.fp32_params,
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
            )
            if not getattr(adamw, "_step_supports_amp_scaling", False):
                raise RuntimeError(
                    "BF16Optimizer requires fused AdamW with device-side found_inf support"
                )
            self.optimizer = adamw
            self.optimizers = {"adamw": adamw}
            self.scheduler = LRSchedulerWithWarmup(
                adamw,
                max_lr=lr,
                min_lr=min_lr,
                **scheduler_kwargs,
            )
            self.schedulers = {"adamw": self.scheduler}
        elif optimizer_type == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("optimizer_type=muon requires torch.optim.Muon")
            resolved_muon_lr = 10.0 * lr if muon_lr is None else float(muon_lr)
            named_master_params = list(zip(self.model_param_names, self.fp32_params))
            muon_params, adamw_params = split_named_params_for_muon(named_master_params)
            if not muon_params:
                raise ValueError("optimizer_type=muon found no eligible 2D matrices")
            self.optimizers = {
                "muon": torch.optim.Muon(
                    muon_params,
                    lr=resolved_muon_lr,
                    momentum=muon_momentum,
                    weight_decay=muon_weight_decay,
                    ns_steps=muon_ns_steps,
                    adjust_lr_fn=muon_adjust_lr_fn,
                )
            }
            if adamw_params:
                self.optimizers["adamw"] = torch.optim.AdamW(
                    adamw_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    fused=True,
                )
            self.optimizer = _OptimizerCollection(self.optimizers)
            muon_min_lr = min_lr * resolved_muon_lr / lr
            self.schedulers = {
                name: LRSchedulerWithWarmup(
                    optimizer,
                    max_lr=resolved_muon_lr if name == "muon" else lr,
                    min_lr=muon_min_lr if name == "muon" else min_lr,
                    **scheduler_kwargs,
                )
                for name, optimizer in self.optimizers.items()
            }
            self.scheduler = _SchedulerCollection(self.schedulers)
            self.muon_parameter_names = [name for name, _ in muon_params]
            self.adamw_parameter_names = [name for name, _ in adamw_params]
            self.muon_parameter_count = sum(parameter.numel() for _, parameter in muon_params)
            self.adamw_parameter_count = sum(parameter.numel() for _, parameter in adamw_params)
        else:
            raise ValueError(f"optimizer_type must be adamw or muon, got {optimizer_type!r}")

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            grad_destinations = []
            grad_sources = []
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    grad_destinations.append(g)
                    grad_sources.append(p.grad)
                    mp.grad = g
                else:
                    mp.grad = None
            if grad_destinations:
                torch._foreach_copy_(grad_destinations, grad_sources)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)

        found_inf = (~torch.isfinite(grad_norm)).to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # A sharded rank may be the only one that observes a nonfinite
            # gradient. All training ranks must make the same update decision.
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
        self.found_inf = found_inf
        if self.optimizer_type == "adamw":
            # Fused AdamW consumes this device scalar through the same path used
            # by GradScaler and avoids a device-to-host synchronization.
            self.optimizer.found_inf = found_inf
            self.optimizer.step()
        elif not bool(found_inf.item()):
            # Muon has no AMP found_inf hook. One synchronized branch per long
            # training step guarantees that Muon and AdamW update together or
            # both skip after the distributed nonfinite reduction.
            for optimizer in self.optimizers.values():
                optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            torch._foreach_copy_(self.model_params, self.fp32_params)
            for p in self.model_params:
                p.grad = None

        return grad_norm

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            torch._foreach_copy_(self.fp32_params, self.model_params)

    def state_dict(self):
        return self.optimizer.state_dict()

    def get_learning_rate(self):
        optimizer = self.optimizers.get("adamw") or next(iter(self.optimizers.values()))
        return optimizer.param_groups[0]["lr"]

    def get_optimizer_learning_rates(self) -> dict[str, float]:
        return {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in self.optimizers.items()
        }

    def optimizer_param_groups(self) -> list[dict]:
        return self.optimizer.param_groups

    def clear_optimizer_state(self) -> None:
        if isinstance(self.optimizer, _OptimizerCollection):
            self.optimizer.clear_state()
        else:
            self.optimizer.state.clear()

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
