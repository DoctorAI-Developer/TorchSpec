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

"""DFlash2 training wrapper."""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchspec.models.dflash import (
    DFlashModel,
    _bernoulli_forward_kl_loss,
    _likelihood_overlap_loss,
    _total_variation_loss,
)


class DFlash2Model(DFlashModel):
    def __init__(
        self,
        *args,
        selector_loss_alpha: float = 1.0,
        logits_chunk_size: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        selector_loss_alpha = float(selector_loss_alpha)
        if not math.isfinite(selector_loss_alpha) or selector_loss_alpha <= 0:
            raise ValueError(
                f"dflash2_selector_loss_alpha must be positive, got {selector_loss_alpha}"
            )
        self.selector_loss_alpha = selector_loss_alpha
        self.logits_chunk_size = int(logits_chunk_size)
        if self.logits_chunk_size < 0:
            raise ValueError(f"logits_chunk_size must be non-negative, got {logits_chunk_size}")

        config = self.draft_model.config
        layer_types = list(getattr(config, "layer_types", []) or [])
        if layer_types and len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "DFlash2 layer_types must contain one entry per draft layer, got "
                f"{len(layer_types)} for {config.num_hidden_layers} layers"
            )
        attention_types = set(layer_types or ["full_attention"])
        if not attention_types <= {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported DFlash2 layer types: {sorted(attention_types)}")
        if len(attention_types) > 1:
            raise ValueError("DFlash2 training does not support mixed full and sliding layers")

        uses_sliding_window = "sliding_attention" in attention_types
        explicit_causality = getattr(config, "is_causal", None)
        self.attention_is_causal = (
            uses_sliding_window if explicit_causality is None else bool(explicit_causality)
        )
        configured_window = getattr(config, "sliding_window", None)
        self.sliding_window = (
            int(configured_window)
            if uses_sliding_window and configured_window is not None
            else None
        )
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")

    def _block_mask_options(self) -> dict:
        return {
            "is_causal": self.attention_is_causal,
            "sliding_window": self.sliding_window,
        }

    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        logits = super()._compute_logits(draft_hidden, lm_head_weight)
        logits = logits * self.draft_model.config.output_multiplier
        softcap = self.draft_model.config.final_logit_softcapping
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _compute_token_statistics(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        aligned_target_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        total_tokens = draft_hidden.shape[1]
        if self.logits_chunk_size == 0 or total_tokens <= self.logits_chunk_size:
            return super()._compute_token_statistics(
                draft_hidden,
                lm_head_weight,
                target_ids,
                aligned_target_hidden,
            )

        batch, num_blocks, block_size = target_ids.shape
        if block_size != self.block_size:
            raise ValueError(
                f"target block size {block_size} does not match model block size {self.block_size}"
            )
        blocks_per_chunk = max(1, self.logits_chunk_size // block_size)
        hidden_blocks = draft_hidden.reshape(batch, num_blocks, block_size, -1)
        target_hidden_blocks = (
            None
            if aligned_target_hidden is None
            else aligned_target_hidden.reshape(batch, num_blocks, block_size, -1)
        )
        ce_chunks = []
        pred_chunks = []
        selector_ce_chunks = []
        distribution_loss_chunks = []

        def compute_chunk(
            hidden: torch.Tensor,
            targets: torch.Tensor,
            target_hidden: torch.Tensor | None,
        ):
            chunk_blocks = targets.shape[1]
            chunk_logits = self._compute_logits(
                hidden.reshape(batch, chunk_blocks * block_size, -1),
                lm_head_weight,
            ).reshape(batch, chunk_blocks, block_size, -1)
            flat_logits = chunk_logits.reshape(-1, chunk_logits.shape[-1])
            flat_targets = targets.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape_as(targets)
            pred = torch.argmax(flat_logits, dim=-1).reshape_as(targets)

            selector = self.draft_model.candidate_selector
            scores, candidate_ids = selector.score_candidates(
                hidden[..., 1:, :],
                chunk_logits[..., 1:, :],
                targets[..., :-1],
                training_successor_ids=targets[..., 1:],
            )
            matches = candidate_ids == targets[..., 1:].unsqueeze(-1)
            target_indices = matches.to(torch.int64).argmax(dim=-1)
            selector_ce = F.cross_entropy(
                scores.reshape(-1, scores.shape[-1]),
                target_indices.reshape(-1),
                reduction="none",
            ).reshape_as(target_indices)
            distribution_loss = ce.new_empty(0)
            if self.loss_objective in {"lk", "opd", "path", "tv"}:
                if target_hidden is None:
                    raise ValueError(
                        "DFlash distribution training requires aligned target hidden states"
                    )
                target_logits = F.linear(target_hidden, lm_head_weight)
                if self.loss_objective == "tv":
                    distribution_loss = _total_variation_loss(chunk_logits, target_logits)
                elif self.loss_objective == "opd":
                    distribution_loss = _bernoulli_forward_kl_loss(
                        chunk_logits, target_logits, targets
                    )
                else:
                    distribution_loss = _likelihood_overlap_loss(chunk_logits, target_logits)
            return ce, pred, selector_ce, distribution_loss

        for start in range(0, num_blocks, blocks_per_chunk):
            stop = min(start + blocks_per_chunk, num_blocks)
            hidden_chunk = hidden_blocks[:, start:stop]
            target_chunk = target_ids[:, start:stop]
            target_hidden_chunk = (
                None if target_hidden_blocks is None else target_hidden_blocks[:, start:stop]
            )
            if self.training and hidden_chunk.requires_grad:
                ce, pred, selector_ce, distribution_loss = torch_checkpoint(
                    compute_chunk,
                    hidden_chunk,
                    target_chunk,
                    target_hidden_chunk,
                    use_reentrant=False,
                )
            else:
                ce, pred, selector_ce, distribution_loss = compute_chunk(
                    hidden_chunk,
                    target_chunk,
                    target_hidden_chunk,
                )
            ce_chunks.append(ce)
            pred_chunks.append(pred)
            selector_ce_chunks.append(selector_ce)
            if self.loss_objective in {"lk", "opd", "path", "tv"}:
                distribution_loss_chunks.append(distribution_loss)

        ce_per_token = torch.cat(ce_chunks, dim=1).reshape(-1)
        pred_ids = torch.cat(pred_chunks, dim=1).reshape(-1)
        selector_ce = torch.cat(selector_ce_chunks, dim=1)
        distribution_loss = (
            torch.cat(distribution_loss_chunks, dim=1).reshape(-1)
            if distribution_loss_chunks
            else None
        )
        return ce_per_token, pred_ids, selector_ce, distribution_loss

    def _selected_token_log_probs(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.logits_chunk_size == 0 or hidden_states.shape[0] <= self.logits_chunk_size:
            return super()._selected_token_log_probs(hidden_states, lm_head_weight, token_ids)
        chunks = []
        for start in range(0, hidden_states.shape[0], self.logits_chunk_size):
            stop = min(start + self.logits_chunk_size, hidden_states.shape[0])
            chunks.append(
                super()._selected_token_log_probs(
                    hidden_states[start:stop],
                    lm_head_weight,
                    token_ids[start:stop],
                )
            )
        return torch.cat(chunks)

    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
        native_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # Spec-AUF changes the unary token CE support only. DFlash2's candidate
        # selector is a separate auxiliary objective, so retain its native-valid
        # supervision rather than silently truncating it after the first unary
        # mismatch. Other objectives preserve their historical weighting.
        selector_weights = native_weights if self.loss_objective == "auf" else objective_weights
        eligible_weights = selector_weights[..., 1:]
        eligible_weights = eligible_weights * (eligible_weights > 0).cumprod(dim=-1)
        if logits.shape != eligible_weights.shape:
            batch, num_blocks, block_size = target_ids.shape
            hidden = draft_hidden.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
            unary_logits = logits.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
            predecessor_ids = target_ids[..., :-1]
            successor_ids = target_ids[..., 1:]
            scores, candidate_ids = self.draft_model.candidate_selector.score_candidates(
                hidden,
                unary_logits,
                predecessor_ids,
                training_successor_ids=successor_ids,
            )
            matches = candidate_ids == successor_ids.unsqueeze(-1)
            target_indices = matches.to(torch.int64).argmax(dim=-1)
            selector_ce = F.cross_entropy(
                scores.reshape(-1, scores.shape[-1]),
                target_indices.reshape(-1),
                reduction="none",
            ).reshape_as(eligible_weights)
        else:
            selector_ce = logits
        selector_num = (selector_ce * eligible_weights).sum()
        selector_den = eligible_weights.sum().detach()
        return self.selector_loss_alpha * selector_num, {
            "selector_loss": (selector_num.detach(), selector_den),
        }
