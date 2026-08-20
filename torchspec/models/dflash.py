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

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask
from torchspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {
    "auf",
    "decay",
    "dpace",
    "lk",
    "opd",
    "path",
    "tv",
}


def _distribution_overlap(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Return ``sum_x min(p(x), q(x))`` with a detached FP32 target."""
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            "draft and target logits must have identical shapes, got "
            f"{tuple(draft_logits.shape)} and {tuple(target_logits.shape)}"
        )

    with torch.no_grad():
        target_probs = torch.softmax(target_logits.float(), dim=-1)
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    return torch.minimum(target_probs, draft_probs).sum(dim=-1)


def _likelihood_overlap_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Return Kimi K3's per-token likelihood-overlap loss at temperature 1.

    The overlap ``sum_x min(p(x), q(x))`` is the expected one-step acceptance
    probability of exact speculative sampling.  Unlike token cross entropy,
    maximizing it therefore optimizes the quantity used by the verifier
    directly.  The verifier distribution is a frozen target; gradients flow
    only through the draft distribution.
    """
    overlap = _distribution_overlap(draft_logits, target_logits)
    return -torch.log(overlap.clamp_min(torch.finfo(overlap.dtype).tiny))


def _total_variation_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Directly maximize one-step acceptance via total variation distance."""
    return 1.0 - _distribution_overlap(draft_logits, target_logits)


def _bernoulli_forward_kl_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward KL for the sampled target-token event and its complement.

    Draft-OPD rollouts are target-distributed.  At an accepted/error-position
    replay label, the emitted target token therefore supplies an on-policy
    forward-KL sample.  The local Bernoulli form matches Draft-OPD's public
    implementation while requiring only the selected token probability from
    each normalized full-vocabulary distribution.
    """

    if draft_logits.shape != target_logits.shape:
        raise ValueError("draft and target logits must have identical shapes")
    if target_ids.shape != draft_logits.shape[:-1]:
        raise ValueError("target IDs must match the logits prefix shape")
    draft_log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    with torch.no_grad():
        target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)
    gather_ids = target_ids.to(dtype=torch.long).unsqueeze(-1)
    draft_logp = torch.gather(draft_log_probs, -1, gather_ids).squeeze(-1)
    target_logp = torch.gather(target_log_probs, -1, gather_ids).squeeze(-1)

    eps = torch.finfo(draft_logp.dtype).eps
    max_log_prob = torch.log(draft_logp.new_tensor(1.0 - eps))
    draft_logp = draft_logp.clamp(min=-80.0, max=max_log_prob)
    target_logp = target_logp.clamp(min=-80.0, max=max_log_prob)
    draft_prob = draft_logp.exp()
    target_prob = target_logp.exp()
    return target_prob * (target_logp - draft_logp) + (1.0 - target_prob) * (
        torch.log1p(-target_prob) - torch.log1p(-draft_prob)
    )


def _auf_position_mask(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep valid positions through each block's first detached greedy failure.

    This is the Accept-Until-Fail support from Spec-AUF (arXiv:2607.01893):
    the accepted prefix and its first failing ("breaker") token remain active,
    while the suffix after that failure receives no loss. Fully correct blocks
    retain every natively valid position.
    """
    if predictions.shape != targets.shape or predictions.shape != valid_mask.shape:
        raise ValueError("AUF predictions, targets, and valid_mask must have identical shapes")
    with torch.no_grad():
        valid = valid_mask.bool()
        mismatch = valid & predictions.detach().ne(targets)
        prior_failures = mismatch.to(torch.int32).cumsum(dim=-1) - mismatch.to(torch.int32)
        return valid & prior_failures.eq(0)


def _dpace_position_weights(
    confidences: torch.Tensor,
    alpha: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        if valid_mask is not None:
            valid_prefix = valid_mask.to(torch.int32).cumprod(dim=-1).bool()
            smoothed = torch.where(valid_prefix, smoothed, torch.zeros_like(smoothed))
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _path_overlap_position_weights(
    likelihood_losses: torch.Tensor,
    alpha: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight likelihood overlap by its smoothed expected-path contribution.

    Kimi's likelihood-overlap loss supplies the exact one-step speculative
    acceptance proxy ``a_j = sum_x min(p_j(x), q_j(x))``. D-PACE derives the
    path value of position ``j`` as the suffix sum of prefix products. This
    helper combines the two while retaining D-PACE's detached asymmetric
    smoothing so weak suffix positions do not lose all training signal.
    """
    overlaps = torch.exp(-likelihood_losses.float())
    weights = _dpace_position_weights(overlaps, alpha, valid_mask)
    return weights.to(dtype=likelihood_losses.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    is_causal: bool = False,
    sliding_window: int | None = None,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention follows is_causal and sliding_window
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        q_offset = q_idx % block_size
        kv_offset = (kv_idx - ctx_len) % block_size
        if is_causal:
            mask_draft = mask_draft & (kv_offset <= q_offset)
        if sliding_window is not None:
            query_position = anchor_pos + q_offset
            mask_context = mask_context & (query_position - kv_idx < sliding_window)
            mask_draft = mask_draft & (
                abs(query_position - (anchor_pos + kv_offset)) < sliding_window
            )

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        opd_rejected_stream_weight: float = 1.0,
        opd_rejected_position_decay: float = 0.8,
        opd_rejected_k3_preserve_negative_tail: bool = False,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.opd_rejected_stream_weight = float(opd_rejected_stream_weight)
        self.opd_rejected_position_decay = float(opd_rejected_position_decay)
        self.opd_rejected_k3_preserve_negative_tail = bool(
            opd_rejected_k3_preserve_negative_tail
        )
        if self.opd_rejected_stream_weight < 0:
            raise ValueError("opd_rejected_stream_weight must be non-negative")
        if not 0 < self.opd_rejected_position_decay <= 1:
            raise ValueError("opd_rejected_position_decay must be in (0, 1]")
        if self.loss_objective in {"lk", "opd", "path", "tv"}:
            objective_name = self.loss_objective.upper()
            if self.ce_loss_alpha != 0:
                raise ValueError(
                    f"DFlash {objective_name} uses distribution overlap as the complete "
                    "unary objective; "
                    "set dflash_ce_loss_alpha=0"
                )
            if self.l1_loss_alpha != 0:
                raise ValueError(
                    f"DFlash {objective_name} cannot be combined with L1 distillation; "
                    "set dflash_l1_loss_alpha=0"
                )

    @property
    def uses_target_hidden_states(self) -> bool:
        return self.l1_loss_alpha > 0 or self.loss_objective in {
            "lk",
            "opd",
            "path",
            "tv",
        }

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns exactly ``self.num_anchors`` anchor slots so that
        ``Q_LEN = num_anchors * block_size`` is constant across steps,
        preventing FlexAttention recompilation from shape changes.  Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        excess slots (those blocks are skipped by the block-sparse kernel).

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
            device: torch device

        Returns:
            anchors: [B, num_anchors] — sampled anchor positions (sorted)
            keep_mask: [B, num_anchors] — True for valid sampled anchors
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        max_n = self.num_anchors

        if max_anchor == 0:
            logger.warning(
                f"Sequence too short for anchor sampling (seq_len={seq_len}, "
                f"block_size={bs}). Returning dummy anchors so loss is zero."
            )
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        # An anchor is only usable if its own position and the position right
        # after it are both supervised: the block's first prediction target is
        # ``anchor + 1``, so an isolated supervised token yields no gradient.
        num_candidates = min(max_anchor + 1, seq_len - 1)
        valid = (loss_mask[:, :num_candidates] > 0.5) & (loss_mask[:, 1 : num_candidates + 1] > 0.5)
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take up to num_anchors slots; pad with zeros if fewer valid positions
        take_n = min(max_n, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < max_n:
            pad = torch.zeros(bsz, max_n - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _prepare_opd_anchor_plan(
        self,
        *,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        anchor_positions: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
        segment_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Validate and pad a recorded error-position replay plan."""

        values = (anchor_positions, anchor_mask, segment_lengths)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("DFlash OPD anchor positions, mask, and segments are atomic")
        anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
        anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
        segment_lengths = segment_lengths.to(device=device, dtype=torch.long)
        if not (
            anchor_positions.shape == anchor_mask.shape == segment_lengths.shape
            and anchor_positions.dim() == 2
            and anchor_positions.shape[0] == batch_size
        ):
            raise ValueError("DFlash OPD anchor tensors must be matching [batch, anchors]")
        if anchor_positions.shape[1] > self.num_anchors:
            raise ValueError(
                "DFlash OPD replay contains more anchors than dflash_num_anchors: "
                f"{anchor_positions.shape[1]} > {self.num_anchors}"
            )
        if bool((anchor_mask & ((anchor_positions < 0) | (anchor_positions >= seq_len))).any()):
            raise ValueError("DFlash OPD anchor position is outside the sequence")
        if bool(
            (anchor_mask & ((segment_lengths < 0) | (segment_lengths >= self.block_size))).any()
        ):
            raise ValueError("DFlash OPD segment length is outside the draft block")
        for row_positions, row_mask in zip(anchor_positions, anchor_mask, strict=True):
            valid_positions = row_positions[row_mask]
            if valid_positions.numel() != torch.unique(valid_positions).numel():
                raise ValueError("DFlash OPD replay contains duplicate anchors")

        anchor_positions = torch.where(
            anchor_mask, anchor_positions, torch.zeros_like(anchor_positions)
        )
        segment_lengths = torch.where(
            anchor_mask, segment_lengths, torch.zeros_like(segment_lengths)
        )
        return anchor_positions, anchor_mask, segment_lengths

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def _block_mask_options(self) -> dict:
        return {}

    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.draft_model, "lm_head"):
            return self.draft_model.lm_head(draft_hidden)
        return F.linear(draft_hidden, lm_head_weight)

    def _compute_token_statistics(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        aligned_target_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return per-token CE, predicted IDs, logits payload, and distribution loss.

        Subclasses may return a compact third payload when their auxiliary
        objective can be computed without retaining the full vocabulary
        matrix. The base DFlash L1 path still requires full logits.  The fourth
        value is populated only for LK or direct total-variation training.
        """
        logits = self._compute_logits(draft_hidden, lm_head_weight)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
        distribution_loss = None
        if self.loss_objective in {"lk", "opd", "path", "tv"}:
            if aligned_target_hidden is None:
                raise ValueError(
                    "DFlash distribution training requires aligned target hidden states"
                )
            target_logits = F.linear(aligned_target_hidden, lm_head_weight)
            target_logits = target_logits.reshape_as(flat_logits)
            if self.loss_objective == "tv":
                distribution_loss = _total_variation_loss(flat_logits, target_logits)
            elif self.loss_objective == "opd":
                distribution_loss = _bernoulli_forward_kl_loss(
                    flat_logits, target_logits, flat_targets
                )
            else:
                distribution_loss = _likelihood_overlap_loss(flat_logits, target_logits)
        return ce_per_token, pred_ids, logits, distribution_loss

    def _selected_token_log_probs(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the full vocabulary and return selected-token logprobs."""

        if hidden_states.dim() != 2 or token_ids.shape != hidden_states.shape[:1]:
            raise ValueError("selected hidden states and token IDs must be [tokens, hidden]")
        logits = self._compute_logits(hidden_states, lm_head_weight)
        return torch.gather(
            torch.log_softmax(logits.float(), dim=-1),
            -1,
            token_ids.to(dtype=torch.long).unsqueeze(-1),
        ).squeeze(-1)

    def _opd_rejected_loss(
        self,
        *,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        rejected_anchor_positions: torch.Tensor | None,
        rejected_offsets: torch.Tensor | None,
        rejected_token_ids: torch.Tensor | None,
        rejected_teacher_logprobs: torch.Tensor | None,
        rejected_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Replay rejected draft suffixes with Draft-OPD's reverse-KL K3 loss."""

        rejected_values = (
            rejected_anchor_positions,
            rejected_offsets,
            rejected_token_ids,
            rejected_teacher_logprobs,
            rejected_mask,
        )
        zero = draft_hidden.new_zeros((), dtype=torch.float32)
        if all(value is None for value in rejected_values):
            if self.loss_objective == "opd":
                raise ValueError("DFlash OPD objective requires rejected draft metadata")
            return zero, zero.detach(), {}
        if any(value is None for value in rejected_values):
            raise ValueError("DFlash OPD rejected draft metadata is atomic")
        if self.loss_objective != "opd":
            raise ValueError("rejected OPD tokens require dflash_loss_objective=opd")

        device = draft_hidden.device
        rejected_anchor_positions = rejected_anchor_positions.to(device=device, dtype=torch.long)
        rejected_offsets = rejected_offsets.to(device=device, dtype=torch.long)
        rejected_token_ids = rejected_token_ids.to(device=device, dtype=torch.long)
        rejected_teacher_logprobs = rejected_teacher_logprobs.to(device=device, dtype=torch.float32)
        rejected_mask = rejected_mask.to(device=device, dtype=torch.bool)
        rejected_shape = rejected_mask.shape
        if rejected_mask.dim() != 2 or any(
            value.shape != rejected_shape
            for value in (
                rejected_anchor_positions,
                rejected_offsets,
                rejected_token_ids,
                rejected_teacher_logprobs,
            )
        ):
            raise ValueError("DFlash OPD rejected tensors must be matching [batch, tokens]")
        if rejected_shape[0] != draft_hidden.shape[0]:
            raise ValueError("DFlash OPD rejected batch size does not match draft hidden states")
        if not bool(rejected_mask.any()):
            return zero, zero.detach(), {"opd_rejected_loss": (zero.detach(), zero.detach())}
        if bool(
            (
                rejected_mask & ((rejected_offsets <= 0) | (rejected_offsets >= self.block_size))
            ).any()
        ):
            raise ValueError("DFlash OPD rejected offset is outside the draft block")
        vocab_size = lm_head_weight.shape[0]
        if bool(
            (rejected_mask & ((rejected_token_ids < 0) | (rejected_token_ids >= vocab_size))).any()
        ):
            raise ValueError("DFlash OPD rejected token ID is outside the vocabulary")
        if bool(
            (
                rejected_mask
                & (~torch.isfinite(rejected_teacher_logprobs) | (rejected_teacher_logprobs > 1e-6))
            ).any()
        ):
            raise ValueError("DFlash OPD rejected teacher logprob is invalid")

        anchor_matches = (
            anchor_positions.unsqueeze(-1) == rejected_anchor_positions.unsqueeze(1)
        ) & block_keep_mask.unsqueeze(-1)
        match_count = anchor_matches.sum(dim=1)
        if bool((rejected_mask & match_count.ne(1)).any()):
            raise ValueError("DFlash OPD rejected token does not match exactly one anchor")
        block_indices = anchor_matches.to(dtype=torch.long).argmax(dim=1)
        draft_indices = block_indices * self.block_size + rejected_offsets
        safe_draft_indices = torch.where(
            rejected_mask, draft_indices, torch.zeros_like(draft_indices)
        )
        hidden_size = draft_hidden.shape[-1]
        selected_hidden = torch.gather(
            draft_hidden,
            1,
            safe_draft_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )[rejected_mask]
        selected_token_ids = rejected_token_ids[rejected_mask]
        student_logprobs = self._selected_token_log_probs(
            selected_hidden,
            lm_head_weight,
            selected_token_ids,
        )
        teacher_logprobs = rejected_teacher_logprobs[rejected_mask]

        # Schulman's non-negative K3 estimator, matching Draft-OPD. These
        # rejected tokens were sampled by the draft distribution, so reverse
        # KL is the correct on-policy direction for this stream.
        raw_delta = teacher_logprobs - student_logprobs
        clipped_delta = raw_delta.clamp(min=-20.0, max=20.0)
        rejected_losses = (
            clipped_delta.exp() - clipped_delta - 1.0
        ).clamp(min=-10.0, max=10.0)
        if self.opd_rejected_k3_preserve_negative_tail:
            # K3 = exp(delta) - delta - 1. For delta <= 0 its value approaches
            # -delta - 1 and d(loss)/d(student_logprob) is bounded in [0, 1].
            # Restore that stable tail for top-k-rejected tokens while retaining
            # the published clamps for the overflow-prone positive-ratio tail.
            negative_delta = raw_delta.clamp(max=0.0)
            negative_tail_losses = (
                negative_delta.exp() - negative_delta - 1.0
            )
            rejected_losses = torch.where(
                raw_delta < 0.0,
                negative_tail_losses,
                rejected_losses,
            )
        offsets = rejected_offsets[rejected_mask].to(dtype=torch.float32)
        weights = torch.pow(
            offsets.new_tensor(self.opd_rejected_position_decay),
            (offsets - 1.0).clamp_min(0.0),
        )
        numerator = (rejected_losses * weights).sum()
        denominator = weights.sum().detach()
        return numerator, denominator, {"opd_rejected_loss": (numerator.detach(), denominator)}

    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
        native_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        return logits.new_zeros(()), {}

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Shared DFlash backbone (context features → anchor sampling → noise
        embedding → position ids → block-causal mask → draft model forward).

        Both ``DFlashModel.forward`` and ``DSparkModel.forward`` build the draft
        hidden states this exact way; only the label/loss tail differs. Keeping
        the attention/mask/anchor wiring here gives it a single source of truth.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Extract context features from target hidden states
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions with validity mask
        if anchor_positions is None and block_keep_mask is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        elif anchor_positions is None or block_keep_mask is None:
            raise ValueError("explicit DFlash anchors and keep mask must be provided together")
        n_blocks = anchor_positions.shape[1]

        # 3. Create noise embeddings (anchor token + MASK tokens)
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        # 5. Create block-causal attention mask
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
                **self._block_mask_options(),
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        # 6. Draft model forward — pass embeddings directly
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

        return draft_hidden, anchor_positions, block_keep_mask, n_blocks

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        opd_anchor_positions: Optional[torch.Tensor] = None,
        opd_anchor_mask: Optional[torch.Tensor] = None,
        opd_segment_lengths: Optional[torch.Tensor] = None,
        opd_rejected_anchor_positions: Optional[torch.Tensor] = None,
        opd_rejected_offsets: Optional[torch.Tensor] = None,
        opd_rejected_token_ids: Optional[torch.Tensor] = None,
        opd_rejected_teacher_logprobs: Optional[torch.Tensor] = None,
        opd_rejected_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Full DFlash training forward pass.

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
            loss_components: dict of extra per-component ``(numerator, denominator)``
                pairs for logging, pooled by the trainer just like ``loss_terms``
                (empty for the base DFlash objective; populated by subclasses).
            loss_terms: additive objective numerator and denominator. The trainer
                pools these over the full accumulation window and data-parallel group.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping.
        opd_plan = self._prepare_opd_anchor_plan(
            seq_len=seq_len,
            batch_size=bsz,
            device=device,
            anchor_positions=opd_anchor_positions,
            anchor_mask=opd_anchor_mask,
            segment_lengths=opd_segment_lengths,
        )
        active_segment_lengths = None
        explicit_anchor_positions = None
        explicit_anchor_mask = None
        if opd_plan is not None:
            explicit_anchor_positions, explicit_anchor_mask, active_segment_lengths = opd_plan
        if self.loss_objective == "opd" and opd_plan is None:
            raise ValueError("DFlash OPD objective requires recorded error-position anchors")
        if self.loss_objective != "opd" and opd_plan is not None:
            raise ValueError("recorded OPD anchors require dflash_loss_objective=opd")
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            anchor_positions=explicit_anchor_positions,
            block_keep_mask=explicit_anchor_mask,
        )

        # 7. Compute labels and weight mask (SpecForge pattern)
        # Labels: same-position prediction (position k predicts token at anchor+k)
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        aligned_target_hidden = None
        if self.uses_target_hidden_states:
            if last_hidden_states is None:
                requirement = (
                    f"DFlash {self.loss_objective.upper()}"
                    if self.loss_objective in {"lk", "opd", "path", "tv"}
                    else "DFlash L1 distillation (l1_loss_alpha > 0)"
                )
                raise ValueError(
                    f"{requirement} requires target last_hidden_states; set "
                    "inference.store_last_hidden_states=true in the run config."
                )
            tgt_idx = (safe_label_indices - 1).clamp(min=0)
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_target_hidden = torch.gather(last_hidden_states, 1, gather_idx)

        # 8. Project through the frozen LM head. DFlash2 can override this hook
        # with an exact chunked path to bound the full-vocabulary peak.
        ce_per_token, pred_ids, logits, distribution_loss = self._compute_token_statistics(
            draft_hidden,
            lm_head_weight,
            target_ids,
            aligned_target_hidden,
        )

        # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        if active_segment_lengths is not None:
            weight_mask = weight_mask * (pos_in_block <= active_segment_lengths.unsqueeze(-1)).to(
                dtype=weight_mask.dtype
            )

        # Gather original loss_mask at label positions
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution. SpecForge uses no decay at all;
        # our objective weighting is an addition to the training signal, not the metric.
        binary_eval_mask = weight_mask.view(-1)

        # 9. Per-token unary objective. Distribution objectives replace CE/L1
        # rather than blending them. LK is Kimi K3's geometric objective; TV
        # directly maximizes arithmetic one-step acceptance; path adds
        # D-PACE-style expected-prefix value to LK below.
        flat_targets = target_ids.view(-1)

        if self.loss_objective in {"lk", "opd", "path", "tv"}:
            if distribution_loss is None:
                raise RuntimeError(
                    "DFlash distribution token statistics did not return overlap loss"
                )
            loss_per_token = distribution_loss
        else:
            loss_per_token = self.ce_loss_alpha * ce_per_token
        if self.l1_loss_alpha > 0:
            if logits.shape != (*draft_hidden.shape[:-1], lm_head_weight.shape[0]):
                raise RuntimeError("DFlash L1 distillation requires full draft logits")
            vocab_size = lm_head_weight.shape[0]
            flat_logits = logits.view(-1, vocab_size)
            target_logits = F.linear(aligned_target_hidden, lm_head_weight).view(-1, vocab_size)
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            draft_probs = torch.softmax(flat_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, self.block_size)

        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            # Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                        valid_mask=weight_mask[..., 1:] > 0,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights
        elif self.loss_objective == "path":
            path_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                path_pred_weights = _path_overlap_position_weights(
                    distribution_loss.view(bsz, n_blocks, self.block_size)[..., 1:],
                    self.dpace_alpha,
                    valid_mask=weight_mask[..., 1:] > 0,
                ).to(dtype=weight_mask.dtype)
                path_weights[..., 1:] = path_pred_weights
            objective_weights = weight_mask * path_weights
        elif self.loss_objective == "auf":
            auf_mask = _auf_position_mask(
                pred_ids.view(bsz, n_blocks, self.block_size),
                target_ids,
                weight_mask > 0,
            )
            objective_weights = weight_mask * auf_mask.to(dtype=weight_mask.dtype)

        flat_weights = objective_weights.view(-1)
        loss_numerator = (loss_per_token * flat_weights).sum()
        loss_denominator = flat_weights.sum()
        extra_numerator, loss_components = self._extra_training_loss(
            draft_hidden=draft_hidden,
            logits=logits,
            target_ids=target_ids,
            objective_weights=objective_weights,
            native_weights=weight_mask,
        )
        loss_numerator = loss_numerator + extra_numerator
        rejected_numerator, rejected_denominator, rejected_components = self._opd_rejected_loss(
            draft_hidden=draft_hidden,
            lm_head_weight=lm_head_weight,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            rejected_anchor_positions=opd_rejected_anchor_positions,
            rejected_offsets=opd_rejected_offsets,
            rejected_token_ids=opd_rejected_token_ids,
            rejected_teacher_logprobs=opd_rejected_teacher_logprobs,
            rejected_mask=opd_rejected_mask,
        )
        if self.loss_objective == "opd":
            loss_numerator = loss_numerator + self.opd_rejected_stream_weight * rejected_numerator
            loss_denominator = (
                loss_denominator + self.opd_rejected_stream_weight * rejected_denominator
            )
            loss_components.update(rejected_components)
        loss = loss_numerator / loss_denominator.clamp(min=1e-6)

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            # Matches Eagle3's per-TTT-position breakdown semantically.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            (loss_numerator, loss_denominator.detach()),
        )
