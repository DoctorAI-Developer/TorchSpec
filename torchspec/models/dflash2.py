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

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchspec.models.dflash import DFlashModel

_VALID_SELECTOR_OBJECTIVES = {
    "teacher_ce",
    "sampling_path",
    "sampling_tv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selector_token_map(
    path_value: str,
    *,
    expected_sha256: str,
    vocab_size: int,
    selector_top_k: int,
) -> tuple[torch.Tensor, str]:
    """Load an immutable global-ID map for the deployed reduced draft head."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DFlash2 selector token map does not exist: {path}")
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("dflash2_selector_token_map_sha256 must be 64 lowercase hex characters")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            "DFlash2 selector token map SHA-256 mismatch: "
            f"{actual} != {expected}"
        )
    try:
        token_ids = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot load DFlash2 selector token map {path}: {exc}") from exc
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
        raise ValueError("DFlash2 selector token map must be a one-dimensional tensor")
    if token_ids.dtype != torch.int64:
        raise ValueError("DFlash2 selector token map must use torch.int64 global IDs")
    if token_ids.numel() < selector_top_k:
        raise ValueError(
            "DFlash2 selector token map is smaller than selector_top_k: "
            f"{token_ids.numel()} < {selector_top_k}"
        )
    if bool(((token_ids < 0) | (token_ids >= vocab_size)).any()):
        raise ValueError("DFlash2 selector token map contains an out-of-vocabulary ID")
    if token_ids.numel() > 1 and not bool((token_ids[1:] > token_ids[:-1]).all()):
        raise ValueError("DFlash2 selector token map IDs must be unique and strictly increasing")
    return token_ids.contiguous(), actual


def _renormalize_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply SGLang's inclusive top-p rule to an already top-k distribution."""
    if top_p == 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    exclusive_cumulative = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(exclusive_cumulative > top_p, 0.0)
    denominator = sorted_probs.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("DFlash2 selector top-p filtering removed every target candidate")
    sorted_probs = sorted_probs / denominator
    return torch.zeros_like(sorted_probs).scatter(-1, sorted_indices, sorted_probs)


def _target_sparse_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SGLang's top-k-first/top-p-renormalized verifier distribution."""
    with torch.no_grad():
        if temperature == 0.0 or top_k == 1:
            token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            return token_ids, torch.ones_like(token_ids, dtype=torch.float32)
        top_logits, token_ids = torch.topk(logits.float() / temperature, top_k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        return token_ids, _renormalize_top_p(probs, top_p)


class DFlash2Model(DFlashModel):
    def __init__(
        self,
        *args,
        selector_loss_alpha: float = 1.0,
        selector_objective: str = "teacher_ce",
        selector_token_map_path: str | None = None,
        selector_token_map_sha256: str | None = None,
        selector_temperature: float = 1.0,
        selector_verifier_temperature: float = 1.0,
        selector_verifier_top_k: int = 20,
        selector_verifier_top_p: float = 0.95,
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
        self.selector_objective = str(selector_objective).lower()
        if self.selector_objective not in _VALID_SELECTOR_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_SELECTOR_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash2 selector objective {self.selector_objective!r}; "
                f"expected one of {valid}"
            )
        self.selector_temperature = float(selector_temperature)
        self.selector_verifier_temperature = float(selector_verifier_temperature)
        self.selector_verifier_top_k = int(selector_verifier_top_k)
        self.selector_verifier_top_p = float(selector_verifier_top_p)
        if not math.isfinite(self.selector_temperature) or self.selector_temperature <= 0:
            raise ValueError("dflash2_selector_temperature must be finite and positive")
        if (
            not math.isfinite(self.selector_verifier_temperature)
            or self.selector_verifier_temperature < 0
        ):
            raise ValueError(
                "dflash2_selector_verifier_temperature must be finite and non-negative"
            )
        if (
            not math.isfinite(self.selector_verifier_top_p)
            or not 0 < self.selector_verifier_top_p <= 1
        ):
            raise ValueError("dflash2_selector_verifier_top_p must be in (0, 1]")

        sampling_selector = self.selector_objective != "teacher_ce"
        if sampling_selector and self.loss_objective != "opd":
            raise ValueError(
                "sampling-aligned DFlash2 selector objectives require "
                "dflash_loss_objective=opd"
            )
        if sampling_selector:
            if not 1 <= self.selector_verifier_top_k <= int(config.vocab_size):
                raise ValueError(
                    "dflash2_selector_verifier_top_k must be in "
                    f"[1, {config.vocab_size}]"
                )
            if not selector_token_map_path or not selector_token_map_sha256:
                raise ValueError(
                    "sampling-aligned DFlash2 selector objectives require both "
                    "dflash2_selector_token_map_path and "
                    "dflash2_selector_token_map_sha256"
                )
            selector_token_ids, selector_map_sha256 = _load_selector_token_map(
                selector_token_map_path,
                expected_sha256=selector_token_map_sha256,
                vocab_size=int(config.vocab_size),
                selector_top_k=int(config.selector_top_k),
            )
            self.selector_token_map_sha256 = selector_map_sha256
        else:
            if selector_token_map_path is not None or selector_token_map_sha256 is not None:
                raise ValueError(
                    "DFlash2 selector token-map fields require a sampling-aligned "
                    "selector objective"
                )
            selector_token_ids = torch.empty(0, dtype=torch.int64)
            self.selector_token_map_sha256 = None
        self.register_buffer(
            "selector_token_ids",
            selector_token_ids,
            persistent=False,
        )

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

    @property
    def uses_target_hidden_states(self) -> bool:
        return super().uses_target_hidden_states or self.selector_objective != "teacher_ce"

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

    def _selector_sampling_overlap(
        self,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lossless-speculation overlap on the deployed selector support.

        Candidate IDs are selected from the immutable reduced-head map without
        teacher forcing. The frozen target is transformed using the same
        top-k-first, inclusive-top-p rule as SGLang. Gradients flow through the
        draft unary logits, hidden projection, and transition codebooks only.
        """
        if not (
            hidden_states.shape[:-1]
            == draft_logits.shape[:-1]
            == target_logits.shape[:-1]
            == predecessor_ids.shape
        ):
            raise ValueError("DFlash2 selector sampling inputs must have matching rows")
        if draft_logits.shape[-1] != self.draft_model.config.vocab_size:
            raise ValueError("DFlash2 selector sampling requires full-vocabulary draft logits")
        if target_logits.shape[-1] != self.draft_model.config.vocab_size:
            raise ValueError("DFlash2 selector sampling requires full-vocabulary target logits")
        if self.selector_token_ids.numel() == 0:
            raise RuntimeError("DFlash2 selector sampling token map is empty")

        reduced_logits = torch.index_select(draft_logits, -1, self.selector_token_ids)
        unary_logits, local_ids = torch.topk(
            reduced_logits,
            int(self.draft_model.candidate_selector.top_k),
            dim=-1,
            sorted=False,
        )
        candidate_ids = self.selector_token_ids[local_ids]
        selector = self.draft_model.candidate_selector
        projected_hidden = selector.hidden_projection(hidden_states)
        predecessor = selector.predecessor_codebook[predecessor_ids] * projected_hidden
        successor = selector.successor_codebook[candidate_ids]
        scores = unary_logits + torch.einsum("...r,...kr->...k", predecessor, successor)
        draft_probs = torch.softmax(scores.float() / self.selector_temperature, dim=-1)

        target_ids, target_probs = _target_sparse_distribution(
            target_logits,
            temperature=self.selector_verifier_temperature,
            top_k=self.selector_verifier_top_k,
            top_p=self.selector_verifier_top_p,
        )
        matches = candidate_ids.unsqueeze(-1).eq(target_ids.unsqueeze(-2))
        target_on_draft = (matches * target_probs.unsqueeze(-2)).sum(dim=-1)
        return torch.minimum(draft_probs, target_on_draft).sum(dim=-1)

    def _compute_token_statistics(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        aligned_target_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        total_tokens = draft_hidden.shape[1]
        if (
            self.selector_objective == "teacher_ce"
            and (self.logits_chunk_size == 0 or total_tokens <= self.logits_chunk_size)
        ):
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
        if self.logits_chunk_size == 0:
            blocks_per_chunk = num_blocks
        hidden_blocks = draft_hidden.reshape(batch, num_blocks, block_size, -1)
        target_hidden_blocks = (
            None
            if aligned_target_hidden is None
            else aligned_target_hidden.reshape(batch, num_blocks, block_size, -1)
        )
        ce_chunks = []
        pred_chunks = []
        selector_payload_chunks = []
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

            distribution_loss = ce.new_empty(0)
            target_logits = None
            if (
                self.loss_objective in {"lk", "opd", "path", "tv"}
                or self.selector_objective != "teacher_ce"
            ):
                if target_hidden is None:
                    raise ValueError(
                        "DFlash2 distribution training requires aligned target hidden states"
                    )
                target_logits = F.linear(target_hidden, lm_head_weight)
            if self.selector_objective == "teacher_ce":
                selector = self.draft_model.candidate_selector
                scores, candidate_ids = selector.score_candidates(
                    hidden[..., 1:, :],
                    chunk_logits[..., 1:, :],
                    targets[..., :-1],
                    training_successor_ids=targets[..., 1:],
                )
                matches = candidate_ids == targets[..., 1:].unsqueeze(-1)
                target_indices = matches.to(torch.int64).argmax(dim=-1)
                selector_payload = F.cross_entropy(
                    scores.reshape(-1, scores.shape[-1]),
                    target_indices.reshape(-1),
                    reduction="none",
                ).reshape_as(target_indices)
            else:
                selector_overlap = self._selector_sampling_overlap(
                    hidden[..., 1:, :],
                    chunk_logits[..., 1:, :],
                    target_logits[..., 1:, :],
                    targets[..., :-1],
                )
                selector_payload = (
                    1.0 - selector_overlap
                    if self.selector_objective == "sampling_tv"
                    else selector_overlap
                )
            if self.loss_objective in {"lk", "opd", "path", "tv"}:
                distribution_loss = self._distribution_token_loss(
                    chunk_logits,
                    target_logits,
                    targets,
                )
            return ce, pred, selector_payload, distribution_loss

        for start in range(0, num_blocks, blocks_per_chunk):
            stop = min(start + blocks_per_chunk, num_blocks)
            hidden_chunk = hidden_blocks[:, start:stop]
            target_chunk = target_ids[:, start:stop]
            target_hidden_chunk = (
                None if target_hidden_blocks is None else target_hidden_blocks[:, start:stop]
            )
            if self.training and hidden_chunk.requires_grad:
                ce, pred, selector_payload, distribution_loss = torch_checkpoint(
                    compute_chunk,
                    hidden_chunk,
                    target_chunk,
                    target_hidden_chunk,
                    use_reentrant=False,
                )
            else:
                ce, pred, selector_payload, distribution_loss = compute_chunk(
                    hidden_chunk,
                    target_chunk,
                    target_hidden_chunk,
                )
            ce_chunks.append(ce)
            pred_chunks.append(pred)
            selector_payload_chunks.append(selector_payload)
            if self.loss_objective in {"lk", "opd", "path", "tv"}:
                distribution_loss_chunks.append(distribution_loss)

        ce_per_token = torch.cat(ce_chunks, dim=1).reshape(-1)
        pred_ids = torch.cat(pred_chunks, dim=1).reshape(-1)
        selector_payload = torch.cat(selector_payload_chunks, dim=1)
        distribution_loss = (
            torch.cat(distribution_loss_chunks, dim=1).reshape(-1)
            if distribution_loss_chunks
            else None
        )
        return ce_per_token, pred_ids, selector_payload, distribution_loss

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
            if self.selector_objective != "teacher_ce":
                raise RuntimeError(
                    "sampling-aligned DFlash2 selector objective requires compact "
                    "selector-overlap statistics"
                )
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
            selector_payload = F.cross_entropy(
                scores.reshape(-1, scores.shape[-1]),
                target_indices.reshape(-1),
                reduction="none",
            ).reshape_as(eligible_weights)
        else:
            selector_payload = logits

        loss_components = {}
        if self.selector_objective == "sampling_path":
            overlap = selector_payload
            valid_prefix = (eligible_weights > 0).to(torch.int64).cumprod(dim=-1).bool()
            prefix_survival = torch.cumprod(
                torch.where(valid_prefix, overlap, torch.ones_like(overlap)),
                dim=-1,
            )
            selector_loss = 1.0 - prefix_survival
            overlap_num = (overlap * eligible_weights).sum()
            overlap_den = eligible_weights.sum().detach()
            loss_components["selector_overlap"] = (overlap_num.detach(), overlap_den)
        elif self.selector_objective == "sampling_tv":
            selector_loss = selector_payload
            overlap_num = ((1.0 - selector_payload) * eligible_weights).sum()
            overlap_den = eligible_weights.sum().detach()
            loss_components["selector_overlap"] = (overlap_num.detach(), overlap_den)
        else:
            selector_loss = selector_payload

        selector_num = (selector_loss * eligible_weights).sum()
        selector_den = eligible_weights.sum().detach()
        loss_components["selector_loss"] = (selector_num.detach(), selector_den)
        return self.selector_loss_alpha * selector_num, loss_components
