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
    "sampling_taps",
    "sampling_tree",
    "sampling_tree_listwise",
    "sampling_tv",
}


def _tree_frontier_ranking_loss(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    budget: int,
    depth_log_bias: float,
    margin: float,
    listwise_temperature: float | None = None,
) -> torch.Tensor:
    """Differentiate ranking mistakes made by the deployed bounded tree.

    Allocation itself is discrete and follows the serving builder with
    stop-gradient choices. The default pairwise mode preserves historical
    behavior: when an off-path edge beats the next reachable gold-path edge, a
    softplus margin pushes those two *actual frontier scores* in the opposite
    direction. With ``listwise_temperature`` set, every reachable gold edge is
    compared against all other valid frontier edges using a temperature-scaled
    multiclass margin. That supplies a preservation gradient even when the
    current allocation is correct but close to the B/B+1 cutoff. Loss is
    accumulated at the active gold depth and normalized by the fixed budget.

    Inputs are a flattened batch of Markov lattices: ``candidate_ids`` is
    ``[N, depth, K]``, ``edge_scores`` is ``[N, depth, K, K]``, and
    ``target_ids`` is ``[N, depth]``. The synthetic root uses predecessor row
    zero, exactly as SGLang's DFlash2 selector-tree kernel.
    """
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, depth, top_k]")
    batch, depth_limit, top_k = map(int, candidate_ids.shape)
    if edge_scores.shape != (batch, depth_limit, top_k, top_k):
        raise ValueError("candidate IDs and edge scores have incompatible shapes")
    if target_ids.shape != (batch, depth_limit):
        raise ValueError("target_ids must have shape [batch, depth]")
    if candidate_ids.dtype != torch.int64 or target_ids.dtype != torch.int64:
        raise ValueError("candidate_ids and target_ids must use torch.int64")
    if budget <= 0 or budget > depth_limit * top_k:
        raise ValueError("tree budget must be in [1, depth * top_k]")
    if not math.isfinite(depth_log_bias):
        raise ValueError("tree depth log bias must be finite")
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("tree ranking margin must be finite and non-negative")
    if listwise_temperature is not None and (
        listwise_temperature <= 0 or not math.isfinite(listwise_temperature)
    ):
        raise ValueError("tree listwise temperature must be finite and positive")
    if batch == 0:
        return edge_scores.new_zeros((0, depth_limit), dtype=torch.float32)

    log_probs = torch.log_softmax(edge_scores.float(), dim=-1)
    matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    gold_present = matches.any(dim=-1)
    gold_children = matches.to(torch.int64).argmax(dim=-1)

    device = edge_scores.device
    batch_index = torch.arange(batch, device=device)
    selected_parent: list[torch.Tensor] = []
    selected_depth: list[torch.Tensor] = []
    selected_child: list[torch.Tensor] = []
    selected_cumulative: list[torch.Tensor] = []
    gold_parent = torch.zeros(batch, dtype=torch.int64, device=device)
    gold_depth = torch.zeros(batch, dtype=torch.int64, device=device)
    loss_terms: list[torch.Tensor] = []
    loss_depths: list[torch.Tensor] = []

    for iteration in range(budget):
        parent_slots = iteration + 1
        parent_index = torch.arange(parent_slots, device=device, dtype=torch.int64)
        if iteration == 0:
            next_depth = torch.zeros((batch, 1), dtype=torch.int64, device=device)
            predecessor = torch.zeros_like(next_depth)
            parent_cumulative = edge_scores.new_zeros((batch, 1), dtype=torch.float32)
        else:
            next_depth = torch.stack([torch.zeros_like(selected_depth[0]), *selected_depth], dim=1)
            predecessor = torch.stack([torch.zeros_like(selected_child[0]), *selected_child], dim=1)
            parent_cumulative = torch.stack(
                [torch.zeros_like(selected_cumulative[0]), *selected_cumulative],
                dim=1,
            )

        valid_parent = next_depth < depth_limit
        safe_depth = next_depth.clamp_max(depth_limit - 1)
        score_rows = log_probs[batch_index[:, None], safe_depth, predecessor]
        raw_cumulative = parent_cumulative.unsqueeze(-1) + score_rows
        selection_scores = (
            raw_cumulative + (safe_depth.to(torch.float32).unsqueeze(-1) + 1.0) * depth_log_bias
        )

        valid = valid_parent.unsqueeze(-1).expand(-1, -1, top_k).clone()
        child_grid = torch.arange(top_k, device=device, dtype=torch.int64).view(1, 1, -1)
        parent_grid = parent_index.view(1, -1, 1)
        for old_parent, old_child in zip(selected_parent, selected_child, strict=True):
            valid &= ~(
                parent_grid.eq(old_parent[:, None, None]) & child_grid.eq(old_child[:, None, None])
            )

        # Exact serving tie order: score desc, depth, parent, global token ID,
        # then local child index. Choices are detached; gathered scores retain
        # gradients for the pairwise frontier surrogate.
        detached_scores = selection_scores.detach().masked_fill(~valid, -torch.inf)
        best = detached_scores.amax(dim=(1, 2), keepdim=True)
        tied = valid & detached_scores.eq(best)
        sentinel = torch.iinfo(torch.int64).max

        depth_key = safe_depth.unsqueeze(-1).expand(-1, -1, top_k)
        best_depth = torch.where(tied, depth_key, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= depth_key.eq(best_depth)

        expanded_parent = parent_grid.expand(batch, -1, top_k)
        best_parent = torch.where(tied, expanded_parent, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= expanded_parent.eq(best_parent)

        token_rows = candidate_ids[batch_index[:, None], safe_depth]
        best_token = torch.where(tied, token_rows, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= token_rows.eq(best_token)

        expanded_child = child_grid.expand(batch, parent_slots, -1)
        best_child = torch.where(tied, expanded_child, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= expanded_child.eq(best_child)
        flat_choices = torch.where(
            tied.reshape(batch, -1),
            torch.arange(parent_slots * top_k, device=device, dtype=torch.int64),
            sentinel,
        ).amin(dim=1)
        chosen_parent = flat_choices // top_k
        chosen_child = flat_choices % top_k

        chosen_raw = raw_cumulative[batch_index, chosen_parent, chosen_child]
        chosen_score = selection_scores[batch_index, chosen_parent, chosen_child]
        chosen_depth = next_depth[batch_index, chosen_parent]

        safe_gold_depth = gold_depth.clamp_max(depth_limit - 1)
        next_gold_present = (gold_depth < depth_limit) & gold_present[batch_index, safe_gold_depth]
        next_gold_child = gold_children[batch_index, safe_gold_depth]
        gold_score = selection_scores[batch_index, gold_parent, next_gold_child]
        chose_gold = (
            next_gold_present & chosen_parent.eq(gold_parent) & chosen_child.eq(next_gold_child)
        )
        if listwise_temperature is None:
            rank_loss = F.softplus(chosen_score - gold_score + margin)
            active_loss = next_gold_present & ~chose_gold
        else:
            gold_edge = parent_grid.eq(gold_parent[:, None, None]) & child_grid.eq(
                next_gold_child[:, None, None]
            )
            competitor_scores = selection_scores.masked_fill(~(valid & ~gold_edge), -torch.inf)
            scaled_competitor_lse = torch.logsumexp(
                (competitor_scores + margin) / listwise_temperature,
                dim=(1, 2),
            )
            rank_loss = listwise_temperature * F.softplus(
                scaled_competitor_lse - gold_score / listwise_temperature
            )
            active_loss = next_gold_present
        loss_terms.append(torch.where(active_loss, rank_loss, torch.zeros_like(rank_loss)))
        loss_depths.append(safe_gold_depth)

        selected_parent.append(chosen_parent)
        selected_depth.append(chosen_depth + 1)
        selected_child.append(chosen_child)
        selected_cumulative.append(chosen_raw)
        gold_parent = torch.where(
            chose_gold,
            torch.full_like(gold_parent, iteration + 1),
            gold_parent,
        )
        gold_depth = gold_depth + chose_gold.to(torch.int64)

    losses = torch.stack(loss_terms, dim=1) / float(budget)
    depths = torch.stack(loss_depths, dim=1)
    return torch.zeros((batch, depth_limit), device=device, dtype=losses.dtype).scatter_add(
        1, depths, losses
    )


def _tree_reach_distillation_loss(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    budget: int,
    depth_log_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill local target preference and bounded-tree prefix reach.

    TAPS separates the probability of choosing the target child at a reached
    parent from the probability that the complete prefix reaches a node. This
    helper applies the same decomposition to DFlash2's deployed Markov lattice.
    The positive reach term covers every available target-prefix node. The
    negative term covers off-path nodes that the exact serving allocator spends
    its bounded verification budget on. Allocation choices are stop-gradient;
    the selected cumulative log probabilities retain gradients.

    The target trajectory is greedy and therefore supplies a one-hot local
    target distribution. If the target token is absent from a position's
    candidate support, that position and all later positions are excluded: the
    selector cannot repair a frozen unary-support miss.
    """
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, depth, top_k]")
    batch, depth_limit, top_k = map(int, candidate_ids.shape)
    if edge_scores.shape != (batch, depth_limit, top_k, top_k):
        raise ValueError("candidate IDs and edge scores have incompatible shapes")
    if target_ids.shape != (batch, depth_limit):
        raise ValueError("target_ids must have shape [batch, depth]")
    if candidate_ids.dtype != torch.int64 or target_ids.dtype != torch.int64:
        raise ValueError("candidate_ids and target_ids must use torch.int64")
    if budget <= 0 or budget > depth_limit * top_k:
        raise ValueError("tree budget must be in [1, depth * top_k]")
    if not math.isfinite(depth_log_bias):
        raise ValueError("tree depth log bias must be finite")
    if batch == 0:
        empty = edge_scores.new_zeros((0, depth_limit), dtype=torch.float32)
        return empty, empty

    log_probs = torch.log_softmax(edge_scores.float(), dim=-1)
    matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    gold_present = matches.any(dim=-1)
    gold_children = matches.to(torch.int64).argmax(dim=-1)
    gold_available = gold_present.to(torch.int64).cumprod(dim=-1).bool()

    device = edge_scores.device
    batch_index = torch.arange(batch, device=device)
    depth_index = torch.arange(depth_limit, device=device)

    # Local one-hot target KL (constant target-entropy term omitted) and the
    # positive BCE part for cumulative prefix reach.
    gold_parents = torch.zeros((batch, depth_limit), dtype=torch.int64, device=device)
    if depth_limit > 1:
        gold_parents[:, 1:] = gold_children[:, :-1]
    gold_edge_log_probs = log_probs[
        batch_index[:, None],
        depth_index[None, :],
        gold_parents,
        gold_children,
    ]
    safe_gold_edges = torch.where(
        gold_available,
        gold_edge_log_probs,
        torch.zeros_like(gold_edge_log_probs),
    )
    local_loss = torch.where(gold_available, -gold_edge_log_probs, 0.0)
    positive_reach_loss = torch.where(
        gold_available,
        -torch.cumsum(safe_gold_edges, dim=-1),
        0.0,
    )

    selected_parent: list[torch.Tensor] = []
    selected_depth: list[torch.Tensor] = []
    selected_child: list[torch.Tensor] = []
    selected_cumulative: list[torch.Tensor] = []
    selected_is_gold: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    negative_depths: list[torch.Tensor] = []

    for iteration in range(budget):
        parent_slots = iteration + 1
        parent_index = torch.arange(parent_slots, device=device, dtype=torch.int64)
        if iteration == 0:
            next_depth = torch.zeros((batch, 1), dtype=torch.int64, device=device)
            predecessor = torch.zeros_like(next_depth)
            parent_cumulative = edge_scores.new_zeros((batch, 1), dtype=torch.float32)
            parent_is_gold = torch.ones((batch, 1), dtype=torch.bool, device=device)
        else:
            next_depth = torch.stack([torch.zeros_like(selected_depth[0]), *selected_depth], dim=1)
            predecessor = torch.stack([torch.zeros_like(selected_child[0]), *selected_child], dim=1)
            parent_cumulative = torch.stack(
                [torch.zeros_like(selected_cumulative[0]), *selected_cumulative],
                dim=1,
            )
            parent_is_gold = torch.stack(
                [torch.ones_like(selected_is_gold[0]), *selected_is_gold],
                dim=1,
            )

        valid_parent = next_depth < depth_limit
        safe_depth = next_depth.clamp_max(depth_limit - 1)
        score_rows = log_probs[batch_index[:, None], safe_depth, predecessor]
        raw_cumulative = parent_cumulative.unsqueeze(-1) + score_rows
        selection_scores = (
            raw_cumulative + (safe_depth.to(torch.float32).unsqueeze(-1) + 1.0) * depth_log_bias
        )

        valid = valid_parent.unsqueeze(-1).expand(-1, -1, top_k).clone()
        child_grid = torch.arange(top_k, device=device, dtype=torch.int64).view(1, 1, -1)
        parent_grid = parent_index.view(1, -1, 1)
        for old_parent, old_child in zip(selected_parent, selected_child, strict=True):
            valid &= ~(
                parent_grid.eq(old_parent[:, None, None]) & child_grid.eq(old_child[:, None, None])
            )

        detached_scores = selection_scores.detach().masked_fill(~valid, -torch.inf)
        best = detached_scores.amax(dim=(1, 2), keepdim=True)
        tied = valid & detached_scores.eq(best)
        sentinel = torch.iinfo(torch.int64).max

        depth_key = safe_depth.unsqueeze(-1).expand(-1, -1, top_k)
        best_depth = torch.where(tied, depth_key, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= depth_key.eq(best_depth)

        expanded_parent = parent_grid.expand(batch, -1, top_k)
        best_parent = torch.where(tied, expanded_parent, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= expanded_parent.eq(best_parent)

        token_rows = candidate_ids[batch_index[:, None], safe_depth]
        best_token = torch.where(tied, token_rows, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= token_rows.eq(best_token)

        expanded_child = child_grid.expand(batch, parent_slots, -1)
        best_child = torch.where(tied, expanded_child, sentinel).amin(dim=(1, 2), keepdim=True)
        tied &= expanded_child.eq(best_child)
        flat_choices = torch.where(
            tied.reshape(batch, -1),
            torch.arange(parent_slots * top_k, device=device, dtype=torch.int64),
            sentinel,
        ).amin(dim=1)
        chosen_parent = flat_choices // top_k
        chosen_child = flat_choices % top_k
        chosen_depth = next_depth[batch_index, chosen_parent]
        chosen_raw = raw_cumulative[batch_index, chosen_parent, chosen_child]
        chosen_token = candidate_ids[batch_index, chosen_depth, chosen_child]
        chosen_is_gold = (
            parent_is_gold[batch_index, chosen_parent]
            & gold_available[batch_index, chosen_depth]
            & chosen_token.eq(target_ids[batch_index, chosen_depth])
        )

        # Stable -log(1 - q_reach) for selected off-path nodes. Clamp only the
        # mathematically singular q=1 endpoint; ordinary log probabilities and
        # their gradients are unchanged.
        max_log_reach = chosen_raw.new_tensor(-torch.finfo(chosen_raw.dtype).eps)
        safe_negative_log_reach = torch.minimum(chosen_raw, max_log_reach)
        negative_bce = -torch.log(-torch.expm1(safe_negative_log_reach))
        negative_valid = gold_available[batch_index, chosen_depth] & ~chosen_is_gold
        negative_terms.append(torch.where(negative_valid, negative_bce, 0.0))
        negative_depths.append(chosen_depth)

        selected_parent.append(chosen_parent)
        selected_depth.append(chosen_depth + 1)
        selected_child.append(chosen_child)
        selected_cumulative.append(chosen_raw)
        selected_is_gold.append(chosen_is_gold)

    negative_loss = torch.stack(negative_terms, dim=1) / float(budget)
    negative_depth = torch.stack(negative_depths, dim=1)
    reach_loss = positive_reach_loss.scatter_add(1, negative_depth, negative_loss)
    return local_loss, reach_loss


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
        raise ValueError(f"DFlash2 selector token map SHA-256 mismatch: {actual} != {expected}")
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
        selector_tree_budget: int = 0,
        selector_tree_depth_log_bias: float = 0.0,
        selector_tree_margin: float = 0.0,
        selector_tree_path_weight: float = 0.25,
        selector_tree_listwise_temperature: float = 0.1,
        selector_taps_local_weight: float = 1.0,
        selector_taps_reach_weight: float = 0.25,
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
        self.selector_tree_budget = int(selector_tree_budget)
        self.selector_tree_depth_log_bias = float(selector_tree_depth_log_bias)
        self.selector_tree_margin = float(selector_tree_margin)
        self.selector_tree_path_weight = float(selector_tree_path_weight)
        self.selector_tree_listwise_temperature = float(selector_tree_listwise_temperature)
        self.selector_taps_local_weight = float(selector_taps_local_weight)
        self.selector_taps_reach_weight = float(selector_taps_reach_weight)
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
        if self.selector_objective in {
            "sampling_taps",
            "sampling_tree",
            "sampling_tree_listwise",
        }:
            max_tree_budget = (self.block_size - 1) * int(config.selector_top_k)
            if not 1 <= self.selector_tree_budget <= max_tree_budget:
                raise ValueError(
                    "bounded-tree selector budget must be in "
                    f"[1, {max_tree_budget}], got {self.selector_tree_budget}"
                )
        if not math.isfinite(self.selector_tree_depth_log_bias):
            raise ValueError("dflash2_selector_tree_depth_log_bias must be finite")
        if self.selector_tree_margin < 0 or not math.isfinite(self.selector_tree_margin):
            raise ValueError("dflash2_selector_tree_margin must be finite and non-negative")
        if self.selector_tree_path_weight < 0 or not math.isfinite(self.selector_tree_path_weight):
            raise ValueError("dflash2_selector_tree_path_weight must be finite and non-negative")
        if self.selector_tree_listwise_temperature <= 0 or not math.isfinite(
            self.selector_tree_listwise_temperature
        ):
            raise ValueError(
                "dflash2_selector_tree_listwise_temperature must be finite and positive"
            )
        if self.selector_taps_local_weight < 0 or not math.isfinite(
            self.selector_taps_local_weight
        ):
            raise ValueError("dflash2_selector_taps_local_weight must be finite and non-negative")
        if self.selector_taps_reach_weight < 0 or not math.isfinite(
            self.selector_taps_reach_weight
        ):
            raise ValueError("dflash2_selector_taps_reach_weight must be finite and non-negative")
        if self.selector_objective == "sampling_taps":
            if self.selector_taps_local_weight == self.selector_taps_reach_weight == 0:
                raise ValueError("sampling_taps requires a positive local or reach weight")
            if not (
                self.selector_verifier_temperature == 0.0
                and self.selector_verifier_top_k == 1
                and self.selector_verifier_top_p == 1.0
            ):
                raise ValueError(
                    "sampling_taps currently requires greedy target verification "
                    "(temperature=0, top_k=1, top_p=1)"
                )

        sampling_selector = self.selector_objective != "teacher_ce"
        if sampling_selector and self.loss_objective != "opd":
            raise ValueError(
                "sampling-aligned DFlash2 selector objectives require dflash_loss_objective=opd"
            )
        if sampling_selector:
            if not 1 <= self.selector_verifier_top_k <= int(config.vocab_size):
                raise ValueError(
                    f"dflash2_selector_verifier_top_k must be in [1, {config.vocab_size}]"
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

    def _selector_tree_frontier_loss(
        self,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build the exact serving lattice and rank its bounded gold frontier."""
        if hidden_states.shape[:-1] != draft_logits.shape[:-1]:
            raise ValueError("DFlash2 selector tree hidden/logit rows must match")
        if target_ids.shape != (*hidden_states.shape[:-2], hidden_states.shape[-2] + 1):
            raise ValueError("DFlash2 selector tree targets must include one anchor token")
        if draft_logits.shape[-1] != self.draft_model.config.vocab_size:
            raise ValueError("DFlash2 selector tree requires full-vocabulary draft logits")
        if self.selector_token_ids.numel() == 0:
            raise RuntimeError("DFlash2 selector tree token map is empty")

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
        top_k = int(selector.top_k)
        predecessor_ids = torch.cat(
            (
                target_ids[..., :1, None].expand(*target_ids.shape[:-1], 1, top_k),
                candidate_ids[..., :-1, :],
            ),
            dim=-2,
        )
        predecessor = selector.predecessor_codebook[predecessor_ids] * projected_hidden.unsqueeze(
            -2
        )
        successor = selector.successor_codebook[candidate_ids]
        edge_scores = unary_logits.unsqueeze(-2) + torch.einsum(
            "...dpr,...dcr->...dpc", predecessor, successor
        )

        depth_limit = int(candidate_ids.shape[-2])
        flat_candidates = candidate_ids.reshape(-1, depth_limit, top_k)
        flat_scores = edge_scores.reshape(-1, depth_limit, top_k, top_k)
        flat_targets = target_ids[..., 1:].reshape(-1, depth_limit)
        return _tree_frontier_ranking_loss(
            flat_candidates,
            flat_scores,
            flat_targets,
            budget=self.selector_tree_budget,
            depth_log_bias=self.selector_tree_depth_log_bias,
            margin=self.selector_tree_margin,
            listwise_temperature=(
                self.selector_tree_listwise_temperature
                if self.selector_objective == "sampling_tree_listwise"
                else None
            ),
        ).reshape(*target_ids.shape[:-1], depth_limit)

    def _selector_tree_taps_loss(
        self,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the serving lattice and distill local and reach probabilities."""
        if hidden_states.shape[:-1] != draft_logits.shape[:-1]:
            raise ValueError("DFlash2 selector tree hidden/logit rows must match")
        if target_ids.shape != (*hidden_states.shape[:-2], hidden_states.shape[-2] + 1):
            raise ValueError("DFlash2 selector tree targets must include one anchor token")
        if draft_logits.shape[-1] != self.draft_model.config.vocab_size:
            raise ValueError("DFlash2 selector tree requires full-vocabulary draft logits")
        if self.selector_token_ids.numel() == 0:
            raise RuntimeError("DFlash2 selector tree token map is empty")

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
        top_k = int(selector.top_k)
        predecessor_ids = torch.cat(
            (
                target_ids[..., :1, None].expand(*target_ids.shape[:-1], 1, top_k),
                candidate_ids[..., :-1, :],
            ),
            dim=-2,
        )
        predecessor = selector.predecessor_codebook[predecessor_ids] * projected_hidden.unsqueeze(
            -2
        )
        successor = selector.successor_codebook[candidate_ids]
        edge_scores = unary_logits.unsqueeze(-2) + torch.einsum(
            "...dpr,...dcr->...dpc", predecessor, successor
        )

        depth_limit = int(candidate_ids.shape[-2])
        flat_candidates = candidate_ids.reshape(-1, depth_limit, top_k)
        flat_scores = edge_scores.reshape(-1, depth_limit, top_k, top_k)
        flat_targets = target_ids[..., 1:].reshape(-1, depth_limit)
        local_loss, reach_loss = _tree_reach_distillation_loss(
            flat_candidates,
            flat_scores,
            flat_targets,
            budget=self.selector_tree_budget,
            depth_log_bias=self.selector_tree_depth_log_bias,
        )
        output_shape = (*target_ids.shape[:-1], depth_limit)
        return local_loss.reshape(output_shape), reach_loss.reshape(output_shape)

    def _compute_token_statistics(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        aligned_target_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        total_tokens = draft_hidden.shape[1]
        if self.selector_objective == "teacher_ce" and (
            self.logits_chunk_size == 0 or total_tokens <= self.logits_chunk_size
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
            elif self.selector_objective in {"sampling_tree", "sampling_tree_listwise"}:
                selector_overlap = self._selector_sampling_overlap(
                    hidden[..., 1:, :],
                    chunk_logits[..., 1:, :],
                    target_logits[..., 1:, :],
                    targets[..., :-1],
                )
                tree_loss = self._selector_tree_frontier_loss(
                    hidden[..., 1:, :],
                    chunk_logits[..., 1:, :],
                    targets,
                )
                selector_payload = torch.stack((tree_loss, selector_overlap), dim=-1)
            elif self.selector_objective == "sampling_taps":
                local_loss, reach_loss = self._selector_tree_taps_loss(
                    hidden[..., 1:, :],
                    chunk_logits[..., 1:, :],
                    targets,
                )
                selector_payload = torch.stack((local_loss, reach_loss), dim=-1)
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
        if self.selector_objective in {
            "sampling_taps",
            "sampling_tree",
            "sampling_tree_listwise",
        }:
            expected_shape = (*eligible_weights.shape, 2)
            if logits.shape != expected_shape:
                raise RuntimeError(
                    "bounded-tree DFlash2 selector objective requires compact "
                    f"two-component statistics, got {tuple(logits.shape)} "
                    f"instead of {expected_shape}"
                )
            if self.selector_objective in {"sampling_tree", "sampling_tree_listwise"}:
                tree_loss = logits[..., 0]
                selector_payload = logits[..., 1]
            else:
                taps_local_loss = logits[..., 0]
                taps_reach_loss = logits[..., 1]
                selector_payload = (
                    self.selector_taps_local_weight * taps_local_loss
                    + self.selector_taps_reach_weight * taps_reach_loss
                )
        elif logits.shape != eligible_weights.shape:
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
        if self.selector_objective in {
            "sampling_path",
            "sampling_tree",
            "sampling_tree_listwise",
        }:
            overlap = selector_payload
            valid_prefix = (eligible_weights > 0).to(torch.int64).cumprod(dim=-1).bool()
            prefix_survival = torch.cumprod(
                torch.where(valid_prefix, overlap, torch.ones_like(overlap)),
                dim=-1,
            )
            path_loss = 1.0 - prefix_survival
            if self.selector_objective in {"sampling_tree", "sampling_tree_listwise"}:
                selector_loss = tree_loss + self.selector_tree_path_weight * path_loss
                tree_num = (tree_loss * eligible_weights).sum()
                tree_den = eligible_weights.sum().detach()
                loss_components["selector_tree_loss"] = (tree_num.detach(), tree_den)
            else:
                selector_loss = path_loss
            overlap_num = (overlap * eligible_weights).sum()
            overlap_den = eligible_weights.sum().detach()
            loss_components["selector_overlap"] = (overlap_num.detach(), overlap_den)
        elif self.selector_objective == "sampling_taps":
            selector_loss = selector_payload
            local_num = (taps_local_loss * eligible_weights).sum()
            reach_num = (taps_reach_loss * eligible_weights).sum()
            component_den = eligible_weights.sum().detach()
            loss_components["selector_taps_local_loss"] = (
                local_num.detach(),
                component_den,
            )
            loss_components["selector_taps_reach_loss"] = (
                reach_num.detach(),
                component_den,
            )
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
