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

"""DFlash2 trainer."""

from argparse import Namespace

from torchspec.models.dflash2 import DFlash2Model
from torchspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from torchspec.training.dflash_trainer import DFlashTrainer


class DFlash2Trainer(DFlashTrainer):
    _draft_config_class = DFlash2Config
    _extra_loss_component_keys = [
        "selector_loss",
        "selector_overlap",
        "selector_taps_local_loss",
        "selector_taps_reach_loss",
        "selector_tree_loss",
        "opd_rejected_loss",
    ]

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.logits_chunk_size = getattr(args, "dflash2_logits_chunk_size", 0)
        self.selector_loss_alpha = getattr(args, "dflash2_selector_loss_alpha", 1.0)
        self.trainable_scope = getattr(args, "dflash2_trainable_scope", "all")
        self.selector_objective = getattr(args, "dflash2_selector_objective", "teacher_ce")
        self.selector_token_map_path = getattr(args, "dflash2_selector_token_map_path", None)
        self.selector_token_map_sha256 = getattr(args, "dflash2_selector_token_map_sha256", None)
        self.selector_temperature = getattr(args, "dflash2_selector_temperature", 1.0)
        self.selector_verifier_temperature = getattr(
            args, "dflash2_selector_verifier_temperature", 1.0
        )
        self.selector_verifier_top_k = getattr(args, "dflash2_selector_verifier_top_k", 20)
        self.selector_verifier_top_p = getattr(args, "dflash2_selector_verifier_top_p", 0.95)
        self.selector_tree_budget = getattr(args, "dflash2_selector_tree_budget", 0)
        self.selector_tree_depth_log_bias = getattr(
            args, "dflash2_selector_tree_depth_log_bias", 0.0
        )
        self.selector_tree_margin = getattr(args, "dflash2_selector_tree_margin", 0.0)
        self.selector_tree_path_weight = getattr(args, "dflash2_selector_tree_path_weight", 0.25)
        self.selector_tree_listwise_temperature = getattr(
            args, "dflash2_selector_tree_listwise_temperature", 0.1
        )
        self.selector_taps_local_weight = getattr(args, "dflash2_selector_taps_local_weight", 1.0)
        self.selector_taps_reach_weight = getattr(args, "dflash2_selector_taps_reach_weight", 0.25)
        self.opd_rejected_stream_weight = getattr(args, "dflash2_opd_rejected_stream_weight", 1.0)
        self.opd_rejected_position_decay = getattr(args, "dflash2_opd_rejected_position_decay", 0.8)
        self.opd_rejected_k3_preserve_negative_tail = getattr(
            args, "dflash2_opd_rejected_k3_preserve_negative_tail", False
        )
        self.opd_accepted_objective = getattr(args, "dflash2_opd_accepted_objective", "forward_kl")

    def _configure_trainable_parameters(self, draft_model) -> None:
        if self.trainable_scope == "all":
            return
        if self.trainable_scope != "selector_only":
            raise ValueError("dflash2_trainable_scope must be one of all or selector_only")
        for parameter in draft_model.parameters():
            parameter.requires_grad = False
        for parameter in draft_model.candidate_selector.parameters():
            parameter.requires_grad = True

    def _build_draft_model(self, config):
        if config.block_size != self.block_size:
            raise ValueError(
                "training.dflash_block_size must match dflash_config.block_size "
                f"({self.block_size} != {config.block_size})"
            )
        if config.num_target_layers != self.num_target_layers:
            raise ValueError(
                "training.dflash_num_target_layers must match the number of "
                f"dflash_config.target_layer_ids ({self.num_target_layers} != "
                f"{config.num_target_layers})"
            )
        return DFlash2DraftModel(config)

    def _build_training_wrapper(self, draft_model):
        return DFlash2Model(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            logits_chunk_size=self.logits_chunk_size,
            selector_loss_alpha=self.selector_loss_alpha,
            selector_objective=self.selector_objective,
            selector_token_map_path=self.selector_token_map_path,
            selector_token_map_sha256=self.selector_token_map_sha256,
            selector_temperature=self.selector_temperature,
            selector_verifier_temperature=self.selector_verifier_temperature,
            selector_verifier_top_k=self.selector_verifier_top_k,
            selector_verifier_top_p=self.selector_verifier_top_p,
            selector_tree_budget=self.selector_tree_budget,
            selector_tree_depth_log_bias=self.selector_tree_depth_log_bias,
            selector_tree_margin=self.selector_tree_margin,
            selector_tree_path_weight=self.selector_tree_path_weight,
            selector_tree_listwise_temperature=self.selector_tree_listwise_temperature,
            selector_taps_local_weight=self.selector_taps_local_weight,
            selector_taps_reach_weight=self.selector_taps_reach_weight,
            opd_rejected_stream_weight=self.opd_rejected_stream_weight,
            opd_rejected_position_decay=self.opd_rejected_position_decay,
            opd_rejected_k3_preserve_negative_tail=(self.opd_rejected_k3_preserve_negative_tail),
            opd_accepted_objective=self.opd_accepted_objective,
        )
