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

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from tools.convert_to_hf import (
    _fixup_export_config,
    _prepare_export_tensors,
    _remap_weight_keys,
    _save_without_vocab_pruning,
)
from torchspec.models.dflash import (
    _bernoulli_forward_kl_loss,
    _create_dflash_mask_mod,
    _dpace_position_weights,
    _total_variation_loss,
)
from torchspec.models.dflash2 import DFlash2Model
from torchspec.models.draft.auto import AutoDraftModelConfig
from torchspec.models.draft.dflash import DFlashConfig
from torchspec.models.draft.dflash2 import (
    CandidateSelector,
    DFlash2Config,
    DFlash2DraftModel,
    DFlashGroupedConv,
)
from torchspec.training.trainer_actor import _trainer_class_for_config

ROOT = Path(__file__).resolve().parents[1]


def _load_dflash2_trainer():
    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_store = ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = type("MooncakeDistributedStore", (), {})
    mooncake_store.ReplicateConfig = type("ReplicateConfig", (), {})
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in ("mooncake", "mooncake.store")}
    sys.modules.update({"mooncake": mooncake, "mooncake.store": mooncake_store})
    try:
        from torchspec.training.dflash2_trainer import DFlash2Trainer

        return DFlash2Trainer
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _tiny_config_kwargs(**overrides):
    config = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "num_target_layers": 4,
        "block_size": 2,
        "conv_group_size": 2,
        "selector_rank": 2,
        "selector_top_k": 2,
    }
    config.update(overrides)
    return config


def _make_config(
    hidden_size=16,
    vocab_size=32,
    num_target_layers=2,
    block_size=4,
    selector_rank=4,
    selector_top_k=None,
):
    num_attention_heads = min(4, hidden_size)
    return DFlash2Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=hidden_size,
        target_num_hidden_layers=12,
        target_layer_ids=[1, 9][:num_target_layers],
        mask_token_id=vocab_size - 1,
        block_size=block_size,
        conv_kernel_size=2,
        conv_group_size=min(4, hidden_size),
        selector_rank=selector_rank,
        selector_top_k=selector_top_k if selector_top_k is not None else min(4, vocab_size),
    )


def _make_model(
    selector_loss_alpha=0.4,
    logits_chunk_size=0,
    loss_objective="decay",
    ce_loss_alpha=1.0,
    opd_rejected_k3_preserve_negative_tail=False,
    opd_accepted_objective="forward_kl",
):
    config = _make_config()
    draft = DFlash2DraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DFlash2Model(
        draft_model=draft,
        block_size=config.block_size,
        num_anchors=2,
        loss_objective=loss_objective,
        loss_decay_gamma=4.0,
        ce_loss_alpha=ce_loss_alpha,
        logits_chunk_size=logits_chunk_size,
        selector_loss_alpha=selector_loss_alpha,
        opd_rejected_k3_preserve_negative_tail=(
            opd_rejected_k3_preserve_negative_tail
        ),
        opd_accepted_objective=opd_accepted_objective,
    )


def _batch(seed=0, all_masked=False, with_last_hidden_states=False):
    generator = torch.Generator().manual_seed(seed)
    batch_size, sequence_length, hidden_size, vocab_size = 2, 12, 16, 32
    loss_mask = torch.zeros(batch_size, sequence_length)
    if not all_masked:
        loss_mask[:, 2:] = 1
    batch = {
        "input_ids": torch.randint(
            0, vocab_size - 1, (batch_size, sequence_length), generator=generator
        ),
        "hidden_states_list": [
            torch.randn(batch_size, sequence_length, hidden_size, generator=generator)
            for _ in range(2)
        ],
        "loss_mask": loss_mask,
        "lm_head_weight": torch.randn(vocab_size, hidden_size, generator=generator),
    }
    if with_last_hidden_states:
        batch["last_hidden_states"] = torch.randn(
            batch_size,
            sequence_length,
            hidden_size,
            generator=generator,
        )
    return batch


class TestDFlash2Config(unittest.TestCase):
    def test_repository_config_dispatches_to_dflash2(self):
        config_path = ROOT / "torchspec" / "config" / "dflash2_draft_config.json"
        self.assertTrue(config_path.is_file())

        config = AutoDraftModelConfig.from_file(str(config_path))

        self.assertIsInstance(config, DFlash2Config)
        self.assertEqual(config.architectures, ["DFlash2DraftModel"])
        self.assertGreater(config.block_size, 1)
        trainer_module = ModuleType("torchspec.training.dflash2_trainer")
        trainer_class = type("DFlash2Trainer", (), {})
        trainer_module.DFlash2Trainer = trainer_class
        with mock.patch.dict("sys.modules", {"torchspec.training.dflash2_trainer": trainer_module}):
            self.assertIs(_trainer_class_for_config(config), trainer_class)

    def test_legacy_dflash_dispatch_is_unchanged(self):
        config = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["DFlashDraftModel"],
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 32,
                "num_target_layers": 2,
                "target_hidden_size": 16,
                "target_num_hidden_layers": 12,
            }
        )

        self.assertIs(type(config), DFlashConfig)
        trainer_module = ModuleType("torchspec.training.dflash_trainer")
        trainer_class = type("DFlashTrainer", (), {})
        trainer_module.DFlashTrainer = trainer_class
        with mock.patch.dict("sys.modules", {"torchspec.training.dflash_trainer": trainer_module}):
            self.assertIs(_trainer_class_for_config(config), trainer_class)

    def test_official_nested_config_semantics(self):
        config = AutoDraftModelConfig.from_dict(
            _tiny_config_kwargs(
                architectures=["DFlash2DraftModel"],
                model_type="qwen3",
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=32,
                rope_parameters={"rope_theta": 12345.0},
                num_target_layers=12,
                use_sliding_window=True,
                sliding_window=2048,
                is_causal=False,
                layer_types=["sliding_attention"],
                dflash_config={
                    "block_size": 4,
                    "target_layer_ids": [1, 9],
                    "mask_token_id": 31,
                    "conv_kernel_size": 2,
                    "conv_group_size": 4,
                    "selector_rank": 4,
                    "selector_top_k": 8,
                },
            )
        )

        self.assertIsInstance(config, DFlash2Config)
        self.assertEqual(config.model_type, "qwen3_dflash2")
        self.assertEqual(config.num_target_layers, 2)
        self.assertEqual(config.target_num_hidden_layers, 12)
        self.assertEqual(config.target_layer_ids, [1, 9])
        self.assertEqual(config.rope_theta, 12345.0)
        self.assertEqual(config.block_size, 4)
        self.assertEqual(config.conv_kernel_size, 2)
        self.assertEqual(config.conv_group_size, 4)
        self.assertEqual(config.selector_rank, 4)
        self.assertEqual(config.selector_top_k, 8)
        self.assertEqual(config.sliding_window, 2048)
        self.assertFalse(config.is_causal)

        model = DFlash2Model(
            DFlash2DraftModel(config),
            block_size=config.block_size,
            num_anchors=1,
        )
        self.assertEqual(
            model._block_mask_options(),
            {"is_causal": False, "sliding_window": 2048},
        )

    def test_missing_target_layer_ids_are_derived_from_target_depth(self):
        config = DFlash2Config(
            **_tiny_config_kwargs(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=32,
                num_target_layers=12,
                block_size=4,
                conv_group_size=4,
                selector_rank=4,
                selector_top_k=4,
            )
        )

        self.assertEqual(config.target_num_hidden_layers, 12)
        self.assertEqual(config.num_target_layers, 2)
        self.assertEqual(config.target_layer_ids, [1, 9])
        self.assertEqual(DFlash2DraftModel(config).context_proj.in_features, 32)

        sliding = DFlash2Config(
            **_tiny_config_kwargs(
                num_hidden_layers=2,
                use_sliding_window=True,
                max_window_layers=0,
            )
        )
        self.assertEqual(sliding.layer_types, ["sliding_attention", "sliding_attention"])
        self.assertEqual(sliding.sliding_window, 4096)

        equal_depth = DFlash2Config(
            **_tiny_config_kwargs(
                num_hidden_layers=2,
                num_target_layers=2,
                dflash_config={"target_layer_ids": [0, 1]},
            )
        )
        self.assertEqual(equal_depth.target_num_hidden_layers, 2)
        self.assertEqual(
            DFlash2DraftModel(equal_depth).config_for_serving()["num_target_layers"],
            2,
        )

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            DFlash2Config(**_tiny_config_kwargs(conv_kernel_size=3))

        with self.assertRaisesRegex(ValueError, "sliding_window"):
            DFlash2Config(**_tiny_config_kwargs(layer_types=["sliding_attention"]))

        for target_layer_ids in ([1, 12], [-1, 9], []):
            with self.subTest(target_layer_ids=target_layer_ids):
                with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                    DFlash2Config(
                        **_tiny_config_kwargs(
                            num_hidden_layers=2,
                            num_target_layers=12,
                            target_layer_ids=target_layer_ids,
                        )
                    )

        unsupported_qwen_options = (
            ({"hidden_act": "relu"}, "hidden_act"),
            ({"attention_bias": True}, "attention_bias"),
            ({"attention_dropout": 0.1}, "attention_dropout"),
            ({"rope_scaling": {"factor": 2.0}}, "rope_scaling"),
            ({"rope_parameters": {"rope_type": "linear"}}, "rope_parameters"),
        )
        for options, error in unsupported_qwen_options:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, error):
                    DFlash2Config(**_tiny_config_kwargs(**options))

    def test_explicit_capture_layers_must_match_dflash2_config(self):
        from torchspec.train_entry import _validate_and_configure_dflash

        config = _make_config()
        args = Namespace(
            inference_engine_type="sgl",
            defer_tokenization=False,
            dflash_block_size=4,
            dflash_num_target_layers=2,
            min_loss_tokens=8,
            aux_hidden_states_layers=[1, 8],
        )

        with self.assertRaisesRegex(ValueError, "aux_hidden_states_layers"):
            _validate_and_configure_dflash(args, config)

        args.aux_hidden_states_layers = config.target_layer_ids
        args.draft_accumulation_steps = 2
        _validate_and_configure_dflash(args, config)

        args.attention_backend = "usp"
        with self.assertRaisesRegex(ValueError, "attention_backend=usp"):
            _validate_and_configure_dflash(args, config)


class TestDFlashGroupedConv(unittest.TestCase):
    def setUp(self):
        self.conv = DFlashGroupedConv(
            hidden_size=4,
            kernel_size=2,
            group_size=2,
            block_size=3,
        )
        with torch.no_grad():
            self.conv.kernel_projection.weight.zero_()
            self.conv.base_kernel.zero_()
            self.conv.base_kernel[0, 1].fill_(1)
            self.conv.base_kernel[1, 0].fill_(2)
            self.conv.base_kernel[1, 1].fill_(3)

    def test_parameter_schema_and_block_boundary(self):
        self.assertEqual(self.conv.base_kernel.shape, (2, 2, 4))
        self.assertEqual(self.conv.kernel_projection.weight.shape, (8, 4))
        hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4)

        prepared, finish_kernel = self.conv.prepare(hidden)

        expected = torch.zeros_like(hidden)
        expected[:, 1:3] = hidden[:, 0:2]
        expected[:, 4:6] = hidden[:, 3:5]
        self.assertTrue(torch.equal(prepared, expected))
        self.assertEqual(finish_kernel.shape, (1, 6, 2, 2))

        finish_kernel = torch.zeros_like(finish_kernel)
        finish_kernel[..., 0, 0] = 0.5
        finish_kernel[..., 1, 0] = 1.0
        finish_kernel[..., 0, 1] = 1.5
        finish_kernel[..., 1, 1] = 2.0
        finished = self.conv.finish(prepared, finish_kernel)
        shifted = torch.zeros_like(prepared)
        shifted[:, 1:3] = prepared[:, 0:2]
        shifted[:, 4:6] = prepared[:, 3:5]
        expected = torch.empty_like(finished)
        expected[..., :2] = 2.5 * prepared[..., :2] + 4 * shifted[..., :2]
        expected[..., 2:] = 3.5 * prepared[..., 2:] + 5 * shifted[..., 2:]
        self.assertTrue(torch.equal(finished, expected))

    def test_output_is_causal_within_each_block(self):
        with torch.no_grad():
            self.conv.kernel_projection.weight.copy_(
                torch.arange(32, dtype=torch.float32).view(8, 4) / 100
            )
        hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
        changed = hidden.clone()
        changed[:, 2] = -1000

        baseline, _ = self.conv.prepare(hidden)
        modified, _ = self.conv.prepare(changed)

        self.assertTrue(torch.equal(modified[:, :2], baseline[:, :2]))
        self.assertFalse(torch.equal(modified[:, 2], baseline[:, 2]))
        self.assertTrue(torch.equal(modified[:, 3:], baseline[:, 3:]))


class TestCandidateSelector(unittest.TestCase):
    def test_scores_match_public_inference_lattice_rows(self):
        selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=2)
        with torch.no_grad():
            selector.predecessor_codebook.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            )
            selector.successor_codebook.copy_(
                torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
            )
            selector.hidden_projection.weight.copy_(torch.eye(2))

        hidden = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
        logits = torch.tensor([[[0.1, 0.7, 0.2, 0.4], [0.6, 0.3, 0.9, 0.2]]])
        anchor_ids = torch.tensor([1])

        unary, public_candidate_ids = torch.topk(logits, 2, dim=-1, sorted=False)
        public_predecessor_ids = torch.cat(
            (
                anchor_ids[:, None, None].expand(-1, 1, 2),
                public_candidate_ids[:, :-1],
            ),
            dim=1,
        )
        public_lattice = unary[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            selector.predecessor_codebook[public_predecessor_ids] * hidden[:, :, None],
            selector.successor_codebook[public_candidate_ids],
        )

        previous_indices = torch.zeros(1, dtype=torch.long)
        selected_tokens = []
        realized_rows = []
        for position in range(hidden.shape[1]):
            row = public_lattice[:, position].gather(
                1,
                previous_indices[:, None, None].expand(-1, 1, selector.top_k),
            )[:, 0]
            previous_indices = row.argmax(dim=-1)
            selected_tokens.append(
                public_candidate_ids[:, position].gather(-1, previous_indices[:, None])[:, 0]
            )
            realized_rows.append(row)
        selected_tokens = torch.stack(selected_tokens, dim=1)
        predecessor_ids = torch.cat((anchor_ids[:, None], selected_tokens[:, :-1]), dim=1)

        scores, candidate_ids = selector.score_candidates(hidden, logits, predecessor_ids)

        self.assertTrue(torch.equal(candidate_ids, public_candidate_ids))
        self.assertTrue(torch.allclose(scores, torch.stack(realized_rows, dim=1), atol=1e-6))

    def test_training_candidates_replace_the_weakest_candidate_with_gold(self):
        selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=2)
        hidden = torch.ones(1, 1, 2)
        logits = torch.tensor([[[0.1, 0.9, 0.8, 0.2]]])

        _, candidate_ids = selector.score_candidates(
            hidden,
            logits,
            predecessor_ids=torch.tensor([[3]]),
            training_successor_ids=torch.tensor([[0]]),
        )

        self.assertEqual(set(candidate_ids[0, 0].tolist()), {0, 1})

        present_selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=3)
        top_k = torch.topk(logits, 3, dim=-1, sorted=False)
        weakest_index = top_k.values.argmin(dim=-1).item()
        gold_index = 1 if weakest_index != 1 else 2
        gold_id = top_k.indices[..., gold_index]
        _, present_candidate_ids = present_selector.score_candidates(
            hidden,
            logits,
            predecessor_ids=torch.tensor([[3]]),
            training_successor_ids=gold_id,
        )
        self.assertEqual(
            set(present_candidate_ids[0, 0].tolist()),
            set(top_k.indices[0, 0].tolist()),
        )
        self.assertEqual(present_candidate_ids[0, 0].unique().numel(), 3)


class TestDFlash2Forward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = _make_model()

    def test_selector_loss_uses_teacher_forced_transitions_and_prefix_mask(self):
        config = _make_config(
            hidden_size=4,
            vocab_size=4,
            num_target_layers=1,
            selector_rank=1,
            selector_top_k=2,
        )
        model = DFlash2Model(
            DFlash2DraftModel(config),
            block_size=4,
            num_anchors=1,
            selector_loss_alpha=0.4,
        )
        selector = model.draft_model.candidate_selector
        with torch.no_grad():
            selector.hidden_projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            selector.predecessor_codebook.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
            selector.successor_codebook.copy_(torch.tensor([[0.5], [1.0], [1.5], [2.0]]))

        hidden = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 3.0, 2.0],
                    [0.1, 3.0, 0.0, 2.0],
                    [3.0, 2.0, 1.0, 0.0],
                ]
            ]
        )
        targets = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 0.5, 0.0]]])

        extra_numerator, components = model._extra_training_loss(
            hidden, logits, targets, weights, weights
        )

        first_ce = torch.logsumexp(torch.tensor([4.5, 1.0]), dim=0) - 1.0
        second_ce = torch.logsumexp(torch.tensor([7.0, 6.0]), dim=0) - 6.0
        expected_numerator = first_ce + 0.5 * second_ce
        self.assertTrue(torch.allclose(extra_numerator, 0.4 * expected_numerator, atol=1e-6))
        component_numerator, component_denominator = components["selector_loss"]
        self.assertTrue(torch.allclose(component_numerator, expected_numerator, atol=1e-6))
        self.assertEqual(component_denominator.item(), 1.5)

        gap_weights = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
        gap_numerator, _ = model._extra_training_loss(
            hidden, logits, targets, gap_weights, gap_weights
        )
        self.assertTrue(torch.allclose(gap_numerator, 0.4 * first_ce, atol=1e-6))

    def test_auf_masks_unary_ce_without_truncating_selector_supervision(self):
        config = _make_config(block_size=4)
        model = DFlash2Model(
            DFlash2DraftModel(config),
            block_size=4,
            num_anchors=1,
            selector_loss_alpha=0.4,
            loss_objective="auf",
        )
        hidden = torch.zeros(1, 4, config.hidden_size)
        targets = torch.tensor([[[0, 1, 2, 3]]])
        # Compact selector-CE payload used by the chunked DFlash2 loss path.
        selector_ce = torch.tensor([[[1.0, 2.0, 3.0]]])
        auf_weights = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        native_weights = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        extra_numerator, components = model._extra_training_loss(
            hidden,
            selector_ce,
            targets,
            auf_weights,
            native_weights,
        )

        self.assertTrue(torch.allclose(extra_numerator, torch.tensor(0.4 * 6.0)))
        component_numerator, component_denominator = components["selector_loss"]
        self.assertEqual(component_numerator.item(), 6.0)
        self.assertEqual(component_denominator.item(), 3.0)

    def test_gradients_reach_convolutions_and_selector(self):
        torch.manual_seed(13)
        result = self.model(**_batch(seed=2))
        self.assertEqual(len(result), 7)
        loss = result[0]
        loss_numerator, loss_denominator = result[-1]
        self.assertTrue(torch.allclose(loss, loss_numerator / loss_denominator.clamp(min=1e-6)))
        loss.backward()

        draft = self.model.draft_model

        def has_gradient(parameters):
            return any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in parameters
            )

        self.assertTrue(has_gradient(draft.layers[0].attention_conv.parameters()))
        self.assertTrue(has_gradient(draft.layers[0].mlp_conv.parameters()))
        self.assertGreater(draft.layers[0].attention_conv.base_kernel.grad[1].abs().sum(), 0)
        self.assertGreater(draft.layers[0].mlp_conv.base_kernel.grad[1].abs().sum(), 0)
        attention_projection_grad = draft.layers[0].attention_conv.kernel_projection.weight.grad
        mlp_projection_grad = draft.layers[0].mlp_conv.kernel_projection.weight.grad
        projection_half = attention_projection_grad.shape[0] // 2
        self.assertGreater(attention_projection_grad[projection_half:].abs().sum(), 0)
        self.assertGreater(mlp_projection_grad[projection_half:].abs().sum(), 0)
        self.assertTrue(has_gradient([draft.candidate_selector.predecessor_codebook]))
        self.assertTrue(has_gradient([draft.candidate_selector.successor_codebook]))
        self.assertTrue(has_gradient(draft.candidate_selector.hidden_projection.parameters()))
        self.assertTrue(has_gradient(draft.context_proj.parameters()))
        self.assertIsNone(draft.embed_tokens.weight.grad)

    def test_chunked_logits_match_full_loss_and_gradients(self):
        full_model = _make_model(logits_chunk_size=0)
        chunked_model = _make_model(logits_chunk_size=4)
        chunked_model.load_state_dict(full_model.state_dict())
        batch = _batch(seed=5)

        torch.manual_seed(29)
        full_result = full_model(**batch)
        torch.manual_seed(29)
        chunked_result = chunked_model(**batch)

        for full_value, chunked_value in zip(full_result[:5], chunked_result[:5]):
            self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))
        for key in full_result[5]:
            for full_value, chunked_value in zip(full_result[5][key], chunked_result[5][key]):
                self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))
        for full_value, chunked_value in zip(full_result[6], chunked_result[6]):
            self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))

        full_result[0].backward()
        chunked_result[0].backward()
        full_params = dict(full_model.named_parameters())
        chunked_params = dict(chunked_model.named_parameters())
        self.assertEqual(full_params.keys(), chunked_params.keys())
        for name, full_param in full_params.items():
            chunked_param = chunked_params[name]
            if full_param.grad is None:
                self.assertIsNone(chunked_param.grad, msg=name)
            else:
                self.assertTrue(
                    torch.allclose(full_param.grad, chunked_param.grad, atol=1e-5, rtol=1e-5),
                    msg=name,
                )

    def test_chunked_distribution_losses_match_full_loss_and_gradients(self):
        for objective in ("lk", "path", "tv"):
            with self.subTest(objective=objective):
                full_model = _make_model(
                    logits_chunk_size=0,
                    loss_objective=objective,
                    ce_loss_alpha=0,
                )
                chunked_model = _make_model(
                    logits_chunk_size=4,
                    loss_objective=objective,
                    ce_loss_alpha=0,
                )
                chunked_model.load_state_dict(full_model.state_dict())
                batch = _batch(seed=31, with_last_hidden_states=True)

                torch.manual_seed(37)
                full_result = full_model(**batch)
                torch.manual_seed(37)
                chunked_result = chunked_model(**batch)

                for full_value, chunked_value in zip(full_result[:5], chunked_result[:5]):
                    self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))
                for key in full_result[5]:
                    for full_value, chunked_value in zip(
                        full_result[5][key], chunked_result[5][key]
                    ):
                        self.assertTrue(
                            torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6)
                        )
                for full_value, chunked_value in zip(full_result[6], chunked_result[6]):
                    self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))

                full_result[0].backward()
                chunked_result[0].backward()
                chunked_parameters = dict(chunked_model.named_parameters())
                for name, full_param in full_model.named_parameters():
                    chunked_param = chunked_parameters[name]
                    if full_param.grad is None:
                        self.assertIsNone(chunked_param.grad, msg=name)
                    else:
                        self.assertTrue(
                            torch.allclose(
                                full_param.grad,
                                chunked_param.grad,
                                atol=1e-5,
                                rtol=1e-5,
                            ),
                            msg=name,
                        )

    def test_opd_chunked_replay_matches_full_loss_and_gradients(self):
        full_model = _make_model(
            logits_chunk_size=0,
            loss_objective="opd",
            ce_loss_alpha=0,
        )
        chunked_model = _make_model(
            logits_chunk_size=4,
            loss_objective="opd",
            ce_loss_alpha=0,
        )
        chunked_model.load_state_dict(full_model.state_dict())
        batch = _batch(seed=41, with_last_hidden_states=True)
        batch.update(
            {
                "opd_anchor_positions": torch.tensor([[1, 5], [2, 6]]),
                "opd_anchor_mask": torch.tensor([[True, True], [True, True]]),
                "opd_segment_lengths": torch.tensor([[3, 2], [2, 3]]),
                "opd_rejected_anchor_positions": torch.tensor([[1, 5, 5], [2, 6, 0]]),
                "opd_rejected_offsets": torch.tensor([[1, 2, 3], [3, 1, 0]]),
                "opd_rejected_token_ids": torch.tensor([[3, 7, 11], [5, 13, 0]]),
                "opd_rejected_teacher_logprobs": torch.tensor(
                    [[-1.2, -2.3, -3.4], [-0.7, -4.1, 0.0]]
                ),
                "opd_rejected_mask": torch.tensor([[True, True, True], [True, True, False]]),
            }
        )

        full_result = full_model(**batch)
        chunked_result = chunked_model(**batch)

        for full_value, chunked_value in zip(full_result[:5], chunked_result[:5]):
            self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))
        self.assertEqual(full_result[5].keys(), chunked_result[5].keys())
        for key in full_result[5]:
            for full_value, chunked_value in zip(full_result[5][key], chunked_result[5][key]):
                self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))
        for full_value, chunked_value in zip(full_result[6], chunked_result[6]):
            self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6, rtol=1e-6))

        full_result[0].backward()
        chunked_result[0].backward()
        chunked_parameters = dict(chunked_model.named_parameters())
        for name, full_parameter in full_model.named_parameters():
            chunked_parameter = chunked_parameters[name]
            if full_parameter.grad is None:
                self.assertIsNone(chunked_parameter.grad, msg=name)
            else:
                self.assertTrue(
                    torch.allclose(
                        full_parameter.grad,
                        chunked_parameter.grad,
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                    msg=name,
                )

    def test_opd_tv_token_statistics_match_chunked_loss_and_gradients(self):
        full_model = _make_model(
            logits_chunk_size=0,
            loss_objective="opd",
            ce_loss_alpha=0,
            opd_accepted_objective="tv",
        )
        chunked_model = _make_model(
            logits_chunk_size=4,
            loss_objective="opd",
            ce_loss_alpha=0,
            opd_accepted_objective="tv",
        )
        chunked_model.load_state_dict(full_model.state_dict())
        generator = torch.Generator().manual_seed(109)
        full_hidden = torch.randn(2, 8, 16, generator=generator, requires_grad=True)
        chunked_hidden = full_hidden.detach().clone().requires_grad_(True)
        target_hidden = torch.randn(2, 8, 16, generator=generator)
        target_ids = torch.randint(0, 32, (2, 2, 4), generator=generator)
        full_head = torch.randn(32, 16, generator=generator, requires_grad=True)
        chunked_head = full_head.detach().clone().requires_grad_(True)

        full_result = full_model._compute_token_statistics(
            full_hidden,
            full_head,
            target_ids,
            target_hidden,
        )
        chunked_result = chunked_model._compute_token_statistics(
            chunked_hidden,
            chunked_head,
            target_ids,
            target_hidden,
        )

        for index in (0, 1, 3):
            self.assertTrue(
                torch.allclose(
                    full_result[index],
                    chunked_result[index],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
        (full_result[0].sum() + full_result[3].sum()).backward()
        (chunked_result[0].sum() + chunked_result[3].sum()).backward()
        self.assertTrue(
            torch.allclose(full_hidden.grad, chunked_hidden.grad, atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(full_head.grad, chunked_head.grad, atol=1e-5, rtol=1e-5)
        )

    def test_opd_rejected_stream_uses_k3_and_position_decay(self):
        model = _make_model(loss_objective="opd", ce_loss_alpha=0)
        student_logprobs = torch.tensor([-1.7, -3.2], requires_grad=True)
        draft_hidden = torch.randn(1, 8, 16, requires_grad=True)
        lm_head = torch.randn(32, 16)

        with mock.patch.object(
            model,
            "_selected_token_log_probs",
            return_value=student_logprobs,
        ):
            numerator, denominator, components = model._opd_rejected_loss(
                draft_hidden=draft_hidden,
                lm_head_weight=lm_head,
                anchor_positions=torch.tensor([[2, 6]]),
                block_keep_mask=torch.tensor([[True, True]]),
                rejected_anchor_positions=torch.tensor([[2, 6]]),
                rejected_offsets=torch.tensor([[1, 3]]),
                rejected_token_ids=torch.tensor([[4, 9]]),
                rejected_teacher_logprobs=torch.tensor([[-1.1, -2.5]]),
                rejected_mask=torch.tensor([[True, True]]),
            )

        delta = (torch.tensor([-1.1, -2.5]) - student_logprobs).clamp(-20, 20)
        losses = (delta.exp() - delta - 1).clamp(-10, 10)
        weights = torch.tensor([1.0, 0.8**2])
        expected = (losses * weights).sum()
        self.assertTrue(torch.allclose(numerator, expected))
        self.assertTrue(torch.allclose(denominator, weights.sum()))
        self.assertTrue(torch.allclose(components["opd_rejected_loss"][0], expected))
        numerator.backward()
        self.assertTrue(torch.isfinite(student_logprobs.grad).all())

    def test_opd_rejected_k3_negative_tail_restores_bounded_gradient(self):
        model = _make_model(
            loss_objective="opd",
            ce_loss_alpha=0,
            opd_rejected_k3_preserve_negative_tail=True,
        )
        student_logprobs = torch.tensor([-2.0, -30.0], requires_grad=True)
        draft_hidden = torch.randn(1, 8, 16, requires_grad=True)
        lm_head = torch.randn(32, 16)

        with mock.patch.object(
            model,
            "_selected_token_log_probs",
            return_value=student_logprobs,
        ):
            numerator, denominator, _ = model._opd_rejected_loss(
                draft_hidden=draft_hidden,
                lm_head_weight=lm_head,
                anchor_positions=torch.tensor([[2, 6]]),
                block_keep_mask=torch.tensor([[True, True]]),
                rejected_anchor_positions=torch.tensor([[2, 6]]),
                rejected_offsets=torch.tensor([[1, 1]]),
                rejected_token_ids=torch.tensor([[4, 9]]),
                rejected_teacher_logprobs=torch.tensor([[-50.0, -1.0]]),
                rejected_mask=torch.tensor([[True, True]]),
            )

        # The strongly negative tail keeps exact K3 value and unit-bounded
        # gradient; the positive overflow tail retains the upstream flat cap.
        self.assertTrue(torch.allclose(numerator, torch.tensor(57.0)))
        self.assertTrue(torch.allclose(denominator, torch.tensor(2.0)))
        numerator.backward()
        self.assertTrue(
            torch.allclose(student_logprobs.grad, torch.tensor([1.0, 0.0]))
        )

    def test_opd_anchor_plan_uses_dynamic_observed_width(self):
        model = _make_model(loss_objective="opd", ce_loss_alpha=0)
        model.num_anchors = 8

        positions, keep_mask, segments = model._prepare_opd_anchor_plan(
            seq_len=12,
            batch_size=2,
            device=torch.device("cpu"),
            anchor_positions=torch.tensor([[2, 7], [3, 0]]),
            anchor_mask=torch.tensor([[True, True], [True, False]]),
            segment_lengths=torch.tensor([[3, 2], [1, 0]]),
        )

        self.assertEqual(positions.shape, (2, 2))
        self.assertEqual(keep_mask.shape, (2, 2))
        self.assertEqual(segments.shape, (2, 2))

    def test_opd_local_forward_kl_matches_bernoulli_definition(self):
        draft_logits = torch.tensor([[0.1, 1.2, -0.4]], requires_grad=True)
        target_logits = torch.tensor([[0.7, -0.3, 0.2]])
        target_ids = torch.tensor([2])

        actual = _bernoulli_forward_kl_loss(draft_logits, target_logits, target_ids)

        draft_logp = torch.log_softmax(draft_logits, dim=-1)[0, 2]
        target_logp = torch.log_softmax(target_logits, dim=-1)[0, 2]
        expected = target_logp.exp() * (target_logp - draft_logp) + (1 - target_logp.exp()) * (
            torch.log1p(-target_logp.exp()) - torch.log1p(-draft_logp.exp())
        )
        self.assertTrue(torch.allclose(actual, expected.unsqueeze(0), atol=1e-6))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(draft_logits.grad).all())

    def test_opd_tv_accepted_objective_matches_total_variation(self):
        model = _make_model(
            loss_objective="opd",
            ce_loss_alpha=0,
            opd_accepted_objective="tv",
        )
        draft_logits = torch.tensor([[0.1, 1.2, -0.4]], requires_grad=True)
        target_logits = torch.tensor([[0.7, -0.3, 0.2]])
        target_ids = torch.tensor([2])

        actual = model._distribution_token_loss(
            draft_logits,
            target_logits,
            target_ids,
        )
        expected = _total_variation_loss(draft_logits, target_logits)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(draft_logits.grad).all())

    def test_invalid_opd_accepted_objective_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "opd_accepted_objective"):
            _make_model(opd_accepted_objective="unsupported")

    def test_all_masked_batch_has_zero_losses(self):
        loss, _, _, _, _, components, loss_terms = self.model(**_batch(all_masked=True))

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(loss_terms[0].item(), 0.0)
        self.assertEqual(loss_terms[1].item(), 0.0)
        selector_numerator, selector_denominator = components["selector_loss"]
        self.assertEqual(selector_numerator.item(), 0.0)
        self.assertEqual(selector_denominator.item(), 0.0)

    def test_logit_scale_and_softcap_are_applied(self):
        self.model.draft_model.config.output_multiplier = 2.0
        self.model.draft_model.config.final_logit_softcapping = 0.5
        hidden = torch.tensor([[[1.0] * 16]])
        lm_head = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) / 100

        logits = self.model._compute_logits(hidden, lm_head)

        linear = F.linear(hidden, lm_head) * 2.0
        self.assertTrue(torch.allclose(logits, torch.tanh(linear / 0.5) * 0.5))

    def test_selector_loss_cannot_be_disabled(self):
        for alpha in (0.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    _make_model(selector_loss_alpha=alpha)

    def test_causal_sliding_mask_truth_table(self):
        mask = _create_dflash_mask_mod(
            anchor_positions=torch.tensor([[5]]),
            block_keep_mask=torch.tensor([[True]]),
            ctx_len=8,
            block_size=4,
            is_causal=True,
            sliding_window=3,
        )

        self.assertTrue(mask(0, 0, 0, 4))
        self.assertFalse(mask(0, 0, 2, 4))
        self.assertTrue(mask(0, 0, 2, 8))
        self.assertTrue(mask(0, 0, 2, 10))
        self.assertFalse(mask(0, 0, 2, 11))

        valid_prefix = torch.tensor([[True, False, False]])
        weights_a = _dpace_position_weights(torch.tensor([[0.8, 0.1, 0.2]]), 0.5, valid_prefix)
        weights_b = _dpace_position_weights(torch.tensor([[0.8, 0.9, 1.0]]), 0.5, valid_prefix)
        self.assertTrue(torch.equal(weights_a, weights_b))


class TestDFlash2Export(unittest.TestCase):
    def test_config_exports_official_dflash2_schema(self):
        config = DFlash2Config(
            **{
                **_make_config().to_dict(),
                "use_sliding_window": True,
                "sliding_window": 2048,
                "layer_types": ["sliding_attention"],
            }
        )
        model = DFlash2DraftModel(config)

        exported = _fixup_export_config(config.to_dict())

        self.assertEqual(exported["model_type"], "qwen3")
        self.assertEqual(exported["architectures"], ["DFlash2DraftModel"])
        self.assertEqual(exported["num_target_layers"], 12)
        self.assertEqual(exported["dflash_config"]["target_layer_ids"], [1, 9])
        self.assertEqual(exported["dflash_config"]["block_size"], 4)
        self.assertEqual(exported["dflash_config"]["conv_kernel_size"], 2)
        self.assertEqual(exported["dflash_config"]["selector_top_k"], 4)
        self.assertEqual(exported["sliding_window"], 2048)
        self.assertTrue(exported["use_sliding_window"])
        self.assertEqual(model.config_for_serving(), exported)

        trainer_class = _load_dflash2_trainer()
        trainer = trainer_class.__new__(trainer_class)
        with tempfile.TemporaryDirectory() as output_dir:
            trainer._write_serving_artifacts(model, model.state_dict(), output_dir)
            saved_config = json.loads(Path(output_dir, "config.json").read_text())
            saved_state = torch.load(
                Path(output_dir, "pytorch_model.bin"),
                weights_only=True,
            )
        self.assertEqual(saved_config, exported)
        self.assertTrue(torch.equal(saved_state["fc.weight"], model.context_proj.weight))

    def test_serving_write_errors_reach_every_rank(self):
        trainer_class = _load_dflash2_trainer()
        model = DFlash2DraftModel(_make_config())

        for dp_rank, global_rank, error_type in (
            (0, 0, OSError),
            (0, 1, RuntimeError),
        ):
            with self.subTest(dp_rank=dp_rank, global_rank=global_rank):
                trainer = trainer_class.__new__(trainer_class)
                trainer.args = Namespace(fsdp_strategy="FULL_SHARD")
                trainer.dp_rank = dp_rank
                trainer.cache_rank = global_rank
                trainer.draft_model = model
                writer = mock.Mock(side_effect=OSError("disk full") if global_rank == 0 else None)

                def gather_errors(errors, local_error, group=None):
                    errors[:] = ["OSError: disk full", None]

                method_globals = trainer.save_draft_model_for_serving.__func__.__globals__
                all_gather = mock.Mock(side_effect=gather_errors)
                with (
                    tempfile.TemporaryDirectory() as output_dir,
                    mock.patch.dict(
                        method_globals,
                        {
                            "get_model_state_dict": mock.Mock(return_value=model.state_dict()),
                            "get_gloo_group": mock.Mock(return_value=object()),
                        },
                    ),
                    mock.patch.object(method_globals["dist"], "is_initialized", return_value=True),
                    mock.patch.object(method_globals["dist"], "get_world_size", return_value=2),
                    mock.patch.object(method_globals["dist"], "all_gather_object", all_gather),
                    mock.patch.object(trainer, "_write_serving_artifacts", writer),
                ):
                    with self.assertRaisesRegex(error_type, "disk full"):
                        trainer.save_draft_model_for_serving(output_dir)

                all_gather.assert_called_once()
                self.assertEqual(writer.call_count, 1 if global_rank == 0 else 0)

    def test_codebook_state_keys_are_bare_parameters(self):
        model = DFlash2DraftModel(_make_config())
        state = model.state_dict()
        remapped = _remap_weight_keys(state)
        offline = _prepare_export_tensors(model)

        self.assertEqual(remapped.keys(), model.state_dict_for_serving(state).keys())
        self.assertEqual(remapped.keys(), offline.keys())
        self.assertIn("candidate_selector.predecessor_codebook", remapped)
        self.assertIn("candidate_selector.successor_codebook", remapped)
        self.assertNotIn("candidate_selector.predecessor_codebook.weight", remapped)
        self.assertNotIn("candidate_selector.successor_codebook.weight", remapped)
        self.assertIn("fc.weight", remapped)
        self.assertIn("hidden_norm.weight", remapped)
        self.assertIn("norm.weight", remapped)
        self.assertNotIn("context_proj.weight", remapped)
        self.assertTrue(torch.equal(remapped["fc.weight"], state["context_proj.weight"]))
        self.assertTrue(torch.equal(offline["fc.weight"], state["context_proj.weight"]))

    def test_offline_export_writes_serving_weights(self):
        model = DFlash2DraftModel(_make_config())

        with tempfile.TemporaryDirectory() as output_dir:
            _save_without_vocab_pruning(
                model,
                output_dir,
                model.config.to_dict(),
                model.config.vocab_size,
            )
            saved_state = load_file(str(Path(output_dir, "model.safetensors")))

        self.assertTrue(torch.equal(saved_state["fc.weight"], model.context_proj.weight))


if __name__ == "__main__":
    unittest.main()
