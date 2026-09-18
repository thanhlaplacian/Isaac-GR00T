# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampling at 1, 2 and 4 steps with the shortcut step-size conditioning."""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


BATCH, ENC_SEQ, HORIZON, ACT_DIM, STATE_DIM = 2, 6, 4, 7, 7


def _head(seed=0, **overrides) -> Gr00tN1d7ActionHead:
    defaults = dict(
        backbone_embedding_dim=32,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=STATE_DIM,
        max_action_dim=ACT_DIM,
        action_horizon=HORIZON,
        state_history_length=1,
        max_num_embodiments=4,
        add_pos_embed=True,
        diffusion_model_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "num_layers": 2,
            "output_dim": 64,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    torch.manual_seed(seed)
    return Gr00tN1d7ActionHead(Gr00tN1d7Config(**defaults)).to(torch.float32).eval()


def _features(head, seed=1):
    torch.manual_seed(seed)
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(BATCH, ENC_SEQ, 32),
            "backbone_attention_mask": torch.ones(BATCH, ENC_SEQ, dtype=torch.bool),
            "image_mask": torch.zeros(BATCH, ENC_SEQ, dtype=torch.bool),
        }
    )
    backbone["image_mask"][:, : ENC_SEQ // 2] = True
    action_input = BatchFeature(
        data={
            "state": torch.randn(BATCH, 1, STATE_DIM),
            "embodiment_id": torch.zeros(BATCH, dtype=torch.long),
        }
    )
    encoded = head._encode_features(backbone, action_input)
    return backbone, action_input, encoded


def _sample(head, seed=7):
    backbone, action_input, encoded = _features(head)
    torch.manual_seed(seed)
    return head.get_action_with_features(
        backbone_features=encoded["backbone_features"],
        state_features=encoded["state_features"],
        embodiment_id=action_input["embodiment_id"],
        backbone_output=backbone,
        action_input=BatchFeature(data={}),
    )["action_pred"]


def _count_denoiser_calls(head, **kwargs):
    calls = []
    original = head._denoise_step

    def spy(**call_kwargs):
        calls.append(call_kwargs.get("dt_level"))
        return original(**call_kwargs)

    object.__setattr__(head, "_denoise_step", spy)
    try:
        _sample(head, **kwargs)
    finally:
        object.__delattr__(head, "_denoise_step")
    return calls


class TestStepCount:
    """Nothing in the suite asserted how many denoiser forwards sampling makes, so a
    model that silently collapsed to one step would have passed everything."""

    @pytest.mark.parametrize("steps", [1, 2, 4])
    def test_denoiser_runs_exactly_once_per_step(self, steps):
        head = _head(num_inference_timesteps=steps)
        assert len(_count_denoiser_calls(head)) == steps

    @pytest.mark.parametrize("steps", [1, 2, 4])
    def test_shortcut_model_announces_the_matching_level(self, steps):
        head = _head(shortcut_enabled=True, num_inference_timesteps=steps)
        levels = _count_denoiser_calls(head)
        assert len(levels) == steps
        expected = steps.bit_length() - 1  # log2
        for level in levels:
            assert level is not None
            assert torch.equal(level, torch.full((BATCH,), expected, dtype=torch.long))

    def test_plain_model_announces_no_level(self):
        head = _head(num_inference_timesteps=4)
        assert all(level is None for level in _count_denoiser_calls(head))


class TestZeroInitIdentity:
    """Given identical weights, a zero-initialised dt branch must change nothing.

    The heads are aligned by copying the plain head's state_dict rather than by
    seeding: the DiT is built before the other submodules of the action head, so the
    extra dt_encoder draw shifts the RNG for everything constructed after it. That
    ordering is irrelevant in production -- a finetune loads every weight except
    dt_encoder from a checkpoint -- but it makes seed-matching the wrong way to state
    the property here.
    """

    @staticmethod
    def _aligned_pair(steps):
        plain = _head(seed=3, num_inference_timesteps=steps)
        shortcut = _head(seed=4, shortcut_enabled=True, num_inference_timesteps=steps)
        missing, unexpected = shortcut.load_state_dict(plain.state_dict(), strict=False)
        assert not unexpected
        assert missing and all(k.startswith("model.dt_encoder.") for k in missing), missing
        return plain, shortcut

    @pytest.mark.parametrize("steps", [1, 2, 4])
    def test_untrained_shortcut_model_matches_plain_model(self, steps):
        plain, shortcut = self._aligned_pair(steps)
        assert torch.equal(_sample(plain), _sample(shortcut))

    def test_the_only_added_weights_are_the_dt_branch(self):
        plain = _head(seed=3)
        shortcut = _head(seed=3, shortcut_enabled=True)
        added = set(shortcut.state_dict()) - set(plain.state_dict())
        assert added and all(key.startswith("model.dt_encoder.") for key in added)
        assert not set(plain.state_dict()) - set(shortcut.state_dict())

    def test_identity_breaks_once_the_branch_is_trained(self):
        """Control: without it, the tests above would also pass on a dt branch that
        was wired up but never actually read."""
        _, shortcut = self._aligned_pair(4)
        before = _sample(shortcut)
        with torch.no_grad():
            shortcut.model.dt_encoder.timestep_embedder.linear_2.weight.normal_(std=0.5)
        assert not torch.equal(before, _sample(shortcut))


class TestUnsupportedBudgets:
    @pytest.mark.parametrize("steps", [3, 5, 6, 100])
    def test_non_power_of_two_is_rejected(self, steps):
        head = _head(shortcut_enabled=True, num_inference_timesteps=steps)
        with pytest.raises(ValueError, match="power of two"):
            _sample(head)

    def test_budget_beyond_the_trained_levels_is_rejected(self):
        """Level 3 (8 steps) was never trained at the default 3 levels."""
        head = _head(shortcut_enabled=True, num_inference_timesteps=8)
        with pytest.raises(ValueError, match="shortcut_num_levels"):
            _sample(head)

    def test_the_same_budget_is_fine_with_more_levels(self):
        head = _head(shortcut_enabled=True, num_inference_timesteps=8, shortcut_num_levels=4)
        assert _sample(head).shape == (BATCH, HORIZON, ACT_DIM)

    @pytest.mark.parametrize("steps", [3, 8])
    def test_plain_model_still_accepts_any_budget(self, steps):
        """The restriction is a property of the shortcut objective, not of the model."""
        head = _head(num_inference_timesteps=steps)
        assert _sample(head).shape == (BATCH, HORIZON, ACT_DIM)
