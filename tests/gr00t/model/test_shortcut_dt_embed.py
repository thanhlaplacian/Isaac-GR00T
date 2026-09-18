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

"""Tests for the optional shortcut-model step-size (dt) conditioning on the DiT.

The whole point of the zero-init is that a model built with
``use_shortcut_dt_embed=True`` is *indistinguishable* from the pretrained flow
model until it is trained. If that is not true, a shortcut experiment cannot be
told apart from a badly initialised branch, so it is checked here rather than
discovered as a mysterious regression.
"""

import inspect

from gr00t.model.modules.dit import AlternateVLDiT, DiT, TimestepEncoder
import pytest
import torch


DIT_KWARGS = dict(
    num_attention_heads=2,
    attention_head_dim=16,
    num_layers=2,
    output_dim=32,
    norm_type="ada_norm",
    dropout=0.0,
    final_dropout=False,
    interleave_self_attention=True,
    cross_attention_dim=24,
)


def _inputs(batch=3, seq=5, enc_seq=7, dim=32):
    torch.manual_seed(0)
    hidden = torch.randn(batch, seq, dim)
    encoder = torch.randn(batch, enc_seq, 24)
    timestep = torch.randint(0, 1000, (batch,))
    image_mask = torch.zeros(batch, enc_seq, dtype=torch.bool)
    image_mask[:, : enc_seq // 2] = True
    attn_mask = torch.ones(batch, enc_seq, dtype=torch.bool)
    return hidden, encoder, timestep, image_mask, attn_mask


def _build(cls, use_shortcut, seed=0, **extra):
    torch.manual_seed(seed)
    model = cls(**DIT_KWARGS, use_shortcut_dt_embed=use_shortcut, **extra)
    return model.eval()


class TestDefaultOff:
    def test_dt_encoder_absent_by_default(self):
        assert _build(DiT, False).dt_encoder is None

    def test_default_forward_signature_unchanged_for_positional_callers(self):
        """scripts/deployment/export_onnx_n1d7.py calls the DiT positionally as
        ``self.dit(sa_embs, vl_embs, timestep, **kwargs)``. dt_level must therefore
        be the LAST parameter, or a new argument silently binds to the wrong slot.
        """
        for cls in (DiT, AlternateVLDiT):
            params = list(inspect.signature(cls.forward).parameters)
            assert params[-1] == "dt_level", f"{cls.__name__}: dt_level must be last, got {params}"
            assert params[1:4] == ["hidden_states", "encoder_hidden_states", "timestep"]


class TestConstructionIsInert:
    """Enabling the flag must not disturb the initialisation of anything else.

    The dt encoder is built last in __init__ precisely so that it draws from the RNG
    after every other submodule. If it were built earlier, every downstream weight
    would shift under the same seed and "shortcut vs. plain" comparisons would be
    confounded by initialisation noise.
    """

    @pytest.mark.parametrize("cls", [DiT, AlternateVLDiT])
    def test_every_shared_parameter_is_bit_identical(self, cls):
        extra = {"attend_text_every_n_blocks": 1} if cls is AlternateVLDiT else {}
        plain = dict(_build(cls, False, seed=11, **extra).named_parameters())
        shortcut = dict(_build(cls, True, seed=11, **extra).named_parameters())

        added = set(shortcut) - set(plain)
        assert added and all(name.startswith("dt_encoder.") for name in added), added
        assert not set(plain) - set(shortcut)

        for name, param in plain.items():
            assert torch.equal(param, shortcut[name]), f"{name} changed when the flag flipped"


class TestZeroInit:
    @pytest.mark.parametrize("cls", [DiT, AlternateVLDiT])
    def test_output_projection_is_zero(self, cls):
        model = _build(cls, True)
        embedder = model.dt_encoder.timestep_embedder
        assert embedder.linear_2.weight.abs().max().item() == 0.0
        assert embedder.linear_2.bias.abs().max().item() == 0.0

    def test_input_projection_is_not_zero(self):
        """Only linear_2 is zeroed; a fully zeroed branch could never train."""
        model = _build(DiT, True)
        assert model.dt_encoder.timestep_embedder.linear_1.weight.abs().max().item() > 0.0

    @pytest.mark.parametrize("cls", [DiT, AlternateVLDiT])
    def test_zero_init_is_idempotent_and_public(self, cls):
        """Checkpoint loading re-runs this after HF re-initialises missing keys."""
        model = _build(cls, True)
        torch.nn.init.normal_(model.dt_encoder.timestep_embedder.linear_2.weight)
        torch.nn.init.normal_(model.dt_encoder.timestep_embedder.linear_2.bias)
        model.zero_init_dt_embed()
        assert model.dt_encoder.timestep_embedder.linear_2.weight.abs().max().item() == 0.0
        assert model.dt_encoder.timestep_embedder.linear_2.bias.abs().max().item() == 0.0

    def test_zero_init_is_a_no_op_when_disabled(self):
        model = _build(DiT, False)
        model.zero_init_dt_embed()  # must not raise
        assert model.dt_encoder is None


class TestIdentityAtInit:
    """A zero-initialised dt branch must not change a single output bit."""

    def test_dit_output_identical_for_every_dt_level(self):
        hidden, encoder, timestep, _, _ = _inputs()
        plain = _build(DiT, False, seed=3)
        shortcut = _build(DiT, True, seed=3)

        reference = plain(hidden, encoder, timestep)
        assert torch.equal(shortcut(hidden, encoder, timestep), reference)
        for level in range(4):
            dt_level = torch.full((hidden.shape[0],), level, dtype=torch.long)
            out = shortcut(hidden, encoder, timestep, dt_level=dt_level)
            assert torch.equal(out, reference), f"dt_level={level} changed the output"

    def test_alternate_vl_dit_output_identical_for_every_dt_level(self):
        hidden, encoder, timestep, image_mask, attn_mask = _inputs()
        plain = _build(AlternateVLDiT, False, seed=3, attend_text_every_n_blocks=1)
        shortcut = _build(AlternateVLDiT, True, seed=3, attend_text_every_n_blocks=1)

        kwargs = dict(image_mask=image_mask, backbone_attention_mask=attn_mask)
        reference = plain(hidden, encoder, timestep, **kwargs)
        assert torch.equal(shortcut(hidden, encoder, timestep, **kwargs), reference)
        for level in range(4):
            dt_level = torch.full((hidden.shape[0],), level, dtype=torch.long)
            out = shortcut(hidden, encoder, timestep, dt_level=dt_level, **kwargs)
            assert torch.equal(out, reference), f"dt_level={level} changed the output"


class TestTrainability:
    def test_dt_branch_receives_gradient_despite_zero_init(self):
        """Zero output weights still produce a non-zero gradient, so the branch
        starts learning at the first optimizer step rather than staying dead."""
        hidden, encoder, timestep, _, _ = _inputs()
        model = _build(DiT, True, seed=3)
        model.train()
        dt_level = torch.full((hidden.shape[0],), 2, dtype=torch.long)
        model(hidden, encoder, timestep, dt_level=dt_level).pow(2).mean().backward()

        grad = model.dt_encoder.timestep_embedder.linear_2.weight.grad
        assert grad is not None
        assert grad.abs().max().item() > 0.0

    def test_parameter_cost_equals_one_timestep_encoder(self):
        """Guards the load-time check in the finetune path: the shortcut model must
        add exactly one TimestepEncoder's worth of parameters and nothing else."""
        plain = _build(DiT, False)
        shortcut = _build(DiT, True)
        delta = sum(p.numel() for p in shortcut.parameters()) - sum(
            p.numel() for p in plain.parameters()
        )
        expected = sum(
            p.numel() for p in TimestepEncoder(embedding_dim=plain.inner_dim).parameters()
        )
        assert delta == expected
