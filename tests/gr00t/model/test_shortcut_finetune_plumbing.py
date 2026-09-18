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

"""The shortcut flags must survive the trip from CLI to model.

Two links in that chain fail silently rather than loudly, which is why they are
pinned here instead of being left to a training run to reveal:

  * ``setup.py`` passes config overrides to ``AutoModel.from_pretrained`` as an
    explicit kwarg allowlist. A field missing from that list is dropped without a
    word, and training runs plain flow matching while every log line says otherwise.
  * ``launch_finetune.py`` copies FinetuneConfig onto the model config field by
    field. A field missing there never leaves the CLI.
"""

import ast
from dataclasses import fields
import inspect
from pathlib import Path

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import pytest
import torch


SHORTCUT_FIELDS = [
    "shortcut_enabled",
    "shortcut_num_levels",
    "shortcut_loss_weight",
    "shortcut_consistency_frac",
    "shortcut_time_distribution",
]


def _source(obj) -> str:
    return Path(inspect.getsourcefile(obj)).read_text()


class TestFieldsExistEverywhere:
    @pytest.mark.parametrize("name", SHORTCUT_FIELDS)
    def test_present_on_the_finetune_cli(self, name):
        assert name in {f.name for f in fields(FinetuneConfig)}

    @pytest.mark.parametrize("name", SHORTCUT_FIELDS)
    def test_present_on_the_model_config(self, name):
        assert hasattr(Gr00tN1d7Config(), name)

    @pytest.mark.parametrize("name", SHORTCUT_FIELDS)
    def test_defaults_agree_between_cli_and_model(self, name):
        cli = {f.name: f.default for f in fields(FinetuneConfig)}[name]
        assert cli == getattr(Gr00tN1d7Config(), name)


class TestAllowlistTrap:
    """The from_pretrained kwarg allowlist drops anything not named in it."""

    @staticmethod
    def _from_pretrained_kwargs() -> set[str]:
        tree = ast.parse(_source(Gr00tN1d7Pipeline))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
            ):
                return {kw.arg for kw in node.keywords if kw.arg}
        raise AssertionError("no AutoModel.from_pretrained call found in setup.py")

    @pytest.mark.parametrize("name", SHORTCUT_FIELDS)
    def test_field_is_forwarded_to_from_pretrained(self, name):
        assert name in self._from_pretrained_kwargs(), (
            f"{name} is missing from the from_pretrained allowlist in setup.py, so it "
            f"would be silently ignored and the run would train plain flow matching."
        )


class TestLaunchCopiesEveryField:
    @pytest.mark.parametrize("name", SHORTCUT_FIELDS)
    def test_field_is_copied_onto_the_model_config(self, name):
        from gr00t.experiment import launch_finetune

        source = Path(launch_finetune.__file__).read_text()
        assert f"config.model.{name} = ft_config.{name}" in source, (
            f"{name} is never copied from FinetuneConfig onto the model config, so the "
            f"CLI flag would have no effect."
        )


class TestCheckpointWithoutTheBranch:
    """Starting a shortcut finetune from an ordinary checkpoint is the normal case."""

    def test_missing_dt_encoder_keys_are_re_zeroed(self):
        head = self._shortcut_head()
        dit = head.model
        # Simulate HuggingFace's random re-initialisation of missing keys.
        with torch.no_grad():
            dit.dt_encoder.timestep_embedder.linear_2.weight.normal_(std=0.5)
            dit.dt_encoder.timestep_embedder.linear_2.bias.normal_(std=0.5)

        Gr00tN1d7Pipeline._init_missing_shortcut_weights(
            _Model(head), ["action_head.model.dt_encoder.timestep_embedder.linear_2.weight"]
        )
        assert dit.dt_encoder.timestep_embedder.linear_2.weight.abs().max().item() == 0.0
        assert dit.dt_encoder.timestep_embedder.linear_2.bias.abs().max().item() == 0.0

    def test_untouched_when_the_checkpoint_already_has_the_branch(self):
        head = self._shortcut_head()
        with torch.no_grad():
            head.model.dt_encoder.timestep_embedder.linear_2.weight.fill_(0.25)
        Gr00tN1d7Pipeline._init_missing_shortcut_weights(_Model(head), [])
        assert head.model.dt_encoder.timestep_embedder.linear_2.weight.abs().max().item() == 0.25

    def test_no_op_on_a_plain_model(self):
        head = self._shortcut_head(shortcut_enabled=False)
        assert head.model.dt_encoder is None
        Gr00tN1d7Pipeline._init_missing_shortcut_weights(
            _Model(head), ["anything.dt_encoder.weight"]
        )

    @staticmethod
    def _shortcut_head(shortcut_enabled=True):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        torch.manual_seed(0)
        return Gr00tN1d7ActionHead(
            Gr00tN1d7Config(
                backbone_embedding_dim=32,
                hidden_size=64,
                input_embedding_dim=64,
                max_state_dim=7,
                max_action_dim=7,
                action_horizon=4,
                state_history_length=1,
                max_num_embodiments=4,
                shortcut_enabled=shortcut_enabled,
                diffusion_model_cfg={
                    "num_attention_heads": 2,
                    "attention_head_dim": 32,
                    "num_layers": 1,
                    "output_dim": 64,
                    "norm_type": "ada_norm",
                    "interleave_self_attention": True,
                },
            )
        )


class _Model:
    """Minimal stand-in for the loaded model: only .action_head is read."""

    def __init__(self, action_head):
        self.action_head = action_head


class TestZeroInitSurvivesHuggingFace:
    """from_pretrained re-initialises missing keys, wiping DiT.__init__'s zero.

    Measured on the real 3B checkpoint before Gr00tN1d7._init_weights existed:
    |w|max = 0.066 on the dt branch and sampling drifted by up to 6e-2 at 4 steps --
    the model no longer started from the pretrained flow policy, so a failed shortcut
    experiment would have been indistinguishable from a bad initialisation.

    The override must cut both ways: zero a branch HuggingFace had to invent, and
    leave a branch that came from the checkpoint alone (HuggingFace marks loaded
    modules _is_hf_initialized and never calls this hook for them).
    """

    @staticmethod
    def _model(shortcut_enabled=True):
        torch.manual_seed(0)
        return _HeadOnlyGr00tN1d7(
            Gr00tN1d7Config(
                backbone_embedding_dim=32,
                hidden_size=64,
                input_embedding_dim=64,
                max_state_dim=7,
                max_action_dim=7,
                action_horizon=4,
                state_history_length=1,
                max_num_embodiments=4,
                shortcut_enabled=shortcut_enabled,
                diffusion_model_cfg={
                    "num_attention_heads": 2,
                    "attention_head_dim": 32,
                    "num_layers": 1,
                    "output_dim": 64,
                    "norm_type": "ada_norm",
                    "interleave_self_attention": True,
                },
            )
        )

    def test_reinitialised_dt_branch_is_forced_back_to_zero(self):
        model = self._model()
        linear_2 = model.action_head.model.dt_encoder.timestep_embedder.linear_2
        with torch.no_grad():
            linear_2.weight.normal_(std=0.5)
            linear_2.bias.normal_(std=0.5)

        # Exactly what HuggingFace does for a key the checkpoint did not carry.
        model._init_weights(linear_2)

        assert linear_2.weight.abs().max().item() == 0.0
        assert linear_2.bias.abs().max().item() == 0.0

    def test_other_layers_are_initialised_normally(self):
        """The override must not become a blanket zeroing of the model."""
        model = self._model()
        other = model.action_head.model.timestep_encoder.timestep_embedder.linear_2
        with torch.no_grad():
            other.weight.zero_()
        model._init_weights(other)
        assert other.weight.abs().max().item() > 0.0

    def test_no_op_on_a_plain_model(self):
        model = self._model(shortcut_enabled=False)
        assert model.action_head.model.dt_encoder is None
        linear = model.action_head.model.timestep_encoder.timestep_embedder.linear_2
        model._init_weights(linear)  # must not raise
        assert linear.weight.abs().max().item() > 0.0


class _HeadOnlyGr00tN1d7:
    """Built lazily so importing this module does not pull in the 2B backbone."""

    def __new__(cls, config):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
        from transformers import PreTrainedModel

        class _Stub(Gr00tN1d7):
            """A real Gr00tN1d7 subclass -- required, because the zero-arg super() in
            _init_weights resolves against Gr00tN1d7 -- with the backbone skipped."""

            def __init__(self, config):
                PreTrainedModel.__init__(self, config)
                self.config = config
                self.action_head = Gr00tN1d7ActionHead(config)

        return _Stub(config)
