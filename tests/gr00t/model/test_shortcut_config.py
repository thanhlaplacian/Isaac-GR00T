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

"""Validation of the shortcut-objective config fields."""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
import pytest


class TestDefaultsAreOff:
    def test_disabled_by_default(self):
        assert Gr00tN1d7Config().shortcut_enabled is False

    def test_defaults_cover_one_two_and_four_steps(self):
        cfg = Gr00tN1d7Config()
        assert cfg.shortcut_num_levels == 3
        assert 2 ** (cfg.shortcut_num_levels - 1) == cfg.num_inference_timesteps == 4

    def test_objective_only_values_are_ignored_while_disabled(self):
        """Checkpoints predating these fields must keep loading, so the fields that
        only matter to the objective are validated only when it is switched on.

        shortcut_time_distribution is deliberately NOT in this group: it changes the
        flow-matching noise schedule on its own, so it is validated unconditionally
        (see test_typo_is_rejected_even_with_the_objective_off).
        """
        cfg = Gr00tN1d7Config(
            shortcut_num_levels=0, shortcut_consistency_frac=9.0, shortcut_loss_weight=-5.0
        )
        assert cfg.shortcut_enabled is False


class TestValidationWhenEnabled:
    def _cfg(self, **overrides):
        return Gr00tN1d7Config(shortcut_enabled=True, **overrides)

    def test_accepts_defaults(self):
        assert self._cfg().shortcut_enabled is True

    @pytest.mark.parametrize("levels", [0, 1, -1])
    def test_rejects_too_few_levels(self, levels):
        with pytest.raises(ValueError, match="shortcut_num_levels"):
            self._cfg(shortcut_num_levels=levels)

    @pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_fraction(self, frac):
        with pytest.raises(ValueError, match="shortcut_consistency_frac"):
            self._cfg(shortcut_consistency_frac=frac)

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="shortcut_loss_weight"):
            self._cfg(shortcut_loss_weight=-1.0)

    def test_rejects_unknown_time_distribution(self):
        with pytest.raises(ValueError, match="shortcut_time_distribution"):
            self._cfg(shortcut_time_distribution="lognormal")

    @pytest.mark.parametrize("dist", ["beta", "uniform"])
    def test_accepts_known_time_distributions(self, dist):
        assert self._cfg(shortcut_time_distribution=dist).shortcut_time_distribution == dist

    def test_rejects_bucket_count_that_does_not_divide_the_grid(self):
        """The production 1000 buckets are 2**3 * 5**3, so they divide the grid up to
        8 steps but not 16. Asking for 5 levels would put the teacher half-steps
        off-grid: the teacher would be trained at a slightly different noise level
        than the student, and nothing downstream would report it."""
        with pytest.raises(ValueError, match="num_timestep_buckets"):
            self._cfg(shortcut_num_levels=5, num_timestep_buckets=1000)

    def test_accepts_bucket_count_that_divides_the_grid(self):
        assert self._cfg(shortcut_num_levels=5, num_timestep_buckets=1024)
        # 1000 still supports up to 8 steps, which is past what we plan to use.
        assert self._cfg(shortcut_num_levels=4, num_timestep_buckets=1000)

    def test_production_defaults_land_on_exact_buckets(self):
        cfg = self._cfg()
        finest = 2 ** (cfg.shortcut_num_levels - 1)
        assert cfg.num_timestep_buckets % finest == 0
        # Every time the sampler visits at 1, 2 or 4 steps, plus the teacher half-steps.
        for level in range(cfg.shortcut_num_levels):
            for k in range(2**level):
                assert (k * cfg.num_timestep_buckets) % (2**level) == 0


class TestUniformTimeSampling:
    """The 'uniform' switch changes the flow-matching noise schedule.

    The 'beta' default must stay byte-identical, since it is the schedule every
    released checkpoint was trained under.
    """

    def _head(self, **overrides):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        cfg = Gr00tN1d7Config(
            backbone_embedding_dim=64,
            hidden_size=64,
            input_embedding_dim=64,
            max_state_dim=7,
            max_action_dim=7,
            action_horizon=4,
            state_history_length=1,
            max_num_embodiments=4,
            diffusion_model_cfg={
                "num_attention_heads": 2,
                "attention_head_dim": 32,
                "num_layers": 1,
                "output_dim": 64,
                "norm_type": "ada_norm",
                "interleave_self_attention": True,
            },
            **overrides,
        )
        return Gr00tN1d7ActionHead(cfg)

    def test_beta_stream_is_unchanged_by_the_new_branch(self):
        import torch

        head = self._head()
        torch.manual_seed(1234)
        got = head.sample_time(50_000, device="cpu", dtype=torch.float32)

        # Recompute the pre-change expression directly from the distribution.
        torch.manual_seed(1234)
        expected = (1 - head.beta_dist.sample([50_000])) * head.config.noise_s
        assert torch.equal(got, expected.to(torch.float32))

    def test_uniform_matches_its_own_moments(self):
        import torch

        head = self._head(shortcut_time_distribution="uniform")
        torch.manual_seed(0)
        sample = head.sample_time(200_000, device="cpu", dtype=torch.float32)
        noise_s = head.config.noise_s

        assert torch.isfinite(sample).all()
        assert (sample >= 0).all() and (sample <= noise_s).all()
        # U[0, noise_s): mean noise_s/2, variance noise_s**2/12.
        assert sample.mean().item() == pytest.approx(noise_s / 2, abs=5e-3)
        assert sample.var(unbiased=False).item() == pytest.approx(noise_s**2 / 12, rel=0.05)

    def test_uniform_and_beta_differ(self):
        """Guards against the switch being wired up but never taking effect."""
        import torch

        beta_head = self._head()
        uniform_head = self._head(shortcut_time_distribution="uniform")
        torch.manual_seed(7)
        beta = beta_head.sample_time(100_000, device="cpu", dtype=torch.float32)
        torch.manual_seed(7)
        uniform = uniform_head.sample_time(100_000, device="cpu", dtype=torch.float32)
        # Beta(1.5, 1.0) transformed has mean ~0.4 * noise_s; uniform has 0.5 * noise_s.
        assert abs(beta.mean().item() - uniform.mean().item()) > 0.05

    def test_typo_is_rejected_even_with_the_objective_off(self):
        with pytest.raises(ValueError, match="shortcut_time_distribution"):
            Gr00tN1d7Config(shortcut_time_distribution="unifrom")
