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

"""Tests for the shortcut self-consistency training objective."""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


BATCH, ENC_SEQ, HORIZON, ACT_DIM, STATE_DIM = 4, 6, 4, 7, 7


def _config(**overrides) -> Gr00tN1d7Config:
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
        num_inference_timesteps=4,
        # Real production values: dropout is exactly what the teacher must avoid.
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
    return Gr00tN1d7Config(**defaults)


def _head(seed=0, **overrides) -> Gr00tN1d7ActionHead:
    torch.manual_seed(seed)
    return Gr00tN1d7ActionHead(_config(**overrides)).to(torch.float32)


def _batch(seed=1):
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
            "action": torch.randn(BATCH, HORIZON, ACT_DIM),
            "embodiment_id": torch.zeros(BATCH, dtype=torch.long),
            "action_mask": torch.ones(BATCH, HORIZON, ACT_DIM),
        }
    )
    return backbone, action_input


class TestDisabledIsUnchanged:
    def test_loss_identical_to_plain_flow_matching_at_the_same_seed(self):
        backbone, action_input = _batch()
        head = _head()
        head.eval()

        torch.manual_seed(42)
        first = head(backbone, action_input)["loss"].item()
        torch.manual_seed(42)
        second = head(backbone, action_input)["loss"].item()
        assert first == second

    def test_no_extra_keys_when_disabled(self):
        backbone, action_input = _batch()
        out = _head().eval()(backbone, action_input)
        assert "consistency_loss" not in out
        assert "flow_loss" not in out

    def test_dit_has_no_dt_encoder_when_disabled(self):
        assert _head().model.dt_encoder is None


class TestEnabled:
    def test_dit_gets_the_dt_encoder(self):
        assert _head(shortcut_enabled=True).model.dt_encoder is not None

    def test_loss_is_finite_and_reports_both_terms(self):
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()
        out = head(backbone, action_input)

        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["consistency_loss"])
        assert torch.isfinite(out["flow_loss"])
        assert out["loss"].item() == pytest.approx(
            out["flow_loss"].item()
            + head.config.shortcut_loss_weight * out["consistency_loss"].item(),
            rel=1e-5,
        )

    def test_consistency_term_is_not_trivially_zero(self):
        """A consistency loss that starts at ~0 means dt_level is being ignored and
        the two branches are computing the same thing."""
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()
        torch.manual_seed(3)
        assert head(backbone, action_input)["consistency_loss"].item() > 1e-8

    def test_dt_encoder_receives_gradient(self):
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()
        head(backbone, action_input)["loss"].backward()

        grad = head.model.dt_encoder.timestep_embedder.linear_2.weight.grad
        assert grad is not None and grad.abs().max().item() > 0.0

    @pytest.mark.parametrize("frac", [0.25, 0.5, 1.0])
    def test_every_consistency_fraction_runs(self, frac):
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True, shortcut_consistency_frac=frac)
        head.train()
        assert torch.isfinite(head(backbone, action_input)["loss"])

    def test_zero_weight_reduces_to_the_flow_loss(self):
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True, shortcut_loss_weight=0.0)
        head.train()
        out = head(backbone, action_input)
        assert out["loss"].item() == pytest.approx(out["flow_loss"].item(), rel=1e-6)


class TestTeacherDropout:
    """The teacher must run in eval() mode, not merely under no_grad().

    diffusion_model_cfg carries dropout=0.2 and final_dropout=True, and forward()
    runs under model.train(). Two teacher half-steps drawn under independent dropout
    masks do not compose into a valid two-step trajectory, so every bootstrap target
    would be corrupted -- and the loss would stay finite and keep decreasing while it
    happened. Neither reference implementation has dropout in its denoiser, so
    neither protects against this.

    These tests deliberately do NOT reset the seed between the two calls: consecutive
    forwards must draw different dropout masks for the check to mean anything. An
    end-to-end seeded comparison passes whether or not the guard is present.
    """

    def _denoise_inputs(self, head, backbone, action_input):
        encoded = head.process_backbone_output(
            BatchFeature(data={k: v.clone() for k, v in backbone.items()})
        )
        state = action_input["state"].view(BATCH, 1, -1)
        return dict(
            actions=action_input["action"],
            timesteps_tensor=torch.full((BATCH,), 250, dtype=torch.long),
            embodiment_id=action_input["embodiment_id"],
            state_features=head.state_encoder(state, action_input["embodiment_id"]),
            vl_embeds=encoded["backbone_features"],
            backbone_output=encoded,
            dt_level=torch.ones(BATCH, dtype=torch.long),
        )

    def test_dropout_really_is_active_in_train_mode(self):
        """Control. Without this, the test below could pass on a dropout-free model
        and prove nothing."""
        head = _head(shortcut_enabled=True)
        head.train()
        kwargs = self._denoise_inputs(head, *_batch())
        torch.manual_seed(0)
        with torch.no_grad():
            first = head._denoise_step(**kwargs)
            second = head._denoise_step(**kwargs)
        assert not torch.equal(first, second)

    def test_teacher_eval_makes_consecutive_forwards_identical(self):
        head = _head(shortcut_enabled=True)
        head.train()
        kwargs = self._denoise_inputs(head, *_batch())
        torch.manual_seed(0)
        with head._teacher_eval(), torch.no_grad():
            first = head._denoise_step(**kwargs)
            second = head._denoise_step(**kwargs)
        assert torch.equal(first, second)

    def test_every_stochastic_module_is_in_eval_inside_the_context(self):
        head = _head(shortcut_enabled=True)
        head.train()
        with head._teacher_eval():
            assert not head.model.training
            assert not head.action_encoder.training
            assert not head.action_decoder.training

    def test_train_mode_is_restored_afterwards(self):
        head = _head(shortcut_enabled=True)
        head.train()
        with head._teacher_eval():
            pass
        assert head.model.training
        assert head.action_encoder.training
        assert head.action_decoder.training

    def test_eval_mode_is_also_restored(self):
        head = _head(shortcut_enabled=True)
        head.eval()
        with head._teacher_eval():
            pass
        assert not head.model.training

    def test_mode_is_restored_even_if_the_body_raises(self):
        head = _head(shortcut_enabled=True)
        head.train()
        with pytest.raises(RuntimeError):
            with head._teacher_eval():
                raise RuntimeError("boom")
        assert head.model.training

    def test_consistency_loss_runs_the_teacher_in_eval_and_the_student_in_train(self):
        """Pins the guard at its call site, not just the context manager itself.

        The context manager can be perfectly correct and still not be used. This spies
        on every denoiser call the consistency loss makes and asserts the exact
        pattern: two teacher half-steps in eval + no_grad, then one student forward in
        train + grad.
        """
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()

        seen = []
        original = head._denoise_step

        def spy(*args, **kwargs):
            seen.append((head.model.training, torch.is_grad_enabled()))
            return original(*args, **kwargs)

        object.__setattr__(head, "_denoise_step", spy)
        try:
            self._run_consistency(head, backbone, action_input)
        finally:
            object.__delattr__(head, "_denoise_step")

        assert seen == [(False, False), (False, False), (True, True)], seen

    @staticmethod
    def _run_consistency(head, backbone, action_input, seed=5):
        encoded = head.process_backbone_output(
            BatchFeature(data={k: v.clone() for k, v in backbone.items()})
        )
        state = action_input["state"].view(BATCH, 1, -1)
        torch.manual_seed(seed)
        return head._shortcut_consistency_loss(
            actions=action_input["action"],
            noise=torch.randn(action_input["action"].shape),
            embodiment_id=action_input["embodiment_id"],
            state_features=head.state_encoder(state, action_input["embodiment_id"]),
            vl_embeds=encoded["backbone_features"],
            backbone_output=encoded,
            action_mask=action_input["action_mask"],
        )

    def test_bootstrap_chain_is_wired_one_level_finer(self):
        """The teacher must run one level FINER than the student, at the half-step.

        If teacher and student share a level, the objective degenerates: the model is
        asked to match itself at the same step size, the level hierarchy is never
        learned, and the loss still looks perfectly healthy while 1-step quality goes
        nowhere.
        """
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()

        calls = []
        original = head._denoise_step

        def spy(*args, **kwargs):
            calls.append((kwargs["dt_level"].clone(), kwargs["timesteps_tensor"].clone()))
            return original(*args, **kwargs)

        object.__setattr__(head, "_denoise_step", spy)
        try:
            self._run_consistency(head, backbone, action_input)
        finally:
            object.__delattr__(head, "_denoise_step")

        (t1_level, t1_time), (t2_level, t2_time), (s_level, s_time) = calls

        # Both teacher half-steps sit at the same, finer level.
        assert torch.equal(t1_level, t2_level)
        assert torch.equal(t1_level, s_level + 1)

        # Student levels stay in the range consistency is defined for.
        num_levels = head.config.shortcut_num_levels
        assert (s_level >= 0).all() and (s_level <= num_levels - 2).all()

        # The student and the first half-step start at the same time; the second
        # half-step is strictly later.
        assert torch.equal(s_time, t1_time)
        assert (t2_time > t1_time).all()

    @staticmethod
    def _record(head, backbone, action_input, seed=5):
        """Run the consistency loss, capturing every denoiser call's input and output."""
        records = []
        original = head._denoise_step

        def spy(**kwargs):
            out = original(**kwargs)
            records.append(
                {
                    "actions": kwargs["actions"].detach().clone(),
                    "dt_level": kwargs["dt_level"].clone(),
                    "timesteps": kwargs["timesteps_tensor"].clone(),
                    "out": out.detach().clone(),
                }
            )
            return out

        object.__setattr__(head, "_denoise_step", spy)
        try:
            loss = TestTeacherDropout._run_consistency(head, backbone, action_input, seed)
        finally:
            object.__delattr__(head, "_denoise_step")
        return loss, records

    def test_target_is_the_mean_of_two_chained_half_steps(self):
        """Numerical oracle for the bootstrap target.

        Reconstructs what the loss must be from the captured calls. Catches the
        mutations that leave the structure intact but the arithmetic wrong: not
        advancing between the half-steps, or averaging fewer than both of them.
        """
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()
        loss, records = self._record(head, backbone, action_input)
        assert len(records) == 3
        first, second, student = records

        # The second half-step must start where the first one landed: one Euler step
        # of the TEACHER's size (half the student's) along the first velocity.
        teacher_step = 1.0 / torch.pow(2.0, first["dt_level"].to(torch.float32))
        expected_mid = first["actions"] + teacher_step[:, None, None] * first["out"]
        assert torch.allclose(second["actions"], expected_mid, atol=1e-5), (
            "the second half-step did not start from the first one's endpoint"
        )

        # The student must be asked about the same starting point as the teacher.
        assert torch.equal(student["actions"], first["actions"])

        # And the loss must be the masked MSE against the mean of BOTH velocities.
        target = (first["out"] + second["out"]) / 2
        mask = torch.ones_like(target)
        expected = (((student["out"] - target) ** 2) * mask).sum() / (mask.sum() + 1e-6)
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_start_times_span_the_level_grid(self):
        """Consistency must be trained away from t=0 too, not only at pure noise."""
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()

        seen = set()
        for seed in range(25):
            _, records = self._record(head, backbone, action_input, seed=seed)
            seen.update(records[2]["timesteps"].tolist())

        assert 0 in seen, f"expected t=0 to be reachable, saw {sorted(seen)}"
        assert any(t > 0 for t in seen), (
            f"every consistency start time was t=0; the on-grid sampler is not "
            f"sampling. Saw: {sorted(seen)}"
        )
        # With 3 levels the student grid is {0} at level 0 and {0, 1/2} at level 1,
        # so 500 is the only other bucket that may appear.
        assert seen <= {0, 500}, f"off-grid start times: {sorted(seen)}"

    def test_teacher_target_carries_no_gradient(self):
        """Stop-grad: the bootstrap target is a constant to the optimizer, while the
        student branch still trains."""
        backbone, action_input = _batch()
        head = _head(shortcut_enabled=True)
        head.train()
        encoded = head.process_backbone_output(
            BatchFeature(data={k: v.clone() for k, v in backbone.items()})
        )
        state = action_input["state"].view(BATCH, 1, -1)
        torch.manual_seed(5)
        loss = head._shortcut_consistency_loss(
            actions=action_input["action"],
            noise=torch.randn(action_input["action"].shape),
            embodiment_id=action_input["embodiment_id"],
            state_features=head.state_encoder(state, action_input["embodiment_id"]),
            vl_embeds=encoded["backbone_features"],
            backbone_output=encoded,
            action_mask=action_input["action_mask"],
        )
        assert loss.requires_grad
        loss.backward()
        assert head.model.dt_encoder.timestep_embedder.linear_2.weight.grad is not None


class TestGridExactness:
    """Consistency times must land on exact timestep buckets.

    Off-grid rounding would train the teacher at a slightly different noise level
    than the student, and nothing downstream would report it.
    """

    def test_all_visited_buckets_are_exact_at_production_settings(self):
        cfg = _config(shortcut_enabled=True)
        buckets = cfg.num_timestep_buckets
        for level in range(cfg.shortcut_num_levels - 1):
            steps = 2**level
            for k in range(steps):
                t = k / steps
                half = 1.0 / (2 * steps)
                for time in (t, t + half):
                    scaled = time * buckets
                    assert scaled == int(scaled), f"level {level}, t={time} is off-grid"
