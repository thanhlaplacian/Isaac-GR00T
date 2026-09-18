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

import contextlib
import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Not part of diffusion_model_cfg: that dict is loaded verbatim from a
        # checkpoint's config.json and would not carry a newly added key.
        self.shortcut_enabled = config.shortcut_enabled
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
                use_shortcut_dt_embed=config.shortcut_enabled,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                use_shortcut_dt_embed=config.shortcut_enabled,
            )
            logger.info("Using DiT for diffusion model")
        if config.shortcut_enabled:
            logger.info(
                "Shortcut objective enabled: %d levels (%s steps), weight %.3g, "
                "consistency on %.0f%% of each batch, t ~ %s",
                config.shortcut_num_levels,
                "/".join(str(2**i) for i in range(config.shortcut_num_levels)),
                config.shortcut_loss_weight,
                100 * config.shortcut_consistency_frac,
                config.shortcut_time_distribution,
            )
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        """Sample the flow-matching noise level t, where t=0 is noise and t=1 is clean.

        The default "beta" path is GR00T's own schedule: Beta(1.5, 1.0) concentrates
        mass near 1, so (1 - sample) concentrates near 0 and the model sees mostly
        high-noise inputs. The shortcut paper instead samples t uniformly; which is
        better here is an open question, so it is a config switch rather than a
        replacement. Both are scaled by noise_s to keep t away from a fully clean
        sample.
        """
        if getattr(self.config, "shortcut_time_distribution", "beta") == "uniform":
            sample = torch.rand(batch_size, device=device, dtype=dtype)
            return sample * self.config.noise_s
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    @contextlib.contextmanager
    def _teacher_eval(self):
        """Run the bootstrap teacher with dropout disabled.

        The production diffusion_model_cfg sets dropout=0.2 and final_dropout=True,
        and forward() runs under model.train(). Two teacher half-steps drawn under
        independent dropout masks do not compose into a valid two-step trajectory, so
        every bootstrap target would be corrupted -- silently, because the loss stays
        finite and still decreases.

        Mirrors set_frozen_modules_to_eval_mode(), which exists for the same reason.
        """
        modules = [self.action_encoder, self.model, self.action_decoder]
        if self.config.add_pos_embed:
            modules.append(self.position_embedding)
        was_training = [m.training for m in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, flag in zip(modules, was_training):
                module.train(flag)

    def _time_to_bucket(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.num_timestep_buckets).long()

    def _shortcut_consistency_loss(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Self-consistency: one step of size d must equal two steps of size d/2.

        The target is built by the model itself one level finer, under stop-grad. The
        chain bootstraps downward: the flow-matching term trains the finest level
        directly, level num_levels-2 learns from it, and so on to level 0 (one step).

        No EMA teacher. Training here is supervised finetuning on a fixed dataset for
        a few thousand steps, not the off-policy RL setting an EMA target is there to
        stabilise; and under ZeRO-2 (which replicates parameters) an EMA copy would be
        a full second replica per rank that also lands in every checkpoint.
        """
        batch_size = actions.shape[0]
        n = max(1, int(round(batch_size * self.config.shortcut_consistency_frac)))
        rows = slice(0, n)  # the sampler already shuffles, so a prefix is a random subset
        device = actions.device
        dtype = actions.dtype

        # Levels 0 .. num_levels-2; each bootstraps from the level one finer.
        levels = torch.randint(0, self.config.shortcut_num_levels - 1, (n,), device=device)
        num_steps = torch.pow(2, levels).to(dtype)  # steps at this level
        step_size = 1.0 / num_steps

        # Start on this level's own grid: consistency only has to hold where the
        # sampler actually lands, and an on-grid t keeps every timestep bucket exact.
        k = (torch.rand(n, device=device, dtype=dtype) * num_steps).floor()
        k = torch.minimum(k, num_steps - 1)  # guard the rand()==1.0 corner
        t = k / num_steps

        t_b = t[:, None, None]
        x_t = (1 - t_b) * noise[rows] + t_b * actions[rows]

        sub_backbone = BatchFeature(
            data={
                key: (value[rows] if torch.is_tensor(value) else value)
                for key, value in backbone_output.items()
            }
        )
        shared = dict(
            embodiment_id=embodiment_id[rows],
            state_features=state_features[rows],
            vl_embeds=vl_embeds[rows],
            backbone_output=sub_backbone,
        )

        half = step_size / 2
        teacher_levels = levels + 1
        with self._teacher_eval(), torch.no_grad():
            v1 = self._denoise_step(
                actions=x_t,
                timesteps_tensor=self._time_to_bucket(t),
                dt_level=teacher_levels,
                **shared,
            )
            x_mid = x_t + half[:, None, None] * v1
            v2 = self._denoise_step(
                actions=x_mid,
                timesteps_tensor=self._time_to_bucket(t + half),
                dt_level=teacher_levels,
                **shared,
            )
            target = (v1 + v2) / 2

        pred = self._denoise_step(
            actions=x_t,
            timesteps_tensor=self._time_to_bucket(t),
            dt_level=levels,
            **shared,
        )

        mask = action_mask[rows]
        loss = F.mse_loss(pred, target, reduction="none") * mask
        return loss.sum() / (mask.sum() + 1e-6)

    def _denoise_step(
        self,
        actions: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
        encoder_attention_mask: torch.Tensor | None = None,
        dt_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity for a single denoising step.

        Shared by training (``forward``) and sampling (``get_action_with_features``)
        so the two cannot drift apart.

        ``encoder_attention_mask`` is threaded through only to keep both call sites
        passing exactly what they passed before this refactor. Neither DiT nor
        AlternateVLDiT actually reads it: both hardcode ``encoder_attention_mask=None``
        on every block, and AlternateVLDiT builds its own cross-attention masks from
        ``image_mask & backbone_attention_mask``.

        Returns:
            Predicted velocity over the action horizon, [B, action_horizon, action_dim].
        """
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=action_features.device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join state and action embeddings along the sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                dt_level=dt_level,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
                dt_level=dt_level,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        # Slice off the leading state token; equals action_horizon at both call sites.
        return pred[:, -action_features.shape[1] :]

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = self._time_to_bucket(t[:, 0, 0])

        # The flow-matching term is the infinitesimal-step limit, so it defines the
        # finest level. Kept on the FULL batch: it is what protects the 4-step
        # behaviour we must not regress, and diluting it would work against that.
        fm_dt_level = None
        if self.shortcut_enabled:
            fm_dt_level = torch.full(
                (actions.shape[0],),
                self.config.shortcut_num_levels - 1,
                device=actions.device,
                dtype=torch.long,
            )

        # NOTE: the previous code passed return_all_hidden_states=True here and then
        # discarded the second return value, so dropping it is a no-op.
        pred_actions = self._denoise_step(
            actions=noisy_trajectory,
            timesteps_tensor=t_discretized,
            embodiment_id=embodiment_id,
            state_features=state_features,
            vl_embeds=vl_embeds,
            backbone_output=backbone_output,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            dt_level=fm_dt_level,
        )

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        flow_loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        loss = flow_loss

        consistency_loss = None
        if self.shortcut_enabled:
            consistency_loss = self._shortcut_consistency_loss(
                actions=actions,
                noise=noise,
                embodiment_id=embodiment_id,
                state_features=state_features,
                vl_embeds=vl_embeds,
                backbone_output=backbone_output,
                action_mask=action_mask,
            )
            loss = loss + self.config.shortcut_loss_weight * consistency_loss

        output = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if self.shortcut_enabled:
            # Reported separately so the two terms can be watched apart: a consistency
            # loss that starts near zero means the model is ignoring dt_level.
            output["flow_loss"] = flow_loss.detach()
            output["consistency_loss"] = consistency_loss.detach()
        return output

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            pred_velocity = self._denoise_step(
                actions=actions,
                timesteps_tensor=timesteps_tensor,
                embodiment_id=embodiment_id,
                state_features=state_features,
                vl_embeds=vl_embeds,
                backbone_output=backbone_output,
            )

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
