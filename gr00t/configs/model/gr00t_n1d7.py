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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    letter_box_transform: bool = False
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Shortcut model parameters (Frans et al., One Step Diffusion via Shortcut Models).
    # All default-off: with shortcut_enabled=False the model trains and samples exactly
    # as plain flow matching, and every field below is ignored.
    shortcut_enabled: bool = False
    """Train with the shortcut self-consistency objective alongside flow matching."""

    shortcut_num_levels: int = 3
    """Number of step-size levels. Level L means 2**L Euler steps, so the default 3
    covers inference at 1, 2 and 4 steps. The finest level (num_levels - 1) is the one
    the flow-matching loss trains; the coarser levels are trained by self-consistency,
    each bootstrapping from the level one finer."""

    shortcut_loss_weight: float = 1.0
    """Weight of the self-consistency term relative to the flow-matching term."""

    shortcut_consistency_frac: float = 0.5
    """Fraction of each batch that also gets a self-consistency target. The
    flow-matching loss is always computed on the whole batch, so this trades training
    FLOPs (two extra denoiser forwards per selected row) against how fast the coarse
    levels learn. 0.25 is close to the reference implementation's batch split; 1.0
    matches SORL."""

    shortcut_time_distribution: str = "beta"
    """Noise-level distribution for the flow-matching term: "beta" keeps GR00T's
    Beta(noise_beta_alpha, noise_beta_beta) sampler, "uniform" uses the shortcut
    paper's U[0, 1). The self-consistency term always samples on its level's grid
    regardless of this setting, since consistency only has to hold where the sampler
    actually lands."""

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

        self._validate_shortcut_config()

    def _validate_shortcut_config(self) -> None:
        """Reject shortcut settings that would fail silently rather than loudly.

        Only runs when the objective is enabled, so checkpoints predating these fields
        keep loading unchanged.
        """
        # Checked unconditionally: shortcut_time_distribution changes the
        # flow-matching noise schedule on its own, with or without the objective, so a
        # typo here must not be silently read as "beta".
        if self.shortcut_time_distribution not in ("beta", "uniform"):
            raise ValueError(
                f"shortcut_time_distribution must be 'beta' or 'uniform' (got "
                f"{self.shortcut_time_distribution!r})."
            )

        if not getattr(self, "shortcut_enabled", False):
            return

        if self.shortcut_num_levels < 2:
            raise ValueError(
                f"shortcut_num_levels must be >= 2 (got {self.shortcut_num_levels}): "
                "self-consistency needs at least one coarse level to bootstrap from a "
                "finer one."
            )
        if not 0.0 < self.shortcut_consistency_frac <= 1.0:
            raise ValueError(
                f"shortcut_consistency_frac must be in (0, 1] (got "
                f"{self.shortcut_consistency_frac}); 0 would enable the objective while "
                "training nothing with it."
            )
        if self.shortcut_loss_weight < 0:
            raise ValueError(
                f"shortcut_loss_weight must be >= 0 (got {self.shortcut_loss_weight})."
            )
        # Consistency targets land on the half-step grid of the finest level, so every
        # visited time must map to an exact timestep bucket. Off-grid rounding would
        # quietly train the teacher at a slightly different noise level than the
        # student, and nothing downstream would report it.
        finest = 2 ** (self.shortcut_num_levels - 1)
        if self.num_timestep_buckets % finest != 0:
            raise ValueError(
                f"num_timestep_buckets ({self.num_timestep_buckets}) must be divisible "
                f"by 2**(shortcut_num_levels - 1) = {finest}, otherwise the denoising "
                "grid does not land on exact buckets."
            )

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
