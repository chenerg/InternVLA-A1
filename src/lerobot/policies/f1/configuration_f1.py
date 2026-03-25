#!/usr/bin/env python

from dataclasses import dataclass, field, replace
import logging

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.f1.transform_f1 import UnifyF1InputsTransformFn
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DeltaActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    TransformGroup,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@DatasetConfig.register_subclass("f1")
@dataclass
class F1DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(height=F1DatasetConfig.height, width=F1DatasetConfig.width),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=F1DatasetConfig.max_state_dim,
                    max_action_dim=F1DatasetConfig.max_action_dim,
                ),
                UnifyF1InputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)

        if self.action_mode == "delta":
            if not has_delta:
                logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
                inputs = [DeltaActionTransformFn(), *inputs]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)
        else:
            if has_delta:
                logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
                inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("f1")
@dataclass
class F1Config(PreTrainedConfig):
    siglip_model_id: str = "google/siglip-base-patch16-224"
    wan_vae_repo_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    wan_vae_ckpt_relpath: str = "cache/vae_step_411000.pth"
    wan_vae_local_path: str | None = None
    wan_vae_local_files_only: bool = False

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    image_delta_indices: list[int] = field(default_factory=lambda: [0, 4, 8, 12, 16])

    max_state_dim: int = 32
    max_action_dim: int = 32

    # DiT backbone.
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Flow-matching schedule.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)

    freeze_siglip: bool = True
    freeze_wan_vae: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.chunk_size != 16:
            raise ValueError(f"f1 requires chunk_size=16, got {self.chunk_size}")

        if self.n_action_steps != 16:
            raise ValueError(f"f1 requires n_action_steps=16, got {self.n_action_steps}")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if len(self.image_delta_indices) == 0:
            raise ValueError("image_delta_indices cannot be empty")

        if self.image_delta_indices[0] != 0:
            raise ValueError(
                f"image_delta_indices must start from current frame 0, got {self.image_delta_indices}"
            )

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        image0_key = f"{OBS_IMAGES}.image0"
        if image0_key not in self.input_features:
            self.input_features[image0_key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
