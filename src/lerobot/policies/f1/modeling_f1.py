#!/usr/bin/env python

import contextlib
import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from lerobot.policies.f1.configuration_f1 import F1Config
from lerobot.policies.f1.wan_vae import WanVAE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import format_big_number


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be shape [batch]")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = (2.0 * math.pi / period)[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


class F1DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.img_norm = nn.LayerNorm(hidden_size)
        self.img_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.vid_norm = nn.LayerNorm(hidden_size)
        self.vid_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        img_ctx: torch.Tensor,
        vid_ctx: torch.Tensor,
        img_key_padding_mask: torch.Tensor | None = None,
        vid_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.self_norm(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h

        h = self.img_norm(x)
        h, _ = self.img_cross_attn(
            h,
            img_ctx,
            img_ctx,
            need_weights=False,
            key_padding_mask=img_key_padding_mask,
        )
        x = x + h

        h = self.vid_norm(x)
        h, _ = self.vid_cross_attn(
            h,
            vid_ctx,
            vid_ctx,
            need_weights=False,
            key_padding_mask=vid_key_padding_mask,
        )
        x = x + h

        x = x + self.mlp(self.mlp_norm(x))
        return x


class F1Model(nn.Module):
    def __init__(self, config: F1Config):
        super().__init__()
        self.config = config

        self.siglip = SiglipVisionModel.from_pretrained(config.siglip_model_id)
        siglip_hidden_size = self.siglip.config.hidden_size
        self.siglip_proj = nn.Linear(siglip_hidden_size, config.hidden_size)

        wan_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.wan_vae = WanVAE(
            z_dim=16,
            repo_id=config.wan_vae_repo_id,
            ckpt_relpath=config.wan_vae_ckpt_relpath,
            local_path=config.wan_vae_local_path,
            local_files_only=config.wan_vae_local_files_only,
            dtype=wan_dtype,
            device=config.device,
        )
        self.wan_proj = nn.Linear(16, config.hidden_size)

        self.state_proj = nn.Linear(config.max_state_dim, config.hidden_size)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.hidden_size)
        self.action_out_proj = nn.Linear(config.hidden_size, config.max_action_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                F1DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

        if config.freeze_siglip:
            self.siglip.eval()
            for p in self.siglip.parameters():
                p.requires_grad = False

        if config.freeze_wan_vae:
            self.wan_vae.model.eval()
            self.wan_vae.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_siglip:
            self.siglip.eval()
        if self.config.freeze_wan_vae:
            self.wan_vae.model.eval()
        return self

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        return time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset

    def _encode_siglip(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        current = images[:, 0] * 2.0 - 1.0
        siglip_ctx = torch.no_grad() if self.config.freeze_siglip else contextlib.nullcontext()
        with siglip_ctx:
            outputs = self.siglip(pixel_values=current)
        img_ctx = self.siglip_proj(outputs.last_hidden_state)

        valid = image_mask.bool()
        img_ctx = torch.where(valid[:, None, None], img_ctx, torch.zeros_like(img_ctx))

        key_padding = (~valid)[:, None].expand(valid.shape[0], img_ctx.shape[1])
        return img_ctx, key_padding

    def _encode_wan(
        self,
        images: torch.Tensor,
        image_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        videos = images * 2.0 - 1.0
        wan_ctx = torch.no_grad() if self.config.freeze_wan_vae else contextlib.nullcontext()
        with wan_ctx:
            latents = self.wan_vae.encode_tensor(videos, pad_mask=image_is_pad.bool())
        # [B, C, T, H, W] -> [B, N, C]
        vid_ctx = rearrange(latents, "b c t h w -> b (t h w) c")
        vid_ctx = self.wan_proj(vid_ctx)

        vid_key_padding = None
        if image_is_pad.ndim == 2:
            temporal_pad = image_is_pad
            latent_t = latents.shape[2]
            if temporal_pad.shape[1] != latent_t:
                src_idx = (
                    torch.linspace(0, temporal_pad.shape[1] - 1, steps=latent_t, device=temporal_pad.device)
                    .round()
                    .long()
                )
                temporal_pad = temporal_pad.index_select(1, src_idx)
            b, t, h, w = latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4]
            vid_key_padding = temporal_pad[:, :, None].expand(b, t, h * w).reshape(b, t * h * w)

        return vid_ctx, vid_key_padding

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        image_is_pad: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        state = state.to(torch.float32)
        actions = actions.to(torch.float32)
        images = images.to(torch.float32)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        u_t = noise - actions

        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
        )
        cond = self.state_proj(state) + self.time_mlp(time_emb)

        x = self.action_in_proj(x_t) + cond[:, None, :]

        img_ctx, img_kpm = self._encode_siglip(images, image_mask)
        vid_ctx, vid_kpm = self._encode_wan(images, image_is_pad)

        for block in self.blocks:
            x = block(x, img_ctx, vid_ctx, img_kpm, vid_kpm)

        v_t = self.action_out_proj(self.final_norm(x)).to(torch.float32)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses


class F1Policy(PreTrainedPolicy):
    config_class = F1Config
    name = "f1"

    def __init__(self, config: F1Config):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = F1Model(config)
        self.model.to(config.device)
        self.reset()

    def __str__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            "=" * 60
            + f"\nPolicy: {self.__class__.__name__}\n"
            + f"Total params     : {total} ({format_big_number(total)})\n"
            + f"Trainable params : {trainable} ({format_big_number(trainable)})\n"
            + "=" * 60
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _preprocess_images(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image0 = batch[f"{OBS_IMAGES}.image0"]
        image0_mask = batch[f"{OBS_IMAGES}.image0_mask"].bool()
        image0_is_pad = batch[f"{OBS_IMAGES}.image0_is_pad"].bool()

        if image0_mask.ndim > 1:
            image0_mask = image0_mask.reshape(image0_mask.shape[0], -1).any(dim=1)

        if image0.ndim == 4:
            image0 = image0[:, None]

        if image0_is_pad.ndim == 1:
            image0_is_pad = image0_is_pad[:, None]

        if image0_is_pad.shape[1] != image0.shape[1]:
            image0_is_pad = torch.zeros(
                (image0.shape[0], image0.shape[1]),
                dtype=torch.bool,
                device=image0.device,
            )

        image0_is_pad = image0_is_pad | (~image0_mask[:, None])
        return image0, image0_mask, image0_is_pad

    def prepare_state(self, batch: dict[str, Tensor]) -> torch.Tensor:
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> torch.Tensor:
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise RuntimeError("f1 is training-only and does not support predict_action_chunk")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise RuntimeError("f1 is training-only and does not support select_action")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        images, image_mask, image_is_pad = self._preprocess_images(batch)

        losses = self.model(
            state=state,
            actions=actions,
            images=images,
            image_mask=image_mask,
            image_is_pad=image_is_pad,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss = losses.mean()

        per_dim = losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        loss_dict = {"loss": loss.item()}
        loss_dict.update({f"loss_action_dim{i}": per_dim[i] for i in range(original_action_dim)})

        return loss, loss_dict
