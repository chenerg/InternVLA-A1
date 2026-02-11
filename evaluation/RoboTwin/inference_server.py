import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES

json_numpy.patch()


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    ckpt_str = str(ckpt_path)
    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        return local_dir.resolve()

    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


def build_policy_and_transforms(ckpt_path: Union[str, Path], stats_key: str, resize_size: int, dtype: torch.dtype):
    ckpt_dir = resolve_ckpt_dir(ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    if not isinstance(config, QwenA1Config):
        raise ValueError(f"Expected QwenA1Config, got {type(config)}")

    policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
    policy.cuda().to(dtype).eval()

    stats = load_json(ckpt_dir / "stats.json")[stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    state_concat = {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}
    state_stat = {"observation.state": state_concat}

    action_concat = {k: np.asarray(stats["action"][k]) for k in stat_keys}
    action_stat = {"action": action_concat}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]
    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        ]
    )

    return policy, input_transforms, unnormalize_fn


class RoboTwinInfer:
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        stats_key: str = "aloha",
        resize_size: int = 224,
        dtype: str = "bfloat16",
        infer_horizon: int = 30,
        action_mode: str = "delta",
        robot_type: tuple[int, ...] = (6, 1, 6, 1),
    ):
        self.dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self.policy, self.input_transforms, self.unnormalize_fn = build_policy_and_transforms(
            ckpt_path, stats_key, resize_size, self.dtype
        )

        self.infer_horizon = infer_horizon
        self.action_mode = action_mode
        self.action_dim = sum(robot_type)
        self.left_gripper_idx = sum(robot_type[0:2]) - 1
        self.right_gripper_idx = sum(robot_type[0:4]) - 1

    def _build_sample(self, payload: Dict[str, Any]):
        required_keys = ["top", "left", "right", "state", "instruction"]
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing payload keys: {missing}")

        # duplicate current frame as 2-step history
        image_head = torch.as_tensor(payload["top"]).contiguous().cuda().to(self.dtype) / 255.0
        image_left = torch.as_tensor(payload["left"]).contiguous().cuda().to(self.dtype) / 255.0
        image_right = torch.as_tensor(payload["right"]).contiguous().cuda().to(self.dtype) / 255.0

        image_head_with_history = torch.stack([image_head, image_head], dim=0)
        image_left_with_history = torch.stack([image_left, image_left], dim=0)
        image_right_with_history = torch.stack([image_right, image_right], dim=0)

        state = torch.from_numpy(np.asarray(payload["state"])).float().cuda()
        task = payload["instruction"]

        sample = {
            f"{OBS_IMAGES}.image0": image_head_with_history,
            f"{OBS_IMAGES}.image1": image_left_with_history,
            f"{OBS_IMAGES}.image2": image_right_with_history,
            "observation.state": state,
            "task": task,
        }

        for key in list(sample.keys()):
            if OBS_IMAGES in key and "mask" not in key:
                sample[key] = sample[key].permute(0, 3, 1, 2)

        sample = self.input_transforms(sample)

        inputs = {}
        for key in sample:
            if key == "task":
                inputs[key] = [sample[key]]
            elif sample[key].dtype == torch.int64:
                inputs[key] = sample[key][None].cuda()
            else:
                inputs[key] = sample[key][None].cuda().to(dtype=self.dtype)

        inputs.update(
            {
                f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
            }
        )
        return inputs, state

    def inference(self, payload: Dict[str, Any]):
        inputs, state = self._build_sample(payload)

        self.policy.reset()
        start_time = time.time()
        with torch.no_grad():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)
        infer_ms = (time.time() - start_time) * 1000
        print(f"Model inference time: {infer_ms:.3f} ms")

        action_pred = action_pred[0, : self.infer_horizon, : self.action_dim]
        action_pred = self.unnormalize_fn({"action": action_pred})["action"]

        if self.action_mode == "delta":
            init_action = state[None]
            init_action[:, self.left_gripper_idx] = 0.0
            init_action[:, self.right_gripper_idx] = 0.0
            action_pred += init_action

        actions = action_pred.float().cpu().numpy()
        actions[:, self.left_gripper_idx] = (actions[:, self.left_gripper_idx] >= 0.5).astype(np.float32)
        actions[:, self.right_gripper_idx] = (actions[:, self.right_gripper_idx] >= 0.5).astype(np.float32)

        return actions


class RoboTwinServer:
    def __init__(self, cfg: "DeployConfig"):
        self.model = RoboTwinInfer(
            ckpt_path=cfg.ckpt_path,
            stats_key=cfg.stats_key,
            resize_size=cfg.resize_size,
            dtype=cfg.dtype,
            infer_horizon=cfg.infer_horizon,
            action_mode=cfg.action_mode,
            robot_type=cfg.robot_type,
        )

    def run(self, host: str = "0.0.0.0", port: int = 9000) -> None:
        app = FastAPI()

        @app.post("/act")
        async def act_endpoint(request: Request):
            payload = await request.json()
            actions = self.model.inference(payload)
            return actions.tolist() if hasattr(actions, "tolist") else actions

        uvicorn.run(app, host=host, port=port)


@dataclass
class DeployConfig:
    ckpt_path: Union[str, Path] = "InternRobotics/InternVLA-A1-3B-RoboTwin"
    stats_key: str = "aloha"
    resize_size: int = 224
    dtype: str = "bfloat16"
    infer_horizon: int = 30
    action_mode: str = "delta"
    robot_type: tuple[int, ...] = (6, 1, 6, 1)

    host: str = "0.0.0.0"
    port: int = 9000


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = RoboTwinServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
