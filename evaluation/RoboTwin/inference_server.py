#!/usr/bin/env python

import base64
import io
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import tyro

from lerobot.utils.constants import OBS_IMAGES

from evaluation.RoboTwin.inference import build_policy_and_transforms, resolve_ckpt_dir


def _encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_array(payload: str) -> np.ndarray:
    buffer = io.BytesIO(base64.b64decode(payload.encode("utf-8")))
    return np.load(buffer, allow_pickle=False)


@dataclass
class ServerArgs:
    ckpt_path: Union[str, Path] = "InternRobotics/InternVLA-A1-3B-RoboTwin"
    stats_key: str = "aloha"
    resize_size: int = 224
    dtype: str = "float32"  # float32 | bfloat16
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class InferenceServer(BaseHTTPRequestHandler):
    policy = None
    input_transforms = None
    unnormalize_fn = None
    dtype = torch.float32

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json(400, {"error": "Empty request"})
            return

        payload = json.loads(self.rfile.read(content_length))
        try:
            images = payload["images"]
            state = _decode_array(payload["state"])
            task = payload["task"]
            init_action = _decode_array(payload["init_action"])
            robot_type = payload["robot_type"]
            infer_horizon = payload["infer_horizon"]
            action_mode = payload["action_mode"]
            decode_image = payload.get("decode_image", False)

            sample = {
                f"{OBS_IMAGES}.image0": torch.from_numpy(_decode_array(images["image0"])),
                f"{OBS_IMAGES}.image1": torch.from_numpy(_decode_array(images["image1"])),
                f"{OBS_IMAGES}.image2": torch.from_numpy(_decode_array(images["image2"])),
                "observation.state": torch.from_numpy(state),
                "task": task,
            }

            for key in list(sample.keys()):
                if OBS_IMAGES in key and "mask" not in key:
                    image = sample[key].permute(0, 3, 1, 2)
                    sample[key] = image

            sample = self.input_transforms(sample)

            inputs: dict[str, Any] = {}
            for key, value in sample.items():
                if key == "task":
                    inputs[key] = [value]
                elif value.dtype == torch.int64:
                    inputs[key] = value[None].cuda()
                else:
                    inputs[key] = value[None].cuda().to(dtype=self.dtype)

            inputs.update(
                {
                    f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
                }
            )

            action_dim = sum(robot_type)
            left_gripper_idx = sum(robot_type[0:2]) - 1
            right_gripper_idx = sum(robot_type[0:4]) - 1

            with torch.no_grad():
                action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=decode_image)

            action_pred = action_pred[0, :infer_horizon, :action_dim]
            action_pred = self.unnormalize_fn({"action": action_pred})["action"]

            if action_mode == "delta":
                init_action = torch.from_numpy(init_action).to(action_pred.device)
                init_action[:, left_gripper_idx] = 0.0
                init_action[:, right_gripper_idx] = 0.0
                action_pred += init_action

            response = {"action": _encode_array(action_pred.cpu().numpy())}
            self._send_json(200, response)
        except Exception as exc:
            logging.exception("Failed to run inference")
            self._send_json(500, {"error": str(exc)})


def main(args: ServerArgs) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
    policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
        ckpt_dir, args.stats_key, args.resize_size, dtype
    )

    InferenceServer.policy = policy
    InferenceServer.input_transforms = input_transforms
    InferenceServer.unnormalize_fn = unnormalize_fn
    InferenceServer.dtype = dtype

    server = ThreadingHTTPServer((args.host, args.port), InferenceServer)
    logging.info("Inference server listening on http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
