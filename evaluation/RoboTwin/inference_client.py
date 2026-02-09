#!/usr/bin/env python

import base64
import io
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from typing import Union
from urllib import request

import imageio
import numpy as np
import torch
import tyro

from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "third_party" / "RoboTwin"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "policy"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "description" / "utils"),
    ]
)

from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

from evaluation.RoboTwin.inference import (
    TASK_NAMES,
    build_task_args,
    class_decorator,
)


def _encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_array(payload: str) -> np.ndarray:
    buffer = io.BytesIO(base64.b64decode(payload.encode("utf-8")))
    return np.load(buffer, allow_pickle=False)


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class ClientArgs:
    server_url: str = "http://127.0.0.1:8000/infer"
    task_idx: int = 0
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    seed: int = 0
    image_history_interval: int = 15
    action_mode: str = "delta"  # delta | abs
    video_dir: Path = Path("videos")
    fps: int = 30
    decode_image_flag: bool = False
    debug: bool = False
    log_level: str = "WARNING"  # DEBUG | INFO | WARNING | ERROR
    infer_horizon: int = 30
    action_horizon_size: int = 50
    test_num: int = 100
    robot_type: tuple[int, ...] = (6, 1, 6, 1)


def infer_once(args: ClientArgs) -> None:
    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name)
    task_env = class_decorator(task_args["task_name"])

    logging.info("=" * 80)
    logging.info("Initializing environment...")
    logging.info("Task: %s, seed: %s", task_name, args.seed)

    task_env.suc = 0
    task_env.test_num = 0
    expert_check = True

    now_id = 0
    succ_seed = 0
    seed = args.seed
    st_seed = 100000 * (1 + seed)
    now_seed = st_seed
    test_num = args.test_num
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    succ_seeds = list(range(st_seed, st_seed * 2))

    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                task_env.setup_demo(
                    now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
                )
                episode_info = task_env.play_once()
                task_env.close_env()
            except (UnStableError, Exception):
                task_env.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

        if (not expert_check) or (task_env.plan_success and task_env.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        task_env.setup_demo(
            now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][args.instruction_type])
        task_env.set_instruction(instruction=instruction)

        succ = False
        action_plan = deque([], maxlen=args.action_horizon_size)
        replay_images = []
        head_color_list = []
        left_wrist_color_list = []
        right_wrist_color_list = []
        image_history_interval = args.image_history_interval
        action_dim = sum(args.robot_type)
        left_gripper_idx = sum(args.robot_type[0:2]) - 1
        right_gripper_idx = sum(args.robot_type[0:4]) - 1

        while task_env.take_action_cnt < task_env.step_lim:
            observation = task_env.get_obs()
            img = observation["observation"]["head_camera"]["rgb"]
            replay_images.append(img.copy())

            if len(action_plan) <= image_history_interval:
                left_wrist_img = observation["observation"]["left_camera"]["rgb"]
                right_wrist_img = observation["observation"]["right_camera"]["rgb"]

                head_color_list.append(torch.as_tensor(img).contiguous().float() / 255.0)
                left_wrist_color_list.append(torch.as_tensor(left_wrist_img).contiguous().float() / 255.0)
                right_wrist_color_list.append(torch.as_tensor(right_wrist_img).contiguous().float() / 255.0)

                while len(head_color_list) > image_history_interval + 1:
                    head_color_list.pop(0)
                    left_wrist_color_list.pop(0)
                    right_wrist_color_list.pop(0)

                past_idx = max(len(head_color_list) - image_history_interval - 1, 0)
                image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
                image_hand_left_with_history = torch.stack(
                    [left_wrist_color_list[past_idx], left_wrist_color_list[-1]], dim=0
                )
                image_hand_right_with_history = torch.stack(
                    [right_wrist_color_list[past_idx], right_wrist_color_list[-1]], dim=0
                )

            if not action_plan:
                init_action = torch.as_tensor(observation["joint_action"]["vector"][None]).contiguous()
                state = torch.from_numpy(observation["joint_action"]["vector"]).float()
                task = task_env.get_instruction()

                payload = {
                    "images": {
                        "image0": _encode_array(image_head_with_history.cpu().numpy()),
                        "image1": _encode_array(image_hand_left_with_history.cpu().numpy()),
                        "image2": _encode_array(image_hand_right_with_history.cpu().numpy()),
                    },
                    "state": _encode_array(state.cpu().numpy()),
                    "task": task,
                    "init_action": _encode_array(init_action.cpu().numpy()),
                    "robot_type": list(args.robot_type),
                    "infer_horizon": args.infer_horizon,
                    "action_mode": args.action_mode,
                    "decode_image": args.decode_image_flag,
                }

                response = _post_json(args.server_url, payload)
                action_pred = _decode_array(response["action"])
                action_plan.extend(action_pred[: args.infer_horizon, :action_dim])

            action = action_plan.popleft()
            action[left_gripper_idx] = 0 if action[left_gripper_idx] < 0.5 else 1
            action[right_gripper_idx] = 0 if action[right_gripper_idx] < 0.5 else 1
            task_env.take_action(action, action_type="qpos")

            if task_env.eval_success:
                succ = True
                break

        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        args.video_dir.mkdir(parents=True, exist_ok=True)
        suffix = "success" if succ else "failure"
        imageio.mimwrite(
            args.video_dir / f"{suffix}_{succ_seed}.mp4",
            replay_images,
            fps=args.fps,
        )

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m |  \033[92m{task_args['task_config']}\033[0m \033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc/task_env.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1


def main(args: ClientArgs) -> None:
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    logging.info("Starting client inference...")
    logging.info("Task index: %s, server: %s", args.task_idx, args.server_url)

    infer_once(args)


if __name__ == "__main__":
    tyro.cli(main)
