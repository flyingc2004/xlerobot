# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from typing import cast

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def main():
    parser = argparse.ArgumentParser(description="Record dataset with XLerobot using keyboard teleop.")
    parser.add_argument("--remote-ip", default="localhost", help="Host IP for XLerobotClient")
    parser.add_argument("--robot-id", default="my_xlerobot", help="Robot ID")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES, help="Number of episodes to record")
    parser.add_argument("--episode-time-sec", type=int, default=EPISODE_TIME_SEC, help="Time per episode in seconds")
    parser.add_argument("--reset-time-sec", type=int, default=RESET_TIME_SEC, help="Reset time between episodes")
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency")
    parser.add_argument("--task-description", default=TASK_DESCRIPTION, help="Task description")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID, help="HuggingFace repo ID for dataset")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Optional output directory for the dataset (full path). "
            "If omitted, defaults to HF_LEROBOT_HOME/<hf-repo-id>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the dataset directory already exists, delete it and recreate without prompting.",
    )
    parser.add_argument("--display-data", action="store_true", help="Display data during recording")
    args = parser.parse_args()

    # Create the robot and teleoperator configurations
    robot_config = XLerobotClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    keyboard_config = KeyboardTeleopConfig()

    # Initialize the robot and teleoperator
    robot = XLerobotClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # TODO: Update this example to use pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Create dataset (handle directory conflicts)
    def _derive_default_dataset_root(repo_id: str) -> Path:
        return HF_LEROBOT_HOME / repo_id

    def _repo_id_with_suffix(repo_id: str, suffix: str) -> str:
        parts = repo_id.split("/")
        parts[-1] = f"{parts[-1]}{suffix}"
        return "/".join(parts)

    if args.dataset_root:
        resolved_repo_id = args.hf_repo_id
        resolved_root_arg: str | Path | None = Path(args.dataset_root).expanduser()
        dataset_dir = Path(resolved_root_arg)
    else:
        resolved_repo_id = args.hf_repo_id
        resolved_root_arg = None
        dataset_dir = _derive_default_dataset_root(resolved_repo_id)

    if dataset_dir.exists():
        if args.overwrite:
            shutil.rmtree(dataset_dir)
        else:
            if not sys.stdin.isatty():
                raise FileExistsError(
                    f"Dataset directory already exists: {dataset_dir}. "
                    "Re-run with --overwrite, or pick a different --dataset-root / --hf-repo-id."
                )
            print(f"[RECORD] Dataset directory already exists: {dataset_dir}")
            choice = input("[RECORD] Choose: (o)verwrite / (n)ew / (q)uit [n]: ").strip().lower() or "n"
            if choice in {"o", "overwrite", "y", "yes"}:
                shutil.rmtree(dataset_dir)
            elif choice in {"q", "quit"}:
                raise SystemExit("[RECORD] Aborted by user.")
            else:
                suffix = "_" + time.strftime("%Y%m%d_%H%M%S")
                if args.dataset_root:
                    resolved_root_arg = dataset_dir.with_name(dataset_dir.name + suffix)
                    print(f"[RECORD] Using new dataset directory: {resolved_root_arg}")
                else:
                    resolved_repo_id = _repo_id_with_suffix(resolved_repo_id, suffix)
                    dataset_dir = _derive_default_dataset_root(resolved_repo_id)
                    print(f"[RECORD] Using new repo_id: {resolved_repo_id}")
                    print(f"[RECORD] New dataset directory: {dataset_dir}")

    # Create dataset
    features = {}
    features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True))
    features.update(hw_to_dataset_features(cast(dict[str, type | tuple], robot.action_features), ACTION, use_video=True))
    dataset = LeRobotDataset.create(
        repo_id=resolved_repo_id,
        fps=args.fps,
        features=features,
        root=resolved_root_arg,
        use_videos=True,
        robot_type=robot.robot_type,
    )
    print(f"[MAIN] Dataset directory: {dataset.root}")

    # Initialize rerun for visualization
    init_rerun(session_name="xlerobot_record")

    # Connect robot and teleop
    robot.connect()
    keyboard.connect()

    # Initialize keyboard listener for events
    events = init_keyboard_listener()

    log_say(f"Starting to record {args.num_episodes} episodes", args.hf_repo_id)

    for episode_idx in range(args.num_episodes):
        log_say(f"Episode {episode_idx + 1}/{args.num_episodes}", args.hf_repo_id)

        start_t = time.time()
        while time.time() - start_t < args.episode_time_sec:
            act = keyboard.get_action()
            obs = robot.get_observation()

            # For base: map keyboard keys to robot base action
            pressed_keys = np.array(list(act.keys()))
            base_action = robot._from_keyboard_to_base_action(pressed_keys) or {}

            robot.send_action(base_action)

            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, base_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": args.task_description}
            dataset.add_frame(frame)

        dataset.save_episode()

        # Reset time between episodes
        if episode_idx < args.num_episodes - 1:
            log_say(f"Reset time {args.reset_time_sec}s", args.hf_repo_id)
            time.sleep(args.reset_time_sec)

    log_say("Recording completed", args.hf_repo_id)

    dataset.finalize()

    # Disconnect
    robot.disconnect()
    keyboard.disconnect()


if __name__ == "__main__":
    main()