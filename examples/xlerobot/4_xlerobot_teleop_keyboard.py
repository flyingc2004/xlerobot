# To Run on the host
# '''powershell
# cd C:\Users\32389\Desktop\lerobot\lerobot
# $env:PYTHONPATH = "src"
# python -m lerobot.robots.xlerobot.xlerobot_host --robot-id my_xlerobot_pc
# '''

# To Run the teleop:
# '''powershell
# cd C:\Users\32389\Desktop\lerobot\lerobot
# $env:PYTHONPATH = "src"

# Local (robot plugged into this PC):
# python .\examples\xlerobot\4_xlerobot_teleop_keyboard.py --mode local

# Remote (robot host on another PC):
# python .\examples\xlerobot\4_xlerobot_teleop_keyboard.py --mode client --remote-ip 192.168.1.123

# With dataset recording:
# python .\examples\xlerobot\4_xlerobot_teleop_keyboard.py --mode client --remote-ip 192.168.1.123 --record --hf-repo-id myuser/my_dataset --task-description "Pick and place task"
# '''

import time
import argparse
import shutil
import sys
from pathlib import Path
from typing import cast
import numpy as np
import math

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig

# Add imports for dataset recording
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR

# Keymaps (semantic action: key)
LEFT_KEYMAP = {
    'shoulder_pan+': 'q', 'shoulder_pan-': 'e',
    'wrist_roll+': 'r', 'wrist_roll-': 'f',
    'gripper+': 't', 'gripper-': 'g',
    'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
    'pitch+': 'z', 'pitch-': 'x',
    'reset': 'c',
    # For head motors
    "head_motor_1+": "<", "head_motor_1-": ">",
    "head_motor_2+": ",", "head_motor_2-": ".",
    
    'triangle': 'y',  # Rectangle trajectory key
}
RIGHT_KEYMAP = {
    'shoulder_pan+': '7', 'shoulder_pan-': '9',
    'wrist_roll+': '/', 'wrist_roll-': '*',
    'gripper+': '+', 'gripper-': '-',
    'x+': '8', 'x-': '2', 'y+': '4', 'y-': '6',
    'pitch+': '1', 'pitch-': '3',
    'reset': '0',

    'triangle': 'Y',  # Rectangle trajectory key
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Head motor mapping
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class RectangularTrajectory:
    """
    Generates a rectangular trajectory on the x-y plane with sinusoidal velocity profiles.
    The rectangle is divided into 4 line segments, each with smooth acceleration/deceleration.
    """
    def __init__(self, width=0.06, height=0.06, segment_duration=0.91):
        """
        Initialize rectangular trajectory parameters.
        
        Args:
            width: Rectangle width in meters
            height: Rectangle height in meters  
            segment_duration: Time for each line segment in seconds
        """
        self.width = width
        self.height = height
        self.segment_duration = segment_duration
        self.total_duration = 4 * segment_duration
        
    def get_trajectory_point(self, current_x, current_y, t):
        """
        Get the target x, y position at time t for the rectangular trajectory.
        
        Args:
            current_x: Starting x position
            current_y: Starting y position
            t: Time since trajectory start (0 to total_duration)
            
        Returns:
            tuple: (target_x, target_y)
        """
        # Determine which segment we're in
        segment = int(t / self.segment_duration)
        segment_t = t % self.segment_duration
        
        # Normalize segment time (0 to 1)
        normalized_t = segment_t / self.segment_duration
        
        # Sinusoidal velocity profile: smooth acceleration and deceleration
        # s(t) = 0.5 * (1 - cos(π * t)) gives smooth 0 to 1 transition
        smooth_t = 0.5 * (1 - math.cos(math.pi * normalized_t))
        
        # Define rectangle corners relative to starting position
        corners = [
            (current_x, current_y),                           # Start (bottom-left)
            (current_x + self.width, current_y),              # Bottom-right
            (current_x + self.width, current_y + self.height), # Top-right  
            (current_x, current_y + self.height),             # Top-left
            (current_x, current_y)                            # Back to start
        ]
        
        # Clamp segment to valid range
        segment = max(0, min(3, segment))
        
        # Interpolate between current corner and next corner
        start_corner = corners[segment]
        end_corner = corners[segment + 1]
        
        target_x = start_corner[0] + smooth_t * (end_corner[0] - start_corner[0])
        target_y = start_corner[1] + smooth_t * (end_corner[1] - start_corner[1])
        
        return target_x, target_y

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=0.81):
        self.kp = kp
        self.degree_step = 1
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 3
        self.xy_step = 0.0081
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        # Rectangular trajectory instance
        self.rectangular_trajectory = RectangularTrajectory(
            width=0.06,          # 6cm wide rectangle
            height=0.06,         # 4cm tall rectangle  
            segment_duration=1.01 # 3 seconds per line segment
        )

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues
        
        # Reset kinematic variables to their initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Don't let handle_keys recalculate wrist_flex - set it explicitly
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def execute_rectangular_trajectory(self, robot, fps=30):
        """
        Execute a blocking rectangular trajectory on the x-y plane.
        
        Args:
            robot: Robot instance to send actions to
            fps: Control loop frequency
        """
        print(f"[{self.prefix}] Starting rectangular trajectory...")
        print(f"[{self.prefix}] Rectangle: {self.rectangular_trajectory.width:.3f}m x {self.rectangular_trajectory.height:.3f}m")
        print(f"[{self.prefix}] Duration: {self.rectangular_trajectory.total_duration:.3f}s total")
        
        # Store starting position
        start_x = self.current_x
        start_y = self.current_y
        
        # Execute trajectory
        start_time = time.time()
        dt = 1.0 / fps
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if trajectory is complete
            if elapsed_time >= self.rectangular_trajectory.total_duration:
                print(f"[{self.prefix}] Rectangular trajectory completed!")
                break
                
            # Get target position from trajectory
            target_x, target_y = self.rectangular_trajectory.get_trajectory_point(
                start_x, start_y, elapsed_time
            )
            
            # Update current position
            self.current_x = target_x
            self.current_y = target_y
            
            # Calculate inverse kinematics
            try:
                joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions["shoulder_lift"] = joint2
                self.target_positions["elbow_flex"] = joint3
                
                # Update wrist_flex coupling
                self.target_positions["wrist_flex"] = (
                    -self.target_positions["shoulder_lift"]
                    -self.target_positions["elbow_flex"]
                    + self.pitch
                )
                
                # Get action
                action = self.p_control_action(robot)
                
                # Determine which arm is executing and send appropriate action structure
                if self.prefix == "left":
                    # Send left arm action with empty actions for other components
                    robot_action = {**action, **{}, **{}, **{}}
                elif self.prefix == "right":
                    # Send right arm action with empty actions for other components
                    robot_action = {**{}, **action, **{}, **{}}
                
                # Send action to robot
                robot.send_action(robot_action)
                
                # Get observation and log data
                obs = robot.get_observation()
                log_rerun_data(obs, robot_action)
                
            except Exception as e:
                print(f"[{self.prefix}] IK failed at x={self.current_x:.4f}, y={self.current_y:.4f}: {e}")
                break
                
            # Maintain control frequency
            # busy_wait(dt)
        
        print(f"[{self.prefix}] Trajectory execution finished.")

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action
    

def main():
    parser = argparse.ArgumentParser(description="XLerobot keyboard teleop (local or remote via ZMQ).")
    parser.add_argument("--mode", choices=["local", "client"], default="local")
    parser.add_argument(
        "--remote-ip",
        default="localhost",
        help="Host IP when --mode client (e.g. 192.168.1.123).",
    )
    parser.add_argument("--port-zmq-cmd", type=int, default=5555)
    parser.add_argument("--port-zmq-observations", type=int, default=5556)
    parser.add_argument("--robot-id", default="my_xlerobot_pc")
    parser.add_argument("--fps", type=int, default=50)
    # Add recording options
    parser.add_argument("--record", action="store_true", help="Enable dataset recording")
    parser.add_argument("--hf-repo-id", default="<hf_username>/<dataset_repo_id>", help="HuggingFace repo ID for dataset")
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
    parser.add_argument("--task-description", default="My task description", help="Task description for dataset")
    parser.add_argument("--episode-time-sec", type=int, default=30, help="Time per episode in seconds")
    args = parser.parse_args()

    # Teleop parameters
    FPS = args.fps

    if args.mode == "client":
        robot_config = XLerobotClientConfig(
            remote_ip=args.remote_ip,
            port_zmq_cmd=args.port_zmq_cmd,
            port_zmq_observations=args.port_zmq_observations,
            id=args.robot_id,
        )
        robot = XLerobotClient(robot_config)
    else:
        robot_config = XLerobotConfig(id=args.robot_id)
        robot = XLerobot(robot_config)
    
    try:
        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        print(robot_config)
        print(robot)
        return
        
    init_rerun(session_name="xlerobot_teleop_v2")

    #Init the keyboard instance
    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    keyboard.connect()

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)

    # Move both arms and head to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    # Initialize dataset recording if enabled
    dataset = None
    episode_start_time = 0.0
    if args.record:
        def _derive_default_dataset_root(repo_id: str) -> Path:
            return HF_LEROBOT_HOME / repo_id

        def _repo_id_with_suffix(repo_id: str, suffix: str) -> str:
            parts = repo_id.split("/")
            parts[-1] = f"{parts[-1]}{suffix}"
            return "/".join(parts)

        def _resolve_recording_location() -> tuple[str, Path, str | Path | None]:
            """Return (repo_id, dataset_dir, root_arg_for_create)."""
            if args.dataset_root:
                dataset_dir = Path(args.dataset_root).expanduser()
                repo_id = args.hf_repo_id
                root_arg = dataset_dir
            else:
                repo_id = args.hf_repo_id
                dataset_dir = _derive_default_dataset_root(repo_id)
                root_arg = None

            if not dataset_dir.exists():
                return repo_id, dataset_dir, root_arg

            if args.overwrite:
                shutil.rmtree(dataset_dir)
                return repo_id, dataset_dir, root_arg

            if not sys.stdin.isatty():
                raise FileExistsError(
                    f"Dataset directory already exists: {dataset_dir}. "
                    "Re-run with --overwrite, or pick a different --dataset-root / --hf-repo-id."
                )

            print(f"[RECORD] Dataset directory already exists: {dataset_dir}")
            choice = input("[RECORD] Choose: (o)verwrite / (n)ew / (q)uit [n]: ").strip().lower() or "n"
            if choice in {"o", "overwrite", "y", "yes"}:
                shutil.rmtree(dataset_dir)
                return repo_id, dataset_dir, root_arg
            if choice in {"q", "quit"}:
                raise SystemExit("[RECORD] Aborted by user.")

            # New dataset
            suffix = "_" + time.strftime("%Y%m%d_%H%M%S")
            if args.dataset_root:
                new_dataset_dir = dataset_dir.with_name(dataset_dir.name + suffix)
                print(f"[RECORD] Using new dataset directory: {new_dataset_dir}")
                return repo_id, new_dataset_dir, new_dataset_dir
            else:
                new_repo_id = _repo_id_with_suffix(repo_id, suffix)
                new_dataset_dir = _derive_default_dataset_root(new_repo_id)
                print(f"[RECORD] Using new repo_id: {new_repo_id}")
                print(f"[RECORD] New dataset directory: {new_dataset_dir}")
                return new_repo_id, new_dataset_dir, None

        resolved_repo_id, resolved_dataset_dir, resolved_root_arg = _resolve_recording_location()

        features = {}
        features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True))
        features.update(
            hw_to_dataset_features(cast(dict[str, type | tuple], robot.action_features), ACTION, use_video=True)
        )
        dataset = LeRobotDataset.create(
            repo_id=resolved_repo_id,
            fps=FPS,
            features=features,
            root=resolved_root_arg,
            use_videos=True,
            robot_type=robot.robot_type,
        )
        print(f"[MAIN] Dataset recording enabled. Repo: {resolved_repo_id}")
        print(f"[MAIN] Dataset directory: {dataset.root}")
        episode_start_time = time.time()

    try:
        while True:
            pressed_keys = set(keyboard.get_action().keys())
            left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
            right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

            # Handle rectangular trajectory for left arm (y key)
            if left_key_state.get('triangle'):
                print("[MAIN] Left arm rectangular trajectory triggered!")
                left_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Handle rectangular trajectory for right arm (Y key)  
            if right_key_state.get('triangle'):
                print("[MAIN] Right arm rectangular trajectory triggered!")
                right_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Handle reset for left arm
            if left_key_state.get('reset'):
                left_arm.move_to_zero_position(robot)
                continue  

            # Handle reset for right arm
            if right_key_state.get('reset'):
                right_arm.move_to_zero_position(robot)
                continue

            # Handle reset for head motors with '?'
            if '?' in pressed_keys:
                head_control.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)  # Head controlled by left arm keymap

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)

            # Base action
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            # print(f"[MAIN] Observation: {obs}")
            log_rerun_data(obs, action)

            # Record data if enabled
            if args.record and dataset is not None:
                obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                frame = {**obs_frame, **action_frame, "task": args.task_description}
                dataset.add_frame(frame)

                # Check if episode time exceeded
                current_time = time.time()
                if current_time - episode_start_time >= args.episode_time_sec:
                    dataset.save_episode()
                    print(f"[MAIN] Episode saved")
                    episode_start_time = current_time

            # busy_wait(1.0 / FPS)
    finally:
        # Save any remaining episode data
        if args.record and dataset is not None and dataset.episode_buffer is not None:
            if dataset.episode_buffer.get("size", 0) > 0:
                dataset.save_episode()
                print(f"[MAIN] Final episode saved")
            dataset.finalize()
        robot.disconnect()
        keyboard.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
