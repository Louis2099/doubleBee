# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
import sys
import time
import numpy as np

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt

# local imports
import cli_args  # isort: skip
from scripts.co_rl.core.runners import OffPolicyRunner

from scripts.co_rl.core.utils.str2bool import str2bool
from scripts.co_rl.core.utils.analyzer import Analyzer

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CO-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1 for inference/play mode).")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--stack_frames", type=int, default=None, help="Number of frames to stack.")
parser.add_argument("--plot", type=str2bool, default="False", help="Plot the data.")
parser.add_argument(
    "--analyze",
    type=str,
    nargs="+",  
    default=None,
    help="Specify which data to analyze (e.g., cmd_vel joint_vel torque)."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")

parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")
parser.add_argument(
    "--plot_velocity",
    action="store_true",
    default=False,
    help="Plot current velocity vs command velocity in real-time.",
)
parser.add_argument(
    "--cmd_vel",
    type=float,
    nargs=4,
    default=None,
    metavar=("VX", "VY", "VZ", "WZ"),
    help="Fixed velocity command for inference: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z]. Example: --cmd_vel 0.5 0.0 0.0 0.0",
)

# append CO-RL cli arguments
cli_args.add_co_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from scripts.co_rl.core.runners import OnPolicyRunner, SRMOnPolicyRunner
from isaaclab.utils.dict import print_dict

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlVecEnvWrapper,
    export_env_as_pdf,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_srm_as_onnx,
)

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import isaaclab.sim as sim_utils


# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401
import lab.doublebee.tasks  # noqa: F401 - Register DoubleBee tasks
from lab.doublebee.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv, ManagerBasedConstraintRLEnvCfg

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def resolve_xy_velocity_to_arrow(vel_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert XY velocity vector to arrow quaternion and scale.
    
    Args:
        vel_xy: Velocity in XY plane [num_envs, 2] in body frame
        
    Returns:
        arrow_scale: Scale vector for arrow [num_envs, 3] - (width, width, length)
        arrow_quat: Quaternion rotation for arrow [num_envs, 4] (w, x, y, z)
    """
    num_envs = vel_xy.shape[0]
    device = vel_xy.device
    
    # Compute velocity magnitude and direction
    vel_magnitude = torch.norm(vel_xy, dim=1)  # [num_envs]
    vel_angle = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])  # [num_envs] - angle in XY plane
    
    # Arrow scale: width proportional to magnitude, length proportional to magnitude
    # Base scale: (0.2, 0.2, 0.6) for zero velocity, scale up with magnitude (increased from 0.1, 0.1, 0.3)
    base_scale = torch.tensor([0.2, 0.2, 0.6], device=device)  # [3] - doubled size
    scale_factor = torch.clamp(vel_magnitude * 0.5 + 0.5, min=0.5, max=3.0)  # [num_envs] - increased max from 2.0 to 3.0
    arrow_scale = base_scale.unsqueeze(0) * scale_factor.unsqueeze(1)  # [num_envs, 3]
    
    # Create quaternion rotation around Z-axis (yaw rotation)
    # Arrow points along +X axis by default, rotate by vel_angle
    half_angle = vel_angle / 2.0  # [num_envs]
    cos_half = torch.cos(half_angle)  # [num_envs]
    sin_half = torch.sin(half_angle)  # [num_envs]
    
    # Quaternion for rotation around Z-axis: (w, x, y, z) = (cos(θ/2), 0, 0, sin(θ/2))
    arrow_quat = torch.zeros(num_envs, 4, device=device)
    arrow_quat[:, 0] = cos_half  # w
    arrow_quat[:, 3] = sin_half  # z
    
    return arrow_scale, arrow_quat


class VelocityPlotter:
    """Real-time velocity plotting for doubleBee tasks."""
    
    def __init__(self, max_history=500):
        self.max_history = max_history
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("DoubleBee Velocity Tracking", fontsize=14)
        
        # Initialize data storage
        self.times = []
        self.cmd_lin_vel_x = []
        self.cmd_lin_vel_y = []
        self.cmd_lin_vel_z = []
        self.cmd_ang_vel_z = []
        self.curr_lin_vel_x = []
        self.curr_lin_vel_y = []
        self.curr_lin_vel_z = []
        self.curr_ang_vel_z = []
        
        # Setup subplots
        self.axes[0, 0].set_title("Linear Velocity X")
        self.axes[0, 0].set_xlabel("Time (s)")
        self.axes[0, 0].set_ylabel("Velocity (m/s)")
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title("Linear Velocity Y")
        self.axes[0, 1].set_xlabel("Time (s)")
        self.axes[0, 1].set_ylabel("Velocity (m/s)")
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title("Linear Velocity Z (Vertical)")
        self.axes[1, 0].set_xlabel("Time (s)")
        self.axes[1, 0].set_ylabel("Velocity (m/s)")
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title("Angular Velocity Z (Yaw)")
        self.axes[1, 1].set_xlabel("Time (s)")
        self.axes[1, 1].set_ylabel("Angular Velocity (rad/s)")
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        
    def update(self, t, cmd_vel, curr_vel):
        """Update plot with new data."""
        # Append new data
        self.times.append(t)
        self.cmd_lin_vel_x.append(cmd_vel[0])
        self.cmd_lin_vel_y.append(cmd_vel[1])
        self.cmd_lin_vel_z.append(cmd_vel[2])
        self.cmd_ang_vel_z.append(cmd_vel[3])
        self.curr_lin_vel_x.append(curr_vel[0])
        self.curr_lin_vel_y.append(curr_vel[1])
        self.curr_lin_vel_z.append(curr_vel[2])
        self.curr_ang_vel_z.append(curr_vel[3])
        
        # Limit history
        if len(self.times) > self.max_history:
            self.times.pop(0)
            self.cmd_lin_vel_x.pop(0)
            self.cmd_lin_vel_y.pop(0)
            self.cmd_lin_vel_z.pop(0)
            self.cmd_ang_vel_z.pop(0)
            self.curr_lin_vel_x.pop(0)
            self.curr_lin_vel_y.pop(0)
            self.curr_lin_vel_z.pop(0)
            self.curr_ang_vel_z.pop(0)
        
        # Update plots
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(self.times, self.cmd_lin_vel_x, 'r-', label='Command', linewidth=2)
        self.axes[0, 0].plot(self.times, self.curr_lin_vel_x, 'b-', label='Current', linewidth=1.5)
        self.axes[0, 0].set_title("Linear Velocity X")
        self.axes[0, 0].set_xlabel("Time (s)")
        self.axes[0, 0].set_ylabel("Velocity (m/s)")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(self.times, self.cmd_lin_vel_y, 'r-', label='Command', linewidth=2)
        self.axes[0, 1].plot(self.times, self.curr_lin_vel_y, 'b-', label='Current', linewidth=1.5)
        self.axes[0, 1].set_title("Linear Velocity Y")
        self.axes[0, 1].set_xlabel("Time (s)")
        self.axes[0, 1].set_ylabel("Velocity (m/s)")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(self.times, self.cmd_lin_vel_z, 'r-', label='Command', linewidth=2)
        self.axes[1, 0].plot(self.times, self.curr_lin_vel_z, 'b-', label='Current', linewidth=1.5)
        self.axes[1, 0].set_title("Linear Velocity Z (Vertical)")
        self.axes[1, 0].set_xlabel("Time (s)")
        self.axes[1, 0].set_ylabel("Velocity (m/s)")
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].clear()
        self.axes[1, 1].plot(self.times, self.cmd_ang_vel_z, 'r-', label='Command', linewidth=2)
        self.axes[1, 1].plot(self.times, self.curr_ang_vel_z, 'b-', label='Current', linewidth=1.5)
        self.axes[1, 1].set_title("Angular Velocity Z (Yaw)")
        self.axes[1, 1].set_xlabel("Time (s)")
        self.axes[1, 1].set_ylabel("Angular Velocity (rad/s)")
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        """Close the plot."""
        plt.close(self.fig)


def main():
    """Play with CO-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks

    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO"] else True
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("co_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
        agent_cfg.use_constraint_rl = True

    # wrap for video recording
    if args_cli.video:
        # Extract epoch number from checkpoint filename if available
        name_prefix = "rl-video"
        if args_cli.checkpoint:
            # Extract number from checkpoint filename (e.g., "model_4999.pt" -> "4999")
            match = re.search(r'(\d+)', args_cli.checkpoint)
            if match:
                epoch = match.group(1)
                name_prefix = f"rl-video-epoch-{epoch}"
        
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": name_prefix,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    if args_cli.analyze is not None:
        analyze_items = args_cli.analyze[0].split()
        analyzer = Analyzer(env=env, analyze_items=analyze_items, log_dir=log_dir)
    
    # set seed BEFORE wrapping (important for deterministic initialization, matches train.py)
    if hasattr(env.unwrapped, 'seed'):
        env.unwrapped.seed(agent_cfg.seed)
    
    # wrap around environment for co-rl
    env = CoRlVecEnvWrapper(env, agent_cfg)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        if args_cli.algo == "srmppo":
            runner = SRMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load the checkpoint and verify it loaded successfully
    try:
        loaded_infos = runner.load(resume_path)
        print(f"[INFO] Model checkpoint loaded successfully from iteration {runner.current_learning_iteration}")
        if loaded_infos:
            print(f"[INFO] Checkpoint info: {loaded_infos}")
    except Exception as e:
        print(f"[ERROR] Failed to load model checkpoint: {e}")
        print("[ERROR] The model may not be loaded correctly. Exiting.")
        raise

    # Initialize GRU model and FC layer
    srm = None
    if hasattr(runner.alg, "srm") and hasattr(runner.alg, "srm_fc"):
        srm = runner.alg.srm

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if is_off_policy:
        export_policy_as_jit(runner.alg, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        try:
            export_policy_as_onnx(
                runner.alg, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
            print("[INFO] Policy exported to ONNX format successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to export policy to ONNX: {e}")
            print("[INFO] To enable ONNX export, install: pip install onnx")
    else:
        export_policy_as_jit(
            runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        try:
            export_policy_as_onnx(
                runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
            print("[INFO] Policy exported to ONNX format successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to export policy to ONNX: {e}")
            print("[INFO] To enable ONNX export, install: pip install onnx")
        if args_cli.algo == "srmppo":
            try:
                export_srm_as_onnx(
                    runner.alg.srm, runner.alg.srm_fc, device=agent_cfg.device, path=export_model_dir, filename="srm.onnx"
                )
                print("[INFO] SRM exported to ONNX format successfully.")
            except Exception as e:
                print(f"[WARNING] Failed to export SRM to ONNX: {e}")
                print("[INFO] To enable ONNX export, install: pip install onnx")
    
    # export environment to pdf
    export_env_as_pdf(yaml_path=os.path.join(log_dir, "params", "env.yaml"), pdf_path=os.path.join(export_model_dir, "env.pdf"))

    # Check if this is a doubleBee velocity tracking task
    is_doublebee_velocity = isinstance(env.unwrapped, ManagerBasedConstraintRLEnv) and "doublebee" in args_cli.task.lower() and "velocity" in args_cli.task.lower()
    
    # Check if using TerrainTargetDirectionCommand (updates commands per step)
    uses_target_command = False
    if is_doublebee_velocity:
        try:
            cmd_manager = env.unwrapped.command_manager
            print(f"[DEBUG] Command manager type: {type(cmd_manager)}")
            print(f"[DEBUG] Command manager has _terms: {hasattr(cmd_manager, '_terms')}")
            
            if hasattr(cmd_manager, "_terms"):
                print(f"[DEBUG] Available command terms: {list(cmd_manager._terms.keys()) if hasattr(cmd_manager._terms, 'keys') else 'N/A'}")
                
                if "base_velocity" in cmd_manager._terms:
                    velocity_cmd = cmd_manager._terms["base_velocity"]
                    print(f"[DEBUG] Velocity command type: {type(velocity_cmd)}")
                    print(f"[DEBUG] Velocity command class name: {velocity_cmd.__class__.__name__}")
                    
                    # Check if this is TerrainTargetDirectionCommand
                    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.velocity_command import TerrainTargetDirectionCommand
                    uses_target_command = isinstance(velocity_cmd, TerrainTargetDirectionCommand)
                    print(f"[DEBUG] Is TerrainTargetDirectionCommand: {uses_target_command}")
                    
                    if uses_target_command:
                        print("[INFO] Detected TerrainTargetDirectionCommand - commands update every step to point toward targets.")
                        print("[INFO] --cmd_vel override will be ignored. Robot will follow terrain targets instead.")
                    else:
                        print(f"[WARNING] Command is {velocity_cmd.__class__.__name__}, not TerrainTargetDirectionCommand.")
                        print("[WARNING] Target visualization will not be enabled. To enable it, configure the environment to use TerrainTargetDirectionCommandCfg.")
                else:
                    print(f"[WARNING] 'base_velocity' not found in command terms.")
            else:
                print("[WARNING] Command manager does not have _terms attribute.")
        except Exception as e:
            print(f"[ERROR] Failed to check command type: {e}")
            import traceback
            traceback.print_exc()
    
    # Set fixed velocity command if specified (only if not using target-based commands)
    if args_cli.cmd_vel is not None and is_doublebee_velocity and not uses_target_command:
        if len(args_cli.cmd_vel) == 4:
            try:
                cmd_manager = env.unwrapped.command_manager
                # Access the velocity command object
                if hasattr(cmd_manager, "_terms") and "base_velocity" in cmd_manager._terms:
                    velocity_cmd = cmd_manager._terms["base_velocity"]
                    # Set the fixed command for all environments
                    velocity_cmd.vel_command_b[:, 0] = args_cli.cmd_vel[0]  # lin_vel_x
                    velocity_cmd.vel_command_b[:, 1] = args_cli.cmd_vel[1]  # lin_vel_y
                    velocity_cmd.vel_command_b[:, 2] = args_cli.cmd_vel[2]  # lin_vel_z
                    velocity_cmd.vel_command_b[:, 3] = args_cli.cmd_vel[3]  # ang_vel_z
                    # Set a very large time_left to prevent auto-resampling
                    velocity_cmd.time_left[:] = 10000.0  # Large value to prevent resampling
                    print(f"[INFO] Fixed velocity command set: vx={args_cli.cmd_vel[0]:.3f}, vy={args_cli.cmd_vel[1]:.3f}, vz={args_cli.cmd_vel[2]:.3f}, wz={args_cli.cmd_vel[3]:.3f}")
                    print("[INFO] Command resampling disabled for inference mode.")
                else:
                    print("[WARNING] Could not access base_velocity command. Using default random sampling.")
            except Exception as e:
                print(f"[WARNING] Failed to set fixed velocity command: {e}. Using default random sampling.")
        else:
            print(f"[ERROR] Expected 4 values for --cmd_vel, got {len(args_cli.cmd_vel)}. Using default random sampling.")
    elif args_cli.cmd_vel is not None and uses_target_command:
        print("[WARNING] --cmd_vel specified but task uses TerrainTargetDirectionCommand.")
        print("[WARNING] Commands update every step to point toward terrain targets. --cmd_vel is ignored.")
    
    # Setup velocity plotter if requested
    plotter = None
    if args_cli.plot_velocity and is_doublebee_velocity:
        plotter = VelocityPlotter(max_history=1000)
        print("[INFO] Real-time velocity plotting enabled.")
    
    # Setup target visualizer if using terrain targets
    target_visualizer = None
    if uses_target_command:
        try:
            # Create a red sphere marker for the target
            target_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.15,  # 30cm radius sphere (increased from 15cm for better visibility)
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),  # Red color
                            metallic=0.0,
                            roughness=0.5,
                        ),
                    ),
                },
            )
            target_visualizer = VisualizationMarkers(target_marker_cfg)
            target_visualizer.set_visibility(True)
            print("[INFO] Target visualization enabled - red sphere shows target position.")
        except Exception as e:
            print(f"[WARNING] Failed to create target visualizer: {e}")
            target_visualizer = None
    
    # Setup command velocity arrow visualizer
    cmd_vel_arrow_visualizer = None
    actual_vel_arrow_visualizer = None
    if is_doublebee_velocity:
        try:
            # Create a blue arrow marker for command velocity
            cmd_vel_arrow_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/CommandVelocityArrow"
            )
            # Set arrow scale (will be updated dynamically) - increased size
            cmd_vel_arrow_cfg.markers["arrow"].scale = (0.3, 0.3, 1.0)  # Doubled from (0.15, 0.15, 0.5)
            cmd_vel_arrow_visualizer = VisualizationMarkers(cmd_vel_arrow_cfg)
            cmd_vel_arrow_visualizer.set_visibility(True)
            print("[INFO] Command velocity arrow visualization enabled - blue arrow shows command direction.")
        except Exception as e:
            print(f"[WARNING] Failed to create command velocity arrow visualizer: {e}")
            cmd_vel_arrow_visualizer = None
        
        try:
            # Create a green arrow marker for actual velocity
            actual_vel_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/ActualVelocityArrow"
            )
            # Set arrow scale (will be updated dynamically) - increased size
            actual_vel_arrow_cfg.markers["arrow"].scale = (0.3, 0.3, 1.0)  # Doubled from (0.15, 0.15, 0.5)
            actual_vel_arrow_visualizer = VisualizationMarkers(actual_vel_arrow_cfg)
            actual_vel_arrow_visualizer.set_visibility(True)
            print("[INFO] Actual velocity arrow visualization enabled - green arrow shows robot's actual velocity.")
        except Exception as e:
            print(f"[WARNING] Failed to create actual velocity arrow visualizer: {e}")
            actual_vel_arrow_visualizer = None
    
    # reset environment
    obs, _ = env.get_observations()
    
    # Update target visualizer after reset (to show new target if resampled)
    if target_visualizer is not None:
        try:
            cmd_manager = env.unwrapped.command_manager
            if hasattr(cmd_manager, "_terms") and "base_velocity" in cmd_manager._terms:
                velocity_cmd = cmd_manager._terms["base_velocity"]
                if hasattr(velocity_cmd, "current_targets_w"):
                    target_pos = velocity_cmd.current_targets_w[0:1, :].clone()  # [1, 3]
                    target_visualizer.visualize(target_pos)
        except Exception:
            pass  # Silently fail if target not accessible
    
    # Update command and actual velocity arrows after reset
    if cmd_vel_arrow_visualizer is not None or actual_vel_arrow_visualizer is not None:
        try:
            robot = env.unwrapped.scene["robot"]
            if robot.is_initialized:
                base_pos_w = robot.data.root_pos_w[0:1, :].clone()  # [1, 3]
                robot_quat_w = robot.data.root_quat_w[0:1, :]  # [1, 4]
                
                # Update command velocity arrow (blue)
                if cmd_vel_arrow_visualizer is not None:
                    cmd_base_pos = base_pos_w.clone()
                    cmd_base_pos[0, 2] += 0.2  # 20cm above base
                    cmd_base_pos[0, 1] -= 0.1  # 10cm to the left
                    
                    cmd_manager = env.unwrapped.command_manager
                    vel_cmd = cmd_manager.get_command("base_velocity")[0:1, :2]  # [1, 2] - XY only
                    
                    arrow_scale, arrow_quat = resolve_xy_velocity_to_arrow(vel_cmd)
                    
                    w1, x1, y1, z1 = robot_quat_w[0, 0], robot_quat_w[0, 1], robot_quat_w[0, 2], robot_quat_w[0, 3]
                    w2, x2, y2, z2 = arrow_quat[0, 0], arrow_quat[0, 1], arrow_quat[0, 2], arrow_quat[0, 3]
                    
                    arrow_quat_w = torch.zeros(1, 4, device=vel_cmd.device)
                    arrow_quat_w[0, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    arrow_quat_w[0, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    arrow_quat_w[0, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                    arrow_quat_w[0, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                    
                    cmd_vel_arrow_visualizer.visualize(cmd_base_pos, arrow_quat_w, arrow_scale)
                
                # Update actual velocity arrow (green)
                if actual_vel_arrow_visualizer is not None:
                    actual_base_pos = base_pos_w.clone()
                    actual_base_pos[0, 2] += 0.2  # 20cm above base
                    actual_base_pos[0, 1] += 0.1  # 10cm to the right
                    
                    vel_actual = robot.data.root_lin_vel_b[0:1, :2]  # [1, 2] - XY only
                    
                    arrow_scale, arrow_quat = resolve_xy_velocity_to_arrow(vel_actual)
                    
                    w1, x1, y1, z1 = robot_quat_w[0, 0], robot_quat_w[0, 1], robot_quat_w[0, 2], robot_quat_w[0, 3]
                    w2, x2, y2, z2 = arrow_quat[0, 0], arrow_quat[0, 1], arrow_quat[0, 2], arrow_quat[0, 3]
                    
                    arrow_quat_w = torch.zeros(1, 4, device=vel_actual.device)
                    arrow_quat_w[0, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    arrow_quat_w[0, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    arrow_quat_w[0, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                    arrow_quat_w[0, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                    
                    actual_vel_arrow_visualizer.visualize(actual_base_pos, arrow_quat_w, arrow_scale)
        except Exception:
            pass  # Silently fail if arrow update fails
    
    # Re-apply fixed command after reset (in case reset changed it)
    # Only if not using target-based commands
    if args_cli.cmd_vel is not None and is_doublebee_velocity and not uses_target_command and len(args_cli.cmd_vel) == 4:
        try:
            cmd_manager = env.unwrapped.command_manager
            if hasattr(cmd_manager, "_terms") and "base_velocity" in cmd_manager._terms:
                velocity_cmd = cmd_manager._terms["base_velocity"]
                velocity_cmd.vel_command_b[:, 0] = args_cli.cmd_vel[0]
                velocity_cmd.vel_command_b[:, 1] = args_cli.cmd_vel[1]
                velocity_cmd.vel_command_b[:, 2] = args_cli.cmd_vel[2]
                velocity_cmd.vel_command_b[:, 3] = args_cli.cmd_vel[3]
                velocity_cmd.time_left[:] = 10000.0
        except Exception:
            pass  # Silently fail if already set
    
    timestep = 0
    start_time = time.time()
    
    # Simulate environment and collect data
    while simulation_app.is_running():
        # Maintain fixed command if specified (prevent resampling)
        # Only if not using target-based commands (which update every step)
        if args_cli.cmd_vel is not None and is_doublebee_velocity and not uses_target_command and len(args_cli.cmd_vel) == 4:
            try:
                cmd_manager = env.unwrapped.command_manager
                if hasattr(cmd_manager, "_terms") and "base_velocity" in cmd_manager._terms:
                    velocity_cmd = cmd_manager._terms["base_velocity"]
                    # Re-apply fixed command and prevent resampling
                    velocity_cmd.vel_command_b[:, 0] = args_cli.cmd_vel[0]
                    velocity_cmd.vel_command_b[:, 1] = args_cli.cmd_vel[1]
                    velocity_cmd.vel_command_b[:, 2] = args_cli.cmd_vel[2]
                    velocity_cmd.vel_command_b[:, 3] = args_cli.cmd_vel[3]
                    velocity_cmd.time_left[:] = 10000.0
            except Exception:
                pass  # Silently fail if command not accessible
        
        with torch.inference_mode():
            if srm is not None:
                encoded_obs = runner.alg.encode_obs(obs)
                actions = policy(encoded_obs)
            else:
                actions = policy(obs)
            # Note: Actions are already bounded to [-1, 1] by tanh activation in actor network
            # No need for explicit clamping
            obs, _, _, extras = env.step(actions)
        
        # Update target visualizer if enabled
        if target_visualizer is not None:
            try:
                cmd_manager = env.unwrapped.command_manager
                if hasattr(cmd_manager, "_terms") and "base_velocity" in cmd_manager._terms:
                    velocity_cmd = cmd_manager._terms["base_velocity"]
                    if hasattr(velocity_cmd, "current_targets_w"):
                        # Get target position for environment 0 (since we're in play mode with num_envs=1)
                        target_pos = velocity_cmd.current_targets_w[0:1, :].clone()  # [1, 3]
                        target_visualizer.visualize(target_pos)
            except Exception as e:
                # Silently fail if target not accessible
                pass
        
        # Update command velocity arrow visualizer if enabled
        if cmd_vel_arrow_visualizer is not None or actual_vel_arrow_visualizer is not None:
            try:
                robot = env.unwrapped.scene["robot"]
                if robot.is_initialized:
                    # Get robot base position in world frame
                    base_pos_w = robot.data.root_pos_w[0:1, :].clone()  # [1, 3]
                    robot_quat_w = robot.data.root_quat_w[0:1, :]  # [1, 4]
                    
                    # Update command velocity arrow (blue)
                    if cmd_vel_arrow_visualizer is not None:
                        # Position command arrow slightly to the left of robot center
                        cmd_base_pos = base_pos_w.clone()
                        cmd_base_pos[0, 2] += 0.2  # 20cm above base
                        cmd_base_pos[0, 1] -= 0.1  # 10cm to the left (negative Y)
                        
                        # Get velocity command (XY components in body frame)
                        cmd_manager = env.unwrapped.command_manager
                        vel_cmd = cmd_manager.get_command("base_velocity")[0:1, :2]  # [1, 2] - XY only
                        
                        # Convert velocity to arrow orientation and scale
                        arrow_scale, arrow_quat = resolve_xy_velocity_to_arrow(vel_cmd)
                        
                        # Transform arrow quaternion from body frame to world frame
                        w1, x1, y1, z1 = robot_quat_w[0, 0], robot_quat_w[0, 1], robot_quat_w[0, 2], robot_quat_w[0, 3]
                        w2, x2, y2, z2 = arrow_quat[0, 0], arrow_quat[0, 1], arrow_quat[0, 2], arrow_quat[0, 3]
                        
                        arrow_quat_w = torch.zeros(1, 4, device=vel_cmd.device)
                        arrow_quat_w[0, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
                        arrow_quat_w[0, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
                        arrow_quat_w[0, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
                        arrow_quat_w[0, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
                        
                        cmd_vel_arrow_visualizer.visualize(cmd_base_pos, arrow_quat_w, arrow_scale)
                    
                    # Update actual velocity arrow (green)
                    if actual_vel_arrow_visualizer is not None:
                        # Position actual arrow slightly to the right of robot center
                        actual_base_pos = base_pos_w.clone()
                        actual_base_pos[0, 2] += 0.2  # 20cm above base
                        actual_base_pos[0, 1] += 0.1  # 10cm to the right (positive Y)
                        
                        # Get actual robot velocity (XY components in body frame)
                        vel_actual = robot.data.root_lin_vel_b[0:1, :2]  # [1, 2] - XY only
                        
                        # Convert velocity to arrow orientation and scale
                        arrow_scale, arrow_quat = resolve_xy_velocity_to_arrow(vel_actual)
                        
                        # Transform arrow quaternion from body frame to world frame
                        w1, x1, y1, z1 = robot_quat_w[0, 0], robot_quat_w[0, 1], robot_quat_w[0, 2], robot_quat_w[0, 3]
                        w2, x2, y2, z2 = arrow_quat[0, 0], arrow_quat[0, 1], arrow_quat[0, 2], arrow_quat[0, 3]
                        
                        arrow_quat_w = torch.zeros(1, 4, device=vel_actual.device)
                        arrow_quat_w[0, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
                        arrow_quat_w[0, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
                        arrow_quat_w[0, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
                        arrow_quat_w[0, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
                        
                        actual_vel_arrow_visualizer.visualize(actual_base_pos, arrow_quat_w, arrow_scale)
            except Exception as e:
                # Silently fail if arrow update fails
                pass
        
        # Update velocity plot if enabled
        if plotter is not None:
            try:
                # Get current velocity command
                cmd_vel = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()
                # Get current robot velocity
                robot_data = env.unwrapped.scene["robot"].data
                curr_lin_vel = robot_data.root_lin_vel_b[0].cpu().numpy()
                curr_ang_vel = robot_data.root_ang_vel_b[0].cpu().numpy()
                curr_vel = np.array([curr_lin_vel[0], curr_lin_vel[1], curr_lin_vel[2], curr_ang_vel[2]])
                
                # Update plot
                current_time = time.time() - start_time
                plotter.update(current_time, cmd_vel, curr_vel)
            except Exception as e:
                print(f"[WARNING] Plot update error: {e}")
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
                
        # Extract the relevant slices and convert to numpy
        if args_cli.analyze is not None:
            analyzer.append(extras['observations']['obs_info'])
    
    # Cleanup
    if plotter is not None:
        plotter.close()
    
    if target_visualizer is not None:
        target_visualizer.set_visibility(False)
    
    if cmd_vel_arrow_visualizer is not None:
        cmd_vel_arrow_visualizer.set_visibility(False)
    
    if actual_vel_arrow_visualizer is not None:
        actual_vel_arrow_visualizer.set_visibility(False)
    
    env.close()

    if args_cli.analyze is not None:
        analyzer.export()
        
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()