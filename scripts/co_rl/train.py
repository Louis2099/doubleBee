# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# from scripts.co_rl.core.runners import OffPolicyRunner
from core.runners import OffPolicyRunner

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CO-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--experiment_description", type=str, default=None, help="Description of the experiment.")
parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")
parser.add_argument("--obs_latency_steps", type=int, default=0, help="Delay policy obs by N steps (0=no delay, 1=one-step latency).")

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
import os
import pickle
import torch
from datetime import datetime

from core.runners import OnPolicyRunner, SRMOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from lab.doublebee.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv

from isaaclab.utils.dict import print_dict
# YAML: IsaacLab provides this
from isaaclab.utils.io import dump_yaml

# Pickle: IsaacLab 0.53.0 does NOT provide dump_pickle, so define it here
import pickle
from pathlib import Path

def dump_pickle(filename: str, data) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

"""Use CO-RL Wrapper."""
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlVecEnvWrapper


# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401  TODO: import orbit.<your_extension_name>
import lab.doublebee.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with CO-RL agent."""
    # Load environment configuration using parse_env_cfg (same as play.py - no Hydra)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 4096,
        use_fabric=False,
    )
    
    # Load agent configuration using the same method as play.py
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_co_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg.experiment_description = (
        args_cli.experiment_description
        if args_cli.experiment_description is not None
        else agent_cfg.experiment_description
    )
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks
    agent_cfg.obs_latency_steps = args_cli.obs_latency_steps if args_cli.obs_latency_steps is not None else agent_cfg.obs_latency_steps

    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO"] else True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Debug: Check environment type before wrapping
    print(f"[DEBUG] Environment type after gym.make: {type(env)}")
    print(f"[DEBUG] Environment unwrapped type: {type(env.unwrapped)}")
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        print(f"[DEBUG] After multi_agent_to_single_agent: {type(env)}, unwrapped: {type(env.unwrapped)}")
    
    if isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
        agent_cfg.use_constraint_rl = True
        print(f"[DEBUG] Detected ManagerBasedConstraintRLEnv, setting use_constraint_rl=True")

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[DEBUG] After RecordVideo wrapper: {type(env)}, unwrapped: {type(env.unwrapped)}")
    
    # set seed BEFORE wrapping (important for deterministic initialization)
    print(f"[DEBUG] Setting seed: {agent_cfg.seed}")
    # For unwrapped environment, call seed directly
    if hasattr(env.unwrapped, 'seed'):
        env.unwrapped.seed(agent_cfg.seed)
    
    # wrap around environment for co-rl
    print(f"[DEBUG] Before CoRlVecEnvWrapper: env type={type(env)}, unwrapped type={type(env.unwrapped)}")
    env = CoRlVecEnvWrapper(env, agent_cfg)
    
    # Now reset the environment after seed is set
    print("[INFO] Resetting environment after seed is set...")
    env.reset()
    
    # create runner from co-rl
    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        if args_cli.algo == "srmppo":
            runner = SRMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Try to dump pickle files, but skip if they contain unpicklable objects (e.g., lambda functions)
    try:
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    except (pickle.PicklingError, AttributeError) as e:
        print(f"[WARNING] Could not pickle env_cfg: {e}")
        print("[INFO] Environment configuration saved as YAML only. This is normal if config contains lambda functions.")
    
    try:
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    except (pickle.PicklingError, AttributeError) as e:
        print(f"[WARNING] Could not pickle agent_cfg: {e}")
        print("[INFO] Agent configuration saved as YAML only.")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
