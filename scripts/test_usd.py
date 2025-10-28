"""
Quick smoke test to load a USD-based robot into Isaac Sim.

This script launches Isaac Sim, creates the requested Isaac Lab task, and
performs a few simulation steps to make sure the asset can be spawned without
errors. It intentionally keeps the interaction minimal so it can be used as a
sanity check while developing new robots.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test."""
    parser = argparse.ArgumentParser(description="Load a USD robot into Isaac Sim and step the simulation.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Flat-DoubleBee-v1-ppo",
        help="Gym task ID to instantiate.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments to create.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of simulation steps to run after reset.",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args_cli = parse_arguments()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env: Any | None = None
    try:
        import gymnasium as gym
        import numpy as np
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_tasks.utils import parse_env_cfg

        import lab.doublebee.tasks  # noqa: F401  # Register DoubleBee tasks

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        env = gym.make(args_cli.task, cfg=env_cfg)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        obs, info = env.reset()
        print(
            f"[INFO] Environment '{args_cli.task}' loaded. Observation shape: {getattr(obs, 'shape', type(obs))}",
            flush=True,
        )
        
        # Debug: Check if robot exists in scene
        if hasattr(env.unwrapped, 'scene'):
            print(f"[DEBUG] Scene entities: {list(env.unwrapped.scene.keys())}", flush=True)
            try:
                robot = env.unwrapped.scene["robot"]
                print(f"[DEBUG] Robot type: {type(robot)}", flush=True)
                print(f"[DEBUG] Robot prim path: {robot.cfg.prim_path}", flush=True)
                print(f"[DEBUG] Robot num instances: {robot.num_instances}", flush=True)
                if hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w'):
                    print(f"[DEBUG] Robot root position: {robot.data.root_pos_w[0]}", flush=True)
                print(f"[DEBUG] Robot is spawned successfully!", flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"VISIBILITY TROUBLESHOOTING:", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"1. If robot is NOT visible in GUI:", flush=True)
                print(f"   - Check USD file visibility settings", flush=True)
                print(f"   - See: scripts/fix_usd_visibility.md for solutions", flush=True)
                print(f"2. Robot USD path: {robot.cfg.spawn.usd_path}", flush=True)
                print(f"3. Try pressing 'F' key in viewport with robot prim selected", flush=True)
                print(f"4. Check Stage panel (left) for: /World/envs/env_0/Robot", flush=True)
                print(f"{'='*80}\n", flush=True)
            except Exception as e:
                print(f"[ERROR] Could not access robot: {e}", flush=True)

        for step in range(args_cli.num_steps):
            action = env.action_space.sample()
            # Convert numpy array to torch tensor if needed
            if isinstance(action, np.ndarray):
                import torch
                action = torch.from_numpy(action).to(env.device)
            obs, reward, terminated, truncated, info = env.step(action)
            term = bool(getattr(terminated, "any", lambda: terminated)())
            trunc = bool(getattr(truncated, "any", lambda: truncated)())
            if term or trunc:
                env.reset()

        print("[INFO] Simulation completed successfully. Close the window or press Ctrl+C to exit.", flush=True)

        while simulation_app.is_running():
            simulation_app.update()
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"[ERROR] {exc}", flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
