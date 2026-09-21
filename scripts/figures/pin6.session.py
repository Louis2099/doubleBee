"""Play rollout on a staircase PINNED to 6 cm, with per-tick policy IO logging.

Replicates eval_climb.py's terrain pinning (lines 293-301) inside a play run,
because eval_climb.py writes per-episode summaries and we need per-tick
total_thrust and pose. See the 2026-09-05 warning in eval_climb.py: pinning
changed measured gain for every arm, so the height trace must be checked before
trusting anything from this.
"""
import argparse, os, sys
sys.argv = [sys.argv[0]]
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import csv, torch, gymnasium as gym
import numpy as np
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import lab.doublebee.tasks  # noqa: F401
from co_rl.core.wrapper import CoRlVecEnvWrapper
from co_rl.core.runners import OffPolicyRunner

TASK = "Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo"
CKPT = ("logs/co_rl/doublebee_velocity/tqc/energy_abl/"
        "2026-09-06_23-10-57_hE4/model_5899.pt")
OUT, STEP_H, NENV, TICKS = "simprof/pinned6.csv", 0.06, 16, 6000

env_cfg = parse_env_cfg(TASK, num_envs=NENV)
tg = env_cfg.scene.terrain.terrain_generator
tg.curriculum = False
tg.num_rows, tg.num_cols = 1, 5
tg.use_cache = False
key = next(k for k in tg.sub_terrains if "stair" in k)
tg.sub_terrains[key].step_height_range = (STEP_H, STEP_H)
print("[pin6] staircase pinned at %.3f m" % STEP_H, flush=True)

env = gym.make(TASK, cfg=env_cfg)
agent_cfg = load_cfg_from_registry(TASK, "co_rl_tqc_cfg_entry_point")
agent_cfg.use_constraint_rl = True
env = CoRlVecEnvWrapper(env, agent_cfg)
runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                         device=agent_cfg.device)
runner.load(os.path.abspath(CKPT), load_optimizer=False)
policy = runner.get_inference_policy(device=env.unwrapped.device)
base = env.unwrapped
robot = base.scene["robot"]

os.makedirs("simprof", exist_ok=True)
fh = open(OUT, "w", newline="")
wr = csv.writer(fh)
wr.writerow(["step", "action_3", "total_thrust",
             "pos_x", "pos_y", "pos_z", "qw", "qx", "qy", "qz"])
obs, _ = env.get_observations()
for k in range(TICKS):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _ = env.step(act)
    T = getattr(base, "_last_propeller_thrust_total", None)
    p = robot.data.root_pos_w[0]
    q = robot.data.root_quat_w[0]
    wr.writerow([k, float(act[0, -1]),
                 float(T[0]) if T is not None else "",
                 float(p[0]), float(p[1]), float(p[2]),
                 float(q[0]), float(q[1]), float(q[2]), float(q[3])])
    if k % 500 == 0:
        fh.flush()
        print("[pin6] %d/%d" % (k, TICKS), flush=True)
fh.close()
print("[pin6] done ->", OUT, flush=True)
app.close()
