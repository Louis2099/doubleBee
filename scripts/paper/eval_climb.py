"""Measure CLIMBING directly, on a fixed staircase, without goal_reached.

WHY THIS EXISTS
  terrain_levels cannot measure climbing ability. The curriculum promotes only
  after three consecutive goal_reached events, and goal_reached requires
  arriving within 0.25 m of the target while the goal command carries no
  distance information. Measured median closest approach is 0.972 m, so the
  gate almost never fires. A policy that climbs 5 cm risers cleanly therefore
  sits at curriculum level 0 indefinitely, which is exactly what play shows for
  the w_E=0 checkpoint. Any figure with terrain_levels on the axis is reporting
  the gate, not the robot.

  This script never calls goal_reached. Clearing a riser is decided from
  kinematics alone.

CLEARED A RISER means, on a staircase of fixed height h:
    the base is >= 0.8*h above its spawn height for >= --hold seconds
    CONTINUOUSLY, and horizontal displacement from spawn is >= --min-xy.
  The hold requirement is what separates climbing from flying. This platform can
  simply lift itself over a step with thrust; a transient altitude spike does
  that, sustained height while displaced does not. The displacement requirement
  rejects hovering in place above the spawn platform.

ENERGY is read from episode_energy_buf, the same power model the reward
integrates (rewards.py::penalize_energy_consumption), so joules here and joules
in the objective are the same quantity.

One step height per process, because pinning the terrain generator is far
easier to reason about than decoding which row each env landed on. Loop in the
shell; see the banner printed at the end.

    python3 eval_climb.py --checkpoint <ckpt> --step-height 0.05 --episodes 60 \
        --out climb_wE0_h5.csv
    python3 eval_climb.py --summarise "climb_*.csv"
"""
import argparse
import csv
import glob
import os
import re
import sys


def summarise(pattern):
    import numpy as np
    rows = []
    for path in sorted(glob.glob(pattern)):
        recs = list(csv.DictReader(open(path)))
        if not recs:
            continue
        m = re.search(r"climb_(.+?)_h(\d+)\.csv$", os.path.basename(path))
        tag, h = (m.group(1), int(m.group(2))) if m else (os.path.basename(path), 0)
        cleared = np.array([float(r["cleared"]) for r in recs])
        energy = np.array([float(r["energy_J"]) for r in recs])
        gain = np.array([float(r["max_gain_m"]) for r in recs])
        ok = cleared > 0.5
        rows.append((tag, h, len(recs), cleared.mean(),
                     energy[ok].mean() if ok.any() else float("nan"),
                     gain.mean()))
    if not rows:
        sys.exit("no climb CSVs matched %r" % pattern)
    rows.sort(key=lambda r: (r[0], r[1]))
    print("%-10s %-7s %-6s %-10s %-14s %s"
          % ("policy", "h_cm", "n_ep", "cleared", "energy_J", "mean_gain_m"))
    print("-" * 62)
    for t, h, n, c, e, g in rows:
        print("%-10s %-7d %-6d %-10.3f %-14.0f %.3f" % (t, h, n, c, e, g))
    print("\nenergy is over CLEARED episodes only. A policy that never leaves the\n"
          "platform looks cheap, and that is not efficiency.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summarise")
    p.add_argument("--checkpoint")
    p.add_argument("--task", default="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo")
    p.add_argument("--step-height", type=float, default=0.05, help="riser height, m")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--hold", type=float, default=0.5,
                   help="seconds the height gain must be held; separates climbing "
                        "from a thrust-driven altitude spike")
    p.add_argument("--min-xy", dest="min_xy", type=float, default=0.35,
                   help="metres of horizontal displacement required, so hovering "
                        "above the spawn platform does not count")
    p.add_argument("--frac", type=float, default=0.8,
                   help="fraction of the step height that counts as up")
    p.add_argument("--out", default="climb.csv")
    a = p.parse_args()

    if a.summarise:
        return summarise(a.summarise)
    if not a.checkpoint:
        sys.exit("--checkpoint required (or --summarise)")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch
    import gymnasium as gym
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import co_rl  # noqa: F401
    from co_rl.core.runners import OffPolicyRunner
    from co_rl.core.wrapper import CoRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

    env_cfg = parse_env_cfg(a.task, num_envs=a.num_envs)

    # Pin the staircase. step_height_range is normally (0.03, 0.09) spread over
    # five curriculum rows; collapsing it to a single value and switching the
    # curriculum off makes every environment identical and the difficulty known.
    tg = env_cfg.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.num_rows, tg.num_cols = 1, 5
    tg.use_cache = False
    key = next(k for k in tg.sub_terrains if "stair" in k)
    tg.sub_terrains[key].step_height_range = (a.step_height, a.step_height)
    print("[climb] staircase pinned at %.3f m (%s)" % (a.step_height, key), flush=True)

    env = gym.make(a.task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(a.task, "co_rl_tqc_cfg_entry_point")
    # ManagerBasedConstraintRLEnv returns FLOAT terminated/truncated; the wrapper
    # bitwise-ors them unless this is set. See play.py:330.
    agent_cfg.use_constraint_rl = True
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[climb] loading %s" % os.path.abspath(a.checkpoint), flush=True)
    runner.load(os.path.abspath(a.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    dev, n = base.device, a.num_envs
    hold_steps = max(1, int(a.hold / base.step_dt))
    up_th = a.frac * a.step_height

    robot = base.scene["robot"]
    spawn = robot.data.root_pos_w.clone()
    run = torch.zeros(n, device=dev)          # consecutive steps above up_th
    cleared = torch.zeros(n, dtype=torch.bool, device=dev)
    max_gain = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    rows = []

    obs, _ = env.get_observations()
    while len(rows) < a.episodes:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            pos = robot.data.root_pos_w
            steps += 1
            gain = pos[:, 2] - spawn[:, 2]
            max_gain = torch.maximum(max_gain, gain)
            disp = torch.norm(pos[:, :2] - spawn[:, :2], dim=1)

            up = gain >= up_th
            run = torch.where(up, run + 1, torch.zeros_like(run))
            cleared |= (run >= hold_steps) & (disp >= a.min_xy)

            for k in (dones > 0.5).nonzero(as_tuple=False).flatten().tolist():
                rows.append({
                    "cleared": int(cleared[k].item()),
                    "max_gain_m": round(float(max_gain[k].item()), 4),
                    "energy_J": round(float(base.episode_energy_buf[k].item()), 2),
                    "steps": int(steps[k].item()),
                })
                # reset this env's bookkeeping; it has already been respawned
                spawn[k] = pos[k]
                run[k] = 0.0
                cleared[k] = False
                max_gain[k] = 0.0
                steps[k] = 0.0
            if rows:
                print("\r[climb] %d/%d" % (len(rows), a.episodes), end="", flush=True)
    print()

    rows = rows[:a.episodes]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    frac = sum(r["cleared"] for r in rows) / len(rows)
    print("wrote %s   cleared %.0f%% of %d episodes at h=%.0f cm"
          % (a.out, 100 * frac, len(rows), 100 * a.step_height))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
