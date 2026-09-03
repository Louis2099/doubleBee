"""Why doesn't the goal fire? Record every episode's CLOSEST APPROACH and the
four criterion values at that exact instant.

The four criteria in constraints.py::goal_reached are ANDed, so a failure tells
you nothing about which one blocked. This logs all four continuously, finds the
step where XY distance was smallest, and reports the state there. If the robot
genuinely passes the target upright and settled and still nothing fires, this
shows it; if one criterion is always the blocker, this names it.

Reported per episode:
    min_xy        smallest XY distance to target achieved (m)
    up_at_min     uprightness at that step        (needs > 0.85)
    w_at_min      body rate at that step, rad/s   (needs < 3.5)
    dz_at_min     |height error| at that step, m  (needs < 0.15)
    all4_ever     did all four hold SIMULTANEOUSLY at any step of the episode
    env_success   what the env's own episode_success_buf recorded

all4_ever and env_success disagreeing means the constraint and the env's
bookkeeping are out of step, which is worth knowing before any number is
reported.

    python3 diagnose_success.py --checkpoint <path> --episodes 40
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--out", default="diagnose_success.csv")
    a = p.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import csv
    import torch
    import gymnasium as gym
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import co_rl  # noqa: F401
    from co_rl.core.runners import OffPolicyRunner
    from co_rl.core.wrapper import CoRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

    env_cfg = parse_env_cfg(a.task, num_envs=a.num_envs)
    env = gym.make(a.task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(a.task, "co_rl_tqc_cfg_entry_point")
    agent_cfg.use_constraint_rl = True          # float terminated/truncated
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[diagnose] loading %s" % os.path.abspath(a.checkpoint), flush=True)
    runner.load(os.path.abspath(a.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    dev = base.device
    n = a.num_envs
    BIG = 1e9

    # thresholds, mirroring constraints.py::goal_reached defaults
    D_TH, UP_TH, W_TH, DZ_TH, Z_OFF = 0.25, 0.85, 3.5, 0.15, 0.3

    min_d = torch.full((n,), BIG, device=dev)
    up_at, w_at, dz_at = (torch.zeros(n, device=dev) for _ in range(3))
    all4 = torch.zeros(n, dtype=torch.bool, device=dev)
    rows = []

    obs, _ = env.get_observations()
    while len(rows) < a.episodes:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            robot = base.scene["robot"]
            cmd = base.command_manager._terms["base_velocity"]
            tgt = cmd.current_targets_w

            d = torch.norm(robot.data.root_pos_w[:, :2] - tgt[:, :2], dim=1)
            up = -robot.data.projected_gravity_b[:, 2]
            wmag = torch.norm(robot.data.root_ang_vel_w, dim=1)
            dz = ((tgt[:, 2] - Z_OFF) - robot.data.root_pos_w[:, 2]).abs()

            all4 |= (d <= D_TH) & (up > UP_TH) & (wmag < W_TH) & (dz < DZ_TH)

            closer = d < min_d
            min_d = torch.where(closer, d, min_d)
            up_at = torch.where(closer, up, up_at)
            w_at = torch.where(closer, wmag, w_at)
            dz_at = torch.where(closer, dz, dz_at)

            for k in (dones > 0.5).nonzero(as_tuple=False).flatten().tolist():
                rows.append({
                    "min_xy": round(float(min_d[k]), 4),
                    "up_at_min": round(float(up_at[k]), 4),
                    "w_at_min": round(float(w_at[k]), 4),
                    "dz_at_min": round(float(dz_at[k]), 4),
                    "all4_ever": int(all4[k].item()),
                    "env_success": float(base.episode_success_buf[k].item()),
                })
                min_d[k] = BIG
                all4[k] = False
            if rows:
                print("\r[diagnose] %d/%d" % (len(rows), a.episodes), end="", flush=True)
    print()

    rows = rows[:a.episodes]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import statistics as st
    med = lambda k: st.median(r[k] for r in rows)
    within = sum(r["min_xy"] <= D_TH for r in rows)
    print("\n%d episodes from %s" % (len(rows), os.path.basename(a.checkpoint)))
    print("  came within %.2f m of the target : %d  (%.0f%%)" % (D_TH, within, 100*within/len(rows)))
    print("  median closest approach          : %.3f m" % med("min_xy"))
    print("  at closest approach, median: uprightness %.3f (need >%.2f) | "
          "rate %.2f (need <%.1f) | dz %.3f (need <%.2f)"
          % (med("up_at_min"), UP_TH, med("w_at_min"), W_TH, med("dz_at_min"), DZ_TH))
    print("  all four held at some step       : %d episodes" % sum(r["all4_ever"] for r in rows))
    print("  env recorded success             : %d episodes" % sum(r["env_success"] > 0.5 for r in rows))
    print()
    blockers = {"too far (XY)": 0, "not upright": 0, "not settled": 0, "wrong height": 0}
    for r in rows:
        if r["all4_ever"]:
            continue
        if r["min_xy"] > D_TH:
            blockers["too far (XY)"] += 1
        elif r["up_at_min"] <= UP_TH:
            blockers["not upright"] += 1
        elif r["w_at_min"] >= W_TH:
            blockers["not settled"] += 1
        else:
            blockers["wrong height"] += 1
    print("  of the failures, the blocking criterion at closest approach was:")
    for k, v in blockers.items():
        print("    %-14s %d" % (k, v))
    print("\nwrote %s" % a.out)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
