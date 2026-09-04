"""Success-criterion ablation: one policy, one set of trajectories, four scoring rules.

WHAT IT SHOWS
  goal_reached ANDs four criteria: XY < 0.25 m, |dz| < 0.15 m, uprightness > 0.85,
  and body rate < 3.5 rad/s. constraints.py counts all four NESTED SUBSETS on the
  same calls, so a single rollout yields what each successively looser criterion
  would have reported for identical behaviour. No retraining, no separate runs.

  It also accumulates the attitude of the states that XY-only ACCEPTS and the
  full conjunction REJECTS: mean lean angle and mean body rate. Those are the
  states a naive success rate counts as a climb and the hardware experiences as
  a fall onto the target.

WHY RELATIVE AND NOT ABSOLUTE
  The counters increment per env per call, and goal_reached is called from the
  reward term, the constraint manager and the curriculum, so absolute counts are
  multiply-counted and are NOT episode success rates. The RATIO is unaffected,
  because all four subsets are counted on exactly the same calls. So this
  reports the inflation factor alpha = XY / all-four, normalised to the strict
  criterion at 1.0, and never claims an absolute rate.

CHECKPOINT
  Use the 41-dim checkpoint. baseline_4000/model_*.pt predates the range channel
  added 2026-09-03 and will fail to load; warm_start.pt from transplant_obs.py
  computes identically and has the right width.

    python3 succ_ablation.py --checkpoint <...>/warm_start.pt --steps 3000
    python3 succ_ablation.py --from-counts 4812,3907,2731,1994 --lean 38.4 --rate 2.91
"""
import argparse
import os
import sys


def plot(counts, lean, rate, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy, xyz, xyzu, allf = counts
    if allf <= 0:
        sys.exit("no strict successes recorded; run longer or check the checkpoint")
    rel = [xy / allf, xyz / allf, xyzu / allf, 1.0]
    labels = ["XY only\n$d_{xy}<0.25$",
              "+ elevation\n$|\\Delta z|<0.15$",
              "+ upright\n$u>0.85$",
              "+ settled\n$\\|\\omega\\|<3.5$"]
    colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(range(4), rel, color=colors, width=0.66)
    for i, v in enumerate(rel):
        ax.text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("reported success,\nrelative to full criterion")
    ax.set_ylim(0, max(rel) * 1.30)
    ax.grid(alpha=0.3, axis="y")
    ax.annotate("states XY-only accepts and\nthe full criterion rejects:\n"
                "mean lean %.0f$\\degree$, %.1f rad/s" % (lean, rate),
                xy=(0.98, 0.97), xycoords="axes fraction", va="top", ha="right",
                fontsize=6.5, color="#b2182b")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=220, bbox_inches="tight")
    print("wrote %s (+ .png)" % out)
    print("\nalpha = XY / all-four = %.2f  (a proximity-only metric reports %.0f%% more)"
          % (rel[0], 100 * (rel[0] - 1)))
    print("elevation removes %.0f%%, uprightness %.0f%%, settling %.0f%% of the inflation"
          % (100 * (rel[0] - rel[1]) / (rel[0] - 1) if rel[0] > 1 else 0,
             100 * (rel[1] - rel[2]) / (rel[0] - 1) if rel[0] > 1 else 0,
             100 * (rel[2] - rel[3]) / (rel[0] - 1) if rel[0] > 1 else 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--task", default="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--step-height", dest="step_height", type=float, default=None,
                   help="pin every staircase to this riser height in metres. On the "
                        "easy end of the play terrain the goal sits at the robot's own "
                        "height, so |dz| < 0.15 is satisfied for free and the elevation "
                        "criterion removes nothing (measured 2026-09-04: XY 3846, +Z "
                        "3846, identical). Pinning a tall riser is what makes the "
                        "criteria separable.")
    p.add_argument("-o", "--out", default="fig_succ_ablation.pdf")
    p.add_argument("--from-counts", help="xy,xyz,xyzu,all -- plot without running sim")
    p.add_argument("--lean", type=float, default=float("nan"))
    p.add_argument("--rate", type=float, default=float("nan"))
    a = p.parse_args()

    if a.from_counts:
        c = [float(x) for x in a.from_counts.split(",")]
        return plot(c, a.lean, a.rate, a.out)
    if not a.checkpoint:
        sys.exit("--checkpoint required (or --from-counts)")

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
    if a.step_height is not None:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.curriculum = False
        tg.num_rows, tg.num_cols = 1, 5
        tg.use_cache = False
        _k = next(k for k in tg.sub_terrains if "stair" in k)
        tg.sub_terrains[_k].step_height_range = (a.step_height, a.step_height)
        print("[succ] staircase pinned at %.3f m" % a.step_height, flush=True)
    env = gym.make(a.task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(a.task, "co_rl_tqc_cfg_entry_point")
    agent_cfg.use_constraint_rl = True      # float terminated/truncated; see play.py:330
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[succ] loading %s" % os.path.abspath(a.checkpoint), flush=True)
    runner.load(os.path.abspath(a.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    for i in range(a.steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if i % 250 == 0:
            print("\r[succ] %d/%d" % (i, a.steps), end="", flush=True)
    print()

    ab = getattr(env.unwrapped, "_succ_abl", None)
    if ab is None:
        sys.exit("no counters found. DOUBLEBEE_SUCCESS_ABLATION must not be 0.")
    counts = [ab["xy"].item(), ab["xyz"].item(), ab["xyzu"].item(), ab["all"].item()]
    ln = ab["loose_n"].item()
    lean = ab["lean_sum"].item() / ln if ln else float("nan")
    rate = ab["rate_sum"].item() / ln if ln else float("nan")
    print("counts  XY %.0f | +Z %.0f | +upright %.0f | +settled %.0f"
          % tuple(counts))
    print("loose-but-not-strict states: n=%.0f  mean lean %.1f deg  mean rate %.2f rad/s"
          % (ln, lean, rate))
    plot(counts, lean, rate, a.out)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
