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

ENERGY is integrated here from the same power model the reward
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
    p.add_argument("--step-height", type=float, default=None,
                   help="pin every staircase to this riser height. DEFAULT None = "
                        "use the play terrain unchanged, which is what you want: "
                        "pinning breaks the spawn/target patch sampling.")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--hold", type=float, default=0.5,
                   help="seconds the height gain must be held; separates climbing "
                        "from a thrust-driven altitude spike")
    p.add_argument("--min-xy", dest="min_xy", type=float, default=0.35,
                   help="metres of horizontal displacement required, so hovering "
                        "above the spawn platform does not count")
    p.add_argument("--clear-gain", dest="clear_gain", type=float, default=0.04,
                   help="height gain in metres counting as cleared when the "
                        "terrain is NOT pinned. 0.04 sits above the smallest "
                        "riser (0.03) and below the largest (0.09).")
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

    # DO NOT pin the terrain by default.
    #
    # 2026-09-05: pinning it (num_rows=1, curriculum=False, single
    # step_height_range) broke the task. Spawn and target flat patches are
    # sampled per curriculum tile, so collapsing the terrain changed the
    # geometry the policies were trained in: mean height gain came out at
    # 0.018-0.044 m for EVERY arm, including one that reaches 5.2 cm risers in
    # training. That measured the harness, not the policies.
    #
    # The play terrain is identical for every arm, so it is already a matched
    # comparison. Use it as-is and report the height-gain distribution.
    if a.step_height is not None:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.curriculum = False
        tg.num_rows, tg.num_cols = 1, 5
        tg.use_cache = False
        key = next(k for k in tg.sub_terrains if "stair" in k)
        tg.sub_terrains[key].step_height_range = (a.step_height, a.step_height)
        print("[climb] staircase PINNED at %.3f m -- verify the policies still "
              "climb before trusting this" % a.step_height, flush=True)
    else:
        print("[climb] play terrain as trained (steps 0.03-0.09 m over 5 rows)", flush=True)

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
    # With the terrain unpinned the riser height varies per environment, so a
    # fraction-of-step threshold is undefined. Use an absolute height gain:
    # 0.04 m is above the smallest riser (0.03) and below the largest (0.09), so
    # it means "got up at least one real step" across the whole curriculum.
    up_th = (a.frac * a.step_height) if a.step_height is not None else a.clear_gain

    robot = base.scene["robot"]
    spawn = robot.data.root_pos_w.clone()
    energy_j = torch.zeros(n, device=dev)     # our own integral of the power model
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
            # Integrate the SAME power model the reward uses. Reading
            # episode_energy_buf at this point returns 0, because env.step() has
            # already reset the finished environments and cleared it -- which is
            # why every cleared episode reported 0 J on 2026-09-05.
            energy_j += _step_joules(env, robot)
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
                    "energy_J": round(float(energy_j[k].item()), 2),
                    "steps": int(steps[k].item()),
                })
                # reset this env's bookkeeping; it has already been respawned
                spawn[k] = pos[k]
                run[k] = 0.0
                cleared[k] = False
                max_gain[k] = 0.0
                steps[k] = 0.0
                energy_j[k] = 0.0
            if rows:
                print("\r[climb] %d/%d" % (len(rows), a.episodes), end="", flush=True)
    print()

    rows = rows[:a.episodes]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    frac = sum(r["cleared"] for r in rows) / len(rows)
    import statistics as st
    g = [r["max_gain_m"] for r in rows]
    e = [r["energy_J"] for r in rows if r["cleared"]]
    print("wrote %s   cleared %.0f%% of %d episodes%s"
          % (a.out, 100 * frac, len(rows),
             ("  at h=%.0f cm" % (100 * a.step_height)) if a.step_height else
             "  (play terrain, gain >= %.3f m)" % a.clear_gain))
    print("  height gain: mean %.3f  median %.3f  p90 %.3f  max %.3f m"
          % (st.mean(g), st.median(g), sorted(g)[int(0.9 * len(g))], max(g)))
    if e:
        print("  energy on cleared episodes: mean %.0f  median %.0f J"
              % (st.mean(e), st.median(e)))
    env.close()
    app.close()


def _step_joules(env, robot):
    """Joules this control step, from the reward's own power model.

    Mirrors rewards.py::penalize_energy_consumption exactly, including the
    is_exponential flags carried in the fitted-model JSON. Dropping those
    silently reports log-watts as watts.
    """
    import torch
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import (
        _PWM_POWER_COEFFS, _PWM_POWER_IS_EXP,
        _RPM_POWER_COEFFS, _RPM_POWER_IS_EXP,
        _torch_polyval,
    )
    d = robot.device
    jn = robot.joint_names
    pv = robot.data.joint_vel[:, [jn.index("leftPropeller"), jn.index("rightPropeller")]]
    pwm = torch.clamp(1000.0 + (pv.abs() / 500.0) * 650.0, 1000.0, 2000.0)
    pp = _torch_polyval(_PWM_POWER_COEFFS.to(d), pwm)
    if _PWM_POWER_IS_EXP:
        pp = torch.exp(pp)
    pp = torch.clamp(pp, min=0.0).sum(dim=1)

    wv = robot.data.joint_vel[:, [jn.index("leftWheel"), jn.index("rightWheel")]]
    rpm = torch.clamp(wv.abs() * (60.0 / (2.0 * torch.pi)), 0.0, 300.0)
    wp = _torch_polyval(_RPM_POWER_COEFFS.to(d), rpm)
    if _RPM_POWER_IS_EXP:
        wp = torch.exp(wp)
    wp = torch.clamp(wp, min=0.0).sum(dim=1)

    return (pp + wp) * env.unwrapped.step_dt



if __name__ == "__main__":
    main()
