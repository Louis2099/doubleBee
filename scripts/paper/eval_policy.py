"""Evaluate a checkpoint on the FIXED play terrain and emit one row per episode.

Why this exists: --log_policy_io dumps observations and actions per step. It has
no episode boundaries, no success flag and no energy, so it cannot produce the
success-vs-energy comparison. This writes what the comparison needs instead.

Why fixed terrain: the w_E sweep arms finished at different curriculum levels
(w_E=0 at 0.07, w_E=4 at 1.57), so they were being scored on different step
heights -- roughly 3.7 cm versus 5.5 cm. Comparing their training-time energy is
comparing different tasks. The play task pins one difficulty for every policy,
which is what makes the numbers rankable.

Energy uses the SAME power model as the training reward
(rewards.py::penalize_energy_consumption): propeller joint speed -> PWM -> watts
and wheel speed -> RPM -> watts, both degree-4 fits to bench measurements,
integrated over step_dt. So the reported joules and the optimised objective are
the same quantity, and both are directly comparable to the integral of V*I
measured on hardware.

Success uses the four-criterion conjunction from constraints.py::goal_reached,
not XY proximity alone.

    python3 eval_policy.py --checkpoint <path> --episodes 60 --out eval_wE2.csv

Then:
    python3 eval_policy.py --summarise "eval_*.csv"
"""
import argparse
import csv
import glob
import os
import re
import sys


def summarise(pattern):
    """Aggregate the per-episode CSVs into the table the Pareto plot needs."""
    import numpy as np
    rows = []
    for path in sorted(glob.glob(pattern)):
        recs = list(csv.DictReader(open(path)))
        if not recs:
            continue
        succ = np.array([float(r["success"]) for r in recs])
        energy = np.array([float(r["energy_J"]) for r in recs])
        climb = np.array([float(r["climb_m"]) for r in recs])
        steps = np.array([float(r["steps"]) for r in recs])
        tag = re.sub(r"^eval_|\.csv$", "", os.path.basename(path))
        ok = succ > 0.5
        # Energy on SUCCESSFUL episodes only. A policy that fails early looks
        # cheap; that is not efficiency, it is not doing the task.
        rows.append((tag, len(recs), succ.mean(),
                     energy[ok].mean() if ok.any() else float("nan"),
                     energy[ok].std() if ok.any() else float("nan"),
                     climb[ok].mean() if ok.any() else float("nan"),
                     steps.mean()))
    if not rows:
        sys.exit("no eval CSVs matched %r" % pattern)
    print("%-10s %-7s %-9s %-14s %-10s %-9s" %
          ("policy", "n_ep", "success", "energy_J (SD)", "climb_m", "ep_len"))
    print("-" * 66)
    for t, n, s, e, es, c, l in rows:
        print("%-10s %-7d %-9.3f %6.0f (%5.0f)  %-10.3f %-9.0f" % (t, n, s, e, es, c, l))
    print()
    print("energy is over SUCCESSFUL episodes only, on identical terrain")
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summarise", help="glob of eval CSVs; aggregates and exits")
    p.add_argument("--checkpoint")
    p.add_argument("--task", default="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo")
    p.add_argument("--algo", default="tqc")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--num_envs", type=int, default=64,
                   help="episodes run in parallel; 64 collects 60 episodes fast")
    p.add_argument("--out", default="eval.csv")
    a = p.parse_args()

    if a.summarise:
        return summarise(a.summarise)
    if not a.checkpoint:
        sys.exit("--checkpoint required (or use --summarise)")

    # Isaac Lab must be launched before any isaaclab import.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch
    import gymnasium as gym
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import co_rl  # noqa: F401  registers the tasks
    from co_rl.core.runners import OffPolicyRunner
    from co_rl.core.wrapper import CoRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import goal_reached
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import (
        penalize_energy_consumption,
    )

    env_cfg = parse_env_cfg(a.task, num_envs=a.num_envs)
    env = gym.make(a.task, cfg=env_cfg)
    from isaaclab_tasks.utils import load_cfg_from_registry
    agent_cfg = load_cfg_from_registry(a.task, "co_rl_tqc_cfg_entry_point")
    # This task is a ManagerBasedConstraintRLEnv, so terminated/truncated come
    # back as FLOAT tensors. The wrapper picks `terminated | truncated` unless
    # use_constraint_rl is set, and bitwise-or is not defined for floats --
    # "bitwise_or_cuda not implemented for 'Float'". play.py sets this flag at
    # line 330 for the same reason.
    agent_cfg.use_constraint_rl = True
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Print it: on 2026-09-03 a shell one-liner sorting full paths on "_"
    # silently selected model_999 over model_4000 (tied timestamp field, then
    # string compare where '9' > '4'), and all five arms were evaluated a
    # thousand iterations before they had learned anything. Use `ls -v`.
    print("[eval] loading %s" % os.path.abspath(a.checkpoint), flush=True)
    runner.load(os.path.abspath(a.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    n = a.num_envs
    dev = env.unwrapped.device
    energy = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    z0 = env.unwrapped.scene["robot"].data.root_pos_w[:, 2].clone()
    zmax = z0.clone()
    done_rows = []

    obs, _ = env.get_observations()
    while len(done_rows) < a.episodes:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            # penalize_energy_consumption returns a NEGATIVE shaped penalty; the
            # raw joules are recovered from the same power model it uses.
            robot = env.unwrapped.scene["robot"]
            energy += _step_joules(env, robot)
            steps += 1
            z = robot.data.root_pos_w[:, 2]
            zmax = torch.maximum(zmax, z)
            succ = goal_reached(env.unwrapped, distance_threshold=0.25)

            fin = (dones > 0.5).nonzero(as_tuple=False).flatten()
            for i in fin.tolist():
                done_rows.append({
                    "success": float(succ[i].item()),
                    "energy_J": float(energy[i].item()),
                    "climb_m": float((zmax[i] - z0[i]).item()),
                    "steps": int(steps[i].item()),
                })
                energy[i] = 0.0
                steps[i] = 0.0
                z0[i] = z[i]
                zmax[i] = z[i]

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["success", "energy_J", "climb_m", "steps"])
        w.writeheader()
        w.writerows(done_rows[:a.episodes])
    print("wrote %s (%d episodes)" % (a.out, min(len(done_rows), a.episodes)))
    env.close()
    app.close()


def _step_joules(env, robot):
    """Joules this step, from the same power model the reward uses."""
    import torch
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import (
        _PWM_POWER_COEFFS, _PWM_POWER_IS_EXP,
        _RPM_POWER_COEFFS, _RPM_POWER_IS_EXP,
        _torch_polyval,
    )
    d = robot.device
    jn = robot.joint_names
    # Mirrors penalize_energy_consumption exactly, including the is_exponential
    # flags carried in the fitted-model JSON. Dropping those silently reports
    # log-watts as watts.
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
