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
    """Aggregate per-episode CSVs. Distributions, not thresholds.

    2026-09-05: the `cleared` flag (gain held 0.5 s while displaced 0.35 m) fired
    on 0.5% of episodes for a policy whose MEAN height gain was 60 mm, because
    episodes end at the goal or in a fall before that hold window closes. Any
    energy averaged over `cleared` episodes was therefore n=1. Height gain and
    energy per metre climbed need no threshold and are what the comparison
    should rest on.
    """
    import numpy as np
    rows = []
    for path in sorted(glob.glob(pattern)):
        recs = list(csv.DictReader(open(path)))
        if not recs:
            continue
        gain = np.array([float(r["max_gain_m"]) for r in recs])
        energy = np.array([float(r["energy_J"]) for r in recs])
        steps = np.array([float(r["steps"]) for r in recs])
        tag = re.sub(r"^climb_|\.csv$", "", os.path.basename(path))
        # energy per metre climbed, summed rather than averaged per episode so
        # episodes with ~0 gain cannot blow the ratio up
        jpm = energy.sum() / gain.sum() if gain.sum() > 1e-6 else float("nan")
        rows.append((tag, len(recs), gain.mean(), np.median(gain),
                     np.percentile(gain, 90), gain.max(),
                     energy.mean(), jpm, steps.mean()))
    if not rows:
        sys.exit("no eval CSVs matched %r" % pattern)
    print("%-9s %-6s %-8s %-8s %-8s %-8s %-9s %-10s %s"
          % ("policy", "n", "gain_mean", "median", "p90", "max", "E_mean(J)", "J/m climbed", "ep_len"))
    print("-" * 88)
    for t, n, gm, gmed, g90, gmax, em, jpm, sl in rows:
        print("%-9s %-6d %-8.3f %-8.3f %-8.3f %-8.3f %-9.0f %-10.0f %.0f"
              % (t, n, gm, gmed, g90, gmax, em, jpm, sl))
    # SPLIT BY WHETHER THE EPISODE CLIMBED.
    #
    # Pooled J/m charges a failed episode's energy against zero metres, which
    # penalises a policy for ATTEMPTING less rather than for being inefficient.
    # Measured 2026-09-07: pooled, hE8 looked worst at 8156 J/m; restricted to
    # episodes that cleared 6 cm it is the BEST at 5381, 11% cheaper per metre
    # than the unpenalised arm, and 21% cheaper above 9 cm.
    #
    # Both numbers are needed. Pooled answers "what does this policy cost me per
    # metre of stairs", which is what a deployment cares about. Split answers
    # "when it climbs, how efficiently", which is what the energy term is
    # actually shaping. Reporting only one of them misstates the trade-off.
    print("\nSAME DATA, RESTRICTED TO EPISODES THAT CLIMBED")
    for thr in (0.03, 0.06, 0.09):
        print("\n  gain > %.2f m  (= cleared one riser)" % thr)
        # ENERGY PER RISER CLEARED is the headline number, not J/m.
        #
        # Every episode faces the same riser, so "joules to get up one step" is
        # directly interpretable and needs no normalisation argument. J/m divides
        # by a distance nobody climbs -- a fractional metre -- and it moves when
        # a policy happens to overshoot in height, which is not efficiency.
        # Reported alongside so the two can be cross-checked, not instead of.
        print("  %-9s %5s %7s %11s %9s %9s" %
              ("policy", "n", "rate", "E/riser(J)", "gain_mean", "J/m"))
        for path in sorted(glob.glob(pattern)):
            recs = list(csv.DictReader(open(path)))
            if not recs:
                continue
            g = np.array([float(r["max_gain_m"]) for r in recs])
            e = np.array([float(r["energy_J"]) for r in recs])
            m = g > thr
            tag = re.sub(r"^climb_|\.csv$", "", os.path.basename(path))
            if m.sum() < 3:
                print("  %-9s %5d   (too few to average)" % (tag, m.sum()))
                continue
            print("  %-9s %5d %6.0f%% %11.0f %9.3f %9.0f"
                  % (tag, m.sum(), 100.0 * m.mean(), e[m].mean(),
                     g[m].mean(), e[m].sum() / g[m].sum()))
    print("\n  rate       = fraction of episodes that cleared that height")
    print("  E/riser(J) = mean energy on those episodes -- what one riser costs")
    print("  The pair (rate, E/riser) IS the trade-off. A policy can be cheap and")
    print("  rarely climb, or reliable and expensive; neither column alone decides.")

    print("\nAll episodes, identical terrain (the play curriculum, unmodified).")
    print("J/m climbed = total joules / total height gained: the efficiency number,")
    print("with no success threshold to argue about.")
    return



def _probe_managers(base):
    """Dump whatever decides resets, so the API is read rather than guessed.

    This env is a ManagerBasedConstraintRLEnv: it has a ConstraintManager, not
    the TerminationManager the first two attempts assumed, which is why every
    episode was labelled "?" on 2026-09-08.
    """
    print("\n===== MANAGER PROBE =====", flush=True)
    print("env class: %s" % type(base).__name__)
    names = [x for x in dir(base) if "manager" in x.lower() and not x.startswith("__")]
    print("manager attributes: %s" % names)
    for nm in names:
        try:
            m = getattr(base, nm)
        except Exception as ex:
            print("  %s -> unreadable (%r)" % (nm, ex))
            continue
        if m is None or isinstance(m, (str, int, float, bool)):
            print("  %s = %r" % (nm, m))
            continue
        print("  --- %s : %s" % (nm, type(m).__name__))
        for attr in ("active_terms", "_term_names", "term_names"):
            if hasattr(m, attr):
                try:
                    print("      %s = %s" % (attr, list(getattr(m, attr))))
                except Exception as ex:
                    print("      %s unreadable (%r)" % (attr, ex))
        bufs = [a for a in dir(m)
                if (a.endswith("_buf") or a in ("dones", "terminated", "time_outs"))
                and not a.startswith("__")]
        print("      buffers: %s" % bufs)
        for a in bufs:
            try:
                v = getattr(m, a)
                print("        %s: shape=%s sum=%s"
                      % (a, tuple(getattr(v, "shape", ())), float(v.sum())))
            except Exception:
                pass
        for a in ("_term_dones", "_term_values", "_term_cfgs"):
            if hasattr(m, a):
                try:
                    d = getattr(m, a)
                    print("      %s keys: %s" % (a, list(d) if hasattr(d, "keys")
                                                 else type(d).__name__))
                except Exception:
                    pass
    print("===== END PROBE =====\n", flush=True)
    sys.stdout.flush()


def _term_flags(tmgr, names):
    """name -> per-env bool tensor for this step.

    get_term() is the public API but is not present on every manager version,
    so fall back to the private buffer before giving up. Returning {} here is
    what produced `end=? 100%` on 2026-09-08.
    """
    out = {}
    buf = getattr(tmgr, "_term_dones", None)
    for nm in names:
        t = None
        try:
            t = tmgr.get_term(nm)
        except Exception:
            if isinstance(buf, dict):
                t = buf.get(nm)
        if t is not None:
            out[nm] = t
    if not out:
        # Cruder, but these two always exist: at least separate a real
        # termination from running out the 20 s horizon.
        for nm in ("terminated", "time_outs"):
            t = getattr(tmgr, nm, None)
            if t is not None:
                out[nm] = t
    return out


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
    p.add_argument("--probe", action="store_true",
                   help="step briefly, dump which manager decides resets, exit")
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

    # WHY each episode ends. Mean ep_len is ~110 of 1000 steps, so essentially
    # every episode terminates rather than running out the 20 s horizon, and a
    # fall and a goal arrival are opposite outcomes that look identical in a
    # dones flag. Read the termination manager per step and keep the first term
    # that fired. Wrapped, because a missing manager must not kill the eval.
    tmgr = getattr(base, "termination_manager", None)
    term_names = []
    if tmgr is None:
        print("[climb] NO termination_manager on %s; managers: %s"
              % (type(base).__name__,
                 [x for x in dir(base) if x.endswith("_manager")]))
    else:
        try:
            term_names = list(tmgr.active_terms)
        except Exception as ex:
            print("[climb] active_terms failed: %r" % (ex,))
        print("[climb] termination manager %s, terms: %s"
              % (type(tmgr).__name__, ", ".join(term_names) or "none"))


    robot = base.scene["robot"]
    spawn = robot.data.root_pos_w.clone()
    energy_j = torch.zeros(n, device=dev)     # our own integral of the power model
    run = torch.zeros(n, device=dev)          # consecutive steps above up_th
    cleared = torch.zeros(n, dtype=torch.bool, device=dev)
    max_gain = torch.zeros(n, device=dev)
    # Keep the two quantities the clearance criterion is built from, so the
    # criterion becomes a plotting-time choice instead of a decision frozen at
    # eval time. `cleared` alone is not enough: it fired on 0.5% of episodes on
    # 2026-09-05 and there was no way to tell a spike from a closed hold window.
    max_run = torch.zeros(n, device=dev)      # longest continuous hold, steps
    max_disp = torch.zeros(n, device=dev)     # furthest from spawn, m
    why = ["?"] * n                           # termination term that fired
    # Gain on the last LIVE step. If it tracks max_gain the robot climbed and
    # stayed up; if it decays to ~0 the robot went up and came back down, and
    # max_gain alone would have called that a climb.
    last_gain = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    rows = []

    obs, _ = env.get_observations()
    if a.probe:
        with torch.inference_mode():
            for _ in range(120):
                obs, _, _, _ = env.step(policy(obs))
        _probe_managers(base)
        app.close()
        return

    while len(rows) < a.episodes:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            pos = robot.data.root_pos_w
            # env.step() has ALREADY reset the environments that finished, so
            # for those k, pos is the new spawn on a different terrain tile and
            # `pos - spawn[k]` is the distance between two spawn points, not
            # anything the robot did. Folding that into maxima let one garbage
            # sample win: measured 2026-09-08, every episode reported
            # max_disp_m of 1.0-1.8 m, which is why the 0.35 m displacement
            # test never bound on anything. Fold in live environments only.
            alive = dones <= 0.5
            af = alive.float()

            steps += af
            energy_j += _step_joules(env, robot) * af
            gain = pos[:, 2] - spawn[:, 2]
            disp = torch.norm(pos[:, :2] - spawn[:, :2], dim=1)

            max_gain = torch.where(alive, torch.maximum(max_gain, gain), max_gain)
            max_disp = torch.where(alive, torch.maximum(max_disp, disp), max_disp)
            last_gain = torch.where(alive, gain, last_gain)

            up = alive & (gain >= up_th)
            run = torch.where(up, run + 1,
                              torch.where(alive, torch.zeros_like(run), run))
            max_run = torch.where(alive, torch.maximum(max_run, run), max_run)
            cleared |= alive & (run >= hold_steps) & (disp >= a.min_xy)

            if tmgr is not None:
                for nm, f in _term_flags(tmgr, term_names).items():
                    for k in f.nonzero(as_tuple=False).flatten().tolist():
                        if why[k] == "?":
                            why[k] = nm

            for k in (dones > 0.5).nonzero(as_tuple=False).flatten().tolist():
                rows.append({
                    "cleared": int(cleared[k].item()),
                    "max_gain_m": round(float(max_gain[k].item()), 4),
                    "energy_J": round(float(energy_j[k].item()), 2),
                    "steps": int(steps[k].item()),
                    "hold_s": round(float(max_run[k].item()) * base.step_dt, 3),
                    "max_disp_m": round(float(max_disp[k].item()), 3),
                    "end_gain_m": round(float(last_gain[k].item()), 4),
                    "end": why[k],
                })
                # reset this env's bookkeeping; it has already been respawned
                spawn[k] = pos[k]
                run[k] = 0.0
                cleared[k] = False
                max_gain[k] = 0.0
                steps[k] = 0.0
                energy_j[k] = 0.0
                max_run[k] = 0.0
                max_disp[k] = 0.0
                last_gain[k] = 0.0
                why[k] = "?"
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
    from collections import Counter
    ends = Counter(r["end"] for r in rows)
    max_ep = int(round(base.max_episode_length))
    sl = sorted(r["steps"] for r in rows)
    # The mean is not enough. On 2026-09-08 mean ep_len was 105 while a single
    # env was observed running 603 steps, so the distribution is skewed and a
    # mass of very short resets would dominate every statistic in the CSV.
    print("  episode length of %d steps (%.1f s): p10 %d  median %d  mean %.0f"
          "  p90 %d  max %d" % (max_ep, max_ep * base.step_dt,
                                sl[int(0.10 * len(sl))], sl[len(sl) // 2],
                                st.mean(sl), sl[int(0.90 * len(sl))], sl[-1]))
    tiny = sum(1 for x in sl if x < 25)
    print("  episodes under 25 steps (0.5 s): %d of %d (%.0f%%)"
          % (tiny, len(sl), 100.0 * tiny / len(sl)))
    print("  how episodes ended: %s"
          % ("  ".join("%s %.0f%%" % (k, 100.0 * v / len(rows))
                       for k, v in ends.most_common())))
    hs = [r["hold_s"] for r in rows]
    dp = [r["max_disp_m"] for r in rows]
    up_rows = [r for r in rows if r["max_gain_m"] >= up_th]
    print("  of %d episodes reaching %.3f m: median hold %.2f s, median disp %.2f m"
          % (len(up_rows), up_th,
             st.median([r["hold_s"] for r in up_rows]) if up_rows else 0.0,
             st.median([r["max_disp_m"] for r in up_rows]) if up_rows else 0.0))
    print("  hold_s over all episodes: median %.2f  p90 %.2f  max %.2f s"
          % (st.median(hs), sorted(hs)[int(0.9 * len(hs))], max(hs)))
    print("  max_disp_m: median %.2f  p90 %.2f  max %.2f m"
          % (st.median(dp), sorted(dp)[int(0.9 * len(dp))], max(dp)))
    eg = [r["end_gain_m"] for r in rows]
    mg = [r["max_gain_m"] for r in rows]
    held = sum(1 for r in rows if r["max_gain_m"] >= up_th
               and r["end_gain_m"] >= 0.6 * r["max_gain_m"])
    reached = sum(1 for r in rows if r["max_gain_m"] >= up_th)
    print("  end_gain_m: median %.3f  (max_gain median %.3f)"
          % (st.median(eg), st.median(mg)))
    print("  of %d episodes reaching %.3f m, %d (%.0f%%) were still up at the "
          "end" % (reached, up_th, held,
                   100.0 * held / reached if reached else 0.0))
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
