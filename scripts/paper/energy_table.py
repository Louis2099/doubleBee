"""The w_E sweep table, from TensorBoard scalars. One row per arm.

USE AN EXPLICIT GLOB. The 2026-09-01 cold-start sweep and the 2026-09-05
warm-started sweep both use directory suffixes _wE0, _wE10 and so on. They are
different experiments (one from scratch, one fine-tuned from a common
checkpoint) and merging them produces a curve that means nothing. Pass
--dirs '2026-09-0[45]*_wE*' or whatever selects the runs you actually want, and
check the printed directory list before believing the numbers.

Reports the mean over the final --window iterations, which is more robust than
a peak (one noisy sample). Energy is over SUCCESSFUL episodes: a policy that
fails early looks cheap, and that is not efficiency.

    python3 energy_table.py logs/co_rl/doublebee_velocity/tqc \
        --dirs '2026-09-0[45]*_wE*'
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator as EA
except ImportError:
    sys.exit("need tensorboard: pip install tensorboard")

step_cm = lambda lv: 100 * (0.03 + ((lv + 0.5) / 5.0) * 0.06)

WANT = [
    ("success",   ["metrics/success/rate"]),
    ("terrain",   ["terrain_levels"]),
    ("ep_len",    ["mean_episode_length"]),
    ("energy_ok", ["energy/successful_trajectories"]),
    ("energy_all",["energy/average_consumption"]),
    ("tilt",      ["constraint/tilt"]),
    ("r_energy",  ["reward/energy_consumption"]),
    ("r_goal",    ["reward/terminal_goal_reached"]),
    ("r_reach",   ["reward/reach_terrain_target"]),
]


def tail_mean(ea, tags, needles, window):
    for t in tags:
        low = t.lower()
        if all(n in low for n in needles.split("/")) or needles in low:
            pts = ea.Scalars(t)
            x = np.array([p.step for p in pts], float)
            y = np.array([p.value for p in pts], float)
            m = x >= (x.max() - window)
            return float(y[m].mean()), int(x.max())
    return float("nan"), 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--dirs", default="*_wE*",
                   help="glob INSIDE root selecting the arms; be specific")
    p.add_argument("--window", type=int, default=300,
                   help="average over the final N iterations")
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.root, a.dirs)))
    paths = [d for d in paths if os.path.isdir(d)]
    if not paths:
        sys.exit("no directories matched %r under %s" % (a.dirs, a.root))
    print("using %d directories:" % len(paths))
    for d in paths:
        print("   ", os.path.basename(d))

    rows = []
    for d in paths:
        m = re.search(r"_wE([0-9p]+)(?:_.*)?$", os.path.basename(d))
        if not m:
            print("  skipping unparsable: %s" % os.path.basename(d)); continue
        w = float(m.group(1).replace("p", "."))
        ea = EA.EventAccumulator(d, size_guidance={EA.SCALARS: 0}); ea.Reload()
        tags = ea.Tags().get("scalars", [])
        vals, last = {}, 0
        for key, needles in WANT:
            v, li = tail_mean(ea, tags, needles[0], a.window)
            vals[key] = v; last = max(last, li)
        vals["iter"] = last
        rows.append((w, vals))
    rows.sort(key=lambda r: r[0])

    print("\nfinal %d iterations" % a.window)
    print("%-6s %-6s %-8s %-8s %-8s %-7s %-9s %-7s %s"
          % ("w_E", "iter", "success", "terrain", "step_cm", "ep_len", "E_succ(J)", "tilt", "E share"))
    print("-" * 80)
    for w, v in rows:
        task = v["r_goal"] + v["r_reach"]
        share = abs(v["r_energy"]) / task if task > 1e-9 else float("nan")
        print("%-6g %-6d %-8.3f %-8.3f %-8.1f %-7.0f %-9.0f %-7.2f %.0f%%"
              % (w, v["iter"], v["success"], v["terrain"], step_cm(v["terrain"]),
                 v["ep_len"], v["energy_ok"], v["tilt"], 100 * share))

    print("\nE share = |energy penalty| / (terminal_goal + reach_target), the energy\n"
          "term as a fraction of task reward. Quote THIS, not the raw weight:\n"
          "an absolute w_E is meaningless without the reward scale.")
    print("\nEnergy is over SUCCESSFUL episodes. Arms at different curriculum\n"
          "levels are doing different tasks, so cross-arm energy needs the\n"
          "fixed-terrain evaluation (eval_climb.py) before it supports a claim.")


if __name__ == "__main__":
    main()
