"""Build the w_E ablation table from the sweep's TensorBoard logs.

The five sweep runs differ only in DOUBLEBEE_W_E and are tagged by
DOUBLEBEE_RUN_NAME, so their directories look like

    logs/co_rl/doublebee_velocity/tqc/2026-09-02_00-14-11_wE0/
    ...                                              _wE0p25/
    ...                                              _wE1/   _wE2/   _wE4/

Reports PEAK, not final. The baseline run peaked at terrain_levels 1.36
around iteration 4050 and had decayed to 0.99 by 5000 while success stayed
flat -- i.e. it kept "succeeding" on easier ground. Quoting the last
iteration would understate every point in the table and would understate
them unequally, since the runs need not peak at the same iteration.

Peak is chosen by terrain_levels x success, because neither alone is
meaningful: success rises when the curriculum backs off, and terrain_levels
rises when the policy stops finishing. The product is what "climbs hard
terrain reliably" actually means.

Usage:
    python3 sweep_table.py <logs/co_rl/doublebee_velocity/tqc>
    python3 sweep_table.py <dir> --latex        # paper-ready tabular
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    sys.exit("need tensorboard: pip install tensorboard")

TAGS = {
    "success": "Metrics/success/rate",
    "levels": "Curriculum/terrain_levels",
    "ep_len": "Train/mean_episode_length",
    "reward": "Train/mean_reward",
    "energy": "Metrics/energy/successful_trajectories",
}


def scalars(run_dir):
    """tag -> [(step, value)], tolerating the several names co_rl uses."""
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    out = {}
    for name, tag in TAGS.items():
        pick = tag if tag in have else next(
            (t for t in have if t.split("/")[-1] == tag.split("/")[-1]), None)
        out[name] = [(s.step, s.value) for s in ea.Scalars(pick)] if pick else []
    return out


def peak(series, min_iter=500):
    """Index of peak by success x terrain_levels, plus the values there.

    min_iter exists because terrain_levels starts near 2.0 -- Isaac Lab spreads
    fresh environments across all 5 curriculum rows at init, so the MEAN level
    is ~2.0 before any learning -- and then FALLS as the curriculum demotes
    environments that fail. Combined with success being ~0.01 of random noise
    early on, the product success x terrain_levels is maximal at iteration 2,
    which is what this function reported on 2026-09-02 before the guard was
    added. Nothing before the curriculum has sorted itself is a real peak.
    """
    succ = dict(series["success"])
    lvl = dict(series["levels"])
    common = sorted(s for s in (set(succ) & set(lvl)) if s >= min_iter)
    if not common:
        return None
    best = max(common, key=lambda s: succ[s] * lvl[s])
    at = lambda k: dict(series[k]).get(best, float("nan"))
    return {
        "iter": best, "success": succ[best], "levels": lvl[best],
        "ep_len": at("ep_len"), "reward": at("reward"), "energy": at("energy"),
    }


def w_e_of(name):
    """wE0p25 -> 0.25"""
    m = re.search(r"_wE([0-9p]+)$", name)
    return float(m.group(1).replace("p", ".")) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="the tqc/ log directory holding the *_wE* runs")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--min-iter", dest="min_iter", type=int, default=500,
                   help="ignore iterations before this; terrain_levels starts "
                        "near 2.0 from the initial curriculum spread and decays, "
                        "so early points look like spurious peaks (default 500)")
    a = p.parse_args()

    runs = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_wE*"))):
        if not os.path.isdir(d):
            continue
        w = w_e_of(os.path.basename(d))
        if w is None:
            continue
        pk = peak(scalars(d), a.min_iter)
        if pk:
            runs.append((w, os.path.basename(d), pk))
    if not runs:
        sys.exit("no *_wE* run has reached iteration %d yet under %s\n"
                 "(use --min-iter 0 to see raw early numbers, but they are not peaks)"
                 % (a.min_iter, a.root))
    runs.sort()

    if a.latex:
        print(r"\begin{tabular}{lccccc}")
        print(r"\toprule")
        print(r"$w_E$ & success & terrain level & ep.\ length & energy (J) & iter \\")
        print(r"\midrule")
        for w, _, k in runs:
            print(r"%.2f & %.3f & %.2f & %.0f & %.0f & %d \\"
                  % (w, k["success"], k["levels"], k["ep_len"], k["energy"], k["iter"]))
        print(r"\bottomrule")
        print(r"\end{tabular}")
        return

    print("%-7s %-9s %-8s %-9s %-9s %-8s %s"
          % ("w_E", "success", "levels", "ep_len", "energy", "iter", "run"))
    print("-" * 78)
    for w, name, k in runs:
        print("%-7.2f %-9.3f %-8.2f %-9.0f %-9.0f %-8d %s"
              % (w, k["success"], k["levels"], k["ep_len"], k["energy"], k["iter"], name))
    print()
    print("peak selected by success x terrain_levels; see module docstring for why")
    if len(runs) < 5:
        print("NOTE: only %d of 5 runs found -- the rest may still be training" % len(runs))


if __name__ == "__main__":
    main()
