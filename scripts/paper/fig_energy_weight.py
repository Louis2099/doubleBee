"""The w_E figure: terrain level vs iteration, one line per energy weight.

Plots terrain_levels rather than success rate, deliberately. The curriculum
promotes an environment only after three consecutive successes, and a success
requires arriving within 0.25 m of the target, upright, and settled. So a policy
sitting at level 1.57 has cleared ~5.5 cm risers repeatedly under the strict
conjunction; the number cannot be inflated by a lenient threshold the way a raw
success percentage can.

Left panel is the learning curves. Right panel is the peak each run reached,
which is what the text quotes.

    python3 fig_energy_weight.py <logs/co_rl/doublebee_velocity/tqc> -o fig_energy_weight.pdf
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator as EA
except ImportError:
    sys.exit("need tensorboard: pip install tensorboard")

# step height at a given mean curriculum level, from stair_config.py:
#   difficulty = (level + 0.5)/5,  step_h = 0.03 + difficulty*(0.09-0.03)
step_cm = lambda lv: 100 * (0.03 + ((lv + 0.5) / 5.0) * 0.06)


def series(run_dir, tail):
    ea = EA.EventAccumulator(run_dir, size_guidance={EA.SCALARS: 0})
    ea.Reload()
    tag = next((t for t in ea.Tags().get("scalars", []) if t.endswith(tail)), None)
    if not tag:
        return None
    pts = ea.Scalars(tag)
    return np.array([p.step for p in pts]), np.array([p.value for p in pts])


def smooth(y, k=25):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("-o", "--out", default="fig_energy_weight.pdf")
    a = p.parse_args()

    runs = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_wE*"))):
        m = re.search(r"_wE([0-9p]+)$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        s = series(d, "terrain_levels")
        if s:
            runs.append((float(m.group(1).replace("p", ".")), s))
    if not runs:
        sys.exit("no *_wE* runs with terrain_levels under %s" % a.root)
    runs.sort()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.2),
                                 gridspec_kw={"width_ratios": [2.4, 1]})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(runs)))
    peaks = []
    for (w, (x, y)), c in zip(runs, cmap):
        ax.plot(x, smooth(y), lw=1.7, color=c, label="$w_E=%g$" % w)
        peaks.append((w, y.max(), c))

    ax.set_xlabel("iteration")
    ax.set_ylabel("terrain level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_title("(a) curriculum progress", fontsize=9, loc="left")

    ws = [w for w, _, _ in peaks]
    hs = [h for _, h, _ in peaks]
    bx.bar(range(len(ws)), hs, color=[c for _, _, c in peaks], width=0.62)
    bx.set_xticks(range(len(ws)))
    bx.set_xticklabels(["%g" % w for w in ws])
    bx.set_xlabel("$w_E$")
    bx.set_ylabel("peak terrain level")
    bx.grid(alpha=0.3, axis="y")
    bx.set_title("(b) peak reached", fontsize=9, loc="left")
    for i, h in enumerate(hs):
        bx.text(i, h, " %.1f cm" % step_cm(h), ha="center", va="bottom",
                fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")

    r = np.corrcoef(ws, hs)[0, 1]
    print("wrote %s" % a.out)
    print("\n%-7s %-14s %s" % ("w_E", "peak level", "step height there"))
    for w, h, _ in peaks:
        print("%-7g %-14.2f %.1f cm" % (w, h, step_cm(h)))
    print("\ncorr(w_E, peak terrain level) = %+.2f" % r)


if __name__ == "__main__":
    main()
