"""Actuator ablation: the fixed-allocation frontier against the learned policy.

WHY THIS FIGURE AND NOT THE OBVIOUS ONE. "Wheels-only fails and propellers-only
burns energy" is unanswerable because it is true and uninteresting. This plots a
BASELINE SWEPT ACROSS ITS WHOLE OPERATING RANGE instead: constant thrust at
T/W 0.23, 0.31, 0.46 and 0.55. That is a frontier, not a strawman, and it
answers the question a reviewer actually asked, which is how much the learned
policy buys over a well-designed fixed allocation.

Panel (a) is the frontier. Panel (b) is reach against height threshold, which
uses the whole gain distribution instead of one threshold nobody agreed on.

Success is REACH, not goal_reached. The goal counter undercounts: ct10 was
observed passing through the goal marker without terminating.

Error bars and bands are spread ACROSS CHECKPOINTS, ten per arm, 200 episodes
each. A single eval process is one draw of the terrain and spawn configuration
sampled 200 times, so a within-process interval is far too narrow: identical
flags on the same checkpoint gave reach-6cm of 22% and 47% on two occasions.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# label, glob tag, T/W (None = not a constant-thrust arm)
# Arms carried in TEXT only, never drawn. All three are floor results that a
# sentence states better than a marker on an axis: wo 0.0%, ws 0.3%, po 0.0%
# reach at 6 cm. po additionally is bounded by the simulator's 0.2 N.m
# propeller effort limit at T/W ~0.5 rather than the platform's real 1.16, so
# its zero describes Isaac and not the robot.
TEXT_ONLY = {"wo", "po", "ws"}

ARMS = [
    ("Learned modulation", "hE4", None),
    ("0.23",           "ctm45", 0.23),
    ("0.31",           "ctm05", 0.31),
    ("0.46",           "ct050", 0.46),
    ("0.55",           "ct10",  0.55),
    ("Wheels only",    "wo",    None),
    ("Wheels+servos",  "ws",    None),
    ("Propellers only", "po",   None),
]
DT = 0.02
THRESH = np.arange(0.03, 0.091, 0.005)   # overridden by --hmin/--hmax


def load(d, tag):
    """Per-checkpoint stats for one arm. Returns None if nothing on disk."""
    per = []
    for f in sorted(glob.glob(os.path.join(d, "climb_%s_model_*.csv" % tag))):
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        g = np.array([float(x["max_gain_m"]) for x in r])
        e = np.array([float(x["energy_J"]) for x in r])
        s = np.array([float(x["steps"]) for x in r])
        per.append(dict(
            it=int(re.search(r"model_(\d+)", f).group(1)),
            gain=g,
            # Mean power over ALL episodes: total joules over total seconds.
            # Not restricted to climbers, because restricting it would compare
            # arms on different subsets and the zero arms have no climbers.
            power=e.sum() / (s.sum() * DT),
            reach=np.array([100.0 * (g >= t).mean() for t in THRESH]),
        ))
    return per or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="abl_ckpt")
    p.add_argument("-o", "--out", default="fig_actuator_ablation.pdf")
    p.add_argument("--layout", choices=("stacked", "sbs"), default="stacked",
                   help="stacked = one column, panels one below the other, "
                        "which is what the paper uses")
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--hmin", type=float, default=3.0, help="cm")
    p.add_argument("--hmax", type=float, default=9.0, help="cm")
    a = p.parse_args()

    global THRESH
    THRESH = np.arange(a.hmin / 100.0, a.hmax / 100.0 + 1e-9, 0.005)

    data = {tag: load(a.dir, tag) for _, tag, _ in ARMS}
    have = [(lab, tag, tw) for lab, tag, tw in ARMS
            if data[tag] and tag not in TEXT_ONLY]
    if not have:
        raise SystemExit("no CSVs found under %s" % a.dir)

    i6 = int(np.argmin(np.abs(THRESH - 0.06)))
    # Same ramp as fig_energy_tradeoff.py:170 so the two figures read as a set.
    tws = sorted(tw for _, _, tw in have if tw is not None)
    ramp = plt.cm.viridis(np.linspace(0.05, 0.85, max(2, len(tws))))
    cnorm = {tw: ramp[i] for i, tw in enumerate(tws)}
    LEARNED = "#c0392b"
    MEC, MEW = "0.15", 0.7

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.7})
    if a.layout == "stacked":
        w = a.width or 3.45
        h = a.height or 4.55
        fig, ax = plt.subplots(2, 1, figsize=(w, h))
    else:
        w = a.width or 7.0
        h = a.height or 2.75
        fig, ax = plt.subplots(1, 2, figsize=(w, h))

    # ---- (a) frontier: power against reach at 6 cm ------------------------
    ctpts = []
    for lab, tag, tw in have:
        per = data[tag]
        pw = np.array([q["power"] for q in per])
        r6 = np.array([q["reach"][i6] for q in per])
        x, y = pw.mean(), r6.mean()
        xe = pw.std(ddof=1) if len(pw) > 1 else 0.0
        ye = r6.std(ddof=1) if len(r6) > 1 else 0.0
        if tw is not None:
            ctpts.append((tw, x, y, xe, ye))
        else:
            ax[0].errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=LEARNED,
                           elinewidth=0.9, capsize=2, zorder=4)
            ax[0].plot(x, y, "*", ms=15, color=LEARNED, mec="white", mew=0.9,
                       zorder=6)

    ctpts.sort()
    if ctpts:
        ax[0].plot([q[1] for q in ctpts], [q[2] for q in ctpts], "-",
                   color="0.55", lw=1.2, zorder=1)
        for tw, x, y, xe, ye in ctpts:
            ax[0].errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=cnorm[tw],
                           elinewidth=0.9, capsize=2, zorder=3)
            ax[0].plot(x, y, "o", ms=5.5, color=cnorm[tw], mec=MEC, mew=MEW,
                       zorder=5)
    ax[0].set_xlabel("Mean power (W)")
    ax[0].set_ylabel("Reaching 6 cm (%)")
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[0].set_ylim(bottom=-2)

    # ---- (b) reach against height threshold -------------------------------
    for lab, tag, tw in have:
        per = data[tag]
        R = np.vstack([q["reach"] for q in per])
        m = R.mean(0)
        sd = R.std(0, ddof=1) if len(per) > 1 else np.zeros_like(m)
        col = LEARNED if tw is None else cnorm[tw]
        lw = 1.8 if tw is None else 1.1
        ax[1].plot(100 * THRESH, m, "-", color=col, lw=lw,
                   zorder=5 if tw is None else 3)
        ax[1].fill_between(100 * THRESH, m - sd, m + sd, color=col,
                           alpha=0.13, lw=0)
    ax[1].set_xlabel("Height reached (cm)")
    ax[1].set_ylabel("Episodes (%)")
    ax[1].grid(alpha=0.25, lw=0.5)
    ax[1].set_ylim(bottom=-2)

    # ---- one legend for both panels ---------------------------------------
    #
    # The panels carry the SAME five series, so a legend in each would be the
    # same key printed twice. fig_energy_tradeoff.py puts a single row under
    # the panels for the same reason; matching it keeps the two figures a set.
    handles = [Line2D([], [], ls="none", marker="*", ms=12, color=LEARNED,
                      mec="white", mew=0.8, label="Learned")]
    handles += [Line2D([], [], ls="none", marker="o", ms=5.5, color=cnorm[tw],
                       mec=MEC, mew=MEW, label="$T/W$ %.2f" % tw)
                for tw in tws]
    ncol = len(handles)   # one row
    # tight_layout FIRST, then hang the legend just under the axes. Reserving a
    # band with rect= and then anchoring outside it leaves a visible gap once
    # bbox_inches="tight" crops to include both.
    band = 0.075 if a.layout == "stacked" else 0.13
    fig.tight_layout(pad=0.4, h_pad=1.1, rect=(0, band, 1, 1))
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.3,
               bbox_to_anchor=(0.5, band))

    fig.savefig(a.out, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print("wrote %s and %s  (%s, %.2f x %.2f in)" % (a.out, png, a.layout, w, h))

    print("\n%-20s %5s %8s %9s %8s" % ("arm", "ckpt", "power", "reach6", "sd"))
    for lab, tag, tw in ARMS:
        if not data[tag]:
            continue
        per = data[tag]
        pw = np.array([q["power"] for q in per])
        r6 = np.array([q["reach"][i6] for q in per])
        mark = "" if tag not in TEXT_ONLY else "   (text only)"
        print("%-20s %5d %7.0fW %8.1f%% %7.1f%s" %
              (lab, len(per), pw.mean(), r6.mean(),
               r6.std(ddof=1) if len(r6) > 1 else 0.0, mark))


if __name__ == "__main__":
    main()
