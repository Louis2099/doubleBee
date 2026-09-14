"""Success against staircase height, one line per actuator allocation.

The plot the ablation should have been from the start. The earlier version put
"height reached" on the x axis, which is a distribution over outcomes on ONE
terrain and needs a paragraph to explain. This puts the TERRAIN on the x axis:
each point is a separate 6 cm/5 cm/4 cm staircase and success is "cleared one
step of the staircase it is on". Self-explanatory, and it matches the design of
the energy figure, which also sweeps terrain height.

Error bars are spread ACROSS THE TEN CHECKPOINTS of each arm, the same
convention as fig_energy_tradeoff.py, because a single eval process is one draw
of the terrain and spawn configuration and its internal interval is far too
narrow.
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = [("Learned modulation", "hE4", None),
        ("T/W 0.55", "ct10", 0.55),
        ("T/W 0.46", "ct050", 0.46),
        ("T/W 0.31", "ctm05", 0.31),
        ("T/W 0.23", "ctm45", 0.23)]
HEIGHTS = [3, 4, 5, 6, 7]
LEARNED, MEC, MEW = "#c0392b", "0.15", 0.7


def cell(d, tag, h, mult):
    """(mean, SEM) of clearing `mult` steps, across checkpoints.

    Standard ERROR, not standard deviation. Every claim in the text is about an
    arm's MEAN, so the interval that belongs on it is the uncertainty in that
    mean, not the spread of an individual checkpoint. On the arms that matter
    that is 4.7 and 5.6 points rather than 14.8 and 17.6.

    The honest caveat, which the caption carries: successive checkpoints of one
    training run are not fully independent, so this is a mild underestimate.
    """
    per = []
    for f in sorted(glob.glob(os.path.join(d, "climb_%s_h%02d_*.csv" % (tag, h)))):
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        g = np.array([float(x["max_gain_m"]) for x in r])
        per.append(100.0 * (g >= mult * h / 100.0).mean())
    if len(per) < 2:
        return (np.nan, np.nan)
    return (float(np.mean(per)),
            float(np.std(per, ddof=1) / np.sqrt(len(per))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="abl_h")
    p.add_argument("-o", "--out", default="fig_step_height.pdf")
    p.add_argument("--width", type=float, default=3.45)
    p.add_argument("--height", type=float, default=4.55)
    p.add_argument("--frontier_h", type=int, default=6,
                   help="step height used for the power/capability panel")
    a = p.parse_args()

    tws = sorted(t for _, _, t in ARMS if t is not None)
    ramp = plt.cm.viridis(np.linspace(0.05, 0.85, len(tws)))
    col = {t: ramp[i] for i, t in enumerate(tws)}

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(2, 1, figsize=(a.width, a.height))

    # ---- (a) capability: clears one step, against staircase height --------
    # Error bars, NOT a filled band. There are only five measurement points;
    # a band drawn between them fills ground that was never sampled and implies
    # a smoothness the experiment does not establish. The collision problem
    # that a band solves is better solved by dodging each arm sideways.
    dodge = np.linspace(-0.13, 0.13, len(ARMS))
    for i, (name, tag, tw) in enumerate(ARMS):
        m = np.array([cell(a.dir, tag, h, 1) for h in HEIGHTS])
        if np.all(np.isnan(m[:, 0])):
            continue
        c = LEARNED if tw is None else col[tw]
        xs = np.array(HEIGHTS, dtype=float) + dodge[i]
        ax[0].errorbar(xs, m[:, 0], yerr=m[:, 1], fmt="none", ecolor=c,
                       elinewidth=0.8, capsize=1.6, alpha=0.75, zorder=2)
        ax[0].plot(xs, m[:, 0], "-", color=c,
                   lw=1.1, zorder=3)   # same weight as every other arm
        ax[0].plot(xs, m[:, 0], "*" if tw is None else "o", color=c,
                   ms=11 if tw is None else 5,
                   mec="white" if tw is None else MEC,
                   mew=0.9 if tw is None else MEW, zorder=4)
    ax[0].set_ylabel("Clears one step (%)")
    ax[0].set_xlabel("Staircase step height (cm)")
    ax[0].set_xticks(HEIGHTS)
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[0].set_ylim(bottom=-2)

    # ---- (b) cost: mean power against that capability, at h_max = r -------
    #
    # Deliberately NOT "clears two steps". Over 50k episodes the learned arm
    # and the best fixed allocation sit within +-4 points of each other on
    # two-step clearance at every height -- inside a checkpoint spread of
    # 10-18 -- and the learned arm LOSES at 3 and 7 cm. Modulation buys
    # single-step capability, not multi-step. That bound goes in the text; a
    # panel implying otherwise would not survive a reviewer rerunning this.
    pts = []
    for name, tag, tw in ARMS:
        E = St = 0.0
        for f in sorted(glob.glob(os.path.join(
                a.dir, "climb_%s_h%02d_*.csv" % (tag, a.frontier_h)))):
            r = list(csv.DictReader(open(f)))
            E += sum(float(x["energy_J"]) for x in r)
            St += sum(float(x["steps"]) for x in r)
        if St == 0:
            continue
        m, sd = cell(a.dir, tag, a.frontier_h, 1)
        pts.append((tw, E / (St * 0.02), m, sd))
    ct = sorted([q for q in pts if q[0] is not None])
    if ct:
        ax[1].plot([q[1] for q in ct], [q[2] for q in ct], "-",
                   color="0.55", lw=1.2, zorder=1)
    for tw, pw, m, sd in pts:
        c = LEARNED if tw is None else col[tw]
        ax[1].errorbar(pw, m, yerr=sd, fmt="none", ecolor=c, elinewidth=0.8,
                       capsize=2, zorder=2)
        ax[1].plot(pw, m, "*" if tw is None else "o", color=c,
                   ms=13 if tw is None else 5.5,
                   mec="white" if tw is None else MEC,
                   mew=0.9 if tw is None else MEW, zorder=4)
    ax[1].set_xlabel("Mean power (W)")
    ax[1].set_ylabel("Clears a %d cm step (%%)" % a.frontier_h)
    ax[1].grid(alpha=0.25, lw=0.5)
    ax[1].set_ylim(bottom=-2)
    # The arms span 228-391 W; letting matplotlib start the axis at 220 spent
    # the left third of the panel on empty space.
    if pts:
        xs = [q[1] for q in pts]
        pad = 0.06 * (max(xs) - min(xs))
        ax[1].set_xlim(min(xs) - pad, max(xs) + pad)

    handles = [Line2D([], [], ls="-", lw=1.1, marker="*", ms=10,
                      color=LEARNED, mec="white", mew=0.8, label="Learned")]
    handles += [Line2D([], [], ls="-", lw=1.1, marker="o", ms=5, color=col[t],
                       mec=MEC, mew=MEW, label="$T/W$ %.2f" % t) for t in tws]
    band = 0.075
    fig.tight_layout(pad=0.4, h_pad=1.0, rect=(0, band, 1, 1))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, handlelength=1.0, columnspacing=0.8,
               handletextpad=0.3, bbox_to_anchor=(0.5, band))

    fig.savefig(a.out, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print("wrote %s and %s" % (a.out, png))
    for name, tag, tw in ARMS:
        vals = [cell(a.dir, tag, h, 1)[0] for h in HEIGHTS]
        print("  %-20s %s" % (name, "  ".join("%5.1f%%" % v for v in vals)))


if __name__ == "__main__":
    main()
