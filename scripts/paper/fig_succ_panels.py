"""Combine several succ_ablation runs into one multi-panel figure.

Each panel is one riser height, sharing a y axis, so the reader sees WHICH
criterion carries the inflation at each difficulty. That inversion is the
argument for the conjunction: at a height the robot can climb, it arrives and
topples, so uprightness does the work; at a height above the wheel radius it
parks upright at the base, so elevation does the work. Neither criterion alone
covers both failures.

Every panel is labelled with its own n. A ratio without a sample size beside it
is not reportable, and on 2026-09-04 a run produced alpha = 4.00 from 12 hits
over 3.

    python3 fig_succ_panels.py fig_succ_h5.counts.json fig_succ_h7.counts.json \
        -o fig_succ_ablation.pdf --labels "5 cm,7 cm"
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["XY only", "+ elev.", "+ upright", "+ settled"]
COLORS = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
MIN_N = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("counts", nargs="+", help="*.counts.json from succ_ablation.py")
    p.add_argument("-o", "--out", default="fig_succ_ablation.pdf")
    p.add_argument("--labels", help="comma-separated panel titles, e.g. '5 cm,7 cm'")
    a = p.parse_args()

    data = [json.load(open(f)) for f in a.counts]
    names = (a.labels.split(",") if a.labels
             else [os.path.basename(f).split(".")[0] for f in a.counts])
    if len(names) != len(data):
        sys.exit("--labels count does not match the number of files")

    fig, axes = plt.subplots(1, len(data), figsize=(2.4 * len(data) + 0.6, 2.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    thin = []
    for ax, d, nm in zip(axes, data, names):
        allf = d["all"]
        if allf <= 0:
            sys.exit("%s has no strict successes" % nm)
        if allf < MIN_N:
            thin.append((nm, allf))
        rel = [d["xy"] / allf, d["xyz"] / allf, d["xyzu"] / allf, 1.0]
        ax.bar(range(4), rel, color=COLORS, width=0.66)
        for i, v in enumerate(rel):
            ax.text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=7)
        ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=6.5, rotation=30, ha="right")
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("%s   ($n=%d$)" % (nm, allf), fontsize=8, loc="left")
        ax.annotate("rejected states:\n%.0f$\\degree$, %.1f rad/s"
                    % (d["lean_deg"], d["rate_rad_s"]),
                    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=6.5, color="#b2182b")
    axes[0].set_ylabel("reported success,\nrelative to full criterion")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=220, bbox_inches="tight")
    print("wrote %s (+ .png)" % a.out)
    for nm, n in thin:
        print("*** %s has n = %d strict successes (< %d). Do NOT quote its ratio."
              % (nm, n, MIN_N))


if __name__ == "__main__":
    main()
