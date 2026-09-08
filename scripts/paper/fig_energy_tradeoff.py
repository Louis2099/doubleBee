"""The energy sweep, three panels, from per-episode eval_climb output.

    python3 fig_energy_tradeoff.py "climb_*.csv" -o fig_energy.pdf

WHY THREE PANELS. A five-point line through averaged values hides the thing that
actually matters: the arms differ in how OFTEN they climb, not only in how much
they spend when they do. Pooling those into one J/m number charges a failed
episode's energy against zero metres, which penalises a policy for attempting
less rather than for being inefficient -- measured 2026-09-07, that made the
most-penalised arm look worst (8156 J/m) when on the episodes where it climbed
it was the best (5381 J/m).

  (a) every episode as a point, so the reader sees the clouds and not a mean
  (b) the trade-off itself: how often it clears, against what it costs when it does
  (c) the gain distributions, which show the difference lives in the tail

CLEARANCE THRESHOLD. --clear defaults to 0.06 m, the riser height and the
h_max = r limit the whole evaluation is built around. Panels (b) and (c) are
sensitive to it, so it is a flag and it is printed on the figure.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern):
    """[(tag, weight, gain[], energy[])], ordered by weight."""
    out = []
    for path in sorted(glob.glob(pattern)):
        recs = list(csv.DictReader(open(path)))
        if not recs:
            continue
        tag = re.sub(r"^climb_|\.csv$", "", os.path.basename(path))
        m = re.search(r"(\d+(?:\.\d+)?)\s*$", tag)
        w = float(m.group(1)) if m else float("nan")
        out.append((tag, w,
                    np.array([float(r["max_gain_m"]) for r in recs]),
                    np.array([float(r["energy_J"]) for r in recs])))
    if not out:
        sys.exit("no CSVs matched %r" % pattern)
    return sorted(out, key=lambda z: (np.isnan(z[1]), z[1]))


def pareto(pts):
    """Indices on the upper-left frontier of (cost, benefit): cheaper and better."""
    keep = []
    for i, (c, b) in enumerate(pts):
        if not any((c2 <= c and b2 >= b) and (c2 < c or b2 > b)
                   for j, (c2, b2) in enumerate(pts) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", nargs="?", default="climb_*.csv")
    p.add_argument("-o", "--out", default="fig_energy.pdf")
    p.add_argument("--clear", type=float, default=0.06,
                   help="height that counts as clearing the riser, m")
    a = p.parse_args()

    arms = load(a.pattern)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(arms)))
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))

    # ---- (a) every episode -------------------------------------------------
    for (tag, w, g, e), c in zip(arms, cmap):
        ax[0].scatter(100 * g, e, s=9, alpha=0.30, color=c, linewidths=0,
                      label="$w_E$=%g" % w if w == w else tag)
    ax[0].axvline(100 * a.clear, color="0.4", ls="--", lw=1.0)
    ax[0].annotate("riser %.0f cm" % (100 * a.clear),
                   xy=(100 * a.clear, ax[0].get_ylim()[1]),
                   xytext=(3, -10), textcoords="offset points",
                   fontsize=7, color="0.35", va="top")
    ax[0].set_xlabel("height gained (cm)")
    ax[0].set_ylabel("energy per episode (J)")
    ax[0].set_title("(a) every episode, identical terrain", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax[0].grid(alpha=0.25)

    # ---- (b) the trade-off -------------------------------------------------
    pts, labs = [], []
    for tag, w, g, e in arms:
        m = g > a.clear
        if m.sum() < 3:
            print("  %s: only %d episodes cleared, omitted from (b)" % (tag, m.sum()))
            continue
        # x axis is ENERGY PER RISER CLEARED, not J/m. Every episode faces the
        # same riser, so "what one step costs" is the interpretable quantity;
        # dividing by a fractional metre is not.
        pts.append((e[m].mean(), 100.0 * m.mean()))
        labs.append("%g" % w if w == w else tag)
    if pts:
        cost = [q[0] for q in pts]
        rate = [q[1] for q in pts]
        front = pareto(pts)
        ax[1].plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                   "-", color="0.55", lw=1.2, zorder=1, label="Pareto front")
        for (c_, r_), lab, col in zip(pts, labs, cmap):
            ax[1].scatter([c_], [r_], s=95, color=col, zorder=3,
                          edgecolors="white", linewidths=1.2)
            ax[1].annotate("$w_E$=%s" % lab, xy=(c_, r_), xytext=(6, 5),
                           textcoords="offset points", fontsize=8)
        ax[1].set_xlabel("energy per riser cleared (J)")
        ax[1].set_ylabel("episodes clearing %.0f cm (%%)" % (100 * a.clear))
        ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) reliability against cost", fontsize=9, loc="left")
    ax[1].grid(alpha=0.25)

    # ---- (c) distributions -------------------------------------------------
    for (tag, w, g, e), c in zip(arms, cmap):
        xs = np.sort(100 * g)
        ax[2].plot(xs, 100 * np.arange(1, len(xs) + 1) / len(xs),
                   lw=1.7, color=c, label="$w_E$=%g" % w if w == w else tag)
    ax[2].axvline(100 * a.clear, color="0.4", ls="--", lw=1.0)
    ax[2].set_xlabel("height gained (cm)")
    ax[2].set_ylabel("episodes below (%)")
    ax[2].set_title("(c) gain distribution", fontsize=9, loc="left")
    ax[2].legend(fontsize=7, loc="lower right")
    ax[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("wrote %s (and .png)" % a.out)

    print("\n%-8s %5s %8s %12s %10s %9s" %
          ("wE", "n", "clear%", "E/riser(J)", "gain_med", "J/m"))
    for tag, w, g, e in arms:
        m = g > a.clear
        ok = m.sum() >= 3
        print("%-8s %5d %7.0f%% %12s %10.3f %9s"
              % (("%g" % w if w == w else tag), len(g), 100 * m.mean(),
                 "%.0f" % e[m].mean() if ok else "-",
                 np.median(g),
                 "%.0f" % (e[m].sum() / g[m].sum()) if ok else "-"))


if __name__ == "__main__":
    main()
