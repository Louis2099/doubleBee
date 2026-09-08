"""The energy sweep, two panels, from per-episode eval_climb output.

    python3 fig_energy_tradeoff.py "climb_*.csv" -o fig_energy.pdf --riser 0.06

WHY TWO PANELS. A five-point line through averaged values hides the thing that
actually matters: the arms differ in how OFTEN they climb, not only in how much
they spend when they do. Pooling those into one J/m number charges a failed
episode's energy against zero metres, which penalises a policy for attempting
less rather than for being inefficient -- measured 2026-09-07, that made the
most-penalised arm look worst (8156 J/m) when on the episodes where it climbed
it was the best (5381 J/m).

  (a) every episode as a point, binned by HOW MANY RISERS IT CLEARED. Holding
      the work done fixed is what makes the energy numbers comparable: within a
      column every point did the same job, so the vertical spread between the
      colours is the penalty's effect and nothing else.
  (b) the trade-off itself: how often it clears, against what it costs when it
      does. This is the aggregate of (a); the two are the same measurement at
      two zoom levels, which is why they belong side by side.

Continuous "height gained" is deliberately gone. It rewards a policy for ending
an episode mid-riser and it has no units the task cares about -- the robot
either got up a step or it did not.

RISER COUNT. floor(max_gain_m / riser). Conservative: a body 11 cm above spawn
on 6 cm risers counts as one, not two.
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

CAP = 3          # risers beyond this are pooled into the "3+" column


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
    p.add_argument("--riser", type=float, default=0.06,
                   help="riser height the staircase was pinned to, m")
    p.add_argument("--min_n", type=int, default=5,
                   help="cells with fewer episodes than this get no mean marker")
    a = p.parse_args()

    arms = load(a.pattern)
    n = len(arms)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, n))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4),
                           gridspec_kw=dict(width_ratios=[1.85, 1.0]))

    # ---- (a) every episode, columns of equal work --------------------------
    dx = 0.78 / n
    offs = (np.arange(n) - (n - 1) / 2.0) * dx
    counts = np.zeros((n, CAP + 1), dtype=int)

    for i, ((tag, w, g, e), c) in enumerate(zip(arms, cmap)):
        r = np.minimum(np.floor(g / a.riser + 1e-9).astype(int), CAP)
        lab = "$w_E$=%g" % w if w == w else tag
        for k in range(CAP + 1):
            m = r == k
            counts[i, k] = m.sum()
            if not m.any():
                continue
            x = k + offs[i] + rng.uniform(-0.30 * dx, 0.30 * dx, m.sum())
            ax[0].scatter(x, e[m], s=7, alpha=0.22, color=c, linewidths=0,
                          label=lab if k == 0 else None, zorder=2)
            if m.sum() >= a.min_n:
                ax[0].errorbar(k + offs[i], e[m].mean(), yerr=e[m].std(),
                               fmt="o", ms=5.5, color=c, ecolor=c,
                               elinewidth=1.3, capsize=2.5,
                               mec="0.15", mew=0.8, zorder=4)
    for k in range(CAP):
        ax[0].axvline(k + 0.5, color="0.85", lw=0.8, zorder=0)
    ax[0].set_xticks(range(CAP + 1))
    ax[0].set_xticklabels([str(k) for k in range(CAP)] + ["%d+" % CAP])
    ax[0].set_xlim(-0.5 - dx, CAP + 0.5 + dx)
    ax[0].set_xlabel("risers cleared (%.0f cm each)" % (100 * a.riser))
    ax[0].set_ylabel("energy per episode (J)")
    ax[0].set_title("(a) every episode, grouped by work done",
                    fontsize=9, loc="left")
    ax[0].legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=2)
    ax[0].grid(alpha=0.25, axis="y")

    # ---- (b) the trade-off, which is (a) aggregated ------------------------
    pts, labs, cols = [], [], []
    for (tag, w, g, e), c in zip(arms, cmap):
        m = g >= a.riser
        if m.sum() < 3:
            print("  %s: only %d episodes cleared, omitted from (b)" % (tag, m.sum()))
            continue
        # x axis is ENERGY PER RISER CLEARED, not J/m. Every episode faces the
        # same riser, so "what one step costs" is the interpretable quantity;
        # dividing by a fractional metre is not.
        pts.append((e[m].mean(), 100.0 * m.mean()))
        labs.append("%g" % w if w == w else tag)
        cols.append(c)
    if pts:
        front = pareto(pts)
        ax[1].plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                   "-", color="0.55", lw=1.2, zorder=1, label="Pareto front")
        for (c_, r_), lab, col in zip(pts, labs, cols):
            ax[1].scatter([c_], [r_], s=95, color=col, zorder=3,
                          edgecolors="white", linewidths=1.2)
            ax[1].annotate("$w_E$=%s" % lab, xy=(c_, r_), xytext=(6, 5),
                           textcoords="offset points", fontsize=8)
        ax[1].set_xlabel("energy per riser cleared (J)")
        ax[1].set_ylabel("episodes clearing a riser (%)")
        ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) reliability against cost", fontsize=9, loc="left")
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("wrote %s (and .png)" % a.out)

    # ---- the numbers behind (a): mean J in each column ---------------------
    head = ["%d" % k for k in range(CAP)] + ["%d+" % CAP]
    print("\nmean energy per episode (J), by risers cleared   [n in brackets]")
    print("%-8s %s" % ("wE", " ".join("%14s" % h for h in head)))
    for i, (tag, w, g, e) in enumerate(arms):
        r = np.minimum(np.floor(g / a.riser + 1e-9).astype(int), CAP)
        cells = []
        for k in range(CAP + 1):
            m = r == k
            cells.append("%9.0f [%3d]" % (e[m].mean(), m.sum()) if m.sum()
                         else "%14s" % "-")
        print("%-8s %s" % (("%g" % w if w == w else tag),
                           " ".join("%14s" % c for c in cells)))

    print("\n%-8s %5s %8s %12s %11s" %
          ("wE", "n", "clear%", "E/riser(J)", "risers_med"))
    for tag, w, g, e in arms:
        m = g >= a.riser
        ok = m.sum() >= 3
        r = np.floor(g / a.riser + 1e-9)
        print("%-8s %5d %7.0f%% %12s %11.0f"
              % (("%g" % w if w == w else tag), len(g), 100 * m.mean(),
                 "%.0f" % e[m].mean() if ok else "-", np.median(r)))


if __name__ == "__main__":
    main()
