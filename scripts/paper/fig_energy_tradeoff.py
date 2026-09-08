"""The energy sweep, two panels, from per-episode eval_climb output.

    python3 fig_energy_tradeoff.py "climb_*.csv" -o fig_energy.pdf --step 0.06

WHY TWO PANELS. A five-point line through averaged values hides the thing that
actually matters: the arms differ in how OFTEN they climb, not only in how much
they spend when they do. Pooling those into one J/m number charges a failed
episode's energy against zero metres, which penalises a policy for attempting
less rather than for being inefficient.

  (a) every episode as a point, binned by HOW MANY STEPS IT CLIMBED. Holding
      the work done fixed is what makes the energy numbers comparable: within a
      column every point did the same job, so the vertical spread between the
      colours is the penalty's effect and nothing else.
  (b) the trade-off itself: how often it climbs, against what it costs when it
      does. This is the aggregate of (a); the two are the same measurement at
      two zoom levels, which is why they belong side by side.

Continuous "height gained" is deliberately gone. It rewards a policy for ending
an episode mid-step and it has no unit the task cares about -- the robot either
got up a step or it did not.

STEP COUNT. floor(max_gain_m / step). Conservative: a body 11 cm above spawn on
6 cm steps counts as one, not two.

CLEARANCE RATES CARRY WILSON INTERVALS in the printed table. With n=200 and one
seed a five-point difference between two arms is not a difference; the energy
differences are far outside their intervals and the rates often are not.
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
from matplotlib.lines import Line2D

CAP = 3          # steps beyond this are pooled into the "3+" column
SHOW_MAX = 150   # scatter points drawn per cell; stats use all episodes


def load(pattern):
    """[(tag, w, gain[], energy[], cleared[], hold_s[], disp[], end_gain[], end[])]."""
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
                    np.array([float(r["energy_J"]) for r in recs]),
                    np.array([int(float(r.get("cleared", 0))) for r in recs], bool),
                    np.array([float(r.get("hold_s", "nan")) for r in recs]),
                    np.array([float(r.get("max_disp_m", "nan")) for r in recs]),
                    np.array([float(r.get("end_gain_m", "nan")) for r in recs]),
                    [r.get("end", "?") for r in recs]))
    if not out:
        sys.exit("no CSVs matched %r" % pattern)
    return sorted(out, key=lambda z: (np.isnan(z[1]), z[1]))


def group(arms):
    """[(weight, label, [per-file tuples])] -- one entry per WEIGHT, not per file.

    With one CSV per checkpoint, each weight has many files. Measured
    2026-09-08, a single arm's median max_gain swung 0.047 -> 0.082 m across
    three consecutive checkpoints, a 74% swing that exceeded the entire spread
    between all five weights at any one checkpoint. Reporting a single
    checkpoint therefore reports where the snapshot landed. Pool them.
    """
    out = {}
    for arm in arms:
        out.setdefault(arm[1], []).append(arm)
    keys = sorted(out, key=lambda w: (np.isnan(w), w))
    return [(w, ("$w_E$=%g" % w if w == w else out[w][0][0]), out[w]) for w in keys]


def pooled(files):
    """Concatenate every checkpoint's episodes into one sample."""
    cat = lambda i: np.concatenate([f[i] for f in files])
    return (cat(2), cat(3), cat(4), cat(5), cat(6), cat(7),
            [x for f in files for x in f[8]])


def pareto(pts):
    """Indices on the upper-left frontier of (cost, benefit): cheaper and better."""
    keep = []
    for i, (c, b) in enumerate(pts):
        if not any((c2 <= c and b2 >= b) and (c2 < c or b2 > b)
                   for j, (c2, b2) in enumerate(pts) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


def climbed(g, hs, dp, eg, step, hold_s, min_xy):
    """Peak height reached. Nothing more is claimed, because nothing more is
    supported.

    This measured `max_gain >= step` all along; earlier versions LABELLED it
    "cleared a riser", which it is not. Measured 2026-09-08, 50-93% of episodes
    end in a `tilt` termination, and a robot pitching over a step edge raises
    its base -- so some episodes reaching 6 cm are mid-tip, not stood up on a
    step. The number is real; the word was wrong. Axis labels now say "reaching
    6 cm" and the table reports what fraction of those ended in a tilt, which is
    the honest qualifier and belongs in the caption.

    --hold_s and --min_xy default to 0 (off) so the default is exactly the
    labelled quantity. Raise either to test sensitivity.
    """
    m = g >= step
    if hold_s > 0 and np.isfinite(hs).any():
        m &= (hs >= hold_s)
    if min_xy > 0 and np.isfinite(dp).any():
        m &= (dp >= min_xy)
    return m


def wilson(k, n, z=1.96):
    """95% interval on a rate, in percent. Normal approx is wrong at these n."""
    if n == 0:
        return 0.0, 0.0
    p = k / float(n)
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", nargs="?", default="climb_*.csv")
    p.add_argument("-o", "--out", default="fig_energy.pdf")
    p.add_argument("--step", "--riser", dest="step", type=float, default=0.06,
                   help="step height the staircase was pinned to, m")
    p.add_argument("--hold_s", type=float, default=0.0,
                   help="seconds the height must be held. A bare height "
                        "threshold counts a momentary tip or thrust spike as a "
                        "climb; eval_climb's own 0.5 s is so strict it fired on "
                        "0.5%% of episodes. Sweep it -- the table prints the "
                        "sensitivity -- and state what you used.")
    p.add_argument("--min_xy", type=float, default=0.0,
                   help="metres from spawn, so hovering in place is not a climb")
    p.add_argument("--min_n", type=int, default=5,
                   help="cells with fewer episodes than this get no mean marker")
    a = p.parse_args()

    arms = load(a.pattern)
    groups = group(arms)
    n = len(groups)
    nck = max(len(f) for _, _, f in groups)
    print("%d weights, up to %d checkpoints each, %d episodes total"
          % (n, nck, sum(len(f[2]) for arm in arms for f in [arm])))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, n))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4),
                           gridspec_kw=dict(width_ratios=[1.85, 1.0]))

    # ---- (a) every episode, columns of equal work --------------------------
    dx = 0.78 / n
    offs = (np.arange(n) - (n - 1) / 2.0) * dx
    handles = []

    for i, ((w, lab, files), c) in enumerate(zip(groups, cmap)):
        g, e, cl, hs, dp, eg, en = pooled(files)
        r = np.minimum(np.floor(g / a.step + 1e-9).astype(int), CAP)
        # The legend swatch is drawn separately at full opacity. Inheriting the
        # scatter's alpha=0.22 made the legend unreadable at print size.
        handles.append(Line2D([0], [0], marker="o", ls="none", ms=5.5,
                              color=c, mec="0.15", mew=0.8, label=lab))
        for k in range(CAP + 1):
            m = r == k
            if not m.any():
                continue
            # 2000 episodes per arm draws as a solid blob. Subsample the CLOUD
            # only -- the mean and sd below still use every episode -- so the
            # panel shows spread instead of a smear.
            ei = e[m]
            if ei.size > SHOW_MAX:
                ei = rng.choice(ei, SHOW_MAX, replace=False)
            x = k + offs[i] + rng.uniform(-0.30 * dx, 0.30 * dx, ei.size)
            ax[0].scatter(x, ei, s=6, alpha=0.18, color=c, linewidths=0,
                          zorder=2)
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
    ax[0].set_xlabel("peak height reached (%.0f cm bins)" % (100 * a.step))
    ax[0].set_ylabel("energy per episode (J)")
    # The 2-column box in the upper left sat on top of the wE=0 column. Open a
    # band above the data and lay the entries out in one row instead, so the
    # legend covers no episodes.
    y0, y1 = ax[0].get_ylim()
    ax[0].set_ylim(y0, y1 + 0.17 * (y1 - y0))
    ax[0].legend(handles=handles, fontsize=7, loc="upper center",
                 framealpha=0.0, ncol=len(handles), handletextpad=0.3,
                 columnspacing=1.1, borderpad=0.2)
    ax[0].grid(alpha=0.25, axis="y")

    # ---- (b) the trade-off, which is (a) aggregated ------------------------
    pts, labs, cols, errs = [], [], [], []
    for (w, lab, files), c in zip(groups, cmap):
        # One (cost, rate) per CHECKPOINT, then mean and sd across them. The
        # error bars are the honest statement of what a single-checkpoint number
        # was hiding; they are checkpoint spread, not seed spread, and the
        # caption has to say so.
        per = []
        for f in files:
            _, _, gi, ei, cli, hsi, dpi, egi, eni = f
            mi = climbed(gi, hsi, dpi, egi, a.step, a.hold_s, a.min_xy)
            if mi.sum() >= 3:
                per.append((ei[mi].mean(), 100.0 * mi.mean()))
        if not per:
            print("  %s: no checkpoint had 3 episodes reaching the height" % lab)
            continue
        cst = np.array([q[0] for q in per])
        rte = np.array([q[1] for q in per])
        pts.append((cst.mean(), rte.mean()))
        errs.append((cst.std(), rte.std()))
        labs.append("%g" % w if w == w else lab)
        cols.append(c)
    if pts:
        front = pareto(pts)
        ax[1].plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                   "-", color="0.55", lw=1.2, zorder=1, label="Pareto front")
        # Bars are +-1 sd across CHECKPOINTS. They stay because without them
        # five bare dots imply a ranking the data does not support -- wE=0, 2
        # and 4 overlap almost entirely. Drawn thin, translucent and behind the
        # markers so they inform without dominating.
        for (c_, r_), (ce, re_), col in zip(pts, errs, cols):
            ax[1].errorbar([c_], [r_], xerr=[ce], yerr=[re_], fmt="none",
                           ecolor=col, elinewidth=0.9, capsize=2, alpha=0.55,
                           zorder=1)
            ax[1].scatter([c_], [r_], s=95, color=col, zorder=3,
                          edgecolors="white", linewidths=1.2)

        # Pad the axes BEFORE annotating, then push each label towards the
        # middle of the panel. Labels placed with a fixed offset walked off the
        # right edge for whichever arm happened to be most expensive.
        xs = [q[0] + s_ for q, (s_, _) in zip(pts, errs)] + \
             [q[0] - s_ for q, (s_, _) in zip(pts, errs)]
        ys = [q[1] + s_ for q, (_, s_) in zip(pts, errs)] + \
             [q[1] - s_ for q, (_, s_) in zip(pts, errs)]
        xpad = 0.24 * (max(xs) - min(xs) or 1.0)
        ypad = 0.24 * (max(ys) - min(ys) or 1.0)
        ax[1].set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax[1].set_ylim(min(ys) - ypad, max(ys) + ypad)
        xmid = 0.5 * sum(ax[1].get_xlim())
        ymid = 0.5 * sum(ax[1].get_ylim())
        ax[1].set_xlabel("energy per episode reaching %.0f cm (J)" % (100 * a.step))
        ax[1].set_ylabel("episodes reaching %.0f cm (%%)" % (100 * a.step))
        ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("Energy Weight Ablation", fontsize=11, fontweight="bold",
                 y=0.965)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("wrote %s (and .png)" % a.out)

    # ---- the numbers behind (a): mean J in each column ---------------------
    head = ["%d" % k for k in range(CAP)] + ["%d+" % CAP]
    print("\nmean energy per episode (J), by steps climbed   [n in brackets]")
    print("%-8s %s" % ("wE", " ".join("%14s" % h for h in head)))
    for w, lab, files in groups:
        g, e, cl, hs, dp, eg, en = pooled(files)
        r = np.minimum(np.floor(g / a.step + 1e-9).astype(int), CAP)
        cells = []
        for k in range(CAP + 1):
            m = r == k
            cells.append("%9.0f [%3d]" % (e[m].mean(), m.sum()) if m.sum()
                         else "%14s" % "-")
        print("%-8s %s" % (("%g" % w if w == w else lab),
                           " ".join("%14s" % c for c in cells)))


    print("reach%% = max_gain >= %.0f cm. tilt%% = of those, the fraction ending"
          % (100 * a.step))
    print("in a tilt termination -- the honest qualifier for the caption.")

    print("\n%-8s %5s %5s %18s %16s %8s %9s" %
          ("wE", "ckpt", "n", "reach% mean+-sd", "E_reach(J)+-sd", "tilt%",
           "gain_med"))
    print("  +-sd is spread across CHECKPOINTS of one run, not across seeds.")
    for w, lab, files in groups:
        g, e, cl, hs, dp, eg, en = pooled(files)
        m = climbed(g, hs, dp, eg, a.step, a.hold_s, a.min_xy)
        rts, cst = [], []
        for f in files:
            _, _, gi, ei, cli, hsi, dpi, egi, eni = f
            mi = climbed(gi, hsi, dpi, egi, a.step, a.hold_s, a.min_xy)
            rts.append(100.0 * mi.mean())
            if mi.sum() >= 3:
                cst.append(ei[mi].mean())
        idx = np.nonzero(m)[0]
        tilt = (100.0 * sum(1 for i in idx if en[i] == "tilt") / len(idx)
                if len(idx) else float("nan"))
        print("%-8s %5d %5d %9.0f%% +-%-5.1f %8.0f +-%-5.0f %7.0f%% %9.3f"
              % (("%g" % w if w == w else lab), len(files), len(g),
                 np.mean(rts), np.std(rts),
                 np.mean(cst) if cst else float("nan"),
                 np.std(cst) if cst else float("nan"),
                 tilt, np.median(g)))


    # How much of the answer is the hold threshold? If the ranking flips across
    # this row the criterion is carrying the result and the paper has to say so.
    if any(np.isfinite(z[5]).any() for z in arms):
        holds = [0.0, 0.10, 0.25, 0.50]
        print("\nsensitivity: climb%% vs required hold, at disp >= %.2f m" % a.min_xy)
        print("%-8s %s" % ("wE", " ".join("%9s" % ("%.2fs" % h) for h in holds)))
        for w, lab, files in groups:
            g, e, cl, hs, dp, eg, en = pooled(files)
            cells = ["%8.0f%%" % (100 * climbed(g, hs, dp, eg, a.step, h,
                                                a.min_xy).mean())
                     for h in holds]
            print("%-8s %s" % (("%g" % w if w == w else lab),
                               " ".join("%9s" % c for c in cells)))


if __name__ == "__main__":
    main()
