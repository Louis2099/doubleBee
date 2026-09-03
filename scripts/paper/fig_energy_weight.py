"""The w_E ablation figure, from TensorBoard scalars.

DEFAULT OUTPUT IS ONE PANEL. --panels a gives the curriculum curves alone,
sized for a single IEEE column, because that plot is self-contained: it shows
the threshold at w_E >= 2 without needing any auxiliary statistic. The old
panel (b) was redundant, being nothing but the right-hand endpoint of (a).

  a  curriculum progress: terrain level vs iteration, one line per w_E.
     Smoothed line over a faint raw trace, so the post-breakaway oscillation
     stays visible instead of being averaged into a claim of convergence.
  b  bar of the final-window mean. Redundant with (a); kept for slide decks.
  c  energy at MATCHED DIFFICULTY: episode energy binned by terrain level.
     Arms that never climbed are EXCLUDED, not plotted as an invisible dot at
     level 0.1: with no overlap there is nothing to compare them against, and
     drawing them only makes a reader hunt for missing curves.

WHY terrain level AND NOT success rate
  The curriculum promotes an environment only after three consecutive
  successes, and a success needs arrival within 0.25 m, upright, settled, at
  height. A run at level 1.2 has cleared ~5 cm risers repeatedly under that
  conjunction, which no lenient threshold can inflate.

WHY --min-iter EXISTS
  terrain_levels STARTS near 2.0. Isaac Lab spreads fresh environments across
  all five curriculum rows at reset, so the mean begins mid-range and FALLS as
  the curriculum demotes what fails. A plain max() over the whole run reports
  that reset spread for every arm: all five come out at ~2.0 and the
  correlation with w_E reads -0.60, an artefact of the reset distribution
  rather than a result. Scoring starts at --min-iter. The default is 200: the
  transient is over by ~100 iterations, and a larger guard would hide the
  small excursions the flat arms make around iteration 2000, which belong in
  the honest picture.

CHECK THE ENERGY TAG BEFORE BELIEVING PANEL (c). The scalar is matched by
substring. If it resolves to a reward term it is w_E * joules, and comparing
it across w_E is circular. The tag name is printed; --list-tags dumps all.

    python3 fig_energy_weight.py logs/co_rl/doublebee_velocity/tqc -o fig.pdf
    python3 fig_energy_weight.py logs/... --panels ac
    python3 fig_energy_weight.py logs/... --list-tags
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

# step height at a mean curriculum level, from stair_config.py:
#   difficulty = (level + 0.5)/5,  step_h = 0.03 + difficulty*(0.09 - 0.03)
step_cm = lambda lv: 100 * (0.03 + ((lv + 0.5) / 5.0) * 0.06)

# an arm joins panel (c) only if it spent real time above this level; below it
# there is no overlap with the climbing arms and hence nothing to match on.
C_MIN_LEVEL = 0.2
C_MIN_BINS = 3


def load(run_dir):
    ea = EA.EventAccumulator(run_dir, size_guidance={EA.SCALARS: 0})
    ea.Reload()
    return ea, ea.Tags().get("scalars", [])


def series(ea, tags, *needles):
    """First tag whose lowercased name contains every needle."""
    for t in tags:
        low = t.lower()
        if all(n in low for n in needles):
            pts = ea.Scalars(t)
            return t, np.array([p.step for p in pts], float), \
                      np.array([p.value for p in pts], float)
    return None, None, None


def smooth(y, k=25):
    if len(y) < k:
        return y
    pad = np.r_[np.full(k // 2, y[0]), y, np.full(k - k // 2 - 1, y[-1])]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("-o", "--out", default="fig_energy_weight.pdf")
    p.add_argument("--panels", default="a",
                   help="which panels, e.g. 'a', 'ac', 'abc' (default: a)")
    p.add_argument("--min-iter", dest="min_iter", type=int, default=200,
                   help="ignore iterations before this when SCORING; curves still "
                        "plot in full. Guards the reset spread near 2.0.")
    p.add_argument("--window", type=int, default=500,
                   help="headline statistic is the mean over the final N iterations")
    p.add_argument("--units", choices=["cm", "level"], default="cm",
                   help="y axis of panel (a). 'cm' converts terrain level to "
                        "riser height, which is affine in the level and lets the "
                        "h_max = r line be drawn. 'level' keeps raw curriculum units.")
    p.add_argument("--r", type=float, default=0.058,
                   help="wheel radius in m. Sets h_max = r, the geometric climbing "
                        "limit: above it the step edge is higher than the axle and "
                        "no torque or thrust rolls the wheel over. TAPE MEASUREMENT, "
                        "5.5-6.0 cm. The demotion argument rests on it, so measure it.")
    p.add_argument("--r-lo", type=float, default=0.055, help="low end of the r estimate, m")
    p.add_argument("--start-iter", type=int, default=50,
                   help="clip the PLOT here. The reset spike to ~2.0 is not data and "
                        "eats half the vertical range. Scoring is unaffected.")
    p.add_argument("--energy-tag", default=None)
    p.add_argument("--list-tags", action="store_true")
    a = p.parse_args()

    want = [ch for ch in a.panels.lower() if ch in "abc"]
    if not want:
        sys.exit("--panels must contain at least one of a, b, c")

    # A resumed run writes a NEW timestamped directory whose TB steps continue
    # from the parent's last iteration (learn() does start_iter =
    # current_learning_iteration). So one weight can span several directories;
    # collect them all and concatenate by step. Suffix after the weight is free
    # form, e.g. "_wE0p25_ext".
    byw = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*_wE*"))):
        m = re.search(r"_wE([0-9p]+)(?:_.*)?$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        byw.setdefault(float(m.group(1).replace("p", ".")), []).append(d)

    runs = []
    for w, dirs in byw.items():
      xs, ys, exs, eys, etag = [], [], [], [], None
      for d in sorted(dirs):
        ea, tags = load(d)
        if a.list_tags:
            print("scalar tags in %s:" % os.path.basename(d))
            for t in tags:
                print("   ", t)
            return
        _, x, y = series(ea, tags, "terrain_levels")
        if x is None:
            continue
        xs.append(x); ys.append(y)
        if a.energy_tag:
            if a.energy_tag in tags:
                pts = ea.Scalars(a.energy_tag)
                etag = a.energy_tag
                exs.append(np.array([q.step for q in pts], float))
                eys.append(np.array([q.value for q in pts], float))
        else:
            t_, ex_, ey_ = series(ea, tags, "energy")
            if ex_ is not None:
                etag = t_; exs.append(ex_); eys.append(ey_)
      if not xs:
          continue
      x = np.concatenate(xs); y = np.concatenate(ys)
      o = np.argsort(x, kind="stable"); x, y = x[o], y[o]
      ex = ey = None
      if exs:
          ex = np.concatenate(exs); ey = np.concatenate(eys)
          o = np.argsort(ex, kind="stable"); ex, ey = ex[o], ey[o]
      if len(dirs) > 1:
          print("w_E=%g stitched from %d dirs, iters %d-%d"
                % (w, len(dirs), int(x.min()), int(x.max())))
      runs.append([w, x, y, etag, ex, ey])
    if not runs:
        sys.exit("no *_wE* runs with terrain_levels under %s" % a.root)
    runs.sort(key=lambda r: r[0])

    if "c" in want and any(r[4] is None for r in runs):
        print("no energy scalar in every run, dropping panel (c)")
        want = [w for w in want if w != "c"]

    # a alone gets IEEE single-column proportions; more panels widen the strip
    sizes = {"a": 3.5, "b": 1.6, "c": 3.0}
    widths = [sizes[w] for w in want]
    fig, axl = plt.subplots(1, len(want), figsize=(sum(widths) + 0.6, 2.7),
                            gridspec_kw={"width_ratios": widths})
    axl = np.atleast_1d(axl)
    P = dict(zip(want, axl))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.88, len(runs)))

    stats, cplot = [], []
    for (w, x, y, etag, ex, ey), c in zip(runs, cmap):
        if "a" in P:
            conv = step_cm if a.units == "cm" else (lambda v: v)
            vis = x >= a.start_iter
            # raw underneath at low alpha: the smoothed line alone would read as
            # a settled plateau where the run is in fact oscillating.
            P["a"].plot(x[vis], conv(y[vis]), lw=0.6, color=c, alpha=0.25)
            P["a"].plot(x[vis], conv(smooth(y)[vis]), lw=1.7, color=c,
                        label="$w_E=%g$" % w)
        late = x >= a.min_iter
        tail = x >= (x.max() - a.window)
        if not late.any():
            sys.exit("--min-iter %d exceeds the w_E=%g run (%d iters)"
                     % (a.min_iter, w, int(x.max())))
        t = y[tail]
        stats.append((w, float(y[late].max()), float(t.mean()),
                      float(np.percentile(t, 25)), float(np.percentile(t, 75)), c))

        if "c" in P:
            lv = np.interp(ex, x, y)
            keep = ex >= a.min_iter
            lv, e = lv[keep], ey[keep]
            edges = np.arange(0.0, max(2.0, lv.max()) + 0.2, 0.2)
            idx = np.digitize(lv, edges) - 1
            bl, bm = [], []
            for b in range(len(edges) - 1):
                sel = idx == b
                mid = 0.5 * (edges[b] + edges[b + 1])
                if sel.sum() >= 5 and mid >= C_MIN_LEVEL:
                    bl.append(mid)
                    bm.append(float(np.median(e[sel])))
            if len(bl) >= C_MIN_BINS:
                cplot.append((w, bl, bm, c))

    if "a" in P:
        ax = P["a"]
        if a.units == "cm":
            # The geometric wall. h_max = r, so the curriculum promoting a run
            # above this band is promoting it into terrain that cannot be
            # climbed at any torque -- which is what the demotions are.
            ax.axhspan(100 * a.r_lo, 100 * a.r, color="crimson", alpha=0.10, zorder=0)
            ax.axhline(100 * a.r, color="crimson", ls="--", lw=1.2, zorder=1)
            ax.text(0.985, 100 * a.r, " $h_{\max}=r$ ", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.5, color="crimson")
        ax.set_xlabel("iteration")
        ax.set_ylabel("riser height (cm)" if a.units == "cm" else "terrain level")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right", frameon=False, ncol=2,
                  columnspacing=1.0, handlelength=1.4)
        if len(want) > 1:
            ax.set_title("(a) curriculum progress", fontsize=9, loc="left")

    if "b" in P:
        bx = P["b"]
        ws = [s[0] for s in stats]
        fin = [s[2] for s in stats]
        bx.bar(range(len(ws)), fin, color=[s[5] for s in stats], width=0.62)
        bx.set_xticks(range(len(ws)))
        bx.set_xticklabels(["%g" % w for w in ws])
        bx.set_xlabel("$w_E$")
        bx.set_ylabel("terrain level, last %d" % a.window)
        bx.grid(alpha=0.3, axis="y")
        bx.set_title("(b) final level", fontsize=9, loc="left")
        for i, h in enumerate(fin):
            bx.text(i, h, " %.1f cm" % step_cm(h), ha="center", va="bottom",
                    fontsize=7, rotation=90)
        bx.set_ylim(0, max(fin) * 1.45)

    if "c" in P:
        cx = P["c"]
        for w, bl, bm, c in cplot:
            cx.plot(bl, bm, "o-", ms=3.5, lw=1.5, color=c, label="$w_E=%g$" % w)
        cx.set_xlabel("terrain level (task difficulty)")
        cx.set_ylabel("episode energy")
        cx.grid(alpha=0.3)
        cx.legend(fontsize=7.5, frameon=False)
        cx.set_title("(c) energy at matched difficulty", fontsize=9, loc="left")
        skipped = [s[0] for s in stats if s[0] not in [q[0] for q in cplot]]
        if skipped:
            print("panel (c) excludes w_E = %s: never sustained terrain level "
                  ">= %.1f, so no difficulty overlap to compare on."
                  % (", ".join("%g" % s for s in skipped), C_MIN_LEVEL))

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    # PNG too: a vector figure you cannot open is a figure you cannot check.
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print("wrote %s\nwrote %s   panels: %s" % (a.out, png, "".join(want)))
    if runs[0][3]:
        print("energy tag: %s   <-- if this is a REWARD term it is w_E*joules; "
              "panel (c) is then circular" % runs[0][3])

    print("\n%-6s %-12s %-24s %s"
          % ("w_E", "peak(>%d)" % a.min_iter, "last-%d mean [IQR]" % a.window,
             "step height"))
    for w, pk, mu, q1, q3, _ in stats:
        print("%-6g %-12.2f %.2f [%.2f, %.2f]%s%.1f cm"
              % (w, pk, mu, q1, q3, " " * 6, step_cm(mu)))
    ws = [s[0] for s in stats]
    print("\ncorr(w_E, last-window level) = %+.2f"
          % np.corrcoef(ws, [s[2] for s in stats])[0, 1])
    print("corr(w_E, peak after %d)     = %+.2f"
          % (a.min_iter, np.corrcoef(ws, [s[1] for s in stats])[0, 1]))
    print("\nNOTE: n = 1 seed per setting. The trend across five settings is the "
          "result;\nno single value carries a variance estimate. A wide IQR means "
          "the run is\noscillating at the edge of its capability, not settled there.")


if __name__ == "__main__":
    main()
