"""The w_E ablation figure, from TensorBoard scalars.

THREE PANELS
  (a) curriculum progress: terrain level vs iteration, one line per w_E.
  (b) where each run ended up: mean terrain level over the final --window iters.
  (c) energy AT MATCHED DIFFICULTY: mean episode energy binned by terrain level.

Panel (c) is the one that carries the efficiency claim. The arms finish at
different curriculum levels, so their raw energy totals are not comparable: they
were doing different tasks. Plotting energy against terrain level controls for
that, and wherever two arms overlap in level the comparison is like for like.

WHY terrain level AND NOT success rate
  The curriculum promotes an environment only after three consecutive successes,
  and a success needs arrival within 0.25 m, upright, settled, at height. A run
  at level 1.57 has cleared ~5.5 cm risers repeatedly under that conjunction,
  which no lenient threshold can inflate.

WHY --min-iter EXISTS, AND WHY THE DEFAULT IS NOT ZERO
  terrain_levels STARTS near 2.0. Isaac Lab spreads fresh environments across
  all five curriculum rows at reset, so the mean begins mid-range and then FALLS
  as the curriculum demotes whatever fails. A plain max() over the whole run
  therefore reports that initial spread for every arm: all five come out at
  ~2.0 and the correlation with w_E reads -0.60, which is an artefact of the
  reset distribution and not a result. Scoring starts at --min-iter.

    python3 fig_energy_weight.py logs/co_rl/doublebee_velocity/tqc -o fig.pdf
    python3 fig_energy_weight.py logs/... --list-tags     # what is actually logged
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
    p.add_argument("--min-iter", dest="min_iter", type=int, default=500,
                   help="ignore iterations before this when SCORING (curves still "
                        "plot in full). Guards against the initial curriculum "
                        "spread near 2.0; see module docstring.")
    p.add_argument("--window", type=int, default=500,
                   help="headline statistic is the mean over the final N iterations, "
                        "which is more robust than a peak (one noisy sample)")
    p.add_argument("--energy-tag", default=None,
                   help="override the auto-detected energy scalar")
    p.add_argument("--list-tags", action="store_true",
                   help="print every scalar tag in the first run and exit")
    a = p.parse_args()

    runs = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_wE*"))):
        m = re.search(r"_wE([0-9p]+)$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        ea, tags = load(d)
        if a.list_tags:
            print("scalar tags in %s:" % os.path.basename(d))
            for t in tags:
                print("   ", t)
            return
        _, x, y = series(ea, tags, "terrain_levels")
        if x is None:
            continue
        etag, ex, ey = (None, None, None)
        if a.energy_tag:
            if a.energy_tag in tags:
                pts = ea.Scalars(a.energy_tag)
                etag = a.energy_tag
                ex = np.array([q.step for q in pts], float)
                ey = np.array([q.value for q in pts], float)
        else:
            etag, ex, ey = series(ea, tags, "energy")
        runs.append([float(m.group(1).replace("p", ".")), x, y, etag, ex, ey])
    if not runs:
        sys.exit("no *_wE* runs with terrain_levels under %s" % a.root)
    runs.sort(key=lambda r: r[0])

    have_energy = all(r[4] is not None for r in runs)
    ncol = 3 if have_energy else 2
    widths = [2.2, 1.0, 1.5] if have_energy else [2.2, 1.0]
    fig, axes = plt.subplots(1, ncol, figsize=(4.1 * ncol, 3.2),
                             gridspec_kw={"width_ratios": widths})
    ax, bx = axes[0], axes[1]
    cx = axes[2] if have_energy else None
    cmap = plt.cm.viridis(np.linspace(0.05, 0.88, len(runs)))

    stats = []
    for (w, x, y, etag, ex, ey), c in zip(runs, cmap):
        ax.plot(x, smooth(y), lw=1.7, color=c, label="$w_E=%g$" % w)
        late = x >= a.min_iter
        tail = x >= (x.max() - a.window)
        if not late.any():
            sys.exit("--min-iter %d exceeds the length of the w_E=%g run (%d iters)"
                     % (a.min_iter, w, int(x.max())))
        stats.append((w, float(y[late].max()), float(y[tail].mean()), c))

        # (c) energy binned by the terrain level in force at the same iteration
        if cx is not None:
            lv = np.interp(ex, x, y)
            keep = ex >= a.min_iter
            lv, e = lv[keep], ey[keep]
            edges = np.arange(0.0, max(2.0, lv.max()) + 0.2, 0.2)
            idx = np.digitize(lv, edges) - 1
            bl, bm = [], []
            for b in range(len(edges) - 1):
                sel = idx == b
                if sel.sum() >= 5:
                    bl.append(0.5 * (edges[b] + edges[b + 1]))
                    bm.append(np.median(e[sel]))
            if bl:
                cx.plot(bl, bm, "o-", ms=3.5, lw=1.5, color=c, label="$w_E=%g$" % w)

    ax.axvspan(0, a.min_iter, color="0.88", zorder=0)
    ax.text(a.min_iter * 0.5, ax.get_ylim()[1] * 0.97, "reset\nspread",
            fontsize=6.5, ha="center", va="top", color="0.35")
    ax.set_xlabel("iteration")
    ax.set_ylabel("terrain level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=2)
    ax.set_title("(a) curriculum progress", fontsize=9, loc="left")

    ws = [s[0] for s in stats]
    fin = [s[2] for s in stats]
    bx.bar(range(len(ws)), fin, color=[s[3] for s in stats], width=0.62)
    bx.set_xticks(range(len(ws)))
    bx.set_xticklabels(["%g" % w for w in ws])
    bx.set_xlabel("$w_E$")
    bx.set_ylabel("terrain level, last %d iters" % a.window)
    bx.grid(alpha=0.3, axis="y")
    bx.set_title("(b) where each run ended up", fontsize=9, loc="left")
    for i, h in enumerate(fin):
        bx.text(i, h, " %.1f cm" % step_cm(h), ha="center", va="bottom",
                fontsize=7, rotation=90)
    bx.set_ylim(0, max(fin) * 1.45)

    if cx is not None:
        cx.set_xlabel("terrain level (task difficulty)")
        cx.set_ylabel("episode energy")
        cx.grid(alpha=0.3)
        cx.legend(fontsize=7, frameon=False)
        cx.set_title("(c) energy at matched difficulty", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    # Also a PNG. A vector figure you cannot open is a figure you cannot check,
    # and PNG survives every viewer and every scp.
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")

    print("wrote %s\nwrote %s" % (a.out, png))
    if not have_energy:
        print("\nno energy scalar found, panel (c) omitted. "
              "run with --list-tags to see what is logged.")
    else:
        print("energy tag: %s" % runs[0][3])
    print("\n%-7s %-15s %-17s %s"
          % ("w_E", "peak(>%d)" % a.min_iter, "last-%d mean" % a.window, "step height"))
    for w, pk, fn, _ in stats:
        print("%-7g %-15.2f %-17.2f %.1f cm" % (w, pk, fn, step_cm(fn)))
    print("\ncorr(w_E, last-window level) = %+.2f" % np.corrcoef(ws, fin)[0, 1])
    print("corr(w_E, peak after %d)     = %+.2f"
          % (a.min_iter, np.corrcoef(ws, [s[1] for s in stats])[0, 1]))
    print("\nNOTE: n = 1 seed per setting. The trend across five settings is the "
          "result;\nno single value carries a variance estimate.")


if __name__ == "__main__":
    main()
