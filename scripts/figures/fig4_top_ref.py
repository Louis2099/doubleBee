"""Fig. 4 TOP panel from whatever has finished. Reference only.

Cells backed by fewer than 10 checkpoints are drawn hollow with dashed lines --
they are not publishable and the ordering they show is wrong at 3-5 cm.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.expanduser("~/Downloads/DoubleBee_fig4_top_PARTIAL")
LEARNED, MEC, MEW = "#c0392b", "0.15", 0.7
H = [3, 4, 5, 6, 7]
# arm -> (tw, [(mean, se, n_ckpt) per height])
D = {
 "Learned":  (None, [(67.3,5.6,10),(60.1,6.1,10),(52.6,4.8,10),(39.8,2.5,10),(28.1,2.4,10)]),
 "T/W 0.55": (0.55, [(60.9,7.3,10),(54.9,7.7,10),(37.7,6.7,10),(22.5,6.2,10),(8.9,5.9,10)]),
 "T/W 0.46": (0.46, [(26.5,0.0,1),(12.5,0.0,1),(7.5,0.0,1),(7.0,2.3,10),(0.9,0.6,9)]),
 "T/W 0.31": (0.31, [(50.5,15.4,3),(34.0,11.5,2),(13.5,4.3,3),(1.5,0.5,10),(0.2,0.1,10)]),
 "T/W 0.23": (0.23, [(46.0,35.5,2),(11.0,4.5,2),(10.8,3.2,2),(0.6,0.2,10),(0.0,0.0,9)]),
}
tws = sorted({v[0] for v in D.values() if v[0] is not None})
ramp = plt.cm.viridis(np.linspace(0.05, 0.85, len(tws)))
col = {t: ramp[i] for i, t in enumerate(tws)}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(3.45, 2.6))
dodge = np.linspace(-0.13, 0.13, len(D))

for i, (name, (tw, cells)) in enumerate(D.items()):
    c = LEARNED if tw is None else col[tw]
    xs = np.array(H, float) + dodge[i]
    ys = np.array([m for m, _, _ in cells])
    es = np.array([e for _, e, _ in cells])
    ns = np.array([n for _, _, n in cells])
    full = ns >= 10
    # solid where complete, dashed where partial
    ax.plot(xs, ys, "-", color=c, lw=1.1, zorder=3,
            alpha=1.0 if full.all() else 0.45)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=c, elinewidth=0.8,
                capsize=1.6, alpha=0.75, zorder=2)
    m = "*" if tw is None else "o"
    ms = 13 if tw is None else 5
    ax.plot(xs[full], ys[full], m, color=c, ms=ms,
            mec="white" if tw is None else MEC, mew=0.9 if tw is None else MEW,
            zorder=4)
    if (~full).any():
        ax.plot(xs[~full], ys[~full], m, color="white", ms=ms, mec=c, mew=1.1,
                zorder=4)

ax.set_xlabel("Staircase step height (cm)")
ax.set_ylabel("Clears one step (%)")
ax.set_xticks(H)
ax.grid(alpha=0.25, lw=0.5)
ax.set_ylim(-4, 92)
handles = [Line2D([], [], marker="*" if D[k][0] is None else "o",
                  color=LEARNED if D[k][0] is None else col[D[k][0]],
                  ls="none", ms=11 if D[k][0] is None else 5,
                  mec="white" if D[k][0] is None else MEC, label=k) for k in D]
handles.append(Line2D([], [], marker="o", color="white", mec="0.4", mew=1.1,
                      ls="none", ms=5, label="incomplete"))
ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right",
          handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
