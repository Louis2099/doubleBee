"""Fig. 5, old grouped layout, new paired-seed numbers, single column."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.expanduser("~/Downloads/DoubleBee_fig5_grouped")
W = ["0", "2", "4", "6", "8"]
# wE -> [(E, SE, n) for groups 0, 1, 2+]
D = {
    "0": [(656.1, 28.1, 947), (747.4, 25.0, 789), (819.9, 17.4, 264)],
    "2": [(592.0, 19.4, 898), (663.4, 15.6, 839), (747.0, 16.2, 263)],
    "4": [(559.3, 31.4, 1204), (653.0, 39.8, 699), (737.3, 37.8, 97)],
    "6": [(514.2, 7.4, 1174), (586.6, 8.2, 709), (656.8, 38.7, 117)],
    "8": [(426.7, 31.5, 1765), (506.4, 10.5, 221), (529.3, 8.0, 14)],
}
ramp = plt.cm.viridis(np.linspace(0.05, 0.85, len(W)))
col = {w: ramp[i] for i, w in enumerate(W)}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(3.45, 2.6))

dx = 0.80 / len(W)
offs = (np.arange(len(W)) - (len(W) - 1) / 2.0) * dx
for i, w in enumerate(W):
    xs = np.arange(3) + offs[i]
    ys = [D[w][k][0] for k in range(3)]
    es = [D[w][k][1] for k in range(3)]
    ns = [D[w][k][2] for k in range(3)]
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3.8, color=col[w], ecolor=col[w],
                elinewidth=0.8, capsize=1.5, mec="0.15", mew=0.5, zorder=3)
    for x, y, n in zip(xs, ys, ns):
        if n < 50:                       # hollow: too few episodes to trust
            ax.plot(x, y, "o", ms=3.8, color="white", mec=col[w], mew=1.0,
                    zorder=4)

for g in (0.5, 1.5):
    ax.axvline(g, color="0.85", lw=0.6, zorder=0)
ax.set_xticks(range(3))
ax.set_xticklabels(["0", "1", "2+"])
ax.set_xlabel("6 cm steps climbed")
ax.set_ylabel("Energy per episode (J)")
ax.set_xlim(-0.62, 2.62)
ax.grid(alpha=0.25, lw=0.5, axis="y")

handles = [Line2D([], [], ls="none", marker="o", ms=3.8, color=col[w],
                  mec="0.15", mew=0.5, label="$w_E$=%s" % w) for w in W]
ax.legend(handles=handles, frameon=False, ncol=5, loc="upper left",
          bbox_to_anchor=(-0.02, 1.16), handletextpad=0.15,
          columnspacing=0.7, borderpad=0.0)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
