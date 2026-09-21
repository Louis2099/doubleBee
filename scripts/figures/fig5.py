"""Fig. 5 regenerated under the paired-seed protocol (abl_seeded)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.expanduser("~/Downloads/DoubleBee_fig5_energy_ablation")
W = ["0", "2", "4", "6", "8"]
# wE -> [(E, SE) for groups 0, 1, 2+]
LEFT = {
    "0": [(656.1, 28.1), (747.4, 25.0), (819.9, 17.4)],
    "2": [(592.0, 19.4), (663.4, 15.6), (747.0, 16.2)],
    "4": [(559.3, 31.4), (653.0, 39.8), (737.3, 37.8)],
    "6": [(514.2, 7.4), (586.6, 8.2), (656.8, 38.7)],
    "8": [(426.7, 31.5), (506.4, 10.5), (529.3, 8.0)],
}
# wE -> (clears %, SE, E J, SE)
RIGHT = {
    "0": (52.6, 5.1, 766.2, 22.6),
    "2": (55.1, 5.4, 682.5, 16.0),
    "4": (39.8, 2.4, 663.3, 40.4),
    "6": (41.3, 2.9, 599.7, 9.0),
    "8": (11.8, 3.4, 507.0, 10.5),
}
ramp = plt.cm.viridis(np.linspace(0.05, 0.85, len(W)))
col = {w: ramp[i] for i, w in enumerate(W)}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.linewidth": 0.7})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.7),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0]))

# ---- left: energy by steps climbed, dodged within each group ----------------
dx = 0.78 / len(W)
offs = (np.arange(len(W)) - (len(W) - 1) / 2.0) * dx
for i, w in enumerate(W):
    xs = np.arange(3) + offs[i]
    ys = [LEFT[w][k][0] for k in range(3)]
    es = [LEFT[w][k][1] for k in range(3)]
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4.6, color=col[w], ecolor=col[w],
                elinewidth=0.9, capsize=1.8, mec="0.15", mew=0.6, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels(["0", "1", "2+"])
ax.set_xlabel("6 cm steps climbed")
ax.set_ylabel("Energy per episode (J)")
ax.grid(alpha=0.25, lw=0.5)
for g in (0.5, 1.5):
    ax.axvline(g, color="0.8", lw=0.6, zorder=0)
ax.set_xlim(-0.6, 2.6)

# ---- right: clearance against energy, trade-off line ------------------------
# NO connecting line. w_E=4 and w_E=6 differ by 1.5 +- 3.6 points (t=0.41), so
# any line through the points in energy order asserts an ordering the error bars
# deny -- whether drawn as a frontier or through all five. Points only.
for w in W:
    c, cs, e, es = RIGHT[w]
    bx.errorbar(e, c, xerr=es, yerr=cs, fmt="o", ms=5.2, color=col[w],
                ecolor=col[w], elinewidth=0.9, capsize=1.8, mec="0.15",
                mew=0.6, zorder=3)
bx.set_xlabel("Energy at 6 cm (J)")
bx.set_ylabel("Clears a 6 cm step (%)")
bx.grid(alpha=0.25, lw=0.5)
bx.set_ylim(0, 70)

handles = [Line2D([], [], ls="none", marker="o", ms=4.5, color=col[w],
                  mec="0.15", mew=0.6, label="$w_E$=%s" % w) for w in W]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.04), handletextpad=0.3, columnspacing=1.4)
fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
