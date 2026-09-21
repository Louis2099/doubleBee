"""Fig. 5, single column: episode energy against energy weight, by work done.

w_E on x (the independent variable), one line per matched-work group. The claim
"increasing the penalty reduces energy at the same number of steps climbed" is
then three roughly parallel descending lines rather than fifteen dodged points.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/DoubleBee_fig5_energy_ablation")
WE = [0, 2, 4, 6, 8]
# group -> (means, SEs, n_episodes)
G = {
    "0 steps":  ([656.1, 592.0, 559.3, 514.2, 426.7],
                 [28.1, 19.4, 31.4, 7.4, 31.5],
                 [947, 898, 1204, 1174, 1765]),
    "1 step":   ([747.4, 663.4, 653.0, 586.6, 506.4],
                 [25.0, 15.6, 39.8, 8.2, 10.5],
                 [789, 839, 699, 709, 221]),
    "2+ steps": ([819.9, 747.0, 737.3, 656.8, 529.3],
                 [17.4, 16.2, 37.8, 38.7, 8.0],
                 [264, 263, 97, 117, 14]),
}
ramp = plt.cm.viridis(np.linspace(0.08, 0.68, 3))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(3.45, 2.45))

for (name, (m, se, n)), c in zip(G.items(), ramp):
    m, se, n = np.array(m), np.array(se), np.array(n)
    ax.errorbar(WE, m, yerr=se, fmt="-o", ms=4.2, lw=1.2, color=c, ecolor=c,
                elinewidth=0.9, capsize=1.8, mec="0.15", mew=0.6,
                label=name, zorder=3)
    # hollow any point backed by fewer than 50 episodes
    thin = n < 50
    if thin.any():
        ax.plot(np.array(WE)[thin], m[thin], "o", ms=4.2, color="white",
                mec=c, mew=1.1, zorder=4)

ax.set_xlabel("Energy weight $w_E$")
ax.set_ylabel("Energy per episode (J)")
ax.set_xticks(WE)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=False, loc="upper right", handlelength=1.6,
          handletextpad=0.4, labelspacing=0.25, borderpad=0.1)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf  (3.45 x 2.45 in, single column)")
for name, (m, se, n) in G.items():
    print("  %-9s %s   (n=%s)" % (name, " ".join("%.0f" % v for v in m), n))
