"""Fig. 4 bottom panel: 6 cm clearance against mean power, paired seeds.

Numbers come from summ_optionB.py's conventions -- clearance is the mean across
ten checkpoints with a SAMPLE standard error, power is pooled total energy over
total control steps. The learned row reproduces Table IV exactly (39.8 +- 2.5,
286 W), which is the check that these agree with the table by construction.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/DoubleBee_fig4_bottom_6cm")
LEARNED, MEC, MEW = "#c0392b", "0.15", 0.7

# name, T/W (None = learned), clears %, SE, pooled power W
ARMS = [("Learned",  None, 39.8, 2.5, 286),
        ("T/W 0.55", 0.55, 22.5, 6.2, 391),
        ("T/W 0.46", 0.46,  7.0, 2.3, 347),
        ("T/W 0.31", 0.31,  1.5, 0.5, 269),
        ("T/W 0.23", 0.23,  0.6, 0.2, 227)]
tws = sorted({a[1] for a in ARMS if a[1] is not None})
ramp = plt.cm.viridis(np.linspace(0.05, 0.85, len(tws)))
col = {t: ramp[i] for i, t in enumerate(tws)}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(3.45, 2.5))

ct = sorted([a for a in ARMS if a[1] is not None], key=lambda q: q[4])
ax.plot([q[4] for q in ct], [q[2] for q in ct], "-", color="0.55", lw=1.2, zorder=1)

for name, tw, cl, e, pw in ARMS:
    c = LEARNED if tw is None else col[tw]
    m, ms = ("*", 15) if tw is None else ("o", 5.5)
    ax.errorbar(pw, cl, yerr=e, fmt="none", ecolor=c, elinewidth=0.9,
                capsize=2, zorder=3)
    ax.plot(pw, cl, m, color=c, ms=ms, mec="white" if tw is None else MEC,
            mew=0.9 if tw is None else MEW, zorder=4)
    dx, ha = (-8, "right") if name == "T/W 0.55" else (8, "left")
    ax.annotate(name, xy=(pw, cl), xytext=(dx, 2.5), textcoords="offset points",
                fontsize=6.3, color=c, ha=ha)

ax.set_xlabel("Mean power (W)")
ax.set_ylabel("Clears a 6 cm step (%)")
ax.grid(alpha=0.25, lw=0.5)
ax.set_xlim(205, 415)
ax.set_ylim(-3, 50)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
print("learned %.1f%% @ %dW vs best fixed %.1f%% @ %dW -> %.2fx at %.0f%% less power"
      % (39.8, 286, 22.5, 391, 39.8 / 22.5, 100 * (391 - 286) / 391))
