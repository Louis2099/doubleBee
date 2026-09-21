"""Fig. 4 top panel from summ_k5.py output.

    python3 summ_k5.py abl_seeded 4 > fig4_top.csv      # on the box
    python3 fig4_top.py fig4_top.csv                    # here

Arm tags map to thrust-to-weight as verified against the published 6 cm values
(22.5, 7.0, 1.5, 0.6) rather than assumed from the hold action.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = sys.argv[1] if len(sys.argv) > 1 else "fig4_top.csv"
NCKPT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = os.path.expanduser("~/Downloads/DoubleBee_fig4_top")
LEARNED, MEC, MEW = "#c0392b", "0.15", 0.7
H = [3, 4, 5, 6, 7]
TW = {"ct10": 0.55, "ct050": 0.46, "ctm05": 0.31, "ctm45": 0.23}
ORDER = ["hE4", "ct10", "ct050", "ctm05", "ctm45"]

raw = {}
for r in csv.DictReader(open(SRC)):
    raw[(r["tag"], int(r["h"]))] = (float(r["mean"]), float(r["se"]), int(r["n"]))

ns = {n for (_, _), (_, _, n) in raw.items()}
print("checkpoints per cell:", sorted(ns))
missing = [(t, h) for t in ORDER for h in H if (t, h) not in raw]
if missing:
    print("MISSING CELLS:", missing)

ramp = plt.cm.viridis(np.linspace(0.05, 0.55, len(TW)))
col = {t: ramp[i] for i, t in enumerate(sorted(TW.values()))}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.7})
# Select first, then draw. Dodge and legend have to describe the arms actually
# plotted. Deriving either from ORDER leaves ghost legend entries for skipped
# arms and shifts the surviving curves off their true step heights.
plotted = []
for tag in ORDER:
    cells = [raw.get((tag, h)) for h in H]
    if any(c is None for c in cells):
        print("skipping arm with missing cells:", tag)
        continue
    # Uniformity, not just presence. An arm with 5/5/7/10/10 checkpoints would
    # mix reporting units inside one curve, and because late checkpoints drift
    # the thin cells are biased rather than merely noisy.
    counts = {n for _, _, n in cells}
    if counts != {NCKPT}:
        print("skipping arm at %d checkpoints (need %d everywhere): %s %s"
              % (min(counts), NCKPT, tag, sorted(counts)))
        continue
    plotted.append((tag, cells))
print("plotting %d arms at %d checkpoints" % (len(plotted), NCKPT))

fig, ax = plt.subplots(figsize=(3.45, 2.6))
span = 0.055 * (len(plotted) - 1)
dodge = np.linspace(-span, span, len(plotted)) if len(plotted) > 1 else [0.0]

for i, (tag, cells) in enumerate(plotted):
    learned = tag == "hE4"
    c = LEARNED if learned else col[TW[tag]]
    xs = np.array(H, float) + dodge[i]
    ys = np.array([m for m, _, _ in cells])
    es = np.array([e for _, e, _ in cells])
    ax.plot(xs, ys, "-", color=c, lw=1.1, zorder=3)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=c, elinewidth=0.8,
                capsize=1.6, alpha=0.75, zorder=2)
    ax.plot(xs, ys, "*" if learned else "o", color=c, ms=13 if learned else 5,
            mec="white" if learned else MEC, mew=0.9 if learned else MEW,
            zorder=4)

ax.set_xlabel("Staircase step height (cm)")
ax.set_ylabel("Clears one step (%)")
ax.set_xticks(H)
ax.grid(alpha=0.25, lw=0.5)
ax.set_ylim(-4, 92)
lab = {"hE4": "Learned"}
lab.update({t: "T/W %.2f" % v for t, v in TW.items()})
handles = [Line2D([], [], marker="*" if t == "hE4" else "o",
                  color=LEARNED if t == "hE4" else col[TW[t]], ls="none",
                  ms=11 if t == "hE4" else 5,
                  mec="white" if t == "hE4" else MEC, label=lab[t])
           for t, _ in plotted]
ax.legend(handles=handles, frameon=False, ncol=1, loc="upper right",
          handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
