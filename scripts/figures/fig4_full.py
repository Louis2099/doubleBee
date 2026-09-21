"""Fig. 4, both panels, from whatever has finished.

    python3 summ_k5.py abl_seeded 9 > fig4_all.csv    # on the box
    python3 fig4_full.py fig4_all.csv

Top    single-step clearance against step height, all five arms
Bottom clearance of a 6 cm step against mean power

Cells resting on fewer than NCKPT checkpoints are drawn HOLLOW with a dashed
segment. They are not publishable. Late checkpoints drift hard, so a thin cell
is biased rather than merely noisy, and at 3-5 cm the thin cells currently
invert the ordering of the fixed allocations.

Power at 6 cm is fixed and comes from summ_arms.py (total energy / total
control steps / 0.02 s). No outstanding evaluation can move it, since every
evaluation still running is at 3, 4 or 5 cm.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = sys.argv[1] if len(sys.argv) > 1 else "fig4_all.csv"
NCKPT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = os.path.expanduser(sys.argv[3] if len(sys.argv) > 3
                         else "~/Downloads/DoubleBee_fig4_full")
LEARNED, MEC, MEW = "#c0392b", "0.15", 0.7
H = [3, 4, 5, 6, 7]
TW = {"ct10": 0.55, "ct050": 0.46, "ctm05": 0.31, "ctm45": 0.23}
POWER = {"hE4": 286, "ct10": 391, "ct050": 347, "ctm05": 269, "ctm45": 227}
ORDER = ["hE4", "ct10", "ct050", "ctm05", "ctm45"]
# Legend runs up the thrust ladder. Plotting order and colour assignment are
# deliberately left alone, so only the legend sequence changes.
LEGEND_ORDER = ["hE4", "ctm45", "ctm05", "ct050", "ct10"]

raw = {}
for r in csv.DictReader(open(SRC)):
    raw[(r["tag"], int(r["h"]))] = (float(r["mean"]), float(r["se"]), int(r["n"]))

thin = [(t, h, raw[(t, h)][2]) for t in ORDER for h in H
        if (t, h) in raw and raw[(t, h)][2] < NCKPT]
print("cells below %d checkpoints: %d" % (NCKPT, len(thin)))
for t, h, n in thin:
    print("   %-6s %d cm  n=%d" % (t, h, n))

ramp = plt.cm.viridis(np.linspace(0.05, 0.80, len(TW)))
col = {t: ramp[i] for i, t in enumerate(sorted(TW.values()))}
colour = lambda t: LEARNED if t == "hE4" else col[TW[t]]
mark = lambda t: "*" if t == "hE4" else "o"
size = lambda t: 12 if t == "hE4" else 5.5

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.0,
                     "axes.linewidth": 0.7})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.45, 4.35))

# ---- top: clearance against step height -----------------------------------
dodge = np.linspace(-0.13, 0.13, len(ORDER))
for i, tag in enumerate(ORDER):
    cells = [raw.get((tag, h)) for h in H]
    if any(c is None for c in cells):
        print("arm has missing cells, skipped:", tag)
        continue
    c = colour(tag)
    xs = np.array(H, float) + dodge[i]
    ys = np.array([m for m, _, _ in cells])
    es = np.array([e for _, e, _ in cells])
    ns = np.array([n for _, _, n in cells])
    full = ns >= NCKPT
    # Solid only where every endpoint of the segment is complete.
    for a, b in zip(range(len(H) - 1), range(1, len(H))):
        ok = full[a] and full[b]
        ax.plot(xs[[a, b]], ys[[a, b]], "-" if ok else "--", color=c,
                lw=1.1 if ok else 0.9, alpha=1.0 if ok else 0.5, zorder=3)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=c, elinewidth=0.8,
                capsize=1.6, alpha=0.75, zorder=2)
    ax.plot(xs[full], ys[full], mark(tag), color=c, ms=size(tag),
            mec="white" if tag == "hE4" else MEC,
            mew=0.9 if tag == "hE4" else MEW, zorder=4)
    if (~full).any():
        ax.plot(xs[~full], ys[~full], mark(tag), color="white", ms=size(tag),
                mec=c, mew=1.1, zorder=4)

ax.set_xlabel("Staircase step height (cm)", labelpad=2)
ax.set_ylabel("Clears one step (%)", labelpad=2)
ax.set_xticks(H)
ax.grid(alpha=0.25, lw=0.5)
ax.set_ylim(-4, 82)
ax.tick_params(length=2, pad=1.5)

# ---- bottom: clearance of a 6 cm step against mean power -------------------
fixed = sorted((t for t in ORDER if t != "hE4"), key=lambda t: POWER[t])
fx = [POWER[t] for t in fixed if (t, 6) in raw]
fy = [raw[(t, 6)][0] for t in fixed if (t, 6) in raw]
bx.plot(fx, fy, "-", color="0.55", lw=1.0, zorder=1)
for tag in ORDER:
    if (tag, 6) not in raw:
        continue
    m, e, n = raw[(tag, 6)]
    c = colour(tag)
    bx.errorbar([POWER[tag]], [m], yerr=[e], fmt="none", ecolor=c,
                elinewidth=0.8, capsize=1.6, alpha=0.8, zorder=2)
    hollow = n < NCKPT
    bx.plot([POWER[tag]], [m], mark(tag), ms=size(tag),
            color="white" if hollow else c, mec=c if hollow else
            ("white" if tag == "hE4" else MEC),
            mew=1.1 if hollow else (0.9 if tag == "hE4" else MEW), zorder=4)

bx.set_xlabel("Mean power (W)", labelpad=2)
bx.set_ylabel("Clears a 6 cm step (%)", labelpad=2)
bx.grid(alpha=0.25, lw=0.5)
bx.set_ylim(-4, 50)
bx.tick_params(length=2, pad=1.5)

lab = {"hE4": "Learned"}
lab.update({t: "$T/W$ %.2f" % v for t, v in TW.items()})
handles = [Line2D([], [], marker=mark(t), color=colour(t), ls="-", lw=1.1,
                  ms=10 if t == "hE4" else 5.5,
                  mec="white" if t == "hE4" else MEC, label=lab[t])
           for t in LEGEND_ORDER]
fig.legend(handles=handles, frameon=False, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.035), handletextpad=0.25,
           columnspacing=0.7)
fig.tight_layout(pad=0.3, h_pad=1.4, rect=(0, 0.035, 1, 1))
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
