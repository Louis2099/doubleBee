"""Lower row of the generalization figure: base-link position over time.

One column per terrain, mirroring the layout of the brachiation paper's Fig 7:
horizontal progress on top, height underneath, sharing a time axis. The claim
the panels support is "it keeps making progress on ground it never trained on",
so those are the two traces that matter.

Traces come from eval_climb_gen.py --traj_out, which dumps env 0's FIRST
episode. Which checkpoint gets plotted is chosen by LOWEST MEAN PITCH across
the matching gen_eval CSVs, not by hand: every number in the paper still pools
all ten checkpoints, only the rendered trace is selected, the same way you pick
one of ten trial videos.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# terrain tag -> (greek label, human name)
TERRAINS = [
    ("GenRough",     ("alpha",   "Irregular ground")),
    ("GenSlopeUp",   ("beta",    "Slope, ascending")),
    ("GenStairDown", ("gamma",   "Steps, descending")),
    ("GenWave",      ("delta",   "Undulating")),
    ("GenSlopeDown", ("epsilon", "Slope, descending")),
]
GREEK = {"alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$",
         "delta": r"$\delta$", "epsilon": r"$\varepsilon$"}
C_DIST, C_Z, C_PITCH = "#1f5fa9", "#2e8b57", "#c0392b"


def pitch_rank(eval_dir, tag):
    """(checkpoint, mean pitch) sorted by mean pitch, lowest first."""
    out = []
    for f in sorted(glob.glob(os.path.join(eval_dir, "climb_%s_*.csv" % tag))):
        rows = list(csv.DictReader(open(f)))
        if not rows or "pitch_mean_deg" not in rows[0]:
            continue
        m = re.search(r"_(\d+)\.csv$", f)
        out.append((m.group(1) if m else "?",
                    float(np.mean([float(r["pitch_mean_deg"]) for r in rows]))))
    return sorted(out, key=lambda q: q[1])


def load_traj(traj_dir, tag, ckpt):
    p = os.path.join(traj_dir, "traj_%s_%s.csv" % (tag, ckpt))
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    if len(r) < 5:
        return None
    f = lambda k: np.array([float(x[k]) for x in r])
    t, x, y, z = f("t"), f("x"), f("y"), f("z")
    return dict(t=t, dist=np.hypot(x, y), z=z, pitch=f("pitch_deg"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="gen_eval")
    p.add_argument("--traj_dir", default="gen_traj")
    p.add_argument("-o", "--out", default="fig_generalization.pdf")
    p.add_argument("--only", nargs="*", default=None,
                   help="terrain tags to plot, in order (default: all found)")
    p.add_argument("--pitch", action="store_true",
                   help="add a third row showing body pitch")
    p.add_argument("--ckpt", default=None,
                   help="force one checkpoint instead of the lowest-pitch pick")
    a = p.parse_args()

    order = a.only or [t for t, _ in TERRAINS]
    cols = []
    for tag in order:
        rank = pitch_rank(a.eval_dir, tag)
        pick, mp = (a.ckpt, float("nan")) if a.ckpt else (rank[0] if rank else (None, None))
        if pick is None:
            print("  %-14s no eval CSVs, skipped" % tag)
            continue
        tr = load_traj(a.traj_dir, tag, pick)
        if tr is None:
            print("  %-14s no trajectory for ckpt %s, skipped" % (tag, pick))
            continue
        label = dict(TERRAINS).get(tag, ("", tag))
        cols.append((tag, label, pick, mp, tr, rank))

    if not cols:
        raise SystemExit("nothing to plot: run eval_climb_gen.py with --traj_out first")

    nrow = 3 if a.pitch else 2
    w = max(3.4, 2.15 * len(cols))
    fig, ax = plt.subplots(nrow, len(cols), figsize=(w, 1.35 * nrow + 0.5),
                           squeeze=False, sharex="col")
    plt.rcParams.update({"font.size": 7})

    for j, (tag, (gk, name), pick, mp, tr, rank) in enumerate(cols):
        ax[0][j].plot(tr["t"], tr["dist"], color=C_DIST, lw=1.2)
        ax[1][j].plot(tr["t"], tr["z"], color=C_Z, lw=1.2)
        ax[1][j].axhline(0.0, color="0.75", lw=0.6, ls=":")
        if a.pitch:
            ax[2][j].plot(tr["t"], tr["pitch"], color=C_PITCH, lw=1.0)
            ax[2][j].set_xlabel("time (s)", fontsize=7)
        else:
            ax[1][j].set_xlabel("time (s)", fontsize=7)

        ax[0][j].set_title("%s  %s" % (GREEK.get(gk, ""), name), fontsize=7.5)
        for i in range(nrow):
            ax[i][j].grid(alpha=0.25, lw=0.4)
            ax[i][j].tick_params(labelsize=6)
        if j == 0:
            ax[0][j].set_ylabel("distance (m)", fontsize=7)
            ax[1][j].set_ylabel("height (m)", fontsize=7)
            if a.pitch:
                ax[2][j].set_ylabel("pitch (deg)", fontsize=7)

        print("  %-14s ckpt %s  mean pitch %.1f deg  |  %.2f m in %.1f s, dz %+.3f m"
              % (tag, pick, mp, tr["dist"][-1], tr["t"][-1], tr["z"][-1]))
        if rank and not a.ckpt:
            print("      pitch ranking: %s" % "  ".join(
                "%s:%.1f" % q for q in rank[:5]))

    fig.tight_layout(pad=0.4)
    fig.savefig(a.out, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print("wrote %s and %s" % (a.out, png))


if __name__ == "__main__":
    main()
