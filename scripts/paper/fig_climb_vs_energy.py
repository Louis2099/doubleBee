"""Capability and cost against riser height, from eval_climb.py output.

  (a) fraction of episodes clearing the riser, vs riser height, per w_E.
      This is the capability curve. It uses kinematics only, so unlike anything
      built on terrain_levels it is not gated by goal_reached.
  (b) energy over CLEARED episodes only, vs riser height, per w_E.
      Restricting to cleared episodes is what makes this efficiency rather than
      inactivity: a policy that never leaves the platform spends almost nothing.

Together they answer the question the curriculum plot could not: how high can
each policy get, and what does it pay to get there, on identical terrain.

The geometric limit h_max = r is drawn on (a). No policy can clear a riser
taller than the wheel radius at any thrust, so the capability curve must fall to
zero there; if it does not, the climb detector is counting flight.

    python3 fig_climb_vs_energy.py "climb_*.csv" -o fig_climb_energy.pdf --r 0.058
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help='glob of eval_climb CSVs, e.g. "climb_*.csv"')
    p.add_argument("-o", "--out", default="fig_climb_energy.pdf")
    p.add_argument("--r", type=float, default=0.058, help="wheel radius, m")
    a = p.parse_args()

    # climb_wE0p25_h5.csv -> ("wE0p25", 5)
    data, pid = {}, []
    for path in sorted(glob.glob(a.pattern)):
        base = os.path.basename(path)
        # climb_pid_h5.csv is the decoupled/PID baseline, produced by
        # play_dctrl.py --step_height --climb_episodes on the SAME pinned
        # terrain. It is not a w_E setting and gets its own series.
        mp = re.search(r"climb_pid_h(\d+)\.csv$", base)
        m = re.search(r"climb_wE([0-9p]+)_h(\d+)\.csv$", base)
        if not m and not mp:
            print("skipping unparsable name: %s" % path)
            continue
        w = None if mp else float(m.group(1).replace("p", "."))
        h = int((mp or m).group(1 if mp else 2))
        recs = list(csv.DictReader(open(path)))
        if not recs:
            continue
        cl = np.array([float(r["cleared"]) for r in recs])
        en = np.array([float(r["energy_J"]) for r in recs])
        ok = cl > 0.5
        row = (h, cl.mean(), en[ok].mean() if ok.any() else np.nan, len(recs))
        (pid if w is None else data.setdefault(w, [])).append(row)
    if not data and not pid:
        sys.exit("nothing matched %r (expected climb_wE2_h5.csv or climb_pid_h5.csv)"
                 % a.pattern)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.88, len(data)))
    for (w, pts), c in zip(sorted(data.items()), cmap):
        pts.sort()
        h = [q[0] for q in pts]
        ax.plot(h, [q[1] for q in pts], "o-", ms=4, lw=1.6, color=c,
                label="$w_E=%g$" % w)
        e = [q[2] for q in pts]
        good = ~np.isnan(e)
        if good.any():
            bx.plot(np.array(h)[good], np.array(e)[good], "o-", ms=4, lw=1.6,
                    color=c, label="$w_E=%g$" % w)

    if pid:
        # Black dashed, deliberately distinct: this is the baseline the learned
        # curves are being argued against, not another point in the sweep.
        pid.sort()
        ph = [q[0] for q in pid]
        ax.plot(ph, [q[1] for q in pid], "s--", ms=4, lw=1.6, color="k",
                label="PID baseline")
        pe = np.array([q[2] for q in pid])
        g = ~np.isnan(pe)
        if g.any():
            bx.plot(np.array(ph)[g], pe[g], "s--", ms=4, lw=1.6, color="k",
                    label="PID baseline")

    ax.axvline(100 * a.r, color="crimson", ls="--", lw=1.2)
    ax.text(100 * a.r, 0.5, " $h_{\\max}=r$ ", color="crimson", fontsize=7.5,
            rotation=90, va="center", ha="right")
    ax.set_xlabel("riser height (cm)")
    ax.set_ylabel("fraction of risers cleared")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("(a) capability", fontsize=9, loc="left")

    bx.set_xlabel("riser height (cm)")
    bx.set_ylabel("energy per cleared episode (J)")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=7.5, frameon=False)
    bx.set_title("(b) cost, cleared episodes only", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print("wrote %s\nwrote %s" % (a.out, png))

    print("\n%-7s %-7s %-10s %-12s" % ("w_E", "h_cm", "cleared", "energy_J"))
    for w, pts in sorted(data.items()):
        for h, cl, en, n in sorted(pts):
            print("%-7g %-7d %-10.3f %-12s" % (w, h, cl, "n/a" if np.isnan(en) else "%.0f" % en))
    for h, cl, en, n in sorted(pid):
        print("%-7s %-7d %-10.3f %-12s" % ("PID", h, cl, "n/a" if np.isnan(en) else "%.0f" % en))


if __name__ == "__main__":
    main()
