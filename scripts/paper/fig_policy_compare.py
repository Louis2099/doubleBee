"""Overlay two policies on the same staircase, aligned on the climb.

Replaces stacking two separate strip figures. Both traces share every axis, so
the reader compares directly instead of eyeballing across panels, and it costs
half the page.

Traces are aligned on CLIMB ONSET rather than on step index: two rollouts reach
the riser at different times, so raw step numbers would compare the approach of
one against the climb of the other. Onset is the first sample where the robot
has risen --rise metres above its pre-climb baseline (median height over the
preceding 0.5 s), which needs no step-position bookkeeping and works across
different arena layouts.

Panels: forward speed, height gained, pitch, total thrust, cumulative energy.
The last one is the point: it makes the energy difference visible as a growing
gap rather than a number in a table.

    python3 fig_policy_compare.py wE0.csv wE10.csv --labels "$w_E=0$,$w_E=10$" \
        -o fig_policy_compare.pdf
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.02


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit("empty: %s" % path)
    cols = rows[0].keys()
    f = lambda k: np.array([float(r[k]) for r in rows])
    if "pos_z" not in cols:
        sys.exit("%s has no pos_z; rerun play.py with --log_policy_io" % path)
    d = {"z": f("pos_z")}
    if "pos_x" in cols and "pos_y" in cols:
        x, y = f("pos_x"), f("pos_y")
        d["v"] = np.r_[0.0, np.hypot(np.diff(x), np.diff(y)) / DT] * np.sign(np.r_[0.0, np.diff(x)])
    else:
        d["v"] = np.zeros(len(rows))
    if "total_thrust" in cols:
        d["thrust"] = f("total_thrust")
    elif "u_thr1" in cols and "u_thr2" in cols:
        d["thrust"] = (f("u_thr1") + f("u_thr2")) / 2.0
    else:
        d["thrust"] = np.zeros(len(rows))
    # pitch from projected gravity if present, else from quaternion
    if "obs_13" in cols and "obs_14" in cols:
        gy, gz = f("obs_13") / 1.15, f("obs_14") / 1.15
        d["pitch"] = np.degrees(np.arctan2(gy, -gz))
    elif all(c in cols for c in ("qw", "qx", "qy", "qz")):
        qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
        d["pitch"] = np.degrees(np.arctan2(2 * (qy * qz + qw * qx),
                                           1 - 2 * (qx * qx + qy * qy)))
    else:
        d["pitch"] = np.zeros(len(rows))
    return d


def onset(z, rise, pre):
    base_n = int(0.5 / DT)
    for i in range(base_n, len(z)):
        if z[i] - np.median(z[max(0, i - base_n):i]) >= rise:
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs=2)
    p.add_argument("--labels", default="A,B")
    p.add_argument("-o", "--out", default="fig_policy_compare.pdf")
    p.add_argument("--rise", type=float, default=0.02, help="m of gain marking onset")
    p.add_argument("--pre", type=float, default=1.0, help="s before onset")
    p.add_argument("--post", type=float, default=1.5, help="s after onset")
    p.add_argument("--stair-span", type=float, default=0.5,
                   help="s to shade as the stair interaction, from onset")
    a = p.parse_args()

    labels = a.labels.split(",")
    npre, npost = int(a.pre / DT), int(a.post / DT)
    t = np.arange(-npre, npost) * DT
    colors = ["#2166ac", "#b2182b"]

    panels = [("v", "forward speed (m/s)"), ("dz", "height gained (m)"),
              ("pitch", "pitch (deg)"), ("thrust", "total thrust (N)"),
              ("cume", "cumulative energy (arb.)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.6))

    for path, lab, col in zip(a.logs, labels, colors):
        d = load(path)
        i = onset(d["z"], a.rise, a.pre)
        if i is None or i - npre < 0 or i + npost > len(d["z"]):
            sys.exit("no usable climb in %s (lower --rise, or the rollout is "
                     "too short around the onset)" % path)
        sl = slice(i - npre, i + npost)
        d["dz"] = d["z"] - np.median(d["z"][i - npre:i - npre + int(0.5 / DT)])
        # thrust integrated as a stand-in for energy; the absolute scale is not
        # meaningful, the GAP between the two curves is.
        d["cume"] = np.cumsum(np.clip(d["thrust"], 0, None)) * DT
        d["cume"] -= d["cume"][sl][0]
        for ax, (key, ylab) in zip(axes, panels):
            ax.plot(t, d[key][sl], lw=1.6, color=col, label=lab)

    for ax, (key, ylab) in zip(axes, panels):
        ax.axvspan(0.0, a.stair_span, color="0.85", zorder=0)
        ax.axvline(0.0, color="k", ls="--", lw=0.9)
        ax.set_xlabel("time from climb onset (s)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].text(a.stair_span / 2, axes[0].get_ylim()[1] * 0.92, "stair",
                 ha="center", fontsize=8, color="0.35")

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=220, bbox_inches="tight")
    print("wrote %s (+ .png)" % a.out)
    print("\nNOTE: cumulative energy here is integrated THRUST, not the power\n"
          "model. It shows the gap qualitatively; quote the J/m figures from\n"
          "eval_climb.py in the caption, not numbers read off this axis.")


if __name__ == "__main__":
    main()
