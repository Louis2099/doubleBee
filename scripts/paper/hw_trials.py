"""Aggregate a directory of hardware runs into the numbers VI-B needs.

    python3 hw_trials.py ~/doublebee_PID_JAI/hw_logs
    python3 hw_trials.py ~/doublebee_PID_JAI/hw_logs --since 0907_19

ONE ROW PER RUN, then the summary: clearance rate with a Wilson interval,
energy per trial, and the failure modes counted.

WHY WILSON. The contributions promise "N=15 trials with interval estimates
rather than a success fraction". At N=15 the normal approximation is not valid
-- 3/15 gives a negative lower bound -- so the interval has to be Wilson.

WHY PER-RISER. "Success" on this arena is not binary: clearing the first 6 cm
riser and clearing the second are different achievements, and reporting them
together hides which one the robot actually does. Both are counted.

ENERGY comes from the energy_J column, which integrates battery V*I over LIVE
ticks only. Runs logged before 2026-09-07 have no electrical columns at all and
are reported as n/a rather than silently averaged in.
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

E1, E2 = -0.2688, 0.4312          # riser x positions, mocap frame
TREAD1_Z, TREAD2_Z = 0.09, 0.15   # base height that means "up on that tread"


def wilson(k, n, z=1.96):
    """Wilson score interval. Valid at small n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyse(path):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 40:
        return None
    col = rows[0].keys()
    f = lambda k: np.array([float(r[k]) for r in rows])
    live = np.array([r["gate_reason"] == "" for r in rows])
    idx = np.flatnonzero(live)
    if idx.size < 40:
        return None
    segs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    s = max(segs, key=len)
    if len(s) < 40:
        return None

    x, z = f("pos_x"), f("pos_z")
    qx, qy, qz, qw = (f(c) for c in ("qx", "qy", "qz", "qw"))
    bzx = 2 * (qx * qz + qw * qy)
    bzy = 2 * (qy * qz - qw * qx)
    bzz = 1 - 2 * (qx * qx + qy * qy)
    yaw = np.radians(f("yaw_deg"))
    lean = np.degrees(np.arctan2(bzx * -np.sin(yaw) + bzy * np.cos(yaw), bzz))

    t1 = ((x[s] > E1) & (x[s] < E2) & (z[s] > TREAD1_Z)).sum() * 0.02
    t2 = ((x[s] > E2) & (z[s] > TREAD2_Z)).sum() * 0.02

    # energy over the live segment, if the run logged it
    e = float("nan")
    if "energy_J" in col:
        v = f("energy_J")[s]
        v = v[np.isfinite(v)]
        if v.size > 2 and v.max() > 0:
            e = v.max() - v.min()

    # why it ended -- the failure mode, named
    end = rows[s[-1] + 1]["gate_reason"] if s[-1] + 1 < len(rows) else "end of log"
    if t2 > 0.2:
        mode = "cleared both"
    elif t1 > 0.2:
        mode = "cleared riser 1"
    elif abs(lean[s]).max() > 60:
        mode = "fell (lean > 60)"
    elif (yaw[s].max() - yaw[s].min()) > np.radians(60):
        mode = "lost heading"
    elif x[s].max() < E1 - 0.10:
        mode = "never reached riser"
    else:
        mode = "bounced at riser"

    return dict(name=os.path.basename(path), dur=len(s) * 0.02,
                xmax=x[s].max(), zmax=z[s].max(), t1=t1, t2=t2,
                lean=abs(lean[s]).max(),
                yaw=np.degrees(yaw[s].max() - yaw[s].min()),
                energy=e, mode=mode, end=end[:26])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", nargs="?",
                   default=os.path.expanduser("~/doublebee_PID_JAI/hw_logs"))
    p.add_argument("--since", default=None,
                   help="only files whose name contains this, e.g. 0907_19")
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.logdir, "*.csv")), key=os.path.getmtime)
    if a.since:
        paths = [q for q in paths if a.since in os.path.basename(q)]
    if not paths:
        sys.exit("no logs in %s" % a.logdir)

    runs = [r for r in (analyse(q) for q in paths) if r]
    if not runs:
        sys.exit("no runs with a usable live segment (>0.8 s)")

    print("%-26s %5s %6s %6s %6s %6s %5s %5s %8s  %s" %
          ("log", "live", "x_max", "z_max", "tread1", "tread2",
           "lean", "yaw", "E(J)", "outcome"))
    print("-" * 108)
    for r in runs:
        print("%-26s %5.1f %+6.2f %6.3f %6.2f %6.2f %5.0f %5.0f %8s  %s" %
              (r["name"], r["dur"], r["xmax"], r["zmax"], r["t1"], r["t2"],
               r["lean"], r["yaw"],
               "n/a" if r["energy"] != r["energy"] else "%.0f" % r["energy"],
               r["mode"]))

    n = len(runs)
    k1 = sum(1 for r in runs if r["t1"] > 0.2 or r["t2"] > 0.2)
    k2 = sum(1 for r in runs if r["t2"] > 0.2)
    lo1, hi1 = wilson(k1, n)
    lo2, hi2 = wilson(k2, n)
    print("\nN = %d trials" % n)
    print("  cleared riser 1 : %2d/%d = %.0f%%   95%% CI [%.0f, %.0f]%%"
          % (k1, n, 100 * k1 / n, 100 * lo1, 100 * hi1))
    print("  cleared riser 2 : %2d/%d = %.0f%%   95%% CI [%.0f, %.0f]%%"
          % (k2, n, 100 * k2 / n, 100 * lo2, 100 * hi2))

    es = [r["energy"] for r in runs if r["energy"] == r["energy"]]
    if es:
        print("  energy/trial    : mean %.0f J, median %.0f J  (n=%d logged)"
              % (np.mean(es), np.median(es), len(es)))
    else:
        print("  energy/trial    : NOT LOGGED in any run -- VI-B1's integral of "
              "V*I cannot be computed from these")

    print("\n  outcomes:")
    from collections import Counter
    for mode, c in Counter(r["mode"] for r in runs).most_common():
        print("    %-22s %2d  (%.0f%%)" % (mode, c, 100 * c / n))
    print("\n  max height reached: %.3f m   longest on tread 1: %.2f s"
          % (max(r["zmax"] for r in runs), max(r["t1"] for r in runs)))


if __name__ == "__main__":
    main()
