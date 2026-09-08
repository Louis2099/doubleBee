"""Did the tilt episodes CLIMB, or did they fail to?

    python3 end_state.py "sweep_*/climb_6cm_*.csv"

WHY. `goal_reached` fires only on close & upright & settled & at_height. A robot
that arrives at the target and then topples fails the upright and settled tests,
so goal_reached does not fire and `tilt` does -- which makes "arrived and fell
over at the goal" indistinguishable from "never climbed" in the end column.

Those are opposite outcomes for the paper. This groups every episode by its end
reason and reports how far it climbed and how far it travelled. If the tilt rows
match the goal_reached rows on gain and displacement, the robot finished the
climb and lost stability at the end, and the tilt rate is a terminal-stability
number, not a climbing-failure number. If tilt rows are short and low, it is a
real failure to climb.

Read `disp` against the ~2.0 m target range: a tilt episode at 1.8 m got there.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np


def rows_of(path):
    return list(csv.DictReader(open(path)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", nargs="?", default="climb_*.csv")
    p.add_argument("--step", type=float, default=0.06)
    a = p.parse_args()

    paths = sorted(glob.glob(a.pattern))
    if not paths:
        sys.exit("no CSVs matched %r" % a.pattern)

    for path in paths:
        recs = rows_of(path)
        if not recs or "end" not in recs[0]:
            print("%s: no `end` column (written before 2026-09-08)" % path)
            continue
        tag = re.sub(r"^climb_|\.csv$", "", os.path.basename(path))
        print("\n=== %s  (n=%d) ===" % (tag, len(recs)))
        print("%-20s %5s %8s %9s %9s %8s %8s %8s"
              % ("end reason", "n", "%", "gain_med", "disp_med", "hold_md",
                 "steps_md", ">=%dcm" % (100 * a.step)))
        by = {}
        for r in recs:
            by.setdefault(r["end"], []).append(r)
        for k in sorted(by, key=lambda z: -len(by[z])):
            v = by[k]
            g = np.array([float(x["max_gain_m"]) for x in v])
            d = np.array([float(x.get("max_disp_m", "nan")) for x in v])
            h = np.array([float(x.get("hold_s", "nan")) for x in v])
            s = np.array([float(x["steps"]) for x in v])
            print("%-20s %5d %7.0f%% %9.3f %9.2f %8.2f %8.0f %7.0f%%"
                  % (k, len(v), 100.0 * len(v) / len(recs),
                     np.median(g), np.nanmedian(d), np.nanmedian(h),
                     np.median(s), 100.0 * (g >= a.step).mean()))

        t = by.get("tilt", [])
        gr = by.get("goal_reached", [])
        if t and gr:
            tg = np.median([float(x["max_gain_m"]) for x in t])
            gg = np.median([float(x["max_gain_m"]) for x in gr])
            td = np.nanmedian([float(x.get("max_disp_m", "nan")) for x in t])
            gd = np.nanmedian([float(x.get("max_disp_m", "nan")) for x in gr])
            verdict = ("CLIMBED THEN FELL" if tg >= 0.9 * gg and td >= 0.9 * gd
                       else "tilt episodes really did climb/travel less")
            print("  tilt vs goal_reached: gain %.3f vs %.3f, disp %.2f vs %.2f"
                  "  ->  %s" % (tg, gg, td, gd, verdict))


if __name__ == "__main__":
    main()
