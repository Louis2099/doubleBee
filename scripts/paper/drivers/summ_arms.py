#!/usr/bin/env python3
"""Summarise eval_climb CSVs per arm and step height.

    python3 summ_arms.py abl_h swA2 swB2 hE4 ct050 ...

Clearance: mean +- sd across checkpoints (the paper's interval). Power: total
energy / total control steps / 0.02 s, which reproduces hE4's published 292 W
at 6 cm exactly. Gain p50/max over all pooled episodes.
"""
import csv, glob, os, statistics as st, sys

DT = 0.02
d = sys.argv[1]
tags = sys.argv[2:]
print("%-8s %-4s %-5s %-15s %-8s %-12s %s" %
      ("arm", "h", "ckpts", "clears %", "power W", "gain p50/max", "episodes"))
for tag in tags:
    for h in ("03", "04", "05", "06", "07"):
        files = sorted(glob.glob(os.path.join(d, "climb_%s_h%s_*.csv" % (tag, h))))
        if not files:
            continue
        per, E, S, G, n = [], 0.0, 0.0, [], 0
        for f in files:
            rows = list(csv.DictReader(open(f)))
            if not rows:
                continue
            # The paper's clearance is PEAK GAIN >= STEP HEIGHT, not the CSV's
            # `cleared` column (which also demands a hold and displacement).
            # Verified 2026-09-13: this reproduces hE4's published 67.9 / 56.9 /
            # 53.9 / 43.4 / 33.2 % at 3-7 cm exactly; `cleared` gives 7.7 % at 6.
            hm = int(h) / 100.0
            per.append(100.0 * sum(float(r["max_gain_m"]) >= hm for r in rows) / len(rows))
            E += sum(float(r["energy_J"]) for r in rows)
            S += sum(float(r["steps"]) for r in rows)
            G += [float(r["max_gain_m"]) for r in rows]
            n += len(rows)
        if not per:
            continue
        sd = st.stdev(per) if len(per) > 1 else 0.0
        G.sort()
        print("%-8s %-4s %-5d %5.1f +- %-6.1f %-8.0f %.3f/%.3f  %d" %
              (tag, h, len(per), st.mean(per), sd, E / S / DT if S else float("nan"),
               G[len(G) // 2], G[-1], n))
    print()
