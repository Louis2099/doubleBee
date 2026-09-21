#!/usr/bin/env python3
"""Per-cell clearance for Fig. 4 top, restricted to checkpoints k=0..KMAX.

Conventions are taken from summ_arms.py and must not drift from it:
  clearance = peak gain >= step height (NOT the CSV `cleared` column)
  interval  = sd/sqrt(n) across checkpoints, which is what the paper prints
              (summ_arms.py prints the sd itself, e.g. hE4 6 cm 7.9 -> 2.5)

Restricting to a fixed k range is what makes the panel uniform. Every arm then
rests on the same number of checkpoints and the same paired seeds, since
seed = 1000*H + k.

    python3 summ_k5.py abl_seeded 4 > fig4_top.csv
"""
import csv, glob, os, re, statistics as st, sys

d = sys.argv[1]
KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TAGS = ("hE4", "ct10", "ct050", "ctm05", "ctm45")

print("tag,h,n,mean,sd,se")
for tag in TAGS:
    for h in ("03", "04", "05", "06", "07"):
        per = []
        for f in sorted(glob.glob(os.path.join(d, "climb_%s_h%s_k*.csv" % (tag, h)))):
            m = re.search(r"_k(\d+)_", f)
            if not m or int(m.group(1)) > KMAX:
                continue
            rows = list(csv.DictReader(open(f)))
            if not rows:
                continue
            hm = int(h) / 100.0
            per.append(100.0 * sum(float(r["max_gain_m"]) >= hm for r in rows) / len(rows))
        if not per:
            continue
        sd = st.stdev(per) if len(per) > 1 else 0.0
        print("%s,%s,%d,%.2f,%.2f,%.2f"
              % (tag, h, len(per), st.mean(per), sd, sd / len(per) ** 0.5))
