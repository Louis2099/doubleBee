#!/usr/bin/env python3
"""Option B summary: matched, seeded evaluation.

    python3 summ_optionB.py [abl_seeded]

Per height: clearance (paper metric, peak gain >= step height) mean +- SE over
the 10 seeded evaluations, mean power, and the PAIRED difference learned minus
each baseline (seed k matched), with its SE across k. Then, per height, the
steps-climbed distribution and energy per episode within each bin (equal work).
"""
import csv, glob, math, os, re, statistics as st, sys

DT = 0.02
D = sys.argv[1] if len(sys.argv) > 1 else "abl_seeded"
ARMS = [("hE4", "learned"), ("swA3", "switch idle 0.31"), ("swB3", "switch idle 0.46"), ("ct10", "fixed T/W 0.55")]


def load(tag, h):
    out = {}
    for f in glob.glob(os.path.join(D, "climb_%s_h%s_k*_*.csv" % (tag, h))):
        k = int(re.search(r"_k(\d+)_", f).group(1))
        rows = list(csv.DictReader(open(f)))
        if rows:
            out[k] = rows
    return out


def se(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


for h in ("03", "04", "05", "06", "07"):
    step = int(h) / 100.0
    data = {tag: load(tag, h) for tag, _ in ARMS}
    if not any(data.values()):
        continue
    print("\n=== %d cm" % int(h))
    clr = {}
    for tag, name in ARMS:
        d = data[tag]
        if not d:
            continue
        per = {k: 100.0 * sum(float(r["max_gain_m"]) >= step for r in rows) / len(rows) for k, rows in d.items()}
        clr[tag] = per
        E = sum(float(r["energy_J"]) for rows in d.values() for r in rows)
        S = sum(float(r["steps"]) for rows in d.values() for r in rows)
        v = list(per.values())
        print("  %-18s evals %2d  clears %5.1f +- %4.1f %%   power %4.0f W" % (name, len(v), st.mean(v), se(v), E / S / DT))
    if "hE4" in clr:
        for tag, name in ARMS[1:]:
            if tag not in clr:
                continue
            ks = sorted(set(clr["hE4"]) & set(clr[tag]))
            if len(ks) < 2:
                continue
            diff = [clr["hE4"][k] - clr[tag][k] for k in ks]
            print("  paired learned - %-16s %+5.1f +- %4.1f pts over %d matched seeds (%.1f SE)" % (
                name, st.mean(diff), se(diff), len(ks), st.mean(diff) / se(diff) if se(diff) > 0 else float("nan")))
    print("  steps climbed share | energy per episode (J) at equal work")
    for tag, name in ARMS:
        d = data[tag]
        if not d:
            continue
        bins = {0: [], 1: [], 2: [], 3: []}
        for rows in d.values():
            for r in rows:
                b = min(3, int(math.floor(float(r["max_gain_m"]) / step + 1e-9)))
                bins[b].append(float(r["energy_J"]))
        n = sum(len(x) for x in bins.values())
        share = " ".join("%d:%4.1f%%" % (b, 100.0 * len(bins[b]) / n) for b in range(4))
        eng = " ".join("E%d=%s" % (b, ("%4.0f" % st.mean(bins[b])) if len(bins[b]) >= 20 else "  - ") for b in (1, 2, 3))
        print("    %-18s %s | %s" % (name, share, eng))
