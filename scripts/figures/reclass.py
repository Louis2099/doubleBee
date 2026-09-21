import csv, glob, os
import numpy as np
E1 = -0.2688
T1Z, T2Z = 0.09, 0.15
print("%-28s %6s %7s %7s %7s %7s  %-14s %-14s" % (
    "file","live","dur_s","dist_m","E_J","P_W","both@0.1312","both@0.4312"))
out = []
for p in sorted(glob.glob("hw_final/*.csv"), key=os.path.getmtime):
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        continue
    if len(rows) < 40: continue
    col = rows[0].keys()
    if "watts" not in col or "pos_x" not in col: continue
    f = lambda k: np.array([float(r[k]) for r in rows])
    live = np.array([r["gate_reason"] == "" for r in rows])
    s = np.flatnonzero(live)
    if s.size < 20: continue
    x, z, w = f("pos_x"), f("pos_z"), f("watts")
    e = f("energy_J")
    dur = s.size * 0.02
    dist = x[s].max() - x[s].min()
    ej = e[s].max() - e[s].min()
    pw = w[s][w[s] > 0].mean() if (w[s] > 0).any() else float("nan")
    b1 = ((x[s] > 0.1312) & (z[s] > T2Z)).sum() * 0.02
    b2 = ((x[s] > 0.4312) & (z[s] > T2Z)).sum() * 0.02
    out.append((os.path.basename(p), s.size, dur, dist, ej, pw, b1, b2))
    print("%-28s %6d %7.2f %7.3f %7.0f %7.0f  %-14s %-14s" % (
        out[-1][0], s.size, dur, dist, ej, pw,
        "YES %.2fs" % b1 if b1 > 0.2 else "-",
        "YES %.2fs" % b2 if b2 > 0.2 else "-"))

print("\n=== looking for the paper's set: 2.95+-0.64 s, 275+-21 W, 823+-229 J, dist 1.26-1.40 m")
cand = [r for r in out if 1.20 <= r[3] <= 1.45 and 1.5 <= r[2] <= 5.0 and 400 <= r[4] <= 1400]
for r in cand:
    print("  %-28s dur %.2f dist %.3f E %.0f P %.0f" % (r[0], r[2], r[3], r[4], r[5]))
if cand:
    d = np.array([r[2] for r in cand]); e = np.array([r[4] for r in cand]); p = np.array([r[5] for r in cand])
    print("  n=%d  dur %.2f+-%.2f  E %.0f+-%.0f  P %.0f+-%.0f" % (
        len(cand), d.mean(), d.std(ddof=1) if len(d)>1 else 0,
        e.mean(), e.std(ddof=1) if len(e)>1 else 0,
        p.mean(), p.std(ddof=1) if len(p)>1 else 0))
