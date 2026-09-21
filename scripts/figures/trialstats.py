import csv, glob, os
import numpy as np
E1, E2 = -0.2688, 0.4312
T1Z, T2Z = 0.09, 0.15
print("%-22s %6s %6s %7s %8s %8s %8s %7s  %s" % (
    "file","rows","live","dur_s","x_min","x_max","z_max","E_J","mode"))
for p in sorted(glob.glob("hw_final/trial_*.csv")):
    rows = list(csv.DictReader(open(p)))
    if len(rows) < 40:
        continue
    col = rows[0].keys()
    f = lambda k: np.array([float(r[k]) for r in rows])
    live = np.array([r["gate_reason"] == "" for r in rows])
    s = np.flatnonzero(live)
    if s.size < 20:
        print("%-22s %6d  (no live segment)" % (os.path.basename(p), len(rows)))
        continue
    x, z = f("pos_x"), f("pos_z")
    e = f("energy_J")[s] if "energy_J" in col else None
    ej = (e.max() - e.min()) if e is not None else float("nan")
    t1 = ((x[s] > E1) & (x[s] < E2) & (z[s] > T1Z)).sum() * 0.02
    t2 = ((x[s] > E2) & (z[s] > T2Z)).sum() * 0.02
    mode = "cleared both" if t2 > 0.3 else ("cleared step 1" if t1 > 0.3 else "no climb")
    print("%-22s %6d %6d %7.1f %8.3f %8.3f %8.3f %7.0f  %s" % (
        os.path.basename(p), len(rows), s.size, s.size*0.02,
        x[s].min(), x[s].max(), z[s].max(), ej, mode))
