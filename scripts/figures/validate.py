import csv, glob, os
import numpy as np

C = [4.0792540792478203e-13, -2.2921522921483562e-09,
     2.2550699300690226e-05, -0.026882905982896884, 8.516433566430207]
def pwm2thrust(p):
    p = np.clip(np.asarray(p, float), 1000.0, 1650.0)
    t = np.zeros_like(p)
    for c in C:
        t = t * p + c
    return np.maximum(t, 0.0)

print("=== thrust polynomial anchors (per propeller) ===")
for p in (1000, 1100, 1208, 1300, 1486, 1650):
    print("  pwm %4d -> %6.2f N   (total %6.2f N, T/W %.3f)" % (
        p, pwm2thrust(p), 2*pwm2thrust(p), 2*pwm2thrust(p)/31.57))

print("\n=== logged u_thr ranges, primary ten ===")
for p in sorted(glob.glob("hw_final/trials_final/trial_*.csv")):
    r = list(csv.DictReader(open(p)))
    u1 = np.array([float(x["u_thr1"]) for x in r])
    u2 = np.array([float(x["u_thr2"]) for x in r])
    live = np.array([x["gated"] == "0" for x in r])
    print("  %-20s u1 [%+.3f, %+.3f]  u2 [%+.3f, %+.3f]  live u1 [%+.3f, %+.3f]" % (
        os.path.basename(p), u1.min(), u1.max(), u2.min(), u2.max(),
        u1[live].min(), u1[live].max()))

print("\n=== window: policy takeover -> peak height (gated==0) ===")
print("NOTES.md says 175627 -> gain 0.141, 258 W, travel 2.36 | 185847 -> 0.137, 262 W, 1.75 | 190232 -> 0.064, 251 W, 1.08")
for p in sorted(glob.glob("hw_final/trials_final/trial_*.csv")):
    r = list(csv.DictReader(open(p)))
    f = lambda k: np.array([float(x[k]) for x in r])
    live = np.flatnonzero(np.array([x["gated"] == "0" for x in r]))
    z, x, w, e = f("pos_z"), f("pos_x"), f("watts"), f("energy_J")
    zl = z[live]
    kpk = live[int(np.argmax(zl))]
    seg = live[live <= kpk]
    z0 = zl[:20].mean()
    gain = z[kpk] - z0
    dur = seg.size * 0.02
    trav = x[seg].max() - x[seg].min()
    ws = w[seg]; pw = ws[ws > 0].mean() if (ws > 0).any() else float("nan")
    ej = e[seg].max() - e[seg].min()
    u = (f("u_thr1") + 1) * 325.0 + 1000.0
    v = (f("u_thr2") + 1) * 325.0 + 1000.0
    thr = pwm2thrust(u[seg]) + pwm2thrust(v[seg])
    print("  %-20s dur %5.2f s  gain %.3f m  travel %.2f m  P %3.0f W  E %5.0f J  thrust %.1f-%.1f N (mean %.1f)" % (
        os.path.basename(p), dur, gain, trav, pw, ej, thr.min(), thr.max(), thr.mean()))
