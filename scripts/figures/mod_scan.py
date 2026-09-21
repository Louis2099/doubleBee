"""Scan every episode for GRADED thrust, not saturated thrust.

sim_fig.py picked its episode by max climb/drive thrust ratio, which selects
for exactly the bang-bang profile that looks like a switch. Here we rank the
opposite way and ask whether any episode modulates thrust continuously.

Reported per episode:
  sat   fraction of climb samples pinned at the ceiling (>= SAT of 15.8 N)
  sd    thrust spread during the climb, in N. A switch is flat, so sd is small
        while pinned. Graded control needs sd well above the drive-phase noise
  r     mean climb thrust / mean drive thrust
  rho   correlation of thrust with pitch during the climb. A switch keys off a
        binary detection, so it should not track posture. Genuine modulation
        should show some coupling
"""
import csv, os, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "sim/policy_io_hE4.csv"
DT, SAT = 0.02, 15.8

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
y, z, T = f("pos_y"), f("pos_z"), f("total_thrust")
qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
lean = np.degrees(np.arctan2(2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)))

b = np.flatnonzero((np.abs(np.diff(y)) > 0.8) | (np.abs(np.diff(z)) > 0.05)) + 1
eps = [e for e in np.split(np.arange(len(r)), b) if e.size >= 60]
print("episodes:", len(eps), " ceiling taken as %.1f N" % SAT)


def sm(a, n=9):
    p = n // 2
    return np.convolve(np.pad(a, p, mode="edge"), np.ones(n) / n,
                       mode="valid")[:len(a)]


rows = []
for i, e in enumerate(eps):
    h = z[e] - np.median(z[e][10:25])
    dz = sm(np.gradient(h, DT))
    rise = dz > 0.04
    d = np.diff(rise.astype(int))
    st, en = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if rise[0]:
        st = np.r_[0, st]
    if rise[-1]:
        en = np.r_[en, e.size - 1]
    segs = [(a, c) for a, c in zip(st, en) if c - a >= 4 and h[c] - h[a] > 0.02]
    if not segs or tilt[e][-1] > 30 or h.max() < 0.04:
        continue
    m = np.zeros(e.size, bool)
    for a, c in segs:
        m[a:c] = True
    Tc, Td = T[e][m], T[e][~m]
    if Tc.size < 8 or Td.size < 8:
        continue
    sat = float(np.mean(Tc >= SAT))
    sd = float(np.std(Tc))
    ratio = float(Tc.mean() / max(Td.mean(), 1e-6))
    pc = lean[e][m]
    rho = float(np.corrcoef(Tc, pc)[0, 1]) if pc.std() > 1e-6 and Tc.std() > 1e-6 else 0.0
    rows.append((i, len(segs), h.max(), sat, sd, ratio, rho, Tc.mean(), Td.mean()))

print("\n%4s %4s %7s %6s %6s %6s %7s %7s %7s"
      % ("ep", "asc", "gain", "sat", "sd", "r", "rho", "Tclimb", "Tdrive"))
# Rank by graded-ness: low saturation, high spread.
for t in sorted(rows, key=lambda v: (v[3], -v[4]))[:25]:
    print("%4d %4d %7.3f %6.2f %6.2f %6.2f %7.2f %7.1f %7.1f"
          % (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]))

if rows:
    s = np.array([v[3] for v in rows])
    print("\n%d usable climbs. saturation: min %.2f median %.2f max %.2f"
          % (len(rows), s.min(), np.median(s), s.max()))
    print("episodes with <20%% of climb pinned at ceiling: %d" % int((s < 0.2).sum()))
    print("episodes with <50%% pinned: %d" % int((s < 0.5).sum()))
