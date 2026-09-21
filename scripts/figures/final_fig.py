"""Hardware climb profile, trial 190715, paper style.

Panel (d) is thrust DELIVERED, recovered from measured battery power (total
watts minus wheel watts) inverted through the calibrated PWM-to-power fit.
It is not the command. The command peaked at 36.6 N and the propellers do not
reach that within the duration of the boost. Measured power and thrust on this
airframe follow P ~ T^1.49, so a 30% power surge at a step buys only ~19% more
thrust, which is why delivered thrust moves far less than the command does.

No body-weight line in (d): it sits at 31.6 N while the trace spans 12-20 N,
so drawing it stretches the axis and flattens the modulation. Ratio goes in
the caption instead.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "hw_final/trials_final/trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_hw_climb_profile")
MDP = "doubleBee_isaac/lab/doublebee/tasks/manager_based/locomotion/velocity/mdp"
F1, F2, DT, WEIGHT_N = -0.55, -0.15, 0.02, 31.57
LEARNED = "#c0392b"
TMAX = 5.0                                   # settle holds flat past this
TH = json.load(open(os.path.join(MDP, "pwm2thrust_params.json")))["coeffs"]
PW = json.load(open(os.path.join(MDP, "pwm2power_params.json")))["coeffs"]
GRID = np.arange(1000.0, 1650.1, 0.5)
PPAIR, TPAIR = 2 * np.polyval(PW, GRID), 2 * np.polyval(TH, GRID)


def smooth(a, n=9):
    pad = n // 2
    return np.convolve(np.pad(a, pad, mode="edge"), np.ones(n) / n,
                       mode="valid")[:len(a)]


r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
seg = np.flatnonzero(np.array([x["gated"] == "0" for x in r]))
x, z = f("pos_x")[seg], f("pos_z")[seg]
qx, qy, qz, qw = f("qx")[seg], f("qy")[seg], f("qz")[seg], f("qw")[seg]
pitch = smooth(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))), 5)
vx, vz = smooth(np.gradient(x, DT)), smooth(np.gradient(z, DT))
pr = np.radians(pitch)
vxb = vx * np.cos(pr) + vz * np.sin(pr)
vzb = -vx * np.sin(pr) + vz * np.cos(pr)

bv, wt = f("batt_v")[seg], f("watts")[seg]
wheelW = np.maximum((f("m1_a")[seg] + f("m2_a")[seg]) * bv, 0.0)
propW = smooth(np.maximum(wt - wheelW, 0.0))
T_del = np.interp(np.interp(propW, PPAIR, GRID), GRID, TPAIR)

t = np.arange(seg.size) * DT
z0 = z[:20].mean()
first = lambda m, d=0: (np.flatnonzero(m)[0] if np.any(m) else d)


def held(sig, thr, h=15):
    ok = sig > thr
    for i in range(len(sig) - h):
        if ok[i:i + h].all():
            return i
    return first(ok)


k1a, k1b = first(x > F1 - 0.12), held(z, z0 + 0.045)
k2a, k2b = k1b + first(x[k1b:] > F2 - 0.12), held(z, z0 + 0.105)
B1, B2 = (t[k1a], t[k1b]), (t[k2a], t[k2b])
m = t <= TMAX
print("step1 %.2f-%.2f  step2 %.2f-%.2f" % (*B1, *B2))
print("thrust: peak %.1f N (T/W %.2f), settle %.1f N, swing %.2fx"
      % (T_del.max(), T_del.max() / WEIGHT_N, T_del[k2b:].mean(),
         T_del.max() / T_del[k2b:].mean()))
print("power : peak %.0f W, settle %.0f W, swing %.2fx   <- cube law, bigger swing"
      % (propW.max(), propW[k2b:].mean(), propW.max() / propW[k2b:].mean()))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))
panels = [(vxb, "(a) forward speed", r"$v_x$ (m s$^{-1}$)", ramp[0]),
          (vzb, "(b) vertical speed", r"$v_z$ (m s$^{-1}$)", ramp[1]),
          (pitch, "(c) pitch angle", r"$\theta$ (deg)", ramp[2]),
          (propW, "(d) propeller power", r"$P$ (W)", LEARNED)]

fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.85))
for a, (y, title, ylab, col) in zip(ax, panels):
    a.plot(t, y, color=col, lw=1.0, zorder=3)
    for b in (B1, B2):
        a.axvspan(*b, color="0.55", alpha=0.22, lw=0, zorder=0)
    a.set_title(title, fontsize=8, loc="left", pad=3)
    a.set_ylabel(ylab, labelpad=2)
    a.set_xlabel("time (s)", labelpad=2)
    a.set_xlim(0, TMAX)
    a.grid(alpha=0.25, lw=0.5)
    a.tick_params(length=2, pad=1.5)
    lo, hi = y[m].min(), y[m].max()
    a.set_ylim(lo - 0.12 * (hi - lo), hi + 0.16 * (hi - lo))
    ytxt = a.get_ylim()[1] - 0.06 * (a.get_ylim()[1] - a.get_ylim()[0])
    for b, lab in ((B1, "1"), (B2, "2")):
        a.text(0.5 * sum(b), ytxt, lab, color="0.25", ha="center", va="top",
               fontsize=6.5)
# Twin thrust axis on (d). The PWM-to-power and PWM-to-thrust fits are both
# monotone over [1000, 1650], so power maps to thrust exactly rather than
# approximately. Plotting power as the primary quantity keeps the MEASURED
# signal primary and makes the cube law visible: P ~ T^1.49, so the power
# swing is 1.57x where the thrust swing is only 1.38x.
def _p2t(P):
    return np.interp(np.interp(P, PPAIR, GRID), GRID, TPAIR)


def _t2p(T):
    return np.interp(np.interp(T, TPAIR, GRID), GRID, PPAIR)


sec = ax[3].secondary_yaxis("right", functions=(_p2t, _t2p))
# "equivalent steady-state" is not a hedge, it is the correct name. Both bench
# curves are steady-state, so the P->T map assumes the rotor is not
# accelerating. At a fast transient some power goes into rotor inertia rather
# than thrust, and the inversion overstates thrust exactly at the peaks.
sec.set_ylabel("equiv. steady-state $T$ (N)", labelpad=2, fontsize=7)
sec.tick_params(labelsize=7, length=2, pad=1.5)

fig.tight_layout(pad=0.25, w_pad=0.9)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
