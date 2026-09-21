"""Hardware climb profile, one row of four panels, from the primary trial logs.

Mirrors the sim-side demonstration figure: body-frame forward and vertical
speed, pitch, and total propeller thrust, with the step-crossing interval
shaded. Every quantity comes from the logged mocap pose and the logged ESC
command, so nothing here depends on the simulator.
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/hwfig"
os.makedirs(OUT, exist_ok=True)

STEP1_X, STEP2_X = -0.55, -0.15          # front faces, FROZEN_COMMAND.sh
DT = 0.02
WEIGHT_N = 31.57
C = [4.0792540792478203e-13, -2.2921522921483562e-09,
     2.2550699300690226e-05, -0.026882905982896884, 8.516433566430207]


def pwm2thrust(p):
    p = np.clip(np.asarray(p, float), 1000.0, 1650.0)
    t = np.zeros_like(p)
    for c in C:
        t = t * p + c
    return np.maximum(t, 0.0)


def smooth(a, n=5):
    if n <= 1:
        return a
    k = np.ones(n) / n
    return np.convolve(a, k, mode="same")


def load(path):
    r = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(x[k]) for x in r])
    live = np.flatnonzero(np.array([x["gated"] == "0" for x in r]))
    z = f("pos_z")
    kpk = live[int(np.argmax(z[live]))]
    seg = live[live <= kpk]                      # takeover -> peak height
    x, y = f("pos_x")[seg], f("pos_y")[seg]
    z = z[seg]
    qx, qy = f("qx")[seg], f("qy")[seg]
    qz, qw = f("qz")[seg], f("qw")[seg]
    # pitch from quaternion
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    # world velocities by central difference, then rotate into the body frame
    vx = smooth(np.gradient(x, DT))
    vz = smooth(np.gradient(z, DT))
    p = np.radians(pitch)
    vxb = vx * np.cos(p) + vz * np.sin(p)
    vzb = -vx * np.sin(p) + vz * np.cos(p)
    u = (f("u_thr1")[seg] + 1.0) * 325.0 + 1000.0
    v = (f("u_thr2")[seg] + 1.0) * 325.0 + 1000.0
    thrust = pwm2thrust(u) + pwm2thrust(v)
    t = np.arange(seg.size) * DT
    # step window: first approach of step 1 to the moment step 2 is cleared
    i0 = np.argmax(x > STEP1_X - 0.10) if (x > STEP1_X - 0.10).any() else 0
    above = (x > STEP2_X) & (z > z[:20].mean() + 0.10)
    i1 = (np.flatnonzero(above)[0] if above.any() else len(x) - 1)
    return dict(t=t, vxb=vxb, vzb=vzb, pitch=pitch, thrust=thrust,
                band=(t[i0], t[min(i1, len(t) - 1)]), gain=z.max() - z[:20].mean(),
                name=os.path.basename(path))


def draw(d, path):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"],
                         "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.35, "grid.linewidth": 0.5})
    fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.25))
    panels = [
        (d["vxb"], "X speed (forward)", r"$v_x$ (m s$^{-1}$)", "tab:blue"),
        (d["vzb"], "Z speed (vertical)", r"$v_z$ (m s$^{-1}$)", "tab:orange"),
        (d["pitch"], "Pitch angle", r"pitch ($^\circ$)", "tab:green"),
        (d["thrust"], "Total propeller thrust", "thrust (N)", "tab:red"),
    ]
    for a, (y, title, ylab, col) in zip(ax, panels):
        a.plot(d["t"], y, color=col, lw=0.9)
        a.axvspan(d["band"][0], d["band"][1], color="0.6", alpha=0.30, lw=0)
        a.set_title(title, fontsize=9)
        a.set_ylabel(ylab, fontsize=8.5)
        a.set_xlabel("time (s)", fontsize=8.5)
        a.tick_params(labelsize=7.5)
        mid = 0.5 * (d["band"][0] + d["band"][1])
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.16 * (hi - lo))
        a.text(mid, a.get_ylim()[1], "steps ", color="crimson", ha="center",
               va="top", fontsize=8.5)
    ax[3].axhline(WEIGHT_N, color="0.35", ls=":", lw=0.8)
    ax[3].text(d["t"][2], WEIGHT_N, " body weight", va="bottom",
               fontsize=7, color="0.35")
    fig.tight_layout(pad=0.4)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)


cands = sorted(glob.glob("hw_final/trials_final/trial_*.csv"))
rows = []
for p in cands:
    d = load(p)
    rows.append(d)
    print("%-20s gain %.3f  dur %5.2f s  band %.2f-%.2f s  thrust %.1f-%.1f N  pitch %+.1f..%+.1f" % (
        d["name"], d["gain"], d["t"][-1], d["band"][0], d["band"][1],
        d["thrust"].min(), d["thrust"].max(), d["pitch"].min(), d["pitch"].max()))

ok = [d for d in rows if d["gain"] > 0.12]
print("\nsuccessful (gain > 0.12 m): %d of %d" % (len(ok), len(rows)))
for d in ok:
    draw(d, os.path.join(OUT, d["name"].replace(".csv", "")))
print("wrote", len(ok), "figures to", OUT)
