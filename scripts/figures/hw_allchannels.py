"""Every usable channel from trial_190715 (the run in correct.mp4), for picking.

Provenance matters more than looks here, so each panel is tagged:
  [mocap]   measured pose
  [encoder] RoboClaw readback
  [power]   Cube power module
  [policy]  the network's own output, unfiltered
  [command] what the interface SENT -- no feedback exists for it

Servo is [command] only. The logs carry the blended, filtered, slew-limited
command and the servos have no position sensor, so it cannot be called an angle.
Conversion is theta = -servo1 * 90 (db_inference.py: servo = -(2/pi)*theta, with
the frozen command's gain of 0.5 at sim_servo_limit_rad 0.7854).

Pitch convention validated against FROZEN_COMMAND.sh, which records the clean
reference climb as -4.0..+7.4 deg and +0.4 +- 2.0 over the last 0.8 s.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_190715_channels")
DT = 0.02

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
li = np.flatnonzero(live)
a, b = li[0], li[-1]
sl = slice(a, b + 1)

z, px, py = f("pos_z"), f("pos_x"), f("pos_y")
qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
bn = int(0.5 / DT)
onset = next((i for i in range(a + bn, b - 5)
              if z[i] - np.median(z[max(a, i - bn):i]) >= 0.02), a + bn)
base = float(np.median(z[max(a, onset - bn):onset]))

speed = np.concatenate([[0.0], np.hypot(np.diff(px), np.diff(py)) / DT])
wheel = (np.abs(f("wheel_meas_l")) + np.abs(f("wheel_meas_r"))) / 2.0
thrust = (f("u_thr1") + f("u_thr2")) / 2.0
servo = -(f("servo1") + f("servo2")) / 2.0 * 90.0
watts = f("watts")
curr = (f("m1_a") + f("m2_a"))

t = (np.arange(a, b + 1) - a) * DT
LEARNED = "#c0392b"
ramp = plt.cm.viridis(np.linspace(0.05, 0.75, 5))

panels = [
    (z[sl] - base,  "Height gained  [mocap]",       r"$\Delta z$ (m)",   ramp[0]),
    (pitch[sl],     "Body pitch  [mocap]",          r"$\theta$ (deg)",   ramp[1]),
    (speed[sl],     "Forward speed  [mocap]",       r"$v$ (m s$^{-1}$)", ramp[2]),
    (wheel[sl],     "Wheel speed  [encoder]",       r"$\omega$ (rad/s)", ramp[3]),
    (thrust[sl],    "Propeller command  [policy]",  "norm.",             LEARNED),
    (servo[sl],     "Servo command  [command]",     "deg",               ramp[4]),
    (watts[sl],     "Electrical power  [power]",    "W",                 "0.35"),
    (curr[sl],      "Wheel current  [encoder]",     "A",                 "0.55"),
]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
fig, axes = plt.subplots(2, 4, figsize=(13, 4.6))
ons_t = (onset - a) * DT
for ax, (yv, title, ylab, col) in zip(axes.ravel(), panels):
    ax.axvline(ons_t, color="0.6", lw=0.8, ls="--", zorder=1)
    if "pitch" in title:
        ax.axhline(0.0, color="0.78", lw=0.6, zorder=1)
    ax.plot(t, yv, color=col, lw=1.1, zorder=3)
    ax.set_title(title, fontsize=8.5, loc="left", pad=3)
    ax.set_ylabel(ylab, labelpad=2)
    ax.set_xlabel("time from policy takeover (s)", labelpad=2)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(length=2, pad=1.5)
    if np.all(np.isnan(yv)):
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")

fig.tight_layout(pad=0.4, h_pad=1.3, w_pad=0.9)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("live %.2f s | height %.3f m | pitch %.1f..%.1f | power %.0f W mean"
      % (len(t) * DT, (z[sl] - base).max(), pitch[sl].min(), pitch[sl].max(),
         np.nanmean(watts[sl])))
print("wrote", OUT + ".png")
