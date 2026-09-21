"""Animated overlay for trial_190715, the run in correct.mp4 (11 Sep 19:07).

Three panels, all measured or policy-native:
  height gained      mocap pos_z, relative to the pre-climb baseline
  body pitch         asin(2(qw*qy - qz*qx)), validated against FROZEN_COMMAND.sh
                     which records the reference climb as -4.0..+7.4 deg and
                     +0.4 +- 2.0 over the last 0.8 s. This reproduces all four.
  propeller command  (u_thr1 + u_thr2)/2, normalised, the policy's own output

Servo is deliberately absent. The logs carry the blended, filtered, slew-limited
COMMAND only -- there is no position feedback on those servos -- so it cannot
honestly be drawn as an angle.

Window is the whole live period (gate_reason == ""), so the approach, both step
contacts and the settled ending are all in frame.

50 fps, the control rate, so one video second is one real second.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2
                         else "~/Downloads/DoubleBee_overlay_190715.mp4")
DT, FPS = 0.02, 50

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
li = np.flatnonzero(live)
a, b = li[0], li[-1]

z = f("pos_z")
qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
thrust = (f("u_thr1") + f("u_thr2")) / 2.0

bn = int(0.5 / DT)
onset = next((i for i in range(a + bn, b - 5)
              if z[i] - np.median(z[max(a, i - bn):i]) >= 0.02), a + bn)
base = float(np.median(z[max(a, onset - bn):onset]))

t = (np.arange(a, b + 1) - a) * DT
h = z[a:b + 1] - base
p = pitch[a:b + 1]
th = thrust[a:b + 1]
n = len(t)
print("%s  live %.2f s (%d frames)  height %.3f..%.3f m  pitch %.1f..%.1f deg"
      % (SRC, n / FPS, n, h.min(), h.max(), p.min(), p.max()))
print("  align overlay t=0 with the moment the robot starts moving in the clip")

LEARNED = "#c0392b"
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.linewidth": 0.9})
fig, ax = plt.subplots(1, 3, figsize=(16, 2.8), dpi=120)
fig.patch.set_facecolor("white")

specs = [(h, "Height gained", r"$\Delta z$ (m)", ramp[0]),
         (p, "Body pitch", r"$\theta$ (deg)", ramp[2]),
         (th, "Propeller command", "Command (norm.)", LEARNED)]
lines = []
for a_, (yv, title, ylab, col) in zip(ax, specs):
    pad = 0.12 * (yv.max() - yv.min() + 1e-9)
    a_.set_xlim(t[0], t[-1])
    a_.set_ylim(yv.min() - pad, yv.max() + pad)
    a_.set_title(title, fontsize=13, pad=5, loc="left")
    a_.set_xlabel("Time (s)", labelpad=2)
    a_.set_ylabel(ylab, labelpad=2)
    a_.grid(alpha=0.3, lw=0.6)
    if title == "Body pitch":
        a_.axhline(0.0, color="0.75", lw=0.7, zorder=1)
    (ln,) = a_.plot([], [], color=col, lw=2.0, zorder=3, label="Learned policy")
    a_.legend(loc="upper left", frameon=True, fontsize=9)
    lines.append(ln)
fig.tight_layout(pad=0.7)


def update(i):
    for ln, (yv, *_r) in zip(lines, specs):
        ln.set_data(t[:i + 1], yv[:i + 1])
    return lines


FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=True).save(
    OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000,
                             extra_args=["-pix_fmt", "yuv420p"]))
print("wrote", OUT)
