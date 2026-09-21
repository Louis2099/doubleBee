"""Animated overlay strip for the primary hardware climb.

Two panels, servo angle and propeller command, drawn progressively so the
traces grow as the robot climbs. Rendered at 50 fps, the control rate, so one
video second equals one real second and it can be laid straight onto footage
without resampling.

Window matches hw_primary_plot.py: 1 s of pre-roll before climb onset (first
live sample rising >= 2 cm above the preceding 0.5 s median) through to peak
height. For trial_175627 that is log time 10.64 s to 15.56 s.

Thrust stays NORMALISED, as in fig_climb_profile.py. u_thr1/u_thr2 are the
policy's channel commands in [-1, 1], not newtons.

White background, since mp4 has no alpha. Crop or key it in the edit.
"""
import csv, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_175627.csv"
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2
                         else "~/Downloads/DoubleBee_hw_overlay.mp4")
DT, FPS = 0.02, 50

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
z = f("pos_z")
# JAIOut servo units are [-1,1] over +/-pi/2 rad, and servo = -(2/pi)*theta
# (db_inference.py ~line 1797), so theta = -servo * pi/2. With the frozen
# command's sim_servo_limit_rad 0.7854, action_scale 1.0 and servo_scale 1.0
# the gain is 0.5, giving theta_deg = -servo * 90.
# NOT np.degrees(servo): that is wrong by pi/2 and has the sign inverted.
servo = -(f("servo1") + f("servo2")) / 2.0 * 90.0
thrust = (f("u_thr1") + f("u_thr2")) / 2.0

bn = int(0.5 / DT)
onset = next((i for i in range(bn, len(z) - 5)
              if live[i] and z[i] - np.median(z[max(0, i - bn):i]) >= 0.02), None)
if onset is None:
    sys.exit("no climb onset in %s" % SRC)
end = min(len(z) - 1, onset + int(6.0 / DT))
peak = onset + int(np.argmax(z[onset:end + 1]))
s = max(0, onset - int(1.0 / DT))

t = (np.arange(s, peak + 1) - onset) * DT
sv, th = servo[s:peak + 1], thrust[s:peak + 1]
n = len(t)
print("%s  frames=%d  dur=%.2fs  log window %.2f..%.2f s"
      % (SRC, n, n / FPS, s * DT, peak * DT))

# 1920 x 336: same width as a 1080p frame so it drops straight on, but a 6:1
# strip rather than 4:1 so it leaves room for the footage above it.
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.linewidth": 0.9})
fig, ax = plt.subplots(1, 2, figsize=(16, 2.8), dpi=120)
fig.patch.set_facecolor("white")

# Paper scheme (sim_fig.py / fig4_full.py): the learned policy is LEARNED red,
# other channels come off the viridis ramp.
LEARNED = "#c0392b"
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))
specs = [(sv, "Servo angle", "Servo angle (deg)", ramp[2]),
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
    a_.axvline(0.0, color="0.6", lw=1.0, ls="--", zorder=1)
    (ln,) = a_.plot([], [], color=col, lw=2.0, zorder=3, label="Learned policy")
    a_.legend(loc="upper left", frameon=True, fontsize=9)
    lines.append(ln)
fig.tight_layout(pad=0.7)


def update(i):
    for ln, (yv, *_rest) in zip(lines, specs):
        ln.set_data(t[:i + 1], yv[:i + 1])
    return lines


ani = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=True)
ani.save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000,
                                  extra_args=["-pix_fmt", "yuv420p"]))
print("wrote", OUT)
