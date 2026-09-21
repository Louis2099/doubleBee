"""Overlay cut to the editing timeline for correct.mp4 / trial_190715.

Timeline (HH:MM:SS:FF):
    33:30  clip starts        -> overlay t = 0
    34:30  arm                -> overlay t = 1.0 s
    36:30  policy takeover    -> overlay t = 3.0 s   (log live data starts here)
    42:45  clip ends          -> overlay t = 9.25 s

The frame fields at start and takeover are equal, so the 3.0 s offset holds at
either a 50 or 60 fps timebase; only the end moves, by ~0.05 s.

Axes are drawn for the whole 9.25 s with fixed limits so nothing jumps. The
traces stay empty until takeover, then grow in real time. The log has 6.58 s of
live data, which runs past the clip end, so it is truncated at 9.25 s.

Channels are measured or policy-native only. Servo is excluded: the logs hold
the blended, filtered command and the servos have no position feedback.
Pitch convention validated against FROZEN_COMMAND.sh (-4.0..+7.4 deg, and
+0.4 +- 2.0 over the last 0.8 s on the reference climb).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_overlay_timeline.mp4")
DT, FPS = 0.02, 50
TAKEOVER_S, TOTAL_S = 3.0, 9.25

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
li = np.flatnonzero(live)
a, b = li[0], li[-1]

z = f("pos_z")
qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
thrust = (f("u_thr1") + f("u_thr2")) / 2.0
# JAIOut servo units are [-1,1] over +/-pi/2 rad and servo = -(2/pi)*theta
# (db_inference.py ~1797). With the frozen command's action_scale 1.0,
# servo_scale 1.0 and sim_servo_limit_rad 0.7854 the gain is 0.5, so
# theta_deg = -servo * 90. This is the COMMAND sent to the servos -- they have
# no position feedback, and it is blended with attitude hold, low-pass filtered
# and slew limited per Section IV-C. Labelled "command", not "angle".
servo_cmd = -(f("servo1") + f("servo2")) / 2.0 * 90.0
# The COMMAND saturates to +/-45 deg at each step contact (action_2 hits -0.99
# there), but --servo_slew_rad_s 2.0 caps the joint at 114 deg/s, so a 45 deg
# move needs 0.4 s and the command reverses first. Integrating the command
# through that limit gives the angle the arm could actually reach, which is what
# the footage shows. Reconstruction from documented interface parameters, not a
# logged measurement -- these servos have no position feedback.
_STEP = 2.0 / 50.0 * 180.0 / np.pi          # rad/s -> deg per 20 ms tick
servo = np.empty_like(servo_cmd)
_pos = servo_cmd[0]
for _i, _tgt in enumerate(servo_cmd):
    _pos += np.clip(_tgt - _pos, -_STEP, _STEP)
    servo[_i] = _pos

bn = int(0.5 / DT)
onset = next((i for i in range(a + bn, b - 5)
              if z[i] - np.median(z[max(a, i - bn):i]) >= 0.02), a + bn)
base = float(np.median(z[max(a, onset - bn):onset]))

# Log samples, clipped to what fits before the clip ends.
n_fit = min(b - a + 1, int(round((TOTAL_S - TAKEOVER_S) / DT)))
sl = slice(a, a + n_fit)
# x-axis is time SINCE takeover, so it starts at 0 rather than carrying 3 s of
# empty axis. The video is still TOTAL_S long and still starts at the clip head,
# the traces just stay blank until takeover arrives.
t_data = np.arange(n_fit) * DT
# Servo dropped: the logs hold only the command, the servos have no position
# feedback, and a slew-limited reconstruction still disagreed with the footage.
series = [(pitch[sl], "Body pitch", r"$\theta$ (deg)", plt.cm.viridis(0.40)),
          (thrust[sl], "Propeller command", "norm.", "#c0392b")]

n_frames = int(round(TOTAL_S * FPS))
print("%s  overlay %.2f s (%d frames)  takeover at %.1f s  log fits %d/%d samples"
      % (SRC, TOTAL_S, n_frames, TAKEOVER_S, n_fit, b - a + 1))

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.linewidth": 0.9})
fig, ax = plt.subplots(1, 2, figsize=(16, 2.8), dpi=120)
fig.patch.set_facecolor("white")

lines = []
for a_, (yv, title, ylab, col) in zip(ax, series):
    pad = 0.12 * (yv.max() - yv.min() + 1e-9)
    a_.set_xlim(0.0, TOTAL_S - TAKEOVER_S)
    a_.set_ylim(yv.min() - pad, yv.max() + pad)
    a_.set_title(title, fontsize=13, pad=5, loc="left")
    a_.set_xlabel("Time (s)", labelpad=2)
    a_.set_ylabel(ylab, labelpad=2)
    a_.grid(alpha=0.3, lw=0.6)
    # No takeover marker needed now: x = 0 IS takeover.
    if title == "Body pitch":
        a_.axhline(0.0, color="0.78", lw=0.7, zorder=1)
    (ln,) = a_.plot([], [], color=col, lw=2.0, zorder=3, label="Learned policy")
    a_.legend(loc="upper left", frameon=True, fontsize=9)
    lines.append(ln)
fig.tight_layout(pad=0.7)


def update(i):
    now = i / FPS
    k = int(round((now - TAKEOVER_S) / DT))
    for ln, (yv, *_rest) in zip(lines, series):
        if k <= 0:
            ln.set_data([], [])
        else:
            m = min(k, n_fit)
            ln.set_data(t_data[:m], yv[:m])
    return lines


FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS,
              blit=True).save(
    OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000,
                             extra_args=["-pix_fmt", "yuv420p"]))
print("wrote", OUT)
