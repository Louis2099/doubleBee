"""White-background overlay for correct.mp4 / trial_190715, at 4K width.

This is the og hw_timeline.py look, rebuilt at native 4K so the type is sharp on
a 3840x2160 timeline instead of being upscaled 2x from 1920. Same two channels,
same paper colour scheme, same "Learned policy" legend.

Colours follow the paper figures rather than the poppy scheme used for the
transparent cut. Cyan on white reads badly, so pitch stays viridis and the
propeller command stays the paper's red.

Opaque mp4, no alpha needed. Drop it in as a strip alongside the robot.

Servo is absent by design. Those servos have no position feedback, so the logs
carry only the command and nothing that could honestly be drawn as an angle.
Pitch convention validated against FROZEN_COMMAND.sh.
"""
import csv, os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_overlay_white4k.mp4")
TMP = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/_white"
DT, FPS = 0.02, 50
TAKEOVER_S, TOTAL_S = 3.0, 9.25
SCALE = 2                      # 2 -> 3840x672, native for a 4K timeline

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
li = np.flatnonzero(live)
a, b = li[0], li[-1]
qw, qx, qy, qz = f("qw"), f("qx"), f("qy"), f("qz")
pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
thrust = (f("u_thr1") + f("u_thr2")) / 2.0

n_fit = min(b - a + 1, int(round((TOTAL_S - TAKEOVER_S) / DT)))
sl = slice(a, a + n_fit)
t = np.arange(n_fit) * DT
series = [(pitch[sl], "Body pitch", r"$\theta$ (deg)", plt.cm.viridis(0.40)),
          (thrust[sl], "Propeller command", "norm.", "#c0392b")]

n_frames = int(round(TOTAL_S * FPS))
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.linewidth": 0.9})
fig, ax = plt.subplots(1, 2, figsize=(16, 2.8), dpi=120 * SCALE)
fig.patch.set_facecolor("white")

lines = []
for a_, (yv, title, ylab, col) in zip(ax, series):
    pad = 0.12 * (yv.max() - yv.min() + 1e-9)
    a_.set_xlim(0.0, TOTAL_S - TAKEOVER_S)
    a_.set_ylim(yv.min() - pad, yv.max() + pad)
    a_.set_title(title, fontsize=13, pad=5, loc="left")
    a_.set_xlabel("time (s)", labelpad=2)
    a_.set_ylabel(ylab, labelpad=2)
    a_.grid(alpha=0.3, lw=0.6)
    if title == "Body pitch":
        a_.axhline(0.0, color="0.78", lw=0.7, zorder=1)
    (ln,) = a_.plot([], [], color=col, lw=2.0, zorder=3, label="Learned policy")
    a_.legend(loc="upper left", frameon=True, fontsize=9)
    lines.append(ln)
fig.tight_layout(pad=0.7)

px = fig.get_size_inches() * fig.dpi
print("%s  %d frames, %.2f s, takeover %.1f s, %dx%d"
      % (SRC, n_frames, TOTAL_S, TAKEOVER_S, px[0], px[1]))

for i in range(n_frames):
    k = int(round((i / FPS - TAKEOVER_S) / DT))
    for ln, (yv, *_rest) in zip(lines, series):
        if k <= 0:
            ln.set_data([], [])
        else:
            m = min(k, n_fit)
            ln.set_data(t[:m], yv[:m])
    fig.savefig("%s/f%05d.png" % (TMP, i), facecolor="white")
plt.close(fig)

subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
                "-i", "%s/f%%05d.png" % TMP, "-c:v", "libx264", "-crf", "16",
                "-pix_fmt", "yuv420p", OUT], check=True)
shutil.rmtree(TMP, ignore_errors=True)
print("wrote", OUT)
