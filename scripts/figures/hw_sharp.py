"""Transparent overlay for correct.mp4 / trial_190715, maximum text sharpness.

No card, no outlines, no path_effects of any kind. Every glyph is drawn as plain
antialiased type, which is the sharpest matplotlib can give.

Two things were softening the earlier versions:
  1. pe.withStroke centres a black stroke ON the glyph outline, so it eats
     inward and thickens the letterform. All of that is gone here.
  2. The overlay rendered at 1920 wide while correct.mp4 is 3840x2160. Scaling
     a 1920 overlay up to 4K doubles every pixel and blurs the type no matter
     how it was styled. SCALE below renders at native 4K width instead.

Set SCALE = 1 if the edit timeline is actually 1080p. Font sizes are in points,
so they keep the same proportions either way, they just get more pixels.

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
OUT = os.path.expanduser("~/Downloads/DoubleBee_overlay_sharp.mov")
TMP = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/_sharp"
DT, FPS = 0.02, 50
TAKEOVER_S, TOTAL_S = 3.0, 9.25
SCALE = 2                      # 2 -> 3840x696, native for a 4K timeline

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
series = [(pitch[sl], "Body pitch", r"$\theta$  (deg)", "#00E5FF"),
          (thrust[sl], "Propeller command", "norm.", "#FF4D3D")]

n_frames = int(round(TOTAL_S * FPS))
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

W = "#FFFFFF"
plt.rcParams.update({"font.size": 13, "text.color": W, "axes.labelcolor": W,
                     "xtick.color": W, "ytick.color": W})

fig, ax = plt.subplots(1, 2, figsize=(16, 2.9), dpi=120 * SCALE)
fig.patch.set_alpha(0.0)
lines = []
for a_, (yv, title, ylab, col) in zip(ax, series):
    pad = 0.14 * (yv.max() - yv.min() + 1e-9)
    a_.set_facecolor("none")
    a_.set_xlim(0.0, TOTAL_S - TAKEOVER_S)
    a_.set_ylim(yv.min() - pad, yv.max() + pad)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a_.spines[s].set_color(W)
        a_.spines[s].set_linewidth(1.4)
    a_.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=6, color=W)
    a_.set_xlabel("time (s)", labelpad=3)
    a_.set_ylabel(ylab, labelpad=3)
    a_.grid(alpha=0.20, lw=0.8, color=W)
    (ln,) = a_.plot([], [], color=col, lw=3.0, zorder=3, solid_capstyle="round")
    lines.append(ln)
fig.tight_layout(pad=0.8)

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
    fig.savefig("%s/f%05d.png" % (TMP, i), transparent=True,
                facecolor="none", edgecolor="none")
plt.close(fig)

subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
                "-i", "%s/f%%05d.png" % TMP, "-c:v", "prores_ks",
                "-profile:v", "4444", "-pix_fmt", "yuva444p10le", OUT],
               check=True)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", OUT,
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                "-crf", "26", "-b:v", "0", "-row-mt", "1",
                OUT.replace(".mov", ".webm")], check=True)
shutil.rmtree(TMP, ignore_errors=True)
print("wrote", OUT, "and", OUT.replace(".mov", ".webm"))
