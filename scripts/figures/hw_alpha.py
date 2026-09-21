"""Transparent overlay for correct.mp4 / trial_190715, styled to sit ON the video.

Writes a PNG sequence with real alpha, then muxes to ProRes 4444
(yuva444p10le), which Premiere, Resolve and FCP all read with the alpha intact.
No chroma key needed.

Styling is for legibility over moving footage rather than for print:
  - no panel, no background, no axis box
  - bright saturated traces with a dark stroke underneath, so they hold up over
    both the pale plywood and the dark floor mats
  - white text with a black outline, same reason
  - x = 0 is policy takeover

Servo is absent by design. Those servos have no position feedback, so the logs
carry only the command and nothing that could honestly be drawn as an angle.
"""
import csv, os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_overlay_alpha.mov")
TMP = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/_alpha"
DT, FPS = 0.02, 50
TAKEOVER_S, TOTAL_S = 3.0, 9.25

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
series = [(pitch[sl], "BODY PITCH", r"$\theta$  (deg)", "#00E5FF"),
          (thrust[sl], "PROPELLER COMMAND", "norm.", "#FF4D3D")]

n_frames = int(round(TOTAL_S * FPS))
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)
print("%s  %d frames, %.2f s, takeover at %.1f s" % (SRC, n_frames, TOTAL_S, TAKEOVER_S))

WHITE = "#FFFFFF"
stroke = [pe.withStroke(linewidth=3.0, foreground="black", alpha=0.75)]
plt.rcParams.update({"font.size": 13, "text.color": WHITE,
                     "axes.labelcolor": WHITE, "xtick.color": WHITE,
                     "ytick.color": WHITE})

fig, ax = plt.subplots(1, 2, figsize=(16, 2.9), dpi=120)
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
        a_.spines[s].set_color(WHITE)
        a_.spines[s].set_linewidth(1.6)
        a_.spines[s].set_path_effects(stroke)
    a_.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=6,
                 color=WHITE, path_effects=stroke)
    a_.set_xlabel("time since takeover (s)", labelpad=3, path_effects=stroke)
    a_.set_ylabel(ylab, labelpad=3, path_effects=stroke)
    a_.grid(alpha=0.22, lw=0.8, color=WHITE)
    for lab in a_.get_xticklabels() + a_.get_yticklabels():
        lab.set_path_effects(stroke)
    (ln,) = a_.plot([], [], color=col, lw=3.4, zorder=3, solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=6.0, foreground="black",
                                                alpha=0.55)])
    lines.append(ln)
fig.tight_layout(pad=0.8)

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
shutil.rmtree(TMP, ignore_errors=True)
print("wrote", OUT)
