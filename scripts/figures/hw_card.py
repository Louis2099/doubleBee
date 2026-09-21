"""Overlay for correct.mp4 / trial_190715 on a translucent dark card.

The fully transparent version needed a black outline on every glyph to stay
readable over the footage, and that outline ate into the letterforms. Here a
semi-transparent dark card sits behind the plot instead, so the text can be
drawn plain and stays sharp, while the footage still reads through the card.

Exported with real alpha (ProRes 4444), so the card composites over the video
rather than being baked onto a white strip. Card opacity is CARD_ALPHA below.

Servo is absent by design. Those servos have no position feedback, so the logs
carry only the command and nothing that could honestly be drawn as an angle.
Pitch convention validated against FROZEN_COMMAND.sh.
"""
import csv, os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_190715.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_overlay_card.mov")
TMP = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/_card"
DT, FPS = 0.02, 50
TAKEOVER_S, TOTAL_S = 3.0, 9.25
CARD_RGB, CARD_ALPHA = "#0E1218", 0.72

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

W = "#FFFFFF"
plt.rcParams.update({"font.size": 13, "text.color": W, "axes.labelcolor": W,
                     "xtick.color": W, "ytick.color": W})

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
        a_.spines[s].set_color(W)
        a_.spines[s].set_linewidth(1.2)
        a_.spines[s].set_alpha(0.75)
    # No path_effects anywhere now. The card supplies the contrast, so every
    # glyph renders as clean antialiased type.
    a_.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=6, color=W)
    a_.set_xlabel("time since takeover (s)", labelpad=3)
    a_.set_ylabel(ylab, labelpad=3)
    a_.grid(alpha=0.16, lw=0.8, color=W)
    (ln,) = a_.plot([], [], color=col, lw=3.0, zorder=3, solid_capstyle="round")
    lines.append(ln)
fig.tight_layout(pad=1.0)

# Card goes on after tight_layout, in figure coords, behind the axes. savefig's
# transparent=True clears fig.patch and ax.patch but leaves this artist alone.
card = FancyBboxPatch((0.004, 0.02), 0.992, 0.96,
                      boxstyle="round,pad=0,rounding_size=0.018",
                      transform=fig.transFigure, zorder=-10,
                      facecolor=CARD_RGB, alpha=CARD_ALPHA,
                      edgecolor=W, linewidth=1.0)
card.set_edgecolor((1, 1, 1, 0.14))
fig.add_artist(card)

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
