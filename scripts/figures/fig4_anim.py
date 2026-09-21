"""Animated clearance-against-power slide for the ICRA video.

18 s total. Everything is revealed by about 5 s, then it holds for another 13 so
there is room to talk over it.

Same argument as Fig. 4 bottom, plus the mode-switching baselines that the paper
keeps in Table IV, so the whole comparison sits on one axis.

Reveal order is deliberate. Fixed allocations first, tracing the grey trend of
"buy clearance with power". Then the two switches, which beat the fixed rules
but still trail. Then the learned star, off that trend entirely.

Numbers are the paper's, not re-derived here:
  fixed allocations   clearance from fig4_mixed.csv at 6 cm, power from
                      fig4_full.py POWER (summ_arms.py, total energy / total
                      control steps / 0.02 s)
  switches + learned  Table IV
Error bars are one standard error across ten checkpoints, as in the paper.

Callout arithmetic, stated honestly:
  best switch is Switch 0.31. It leads Switch 0.46 at 3, 4, 6 and 7 cm but NOT
  at 5 cm (39.8 against 42.6, a gap inside the error bars), and it draws less
  power, 264 against 321. Best switch, not unbeaten at every height.
    clearance 39.8 / 28.8 = 1.382   -> +38 %
    power     286  / 264  = 1.083   -> +8 % , i.e. the learned policy costs MORE
  best fixed allocation is T/W 0.55.
    clearance 39.8 / 22.5 = 1.769   -> +77 %
    power     286  / 391  = 0.731   -> -27 % , matching the paper's V-A figure

Both rows quote MEAN POWER, and the switch row admits the loss rather than
hiding it. An earlier draft quoted energy per completed climb for the switch
row instead, where the learned policy does win, 658 J against 707 J (V-B). That
was dropped on purpose. This plot's x-axis IS mean power, and Switch 0.31 sits
visibly LEFT of the learned star at 264 W against 286 W, so a caption claiming
a saving would contradict the chart underneath it. The energy figure belongs in
the narration, where the conditioning on a completed climb can be explained.
"""
import os, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/DoubleBee_baselines_anim.mp4")
TMP = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/_f4a"
FPS, DUR = 50, 18.0
SCALE = 2                      # 2 -> 3840x2160

LEARNED = "#c0392b"
SWITCH = "#e67e22"
ramp = plt.cm.viridis(np.linspace(0.05, 0.80, 4))
cf = {0.23: ramp[0], 0.31: ramp[1], 0.46: ramp[2], 0.55: ramp[3]}

# label, power W, clears 6 cm %, se, colour, marker, size, t_appear, (dx, dy, ha)
PTS = [
    ("$T/W$ 0.23",  227,  1.1, 0.82, cf[0.23], "o", 18, 0.80, (0, 3.2, "center")),
    ("$T/W$ 0.31",  269,  1.3, 0.44, cf[0.31], "o", 18, 1.40, (0, 3.2, "center")),
    ("$T/W$ 0.46",  347,  8.3, 1.90, cf[0.46], "o", 18, 2.00, (10, 0.0, "left")),
    ("$T/W$ 0.55",  391, 22.5, 6.23, cf[0.55], "o", 18, 2.60, (-10, 3.0, "right")),
    ("Switch 0.31", 264, 28.8, 5.00, SWITCH,   "s", 17, 3.25, (-11, 0.5, "right")),
    ("Switch 0.46", 321, 25.6, 3.70, SWITCH,   "s", 17, 3.80, (11, -0.5, "left")),
    ("Learned",     286, 39.8, 2.50, LEARNED,  "*", 40, 4.40, (-16, 0.0, "right")),
]
TREND = [(227, 1.1), (269, 1.3), (347, 8.3), (391, 22.5)]
T_TREND, T_NOTE = 0.80, 5.10

n_frames = int(round(DUR * FPS))
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

fade = lambda t, t0, d=0.45: float(np.clip((t - t0) / d, 0.0, 1.0))

plt.rcParams.update({"font.size": 23, "axes.labelsize": 27,
                     "xtick.labelsize": 21, "ytick.labelsize": 21,
                     "axes.linewidth": 1.6})

for i in range(n_frames):
    t = i / FPS
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120 * SCALE)
    fig.patch.set_facecolor("white")
    ax.set_xlim(205, 420)
    ax.set_ylim(-4, 52)
    ax.set_xlabel("Mean power (W)", labelpad=8)
    ax.set_ylabel("Clears a 6 cm step (%)", labelpad=8)
    ax.grid(alpha=0.25, lw=1.0)
    ax.tick_params(length=5, pad=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # grey trend through the fixed allocations, drawn left to right
    a = fade(t, T_TREND, 2.20)
    if a > 0:
        k = a * (len(TREND) - 1)
        j = int(np.floor(k))
        xs = [p[0] for p in TREND[:j + 1]]
        ys = [p[1] for p in TREND[:j + 1]]
        if j < len(TREND) - 1:
            f = k - j
            xs.append(TREND[j][0] + f * (TREND[j + 1][0] - TREND[j][0]))
            ys.append(TREND[j][1] + f * (TREND[j + 1][1] - TREND[j][1]))
        ax.plot(xs, ys, "-", color="0.62", lw=2.2, zorder=1)

    for lab, px, py, se, col, mk, ms, t0, (dx, dy, ha) in PTS:
        al = fade(t, t0)
        if al <= 0:
            continue
        # the learned star lands with a brief scale pop
        grow = 1.0 + 0.55 * (1.0 - fade(t, t0, 0.50)) if mk == "*" else 1.0
        ax.errorbar([px], [py], yerr=[se], fmt="none", ecolor=col,
                    elinewidth=1.8, capsize=5, alpha=0.75 * al, zorder=2)
        ax.plot([px], [py], mk, ms=ms * grow, color=col, alpha=al,
                mec="white" if mk == "*" else "0.15",
                mew=2.0 if mk == "*" else 1.2, zorder=4)
        ax.annotate(lab, (px, py), textcoords="offset points",
                    xytext=(dx, dy * 6), ha=ha, va="center",
                    fontsize=23 if mk != "*" else 27,
                    fontweight="bold" if mk == "*" else "normal",
                    color=col, alpha=al, zorder=5)

    an = fade(t, T_NOTE, 0.70)
    if an > 0:
        ax.annotate("vs best switch    +38 % clearance,  8 % more power\n"
                    "vs best fixed      +77 % clearance,  27 % less power",
                    xy=(286, 39.8), xytext=(340, 47.0), ha="center", va="center",
                    fontsize=19, color=LEARNED, alpha=an, zorder=6, linespacing=1.6,
                    arrowprops=dict(arrowstyle="->", color=LEARNED,
                                    lw=2.0, alpha=an,
                                    shrinkA=6, shrinkB=14))

    fig.tight_layout(pad=1.4)
    fig.savefig("%s/f%05d.png" % (TMP, i), facecolor="white")
    plt.close(fig)

subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
                "-i", "%s/f%%05d.png" % TMP, "-c:v", "libx264", "-crf", "16",
                "-pix_fmt", "yuv420p", OUT], check=True)
shutil.rmtree(TMP, ignore_errors=True)
print("wrote", OUT)
