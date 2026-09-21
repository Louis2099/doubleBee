"""Compare candidate trials so the right one can be picked by eye.

One row per trial, three columns: height gained, body pitch, propeller command.

Window is NOTES.md's convention -- policy takeover (first live sample) through
to peak height -- so the opening lean and the settled ending are both visible.

Pitch convention validated against FROZEN_COMMAND.sh, which records the clean
reference climb as "pitch -4.0 .. +7.4 deg" and "+0.4 +- 2.0 over the last
0.8 s". asin(2(qw*qy - qz*qx)) reproduces all four numbers exactly.

Only genuinely measured or policy-native channels here. Height and pitch are
mocap, propeller command is the policy's own output. Servo is omitted: the logs
carry the command only, with no position feedback, so it cannot be plotted as
an angle.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIALS = sys.argv[1:] or ["trial_190937.csv", "trial_192233.csv", "trial_190715.csv"]
OUT = os.path.expanduser("~/Downloads/DoubleBee_hw_candidates")
DT = 0.02
LEARNED = "#c0392b"
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
fig, axes = plt.subplots(len(TRIALS), 3, figsize=(7.16, 1.7 * len(TRIALS)),
                         squeeze=False)

for row, path in enumerate(TRIALS):
    r = list(csv.DictReader(open(path)))
    g = lambda k: np.array([float(x[k]) for x in r])
    qw, qx, qy, qz = g("qw"), g("qx"), g("qy"), g("qz")
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    z = g("pos_z")
    thrust = (g("u_thr1") + g("u_thr2")) / 2.0
    live = np.array([x.get("gate_reason", "") == "" for x in r])
    li = np.flatnonzero(live)
    a = li[0]

    bn = int(0.5 / DT)
    onset = next((i for i in range(bn, len(z) - 5)
                  if live[i] and z[i] - np.median(z[max(0, i - bn):i]) >= 0.02), None)
    end = min(len(z) - 1, onset + int(6.0 / DT))
    peak = onset + int(np.argmax(z[onset:end + 1]))
    base = float(np.median(z[max(0, onset - bn):onset]))

    t = (np.arange(a, peak + 1) - a) * DT
    tag = os.path.basename(path).replace("trial_", "").replace(".csv", "")
    tail = li[li <= peak][-40:]
    print("%s  gain %.3f m  dur %.2f s  open %.1f deg  end %.1f +- %.2f deg"
          % (tag, z[peak] - base, (peak - a) * DT, pitch[a:a + 50].mean(),
             pitch[tail].mean(), pitch[tail].std()))

    series = [(z[a:peak + 1] - base, r"$\Delta z$ (m)", ramp[0]),
              (pitch[a:peak + 1], r"$\theta$ (deg)", ramp[2]),
              (thrust[a:peak + 1], "Command (norm.)", LEARNED)]
    for col, (yv, ylab, c) in enumerate(series):
        ax = axes[row][col]
        ax.axvline((onset - a) * DT, color="0.6", lw=0.8, ls="--", zorder=1)
        if col == 1:
            ax.axhline(0.0, color="0.75", lw=0.6, zorder=1)
        ax.plot(t, yv, color=c, lw=1.0, zorder=3)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(length=2, pad=1.5)
        ax.set_ylabel(ylab, labelpad=2)
        if col == 0:
            ax.set_title("trial %s" % tag, fontsize=8, loc="left", pad=3)
        if row == len(TRIALS) - 1:
            ax.set_xlabel("time from policy takeover (s)", labelpad=2)

fig.tight_layout(pad=0.3, h_pad=1.0, w_pad=0.9)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("wrote", OUT + ".png")
