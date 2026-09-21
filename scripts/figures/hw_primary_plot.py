"""Primary hardware experiment: what the policy does through a two-step climb.

Window follows trials_final/NOTES.md: policy takeover to peak height. Onset is
the first live sample rising >= 2 cm above the median of the preceding 0.5 s
(same rule as fig_climb_profile.py), with 1 s of pre-roll for context, running
to peak height within 6 s of onset so both steps are inside the window.

Panels: height gained, body pitch, servo angle, propeller command.
Servo is included because thrust magnitude alone cannot show whether the
propellers pushed up or forward -- that is the argument in fig_climb_profile.py.

Thrust is left NORMALISED, matching fig_climb_profile.py, which labels it
"thrust command (norm.)". u_thr1/u_thr2 are the policy's channel commands in
[-1, 1], not newtons.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "trial_175627.csv"
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2
                         else "~/Downloads/DoubleBee_hw_primary")
DT = 0.02

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
live = np.array([x.get("gate_reason", "") == "" for x in r])
z, qw, qx, qy, qz = f("pos_z"), f("qw"), f("qx"), f("qy"), f("qz")
px, py = f("pos_x"), f("pos_y")

# Pitch about the lateral axis. The arcsin form gives a sane +/-14 deg band on
# these logs; the atan2 roll-form gives +/-60 deg, which is roll, not pitch.
pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
# theta = -servo * pi/2 in JAIOut units, and the frozen command's gain of 0.5
# (sim_servo_limit_rad 0.7854) gives theta_deg = -servo * 90. np.degrees() here
# is wrong by pi/2 and sign-inverted.
servo = -(f("servo1") + f("servo2")) / 2.0 * 90.0
thrust = (f("u_thr1") + f("u_thr2")) / 2.0

bn = int(0.5 / DT)
onset = None
for i in range(bn, len(z) - 5):
    if not live[i]:
        continue
    if z[i] - np.median(z[max(0, i - bn):i]) >= 0.02:
        onset = i
        break
if onset is None:
    sys.exit("no climb onset found in %s" % SRC)

end = min(len(z) - 1, onset + int(6.0 / DT))
peak = onset + int(np.argmax(z[onset:end + 1]))
s = max(0, onset - int(1.0 / DT))
base = float(np.median(z[max(0, onset - bn):onset]))

t = (np.arange(s, peak + 1) - onset) * DT
h = z[s:peak + 1] - base
print("%s onset=%.2fs peak=%.2fs  gain=%.3f m  dur=%.2f s"
      % (SRC, onset * DT, peak * DT, h.max(), (peak - onset) * DT))
print("  travel %.2f m, mean power %.0f W"
      % (np.sum(np.hypot(np.diff(px[onset:peak + 1]), np.diff(py[onset:peak + 1]))),
         np.mean(f("watts")[onset:peak + 1])))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))
panels = [(h, "(a) Height gained", r"$\Delta z$ (m)", ramp[0]),
          (pitch[s:peak + 1], "(b) Body pitch", r"$\theta$ (deg)", ramp[1]),
          (servo[s:peak + 1], "(c) Servo angle", r"$\sigma$ (deg)", ramp[2]),
          (thrust[s:peak + 1], "(d) Propeller command", "norm.", "#c0392b")]

fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.85))
for a_, (yv, title, ylab, col) in zip(ax, panels):
    a_.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=1)
    a_.plot(t, yv, color=col, lw=1.0, zorder=3)
    a_.set_title(title, fontsize=8, loc="left", pad=3)
    a_.set_ylabel(ylab, labelpad=2)
    a_.set_xlabel("time from climb onset (s)", labelpad=2)
    a_.set_xlim(t[0], t[-1])
    a_.grid(alpha=0.25, lw=0.5)
    a_.tick_params(length=2, pad=1.5)
fig.tight_layout(pad=0.25, w_pad=0.9)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
