"""Sim thrust modulation, from play.py --log_policy_io.

total_thrust is the simulator's ACHIEVED thrust, from actual propeller joint
speed, not the command. Episode chosen on: two distinct ascents, upright at the
end, elevated thrust through each crossing.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "sim/policy_io_hE4.csv"
OUT = os.path.expanduser("~/Downloads/DoubleBee_sim_thrust_modulation")
DT, LEARNED, CEIL = 0.02, "#c0392b", 15.8

r = list(csv.DictReader(open(SRC)))
f = lambda k: np.array([float(x[k]) for x in r])
y, z, T, a3 = f("pos_y"), f("pos_z"), f("total_thrust"), f("action_3")
qw_, qx, qy, qz_ = f("qw"), f("qx"), f("qy"), f("qz")
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))

b = np.flatnonzero((np.abs(np.diff(y)) > 0.8) | (np.abs(np.diff(z)) > 0.05)) + 1
eps = [e for e in np.split(np.arange(len(r)), b) if e.size >= 60]


def sm(a, n=9):
    p = n // 2
    return np.convolve(np.pad(a, p, mode="edge"), np.ones(n) / n,
                       mode="valid")[:len(a)]


def analyse(e):
    # baseline from ticks 10-25: the first few after a reset still settle
    h = z[e] - np.median(z[e][10:25])
    dz = sm(np.gradient(h, DT))
    rise = dz > 0.04
    d = np.diff(rise.astype(int))
    st, en = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if rise[0]:
        st = np.r_[0, st]
    if rise[-1]:
        en = np.r_[en, e.size - 1]
    segs = [(a, c) for a, c in zip(st, en) if c - a >= 4 and h[c] - h[a] > 0.02]
    return h, dz, segs


print("%3s %6s %7s %5s %8s %s" % ("ep", "gain", "endtilt", "asc", "ratio", "ascents"))
cands = []
for i, e in enumerate(eps):
    h, dz, segs = analyse(e)
    if len(segs) < 2 or tilt[e][-1] > 30:
        continue
    climb = h > 0.03
    ratio = T[e][climb].mean() / max(T[e][~climb].mean(), 1e-6)
    desc = "  ".join("%.3f->%.3f @ %.0fN" % (h[a], h[c], T[e][a:c].mean())
                     for a, c in segs)
    print("%3d %6.3f %7.1f %5d %8.2f  %s" % (i, h.max(), tilt[e][-1], len(segs),
                                             ratio, desc))
    cands.append((ratio, i, e))

ratio, best, e = max(cands)

# Window ends at peak height, matching the hardware protocol of Sec. V-E
# (trials_final/NOTES.md: "policy takeover -> peak height"). The settling that
# follows the peak is what that convention excludes on every hardware trial.
# Cut BEFORE analyse(), so h/dz/segs and every derived array share one length.
_hf = z[e] - np.median(z[e][10:25])
e = e[:int(np.argmax(_hf)) + 1]
print("window cut at peak: %d samples, %.2f s, gain %.1f cm"
      % (e.size, e.size * DT, 100 * _hf[:e.size].max()))
h, dz, segs = analyse(e)
print("\nchosen ep %d: gain %.3f m, %d ascents, end tilt %.1f deg, ratio %.2f"
      % (best, h.max(), len(segs), tilt[e][-1], ratio))
for k, (a, c) in enumerate(segs):
    print("  ascent %d: %.2f-%.2f s, rise %.3f m, thrust %.1f N"
          % (k + 1, a * DT, c * DT, h[c] - h[a], T[e][a:c].mean()))

t = np.arange(e.size) * DT
vy = sm(np.gradient(y[e], DT))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7})
ramp = plt.cm.viridis(np.linspace(0.05, 0.60, 3))
# Body lean along the direction of travel (+y in sim). The aerospace pitch
# convention used for the hardware logs reads ~0 here because forward is +y.
_w, _x, _y, _z = qw_[e], qx[e], qy[e], qz_[e]
lean = np.degrees(np.arctan2(2 * (_y * _z - _w * _x), 1 - 2 * (_x * _x + _y * _y)))
panels = [(vy, "(a) Forward speed", r"$v$ (m s$^{-1}$)", ramp[0]),
          (h, "(b) Height gained", r"$\Delta z$ (m)", ramp[1]),
          (lean, "(c) Pitch", r"$\theta$ (deg)", ramp[2]),
          (T[e], "(d) Thrust", r"$T$ (N)", LEARNED)]
fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.85))
for a_, (yv, title, ylab, col) in zip(ax, panels):
    for k, (s0, s1) in enumerate(segs):
        a_.axvspan(t[s0], t[s1], color="0.55", alpha=0.22, lw=0, zorder=0)
    a_.plot(t, yv, color=col, lw=1.0, zorder=3)
    a_.set_title(title, fontsize=8, loc="left", pad=3)
    a_.set_ylabel(ylab, labelpad=2)
    a_.set_xlabel("time (s)", labelpad=2)
    a_.set_xlim(0, t[-1])
    a_.grid(alpha=0.25, lw=0.5)
    a_.tick_params(length=2, pad=1.5)
    lo, hi = yv.min(), yv.max()
    a_.set_ylim(lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo))
    ytx = a_.get_ylim()[1] - 0.05 * (a_.get_ylim()[1] - a_.get_ylim()[0])
    for k, (s0, s1) in enumerate(segs):
        a_.text(t[(s0 + s1) // 2], ytx, str(k + 1), color="0.25", ha="center",
                va="top", fontsize=6.5)
fig.tight_layout(pad=0.25, w_pad=0.9)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight")
print("wrote", OUT + ".pdf")
