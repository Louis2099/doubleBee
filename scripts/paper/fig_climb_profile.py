"""Event-triggered average of what the policy does at a riser.

Replaces the single-episode trace (IROS'26 Fig. 4) with a distribution. Every
climb in every log is aligned on the instant the robot starts gaining height,
then the traces are averaged with an inter-quartile band.

Why event-triggered rather than one episode: a single trace shows that the
policy CAN spike thrust at a step; it cannot show that it RELIABLY does, and a
reviewer has no way to tell a representative run from a lucky one. Aligning
N climbs on t=0 and plotting median with IQR answers "what does this policy do
at a riser" as a claim about the policy instead of about one rollout.

Alignment: the first sample where the robot has risen >= --rise metres above
its pre-climb baseline, with the baseline taken as the median height over the
0.5 s before. That marker is robust to where the step actually sits, so no
step-position bookkeeping is needed and logs from different arena layouts can
be pooled.

Panels: forward speed, height gained, total thrust, servo angle.
The servo panel is the one IROS'26 Fig. 4 omits, and it is the actuator that
distinguishes this platform -- thrust magnitude alone cannot show whether the
propellers were pushing up or forward.

Works on hardware logs (db_inference.py --log_path) and, once you have rollouts,
on sim logs (play.py --log_policy_io, which now records pos/quat too).

    python3 fig_climb_profile.py hw_*.csv -o fig_climb.pdf
    python3 fig_climb_profile.py policy_io_env0.csv -o fig_climb_sim.pdf
"""
import argparse
import csv
import glob
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("need matplotlib")

DT = 0.02          # 50 Hz control loop, both sim and hardware


def read(path):
    """Load a log and normalise the columns we need across the two formats."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cols = rows[0].keys()
    f = lambda k: np.array([float(r[k]) for r in rows])

    if "pos_z" not in cols:
        return None
    out = {"z": f("pos_z")}

    # live mask: hardware logs gate the wheels; sim logs do not
    out["live"] = (np.array([r.get("gate_reason", "") == "" for r in rows])
                   if "gate_reason" in cols else np.ones(len(rows), bool))

    # forward speed from position (both formats have pos_x/pos_y)
    if "pos_x" in cols and "pos_y" in cols:
        x, y = f("pos_x"), f("pos_y")
        out["v"] = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y)) / DT])
    else:
        out["v"] = np.zeros(len(rows))

    # thrust: hardware logs the two channel commands, sim logs the total
    if "u_thr1" in cols and "u_thr2" in cols:
        out["thrust"] = (f("u_thr1") + f("u_thr2")) / 2.0
        out["thrust_label"] = "thrust command (norm.)"
    elif "total_thrust" in cols:
        out["thrust"] = f("total_thrust")
        out["thrust_label"] = "total thrust (N)"
    else:
        return None

    # servo: hardware logs the angle, sim logs the action
    if "servo1" in cols and "servo2" in cols:
        out["servo"] = np.degrees((f("servo1") + f("servo2")) / 2.0)
    elif "action_2" in cols:
        out["servo"] = np.degrees(f("action_2") * np.pi / 4)   # SERVO_POS_LIMIT_RAD
    else:
        out["servo"] = np.zeros(len(rows))
    return out


def climbs(d, rise, pre, post):
    """Windows around each climb onset. Returns list of index slices."""
    z, live = d["z"], d["live"]
    npre, npost = int(pre / DT), int(post / DT)
    base_n = int(0.5 / DT)
    events, i = [], base_n
    while i < len(z) - npost:
        if not live[i]:
            i += 1
            continue
        baseline = np.median(z[max(0, i - base_n):i])
        if z[i] - baseline >= rise:
            if i - npre >= 0 and live[i - npre:i + npost].all():
                events.append((i - npre, i + npost))
            i += npost          # don't re-trigger inside the same climb
        else:
            i += 1
    return events


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="CSV logs (globs ok)")
    p.add_argument("-o", "--out", default="fig_climb.pdf")
    p.add_argument("--rise", type=float, default=0.02,
                   help="height gain that marks climb onset, m (default 0.02)")
    p.add_argument("--pre", type=float, default=1.0, help="seconds before onset")
    p.add_argument("--post", type=float, default=1.5, help="seconds after onset")
    a = p.parse_args()

    paths = [q for pat in a.logs for q in sorted(glob.glob(pat))]
    stack = {k: [] for k in ("v", "z", "thrust", "servo")}
    label = "thrust"
    nfiles = 0
    for path in paths:
        d = read(path)
        if d is None:
            continue
        ev = climbs(d, a.rise, a.pre, a.post)
        if ev:
            nfiles += 1
        label = d.get("thrust_label", label)
        for s, e in ev:
            stack["v"].append(d["v"][s:e])
            stack["z"].append(d["z"][s:e] - np.median(d["z"][s:s + int(0.5 / DT)]))
            stack["thrust"].append(d["thrust"][s:e])
            stack["servo"].append(d["servo"][s:e])

    n = len(stack["v"])
    if n < 3:
        sys.exit("only %d climb events found across %d logs -- lower --rise, or "
                 "check that the logs contain actual climbs" % (n, len(paths)))

    t = np.arange(-int(a.pre / DT), int(a.post / DT)) * DT
    panels = [("v", "forward speed (m/s)"), ("z", "height gained (m)"),
              ("thrust", label), ("servo", "servo angle (deg)")]

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, (key, ylab) in zip(axes, panels):
        A = np.vstack(stack[key])
        med = np.median(A, axis=0)
        lo, hi = np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0)
        ax.fill_between(t, lo, hi, alpha=0.25, linewidth=0)
        ax.plot(t, med, linewidth=1.8)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.9)
        ax.set_xlabel("time relative to climb onset (s)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].set_title("n = %d climbs from %d runs" % (n, nfiles), loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote %s  (%d climb events, %d logs)" % (a.out, n, nfiles))
    print("median at onset: v=%.2f m/s  thrust=%.2f  servo=%.1f deg"
          % (np.median(np.vstack(stack["v"])[:, int(a.pre / DT)]),
             np.median(np.vstack(stack["thrust"])[:, int(a.pre / DT)]),
             np.median(np.vstack(stack["servo"])[:, int(a.pre / DT)])))


if __name__ == "__main__":
    main()
