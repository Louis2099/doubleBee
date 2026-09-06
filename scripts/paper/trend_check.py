"""Is this run on track to learn, or is it dead? Reads the tensorboard events.

Success rate is useless before ~3000 iterations on this task -- it is still 0
while the policy is learning to stay upright. The signals that LEAD the
curriculum are tilt terminations falling and episode length rising. This prints
those as trends over the last third of the run against the middle third, so a
flat run is distinguishable from a slow one.

Run it on the training box:
    python3 trend_check.py                     # newest run under logs/
    python3 trend_check.py <path/to/run_dir>
"""
import glob
import os
import sys

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# tag -> (label, direction we want)  +1 = up is good, -1 = down is good
WATCH = [
    ("Curriculum/terrain_levels",                       "curriculum level", +1),
    ("Episode_Constraint/tilt",                         "tilt terminations", -1),
    ("Train/mean_episode_length",                       "episode length", +1),
    ("Episode_Reward/reward_alive_upright",             "alive+upright", +1),
    ("Episode_Reward/reward_thrust_up_at_step",         "thrust up at step", +1),
    ("Episode_Reward/reward_thrust_recovery_under_lean", "thrust under lean", +1),
    ("Metrics/success/rate",                            "success rate", +1),
]


def series(ea, tag):
    try:
        ev = ea.Scalars(tag)
    except KeyError:
        return None, None
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root is None:
        cands = sorted(glob.glob("logs/*/*/"), key=os.path.getmtime)
        if not cands:
            sys.exit("no logs/*/*/ found -- pass the run directory explicitly")
        root = cands[-1]
    evs = sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents*"),
                          recursive=True), key=os.path.getmtime)
    if not evs:
        sys.exit("no event files under %s" % root)
    ea = event_accumulator.EventAccumulator(
        evs[-1], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    print("run: %s" % root)
    steps, _ = series(ea, WATCH[0][0])
    if steps is None or len(steps) < 30:
        avail = [t for t in ea.Tags()["scalars"] if "terrain" in t or "tilt" in t]
        sys.exit("not enough data yet (or tag names differ). candidates: %s" % avail[:5])
    n = len(steps)
    print("iterations logged: %d (latest %d)\n" % (n, steps[-1]))
    mid, late = slice(n // 3, 2 * n // 3), slice(2 * n // 3, n)

    verdict = []
    print("%-22s %10s %10s %10s   %s" % ("signal", "mid-run", "recent", "change", ""))
    print("-" * 68)
    for tag, label, want in WATCH:
        _, v = series(ea, tag)
        if v is None:
            print("%-22s %10s   (tag not found)" % (label, "-"))
            continue
        a, b = float(np.mean(v[mid])), float(np.mean(v[late]))
        d = b - a
        moving = (d * want) > 0 and abs(d) > 0.02 * max(abs(a), 1e-6)
        verdict.append(moving)
        print("%-22s %10.4f %10.4f %+10.4f   %s"
              % (label, a, b, d, "moving" if moving else "flat"))

    print()
    good = sum(verdict)
    if good >= 3:
        print("ON TRACK -- %d of %d precursors moving the right way. Let it run." % (good, len(verdict)))
    elif good >= 1:
        print("AMBIGUOUS -- %d of %d moving. Check again in an hour before deciding." % (good, len(verdict)))
    else:
        print("STALLED -- nothing is moving. Warm-start instead of waiting for 3000.")


if __name__ == "__main__":
    main()
