"""What fraction of the reward is the energy penalty?

    python3 energy_share.py logs/co_rl/doublebee_velocity/tqc/*_hE*

WHY. "$w_E=4$" tells a reader nothing. "The energy term is X% of total positive
reward" is interpretable, and it is the number a reviewer computes themselves
when they compare the reward table against a title claiming energy awareness.
The workshop setting measured 0.48%, which is what made the claim look
unsupported.

Episode_Reward/* in these logs are WEIGHTED episodic sums (on_policy_runner.py
writes weighted_value), so they can be compared directly. We average each term
over the final --last iterations, split into positive and negative, and report

    share = |energy_consumption| / sum(positive terms)

plus where the energy term ranks among the penalties.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

PREFIX = "Episode_Reward/"
ENERGY = PREFIX + "energy_consumption"


def term_means(run_dir, last):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith(PREFIX):
            continue
        ev = ea.Scalars(tag)
        if not ev:
            continue
        vals = [e.value for e in ev][-last:]
        out[tag] = float(np.mean(vals))
    return out, (ev[-1].step if ev else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--last", type=int, default=300,
                   help="average over this many final logged points")
    a = p.parse_args()

    dirs = []
    for r in a.runs:
        dirs.extend(sorted(glob.glob(r)) or [r])
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit("no run directories matched")

    print("%-8s %6s %10s %10s %9s %9s  %s"
          % ("run", "iter", "pos_sum", "neg_sum", "energy", "share%", "rank"))
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        m = re.search(r"_(hE\d+|abl_\w+)$", name)
        name = m.group(1) if m else name
        try:
            t, step = term_means(d, a.last)
        except Exception as ex:
            print("%-8s  unreadable (%r)" % (name, ex))
            continue
        if ENERGY not in t:
            print("%-8s  no energy_consumption term" % name)
            continue
        pos = sum(v for v in t.values() if v > 0)
        negs = {k: -v for k, v in t.items() if v < 0}
        e = negs.get(ENERGY, 0.0)
        order = sorted(negs, key=lambda k: -negs[k])
        rank = order.index(ENERGY) + 1 if ENERGY in order else 0
        print("%-8s %6d %10.2f %10.2f %9.3f %8.1f%%  %d/%d"
              % (name, step, pos, sum(negs.values()), e,
                 100.0 * e / pos if pos else float("nan"), rank, len(negs)))


if __name__ == "__main__":
    main()
