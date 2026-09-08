"""Pull scalar curves out of the TQC runs' tensorboard files.

    # what is logged
    python3 tb_curves.py --list logs/co_rl/doublebee_velocity/tqc/*_hE0

    # dump chosen tags for every arm into one tidy CSV
    python3 tb_curves.py -o curves.csv --tags "Metrics/success" "Train/mean_reward" \
        logs/co_rl/doublebee_velocity/tqc/*_hE*

WHY. Measured 2026-09-08, hE4 degrades across its own final checkpoints --
median max_gain 0.061 -> 0.055 -> 0.051 m at iterations 5700/5800/5899, with
energy falling alongside. That is not checkpoint noise, it is the policy sliding
into the do-nothing optimum the energy penalty admits: spend nothing, climb
nothing. hE8 is the completed version of the same collapse.

A per-checkpoint eval cannot show this; the training curve can. Averaging the
last N checkpoints would actively hide it, because it smooths a trend rather
than noise and reports a number describing no policy that ever existed.

Output is tidy: run,tag,step,value -- one row per logged point.
"""
import argparse
import csv
import glob
import os
import re
import sys


def accumulate(run_dir):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories (globs are fine)")
    p.add_argument("--list", action="store_true", help="print available tags and exit")
    p.add_argument("--tags", nargs="*", default=None,
                   help="substring match; default dumps every scalar tag")
    p.add_argument("-o", "--out", default="curves.csv")
    a = p.parse_args()

    dirs = []
    for r in a.runs:
        dirs.extend(sorted(glob.glob(r)) or [r])
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit("no run directories matched")

    if a.list:
        for d in dirs:
            ea = accumulate(d)
            tags = ea.Tags().get("scalars", [])
            print("\n=== %s  (%d scalar tags) ===" % (os.path.basename(d), len(tags)))
            for t in sorted(tags):
                print("   ", t)
        return

    rows = []
    for d in dirs:
        run = os.path.basename(d.rstrip("/"))
        m = re.search(r"_(hE\d+|abl_\w+)$", run)
        run = m.group(1) if m else run
        ea = accumulate(d)
        tags = ea.Tags().get("scalars", [])
        keep = [t for t in tags
                if a.tags is None or any(s.lower() in t.lower() for s in a.tags)]
        if not keep:
            print("  %s: no tag matched" % run)
            continue
        for t in keep:
            for ev in ea.Scalars(t):
                rows.append({"run": run, "tag": t,
                             "step": ev.step, "value": ev.value})
        print("  %s: %d tags, %d points" % (run, len(keep), len(rows)))

    if not rows:
        sys.exit("nothing extracted")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
