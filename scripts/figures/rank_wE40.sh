#!/bin/bash
# Rank actfix_wE40 checkpoints by 6 cm clearance so the video uses the run's
# BEST climber, not its last one. Final-checkpoint performance says little here
# because reach swings 15-25 points between checkpoints.
#
# 64 episodes is enough to rank; it is not the paper's 200-episode protocol and
# these numbers must not be quoted anywhere.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export IL=../../isaaclab/IsaacLab/isaaclab.sh
export B=logs/co_rl/doublebee_velocity/tqc
export RUN=$B/2026-09-14_23-03-17_actfix_wE40
export TASK=Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo
export OUT=rank40
export LOG=$OUT/rank.log
mkdir -p $OUT
: > $LOG

one () {
  C=$1
  F="$OUT/c${C}.csv"
  N=$(wc -l < "$F" 2>/dev/null || echo 0)
  [ "$N" -gt 30 ] && return
  [ -f "$RUN/model_${C}.pt" ] || { echo "[r] MISSING $C" >> $LOG; return; }
  timeout 900 $IL -p scripts/paper/eval_climb.py \
    --task $TASK --checkpoint "$RUN/model_${C}.pt" \
    --step-height 0.06 --episodes 64 --num_envs 64 --seed 6000 \
    --out "$F" >> $OUT/c${C}.log 2>&1
  echo "[r] $C done $(date -u +%H:%M:%S)" >> $LOG
}
export -f one

echo "[r] START $(date -u)" >> $LOG
for C in 1500 1800 2100 2400 2700 3000 3300 3600 3900 3999; do echo $C; done \
  | xargs -P 3 -I{} bash -c 'one {}'
echo "[r] ALL DONE $(date -u)" >> $LOG

python3 - <<'PYEOF' >> $LOG
import csv, glob, os, re
rows = []
for f in sorted(glob.glob("rank40/c*.csv")):
    m = re.search(r"c(\d+)\.csv", f)
    r = list(csv.DictReader(open(f)))
    if not m or not r:
        continue
    clears = 100.0 * sum(float(x["max_gain_m"]) >= 0.06 for x in r) / len(r)
    gain = sorted(float(x["max_gain_m"]) for x in r)
    rows.append((clears, int(m.group(1)), len(r), gain[len(gain)//2], gain[-1]))
rows.sort(reverse=True)
print("\n=== RANKED (64 eps, 6 cm) ===")
print("%5s %8s %6s %9s %8s" % ("rank", "ckpt", "eps", "clears %", "gain p50/max"))
for i, (c, ck, n, p50, mx) in enumerate(rows, 1):
    print("%5d %8d %6d %8.1f   %.3f/%.3f" % (i, ck, n, c, p50, mx))
PYEOF
