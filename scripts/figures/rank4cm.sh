#!/bin/bash
# Rank BOTH runs at 4 cm, the step height the solo clip uses.
# The earlier wE40 ranking was at 6 cm, so it does not transfer.
# 64 episodes: enough to order checkpoints, NOT the paper's 200-episode
# paired-seed protocol. These numbers are for picking a clip, never to quote.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export IL=../../isaaclab/IsaacLab/isaaclab.sh
export B=logs/co_rl/doublebee_velocity/tqc
export TASK=Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo
export OUT=rank4
export LOG=$OUT/rank.log
mkdir -p $OUT
: > $LOG

one () {
  set -- $1; TAG=$1; RUN=$2; C=$3
  F="$OUT/${TAG}_${C}.csv"
  N=$(wc -l < "$F" 2>/dev/null || echo 0)
  [ "$N" -gt 30 ] && return
  [ -f "$RUN/model_${C}.pt" ] || { echo "[r] MISSING $TAG $C" >> $LOG; return; }
  timeout 900 $IL -p scripts/paper/eval_climb.py \
    --task $TASK --checkpoint "$RUN/model_${C}.pt" \
    --step-height 0.04 --episodes 64 --num_envs 64 --seed 4000 \
    --out "$F" >> $OUT/${TAG}_${C}.log 2>&1
  echo "[r] $TAG $C done $(date -u +%H:%M:%S)" >> $LOG
}
export -f one

H=$B/energy_abl/2026-09-06_23-10-57_hE4
W=$B/2026-09-14_23-03-17_actfix_wE40

echo "[r] START $(date -u)" >> $LOG
{
  for C in 2500 3000 3500 4000 4500 5000 5500 5899; do echo "hE4 $H $C"; done
  for C in 2400 2700 3000 3300 3600;                do echo "wE40 $W $C"; done
} | xargs -P 3 -I{} bash -c 'one "{}"'
echo "[r] ALL DONE $(date -u)" >> $LOG

python3 - <<'PYEOF' >> $LOG
import csv, glob, os, re
rows = []
for f in sorted(glob.glob("rank4/*.csv")):
    m = re.search(r"(hE4|wE40)_(\d+)\.csv", os.path.basename(f))
    r = list(csv.DictReader(open(f)))
    if not m or not r:
        continue
    clears = 100.0 * sum(float(x["max_gain_m"]) >= 0.04 for x in r) / len(r)
    g = sorted(float(x["max_gain_m"]) for x in r)
    rows.append((clears, m.group(1), int(m.group(2)), len(r), g[len(g)//2], g[-1]))
rows.sort(reverse=True)
print("\n=== RANKED at 4 cm (64 eps) ===")
print("%4s %6s %7s %5s %9s %s" % ("rk", "run", "ckpt", "eps", "clears %", "gain p50/max"))
for i, (c, run, ck, n, p50, mx) in enumerate(rows, 1):
    print("%4d %6s %7d %5d %8.1f   %.3f/%.3f" % (i, run, ck, n, c, p50, mx))
for run in ("hE4", "wE40"):
    best = [r for r in rows if r[1] == run]
    if best:
        print("BEST %s: model_%d at %.1f%%" % (run, best[0][2], best[0][0]))
PYEOF
