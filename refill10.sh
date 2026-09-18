#!/bin/bash
# Fig.4 top panel, fill to TEN checkpoints (k=0..9), arm by arm.
#
# Ordering is the whole point. breadth.sh went breadth-first across arms and
# fig4_fill.sh only ran k=0..4, so neither ever COMPLETED an arm. Here the
# queue is arm-major and depth-complete, so at any cut-off the maximum number
# of arms are finished to ten checkpoints and are publishable.
#
# Priority is cheapest-first: ctm05 needs 13 evals, ct050 20, ctm45 20.
#   ctm05 (T/W 0.31) done  ~1.1 h
#   ct050 (T/W 0.46) done  ~3.1 h
#   ctm45 (T/W 0.23) done  ~4.8 h
# h06 is already complete for every arm and is not queued.
#
# Guard is a line count, not [ -s ]. A killed eval can leave a short but
# nonempty CSV and [ -s ] would skip it forever. A finished run has ~201 lines.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export IL=../../isaaclab/IsaacLab/isaaclab.sh
export B=logs/co_rl/doublebee_velocity/tqc
export LOG=sweep_logs/switch/refill10.log
export CONST=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo
mkdir -p abl_seeded sweep_logs/switch

one () {                     # "TAG HOLD H k"
  set -- $1; TAG=$1; HOLD=$2; H=$3; k=$4
  RUN=$(ls -dt $B/*_abl_${TAG} | head -1)
  CK=($(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10))
  C=${CK[$k]}
  OUT="abl_seeded/climb_${TAG}_h${H}_k${k}_${C}.csv"
  N=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  [ "$N" -gt 100 ] && return
  env DOUBLEBEE_HOLD_ACTION=$HOLD timeout 3000 $IL -p scripts/paper/eval_climb.py \
    --task $CONST --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
    --episodes 200 --num_envs 64 --seed $((1000 * 10#$H + k)) --out "$OUT" 2>&1 \
    | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
  echo "[refill] $TAG h$H k$k $(date -u +%H:%M:%S)" >> $LOG
}
export -f one

echo "[refill] START $(date -u)" >> $LOG
for TAG_HOLD in "ctm05 -0.05" "ct050 0.5" "ctm45 -0.45"; do
  for H in 03 04 05 07; do
    for k in 0 1 2 3 4 5 6 7 8 9; do echo "$TAG_HOLD $H $k"; done
  done
done | xargs -P 3 -I{} bash -c 'one "{}"'
echo "[refill] ALL DONE $(date -u)" >> $LOG
