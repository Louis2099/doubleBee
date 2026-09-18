#!/bin/bash
# Fig.4 top panel, fill to a uniform 5 checkpoints (k=0..4).
#
# Work is split by CELL across 3 workers, not by arm, so ct050 stops being the
# serial critical path the way it was under breadth.sh.
#
# Guard is a line count, not [ -s ]. Killing breadth.sh mid-eval can leave a
# short-but-nonempty CSV, and [ -s ] would skip it forever. A finished run has
# ~201 lines (200 episodes + header), so anything under 100 is re-run.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export IL=../../isaaclab/IsaacLab/isaaclab.sh
export B=logs/co_rl/doublebee_velocity/tqc
export LOG=sweep_logs/switch/fill5.log
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
  echo "[fill] $TAG h$H k$k $(date -u +%H:%M:%S)" >> $LOG
}
export -f one

echo "[fill] START $(date -u)" >> $LOG
# h04 and h03 first: those are the emptiest cells and the critical path.
for TAG_HOLD in "ct050 0.5" "ctm45 -0.45" "ctm05 -0.05"; do
  for H in 04 03 05; do for k in 0 1 2 3 4; do echo "$TAG_HOLD $H $k"; done; done
done | xargs -P 3 -I{} bash -c 'one "{}"'
echo "[fill] ALL DONE $(date -u)" >> $LOG
