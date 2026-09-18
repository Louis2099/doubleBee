#!/bin/bash
# Fig.4 top panel, breadth-first over checkpoints.
#
# Three workers (measured optimum: 3 gave ~12 evals/h, 12 gave ~10). One worker
# per arm. Each walks heights 05,04,03 taking checkpoints k=0..4 first, then
# k=5..9. So at any cut-off every cell has the same depth rather than some cells
# having 10 and others 0. Existing files are skipped by the [ ! -s ] guard.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
LOG=sweep_logs/switch/breadth.log
CONST=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo
mkdir -p abl_seeded sweep_logs/switch

arm () {   # tag hold
  TAG=$1; HOLD=$2
  RUN=$(ls -dt $B/*_abl_${TAG} 2>/dev/null | head -1)
  CK=($(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10))
  for PASS in "0 1 2 3 4" "5 6 7 8 9"; do
    for H in 05 04 03; do
      for k in $PASS; do
        C=${CK[$k]}
        SEED=$((1000 * 10#$H + k))
        OUT="abl_seeded/climb_${TAG}_h${H}_k${k}_${C}.csv"
        [ -s "$OUT" ] && continue
        env DOUBLEBEE_HOLD_ACTION=$HOLD timeout 3000 $IL -p scripts/paper/eval_climb.py \
          --task $CONST --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
          --episodes 200 --num_envs 64 --seed $SEED --out "$OUT" 2>&1 \
          | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
        echo "[bf] $TAG h$H k$k $(date -u +%H:%M:%S)" >> $LOG
      done
    done
    echo "[bf] $TAG PASS done $(date -u)" >> $LOG
  done
}

echo "[bf] START $(date -u)" >> $LOG
arm ct050  0.5   &
arm ctm05  -0.05 &
arm ctm45  -0.45 &
wait
echo "[bf] ALL DONE $(date -u)" >> $LOG
