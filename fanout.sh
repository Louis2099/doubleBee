#!/bin/bash
# Fan out the remaining Fig.4 arm-height pairs as concurrent workers.
#
# optionC_driver.sh walks heights sequentially per arm (3 workers), which is why
# the remaining 95 evaluations project to 07:28. Each (arm, height) pair is
# independent, so run the nine outstanding pairs concurrently instead.
#
# Same seed convention (1000*H + k), same output directory and filenames, and
# the same [ ! -s ] guard, so these are indistinguishable from the driver's own
# output and cannot collide with it when it reaches those heights later.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
LOG=sweep_logs/switch/fanout.log
CONST=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo
mkdir -p abl_seeded sweep_logs/switch

pair () {   # tag hold height
  TAG=$1; HOLD=$2; H=$3
  RUN=$(ls -dt $B/*_abl_${TAG} 2>/dev/null | head -1)
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  k=0
  for C in $CK; do
    SEED=$((1000 * 10#$H + k))
    OUT="abl_seeded/climb_${TAG}_h${H}_k${k}_${C}.csv"
    if [ ! -s "$OUT" ]; then
      env DOUBLEBEE_HOLD_ACTION=$HOLD timeout 3000 $IL -p scripts/paper/eval_climb.py \
        --task $CONST --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
        --episodes 200 --num_envs 64 --seed $SEED --out "$OUT" 2>&1 \
        | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
    fi
    k=$((k + 1))
  done
  echo "[fan] $TAG h$H DONE $(date -u)" >> $LOG
}

echo "[fan] START $(date -u)" >> $LOG
for H in 05 04 03; do
  pair ct050  0.5   $H &
  pair ctm05  -0.05 $H &
  pair ctm45  -0.45 $H &
done
wait
echo "[fan] ALL DONE $(date -u)" >> $LOG
