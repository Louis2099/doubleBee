#!/bin/bash
# Pool the sweep's best switch settings over all 10 checkpoints (same protocol as
# abl_h), because a single-checkpoint 200-episode cell can swing 15-25 points.
# 6 cm first. Output: abl_confirm/climb_<tag>_h<HH>_<ckpt>.csv
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo
B=logs/co_rl/doublebee_velocity/tqc
LOG=sweep_logs/switch/confirm.log
mkdir -p abl_confirm
arm () {  # tag run_suffix low high thresh latch
  TAG=$1; RUN=$(ls -dt $B/*_abl_$2 | head -1); LOW=$3; HI=$4; TH=$5; LA=$6
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  echo "[confirm] $TAG low=$LOW high=$HI thresh=$TH latch=$LA ckpts $(echo $CK)  $(date -u)" >> $LOG
  for H in 06 07 05 04 03; do
    for C in $CK; do
      OUT="abl_confirm/climb_${TAG}_h${H}_${C}.csv"
      [ -s "$OUT" ] && continue
      DOUBLEBEE_SWITCH_LOW=$LOW DOUBLEBEE_SWITCH_HIGH=$HI DOUBLEBEE_SWITCH_THRESH=$TH DOUBLEBEE_SWITCH_LATCH=$LA \
      $IL -p scripts/paper/eval_climb.py --task $TASK --checkpoint "$RUN/model_${C}.pt" \
        --step-height "0.$H" --episodes 200 --num_envs 64 --out "$OUT" 2>&1 \
        | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
    done
    echo "[confirm] $TAG done at $H cm  $(date -u)" >> $LOG
  done
}
arm swA3t04 swA3 -0.05 1.0 0.04 3.0
arm swB3best swB3 0.50 1.0 0.01 1.5
echo "[confirm] ALL DONE $(date -u)" >> $LOG
