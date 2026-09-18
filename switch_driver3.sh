#!/bin/bash
# Evaluate the axis-fixed switched arms (swA2, swB2).
#
# CRITICAL, and the reason this file exists rather than reusing eval_switch.sh:
# the switch parameters are read from the ENVIRONMENT at config construction
# (actions.py:1122-1126) and default to low -0.45 / high 1.0 / latch 0.5 s.
# eval_switch.sh exported none of them, so swA and swB were evaluated as a
# DIFFERENT controller than the one they trained under. Each arm's own values
# are exported here.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
LOG=sweep_logs/switch/driver3.log
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo

run_arm () {   # dir tag low high latch
  RUN="$1"; TAG="$2"; LOW="$3"; HIGH="$4"; LATCH="$5"
  echo "[d3] $TAG waiting for $RUN/model_5899.pt  $(date -u)" >> $LOG
  while [ ! -s "$RUN/model_5899.pt" ]; do sleep 120; done
  sleep 30
  echo "[d3] $TAG evaluating low=$LOW high=$HIGH latch=$LATCH  $(date -u)" >> $LOG
  CKPTS=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  for H in 03 04 05 06 07; do
    for C in $CKPTS; do
      OUT="abl_h/climb_${TAG}_h${H}_${C}.csv"
      if [ -s "$OUT" ]; then continue; fi
      DOUBLEBEE_SWITCH_LOW=$LOW DOUBLEBEE_SWITCH_HIGH=$HIGH DOUBLEBEE_SWITCH_LATCH=$LATCH \
      $IL -p scripts/paper/eval_climb.py --task $TASK \
        --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
        --episodes 200 --num_envs 64 --out "$OUT" 2>&1 \
        | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
    done
    echo "[d3] $TAG done at $H cm  $(date -u)" >> $LOG
  done
  echo "[d3] $TAG EVAL DONE  $(date -u)" >> $LOG
}

B=logs/co_rl/doublebee_velocity/tqc
run_arm $B/2026-09-12_14-27-48_abl_swA2 swA2 -0.05 1.0 3.0
run_arm $B/2026-09-12_14-27-58_abl_swB2 swB2 0.50 1.0 3.0
echo "[d3] ALL DONE $(date -u)" >> $LOG
