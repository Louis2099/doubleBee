#!/bin/bash
# Settle the two servo sign conventions by experiment, as the baseline file has
# said to do since it was written. Metric: how long the robot survives. The
# faithful controller balanced on hardware (Decouple 2 held level pitch to
# 3.6 deg sd), so the correct signs should keep episodes long.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p pid_eval
for FF in 1 -1; do
  for BI in 1 -1; do
    OUT="pid_eval/sign_ff${FF}_bi${BI}.csv"
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    DOUBLEBEE_BASELINE=faithful DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
    DOUBLEBEE_YAW_SIGN=-1 DOUBLEBEE_LEAN_MAX=0.0 \
    DOUBLEBEE_SERVO_FF_SIGN=$FF DOUBLEBEE_SERVO_BIAS_SIGN=$BI \
    $IL -p scripts/co_rl/play_dctrl.py \
      --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
      --num_envs 16 --headless --step_height 0.06 \
      --climb_episodes 16 --climb_out "$OUT" > /tmp/sign_${FF}_${BI}.log 2>&1
    N=$(python3 -c "
import csv,sys
try:
    r=list(csv.DictReader(open('$OUT')))
    s=[float(x['steps']) for x in r]
    print('mean %5.1f steps  max %5.1f  (n=%d)' % (sum(s)/len(s), max(s), len(s)))
except Exception as e: print('no data:', e)
")
    echo "ff=$FF bias=$BI -> $N"
  done
done
echo "[sign] DONE $(date -u)"
