#!/bin/bash
# Which way should the baseline lean at a step?
#
# play_dctrl sets theta_desired = -LEAN_MAX * step_ahead, so a POSITIVE
# LEAN_MAX commands NOSE DOWN at a step (theta<0 is nose down with
# THETA_SIGN=+1). The paper describes the climbing behaviour as leaning
# BACKWARD. A negative LEAN_MAX commands that instead.
#
# Signs pinned from the 2026-09-11 matrices: servo ff/bias -1, wheel -1.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p pid_eval
run () {  # lean boost blend tag
  local OUT="pid_eval/lean_$4.csv"
  [ -s "$OUT" ] && { echo "skip $4"; return; }
  DOUBLEBEE_BASELINE=augmented DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
  DOUBLEBEE_YAW_SIGN=-1 DOUBLEBEE_SERVO_FF_SIGN=-1 DOUBLEBEE_SERVO_BIAS_SIGN=-1 \
  DOUBLEBEE_WHEEL_SIGN=-1 DOUBLEBEE_V_MAX=0.60 DOUBLEBEE_V_STEP=0.60 \
  DOUBLEBEE_LEAN_MAX=$1 DOUBLEBEE_T_STEP_BOOST=$2 DOUBLEBEE_STEP_SIGMA_BLEND=$3 \
  $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 32 --headless --step_height 0.06 \
    --climb_episodes 32 --climb_out "$OUT" > /tmp/lean_$4.log 2>&1
  python3 -c "
import csv
r=list(csv.DictReader(open('$OUT')))
g=[float(x['max_gain_m']) for x in r]; s=[float(x['steps']) for x in r]
n=len(r); up=sum(1 for x in g if x>=0.06)
print('  %-14s clears %2d/%2d (%3.0f%%)  gain p50=%.3f max=%.3f  steps %5.0f'
      % ('$4', up, n, 100*up/n, sorted(g)[n//2], max(g), sum(s)/n))
"
}
echo "=== lean direction, with step boost ==="
run  0.6981 3.0 0.8 nosedown_b3
run -0.6981 3.0 0.8 leanback_b3
echo "=== and without the boost, to isolate lean direction alone ==="
run -0.6981 1.0 0.0 leanback_nob
run  0.6981 1.0 0.0 nosedown_nob
echo "[lean] DONE $(date -u)"
