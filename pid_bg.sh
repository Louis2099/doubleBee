#!/bin/bash
# Bounded PID experiment. Runs ONLY after the switched-thrust arms have finished
# and been evaluated, so it never competes with them for the GPU.
#
# One question: with every sign convention corrected, can the decoupled
# controller reach a step and climb it at all? Six cells, 32 episodes each.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
echo "[pidbg] waiting for the arms + driver2 to finish  $(date -u)"
while pgrep -f "co_rl/train.py" >/dev/null 2>&1 || pgrep -f "switch_driver2" >/dev/null 2>&1; do
  sleep 300
done
echo "[pidbg] GPU free, starting  $(date -u)"
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p pid_bg
run () {  # mode boost blend lean tag
  local OUT="pid_bg/$5.csv"
  [ -s "$OUT" ] && { echo "skip $5"; return; }
  DOUBLEBEE_BASELINE=$1 DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
  DOUBLEBEE_YAW_SIGN=-1 DOUBLEBEE_SERVO_FF_SIGN=-1 DOUBLEBEE_SERVO_BIAS_SIGN=-1 \
  DOUBLEBEE_WHEEL_SIGN=-1 DOUBLEBEE_V_MAX=0.60 DOUBLEBEE_V_STEP=0.60 \
  DOUBLEBEE_T_STEP_BOOST=$2 DOUBLEBEE_STEP_SIGMA_BLEND=$3 DOUBLEBEE_LEAN_MAX=$4 \
  timeout 1800 $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 32 --headless --step_height 0.06 \
    --climb_episodes 32 --climb_out "$OUT" > /tmp/pidbg_$5.log 2>&1
  python3 -c "
import csv
try:
    r=list(csv.DictReader(open('$OUT')))
    g=[float(x['max_gain_m']) for x in r]; s=[float(x['steps']) for x in r]
    n=len(r); up=sum(1 for x in g if x>=0.06)
    print('  %-18s clears %2d/%2d (%3.0f%%)  gain p50=%.3f max=%.3f  steps %4.0f'
          % ('$5', up, n, 100*up/n, sorted(g)[n//2], max(g), sum(s)/n))
except Exception as e: print('  $5: no data (%s)' % e)
"
}
run augmented 3.0 0.8 -0.6981 aug_b3
run augmented 6.0 0.8 -0.6981 aug_b6
run augmented 6.0 0.8  0.6981 aug_b6_nosedown
run augmented 3.0 0.0 -0.6981 aug_b3_noblend
run faithful  6.0 0.8 -0.6981 fai_b6
run augmented 1.0 0.0 -0.6981 aug_stock
echo "[pidbg] DONE $(date -u)"
