#!/bin/bash
# Find a decoupled-baseline configuration that actually climbs a 6 cm step.
#
# Three levers, in increasing order of deviation from [7]:
#   V_MAX          approach speed. NOT a deviation: the velocity command is an
#                  INPUT to Eq. (23). Hardware Decouple 1 peaked at 0.65 m/s.
#   T_STEP_BOOST   thrust multiplier at a detected step. A declared deviation,
#                  and the same affordance db_inference.py already uses on the
#                  real robot (--prop_scale_step 4-5).
#   SIGMA_BLEND    pull thrust toward vertical at a step so the boost unloads
#                  the wheels instead of pushing sideways. Declared deviation.
#
# Signs pinned from the 2026-09-11 matrix: both -1 (our USD's servo-positive is
# inverted relative to the paper figure). EQ20 off, as the paper declares.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p pid_eval

run () {   # mode vmax boost blend lean tag
  local OUT="pid_eval/tune_$6.csv"
  [ -s "$OUT" ] && { echo "skip $6"; return; }
  DOUBLEBEE_BASELINE=$1 DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
  DOUBLEBEE_YAW_SIGN=-1 DOUBLEBEE_SERVO_FF_SIGN=-1 DOUBLEBEE_SERVO_BIAS_SIGN=-1 \
  DOUBLEBEE_V_MAX=$2 DOUBLEBEE_V_STEP=$2 \
  DOUBLEBEE_T_STEP_BOOST=$3 DOUBLEBEE_STEP_SIGMA_BLEND=$4 \
  DOUBLEBEE_LEAN_MAX=$5 \
  $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 32 --headless --step_height 0.06 \
    --climb_episodes 32 --climb_out "$OUT" > /tmp/tune_$6.log 2>&1
  python3 -c "
import csv
r=list(csv.DictReader(open('$OUT')))
g=[float(x['max_gain_m']) for x in r]; s=[float(x['steps']) for x in r]
n=len(r); up=sum(1 for x in g if x>=0.06)
print('  %-22s clears %2d/%2d (%3.0f%%)  maxgain p50=%.3f max=%.3f  steps %5.1f'
      % ('$6', up, n, 100*up/n, sorted(g)[n//2], max(g), sum(s)/n))
"
}

echo "=== baseline reference: augmented, stock speed, no boost ==="
run augmented 0.30 1.0 0.0 0.6981 aug_stock
echo "=== lever 1: faster approach (not a deviation) ==="
run augmented 0.60 1.0 0.0 0.6981 aug_v060
echo "=== lever 2: + step thrust boost ==="
run augmented 0.60 2.0 0.0 0.6981 aug_v060_b2
run augmented 0.60 3.0 0.0 0.6981 aug_v060_b3
echo "=== lever 3: + thrust held vertical at the step ==="
run augmented 0.60 3.0 0.8 0.6981 aug_v060_b3_s08
run augmented 0.60 2.0 0.8 0.6981 aug_v060_b2_s08
echo "=== and the faithful variant with the same help ==="
run faithful  0.60 3.0 0.8 0.6981 fai_v060_b3_s08
echo "[tune] DONE $(date -u)"
