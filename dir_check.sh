#!/bin/bash
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p dir_check
for WS in 1 -1; do for YS in 1 -1; do
  TAG="ws${WS}_ys${YS}"
  OUT="dir_check/${TAG}.csv"
  [ -s "$OUT" ] && { echo "skip $TAG"; continue; }
  DOUBLEBEE_BASELINE=augmented DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
  DOUBLEBEE_YAW_SIGN=$YS DOUBLEBEE_SERVO_FF_SIGN=-1 DOUBLEBEE_SERVO_BIAS_SIGN=-1 \
  DOUBLEBEE_WHEEL_SIGN=$WS DOUBLEBEE_V_MAX=0.60 DOUBLEBEE_V_STEP=0.60 \
  DOUBLEBEE_T_STEP_BOOST=3.0 DOUBLEBEE_STEP_SIGMA_BLEND=0.8 DOUBLEBEE_LEAN_MAX=-0.6981 \
  timeout 900 $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 8 --headless --step_height 0.06 \
    --climb_episodes 8 --climb_out "$OUT" > /tmp/dir_${TAG}.log 2>&1
  python3 -c "
import csv,statistics as st
try:
    r=list(csv.DictReader(open(\"$OUT\")))
    t=[float(x[\"toward_goal_m\"]) for x in r]
    d=[float(x[\"max_disp_m\"]) for x in r]
    g=[float(x[\"max_gain_m\"]) for x in r]
    print(\"  %-12s toward_goal mean %+6.2f m (max %+5.2f)  disp %4.2f  gain max %.3f  n=%d\"
          % (\"$TAG\", st.mean(t), max(t), st.mean(d), max(g), len(r)))
except Exception as e: print(\"  $TAG FAILED:\", e)
"
done; done
echo "[dircheck] DONE $(date -u)"
