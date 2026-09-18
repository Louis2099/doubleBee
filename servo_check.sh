#!/bin/bash
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p servo_check
for FF in 1 -1; do for BI in 1 -1; do for TH in 1 -1; do
  TAG="ff${FF}_bi${BI}_th${TH}"
  OUT="servo_check/${TAG}.csv"
  [ -s "$OUT" ] && { echo "skip $TAG"; continue; }
  DOUBLEBEE_BASELINE=augmented DOUBLEBEE_EQ20=0 DOUBLEBEE_SKIP_POLICY=1 \
  DOUBLEBEE_YAW_SIGN=1 DOUBLEBEE_WHEEL_SIGN=-1 \
  DOUBLEBEE_SERVO_FF_SIGN=$FF DOUBLEBEE_SERVO_BIAS_SIGN=$BI DOUBLEBEE_THETA_SIGN=$TH \
  DOUBLEBEE_V_MAX=0.60 DOUBLEBEE_V_STEP=0.60 \
  DOUBLEBEE_T_STEP_BOOST=3.0 DOUBLEBEE_STEP_SIGMA_BLEND=0.8 DOUBLEBEE_LEAN_MAX=-0.6981 \
  timeout 900 $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 8 --headless --step_height 0.06 \
    --climb_episodes 8 --climb_out "$OUT" > /tmp/servo_${TAG}.log 2>&1
  python3 -c "
import csv,statistics as st
try:
    r=list(csv.DictReader(open(\"$OUT\")))
    t=[float(x[\"toward_goal_m\"]) for x in r]; g=[float(x[\"max_gain_m\"]) for x in r]
    d=[float(x[\"max_disp_m\"]) for x in r]; c=sum(int(x[\"cleared\"]) for x in r)
    print(\"  %-16s toward %+5.2f m (max %+5.2f) | gain p50 %.3f max %.3f | disp %4.2f | cleared %d/%d\"
          % (\"$TAG\", st.mean(t), max(t), sorted(g)[len(g)//2], max(g), st.mean(d), c, len(r)))
except Exception as e: print(\"  $TAG FAILED:\", e)
"
done; done; done
echo "[servocheck] DONE $(date -u)"
