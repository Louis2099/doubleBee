#!/bin/bash
# Faithful decoupled baseline, pitch-setpoint sweep at 6 cm.
#
# theta_desired = -LEAN_MAX * step_ahead, so LEAN_MAX in radians is the sweep
# parameter the paper asks for: {0, -20, -40, -60, -80} degrees. Sweeping rather
# than picking one value is what turns the IROS anecdote (-80 climbs, 0 fails)
# into the controller's actual frontier, and it is the answer to "the baseline
# was undertuned".
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
MODE="${1:-faithful}"
mkdir -p pid_eval

for DEG in 0 20 40 60 80; do
  RAD=$(python3 -c "import math;print('%.4f' % math.radians($DEG))")
  OUT="pid_eval/climb_pid_${MODE}_d${DEG}_h06.csv"
  [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
  echo "===== $MODE, theta_desired = -$DEG deg (LEAN_MAX=$RAD) ====="
  DOUBLEBEE_BASELINE=$MODE DOUBLEBEE_LEAN_MAX=$RAD \
  DOUBLEBEE_YAW_SIGN=-1 DOUBLEBEE_SKIP_POLICY=1 \
  $IL -p scripts/co_rl/play_dctrl.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --num_envs 64 --headless \
    --step_height 0.06 --climb_episodes 200 \
    --climb_out "$OUT" 2>&1 \
    | grep -aE "^\[BASELINE\]|^wrote|cleared|Traceback|RuntimeError" | head -20
done
echo "[pid] DONE $MODE $(date -u)"
ls -la pid_eval/ | tail -6
