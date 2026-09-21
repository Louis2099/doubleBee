#!/bin/bash
# EARLY checkpoint only. No target balls. Diagonal camera so the terrain corner
# makes the triangle apex at the top of frame.
#
# Spawn: ONE spot at the back of the flat base. platform_width=3.0 so the base
# runs to +/-1.5, and the robots travel +y toward the steps, so y -1.4..-1.0 is
# the back edge and still on the base. z clamped to +/-0.08 keeps every patch at
# base level rather than on a step tread.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
SC=lab/doublebee/tasks/manager_based/locomotion/velocity/terrain_config/stair_config.py
PY=scripts/co_rl/play.py
CK=$B/energy_abl/2026-09-06_07-59-52_wE0/model_400.pt
OUT=vids/final
LOG=$OUT/early.log
mkdir -p $OUT
: > $LOG

# Hide the goal balls.
cp "$PY" "$PY.vidbak"
sed -i 's/^    if uses_target_command:$/    if uses_target_command and os.environ.get("DOUBLEBEE_NO_TARGET_BALLS", "0") in ("0", "", "false", "False"):/' "$PY"

# One spawn spot, back of the base.
cp "$SC" "$SC.videobak"
sed -i '183,192 s/num_patches=8/num_patches=4/; 183,192 s/x_range=(-0.5, 0.5)/x_range=(-0.25, 0.25)/; 183,192 s/y_range=(-0.5, 0.5)/y_range=(-1.4, -1.0)/; 183,192 s/z_range=(-0.5, 0.5)/z_range=(-0.08, 0.08)/' "$SC"
echo "--- spawn block:" >> $LOG
sed -n '183,192p' "$SC" >> $LOG

export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_NO_TARGET_BALLS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000

echo "[v] early start $(date -u +%H:%M:%S) ckpt=$(basename $CK)" >> $LOG
timeout 1500 $IL -p scripts/co_rl/play.py \
  --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
  --checkpoint "$CK" \
  --num_envs 15 --headless \
  --video --video_length 200 --render_res 1920 1080 \
  --cam_eye -6 -6 5.15 --cam_lookat 0 0 0 \
  >> $OUT/early_run.log 2>&1

NEW=$(find logs -name "*.mp4" -newermt "-25 min" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
if [ -n "$NEW" ]; then
  cp "$NEW" "$OUT/early.mp4"; mv "$NEW" "${NEW}.used"
  echo "[v] early ok $(stat -c%s "$OUT/early.mp4") bytes" >> $LOG
else
  echo "[v] early NO MP4" >> $LOG
  grep -iE "failed to find valid patches|Traceback|Error" "$OUT/early_run.log" | head -3 >> $LOG
fi

mv "$SC.videobak" "$SC"
mv "$PY.vidbak" "$PY"
echo "[v] restored terrain + play.py" >> $LOG
echo "[v] DONE $(date -u)" >> $LOG
