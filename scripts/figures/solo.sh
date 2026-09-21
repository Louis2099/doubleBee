#!/bin/bash
# Single robot, red target ball, THREE 4 cm steps, static camera.
#
# Takes a terrain seed as $1 (default 42). The play terrain hard-codes seed=42,
# which fixes the flat-patch positions, so every render spawned the robot in the
# same place. Varying the seed varies the patches and therefore the spawn.
#
# Geometry: tile 7.4 m, sub-terrain border 1.0 -> half-extent 2.7. Steps run
# platform_width/2 (1.5) to 2.7 at 0.4 m tread = 3 steps.
# Target band must sit ON those steps (1.6..2.6), not on the flat platform, or
# find_flat_patches exhausts and the run dies.
set -u
SEEDVAL=${1:-42}
TAG=${2:-solo}
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
SC=lab/doublebee/tasks/manager_based/locomotion/velocity/terrain_config/stair_config.py
CK=$B/energy_abl/2026-09-06_23-10-57_hE4/model_3000.pt
OUT=vids/solo
LOG=$OUT/${TAG}.log
mkdir -p $OUT
: > $LOG

cp "$SC" "$SC.videobak"
sed -i "147,175 s/seed=42/seed=${SEEDVAL}/; 147,175 s/size=(10.0, 10.0)/size=(7.4, 7.4)/; 147,175 s/difficulty_range=(0.26, 0.40)/difficulty_range=(0.10, 0.24)/" "$SC"
sed -i '183,192 s/num_patches=8/num_patches=12/; 183,192 s/x_range=(-0.5, 0.5)/x_range=(-1.3, 1.3)/; 183,192 s/y_range=(-0.5, 0.5)/y_range=(0.2, 1.3)/; 183,192 s/z_range=(-0.5, 0.5)/z_range=(-0.08, 0.08)/' "$SC"
echo "--- seed=${SEEDVAL} geometry:" >> $LOG
sed -n '147,195p' "$SC" | grep -E "seed=|size=\(7.4|difficulty_range=\(0|platform_width|x_range|y_range" >> $LOG

export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000
export DOUBLEBEE_PLAY_TARGET_Y=1.6,2.6
export DOUBLEBEE_PLAY_TARGET_Z=0.0,0.30
export DOUBLEBEE_TARGET_BALL_R=0.14

echo "[s] start $(date -u +%H:%M:%S) seed=${SEEDVAL} ckpt=$(basename $CK)" >> $LOG
timeout 1500 $IL -p scripts/co_rl/play.py \
  --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
  --checkpoint "$CK" --num_envs 1 --headless \
  --video --video_length 400 --render_res 1920 1080 \
  --cam_eye -4.4 -4.4 3.8 --cam_lookat 0 0 0.1 \
  >> $OUT/${TAG}_run.log 2>&1

NEW=$(find logs -name "*.mp4" -newermt "-25 min" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
if [ -n "$NEW" ]; then
  cp "$NEW" "$OUT/${TAG}.mp4"; mv "$NEW" "${NEW}.used"
  echo "[s] ok $(stat -c%s "$OUT/${TAG}.mp4") bytes" >> $LOG
else
  echo "[s] NO MP4" >> $LOG
  grep -iE "valid patches|Traceback|Error" "$OUT/${TAG}_run.log" | head -5 >> $LOG
fi

mv "$SC.videobak" "$SC"
echo "[s] restored, DONE $(date -u)" >> $LOG
