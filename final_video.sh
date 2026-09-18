#!/bin/bash
# ICRA video: early vs converged, one staircase, spread spawns.
# Backs up stair_config.py, widens ONLY the play init_pos block, restores after.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
SC=lab/doublebee/tasks/manager_based/locomotion/velocity/terrain_config/stair_config.py
OUT=vids/final
LOG=$OUT/render.log
mkdir -p $OUT

cp "$SC" "$SC.videobak"
# Play init_pos block only (starts ~line 185). Spread spawns so 15 robots do
# not land in one 1 m box.
sed -i '183,200 s/num_patches=8/num_patches=40/; 183,200 s/x_range=(-0.5, 0.5)/x_range=(-2.5, 2.5)/; 183,200 s/y_range=(-0.5, 0.5)/y_range=(-2.5, 2.5)/' "$SC"
grep -n -A6 '"init_pos"' "$SC" | sed -n '10,20p' >> $LOG

export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000
export DOUBLEBEE_TERRAIN_RGB=0.13,0.13,0.16
export DOUBLEBEE_TARGET_BALL_R=0.02

shoot () {   # tag  checkpoint
  TAG=$1; CK=$2
  [ -f "$CK" ] || { echo "[v] MISSING $CK" >> $LOG; return; }
  echo "[v] $TAG start $(date -u +%H:%M:%S)" >> $LOG
  timeout 1500 $IL -p scripts/co_rl/play.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
    --checkpoint "$CK" --num_envs 15 --headless \
    --video --video_length 200 --render_res 1920 1080 \
    --cam_eye 4 -4 2 --cam_lookat 0 0 0 \
    >> $OUT/${TAG}.log 2>&1
  NEW=$(find logs -name "*.mp4" -newermt "-25 min" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  if [ -n "$NEW" ]; then
    cp "$NEW" "$OUT/${TAG}.mp4"; mv "$NEW" "${NEW}.used"
    echo "[v] $TAG ok $(stat -c%s "$OUT/${TAG}.mp4") bytes" >> $LOG
  else
    echo "[v] $TAG NO MP4" >> $LOG
  fi
}

echo "[v] START $(date -u)" >> $LOG
shoot early     "$B/energy_abl/2026-09-06_07-59-52_wE0/model_1000.pt"
shoot converged "$B/energy_abl/2026-09-06_23-10-57_hE4/model_5899.pt"

mv "$SC.videobak" "$SC"      # always restore the eval terrain
echo "[v] restored $SC" >> $LOG
echo "[v] ALL DONE $(date -u)" >> $LOG
