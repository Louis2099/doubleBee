#!/bin/bash
# ICRA video: early vs converged, one staircase, one shared goal.
# Backs up stair_config.py, edits ONLY the play block, restores after.
#
# init_pos  x/y widened to +/-1.2 so 15 robots spread across the flat platform
#           (platform_width=3.0, so the flat centre is +/-1.5 -- going wider put
#           robots on the step treads). z tightened to +/-0.08 so a patch sitting
#           flat on a RISER is rejected: max_height_diff only checks flatness
#           within patch_radius, not height above the base.
# target    num_patches=5 -> 1, so every env samples the SAME patch and the 15
#           goal balls collapse onto one.
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
sed -i '183,192 s/num_patches=8/num_patches=40/; 183,192 s/x_range=(-0.5, 0.5)/x_range=(-1.2, 1.2)/; 183,192 s/y_range=(-0.5, 0.5)/y_range=(-1.2, 1.2)/; 183,192 s/z_range=(-0.5, 0.5)/z_range=(-0.08, 0.08)/; 190,215 s/num_patches=5/num_patches=1/' "$SC"
echo "--- edited play block:" >> $LOG
sed -n '183,200p' "$SC" >> $LOG

export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000

shoot () {   # tag  checkpoint
  TAG=$1; CK=$2
  [ -f "$CK" ] || { echo "[v] MISSING $CK" >> $LOG; return; }
  echo "[v] $TAG start $(date -u +%H:%M:%S)" >> $LOG
  # Pulled back on -Y (behind the robots, which travel +Y), lower, and looking
  # slightly UP so the horizon and dark sky are in frame.
  timeout 1500 $IL -p scripts/co_rl/play.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
    --checkpoint "$CK" --num_envs 15 --headless \
    --video --video_length 200 --render_res 1920 1080 \
    --cam_eye 2.0 -6.5 2.0 --cam_lookat 0 1.0 0.5 \
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
