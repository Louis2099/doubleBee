#!/bin/bash
# ICRA video: early-training vs converged, both from the PAPER's lineage.
#
#   early      wE0/model_0        start of pre-training (balancing warm start)
#   converged  hE4/model_5899     the network deployed on hardware
#
# Training task, NOT the -Play- variant: STAIR_TERRAINS_CFG_PLAY is num_rows=1,
# num_cols=1, so 100 envs would pile onto a single staircase.
#
# play.py writes to <log_dir>/videos/play/rl-video-*.mp4 with a fixed prefix,
# so each clip is moved into vids/ immediately or the next run overwrites it.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
OUT=vids
LOG=$OUT/render.log
mkdir -p $OUT

# Dome light IS the visible sky, so these set background and illumination
# together. Pale default leaves a light robot with no separation from horizon.
export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000

shoot () {   # tag  checkpoint
  TAG=$1; CK=$2
  if [ ! -f "$CK" ]; then echo "[vid] MISSING $CK" >> $LOG; return; fi
  echo "[vid] $TAG start $(date -u +%H:%M:%S)" >> $LOG
  timeout 1800 $IL -p scripts/co_rl/play.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --algo tqc \
    --checkpoint "$CK" --num_envs 100 --headless \
    --video --video_length 200 --render_res 1920 1080 \
    --cam_eye 8 -8 5 --cam_lookat 0 0 0 \
    >> $OUT/${TAG}.log 2>&1
  # Collect whichever mp4 this run just produced, wherever it landed.
  NEW=$(find logs -name "*.mp4" -newermt "-40 min" 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1)
  if [ -n "$NEW" ]; then
    cp "$NEW" "$OUT/${TAG}.mp4"
    mv "$NEW" "${NEW}.used"
    echo "[vid] $TAG -> $OUT/${TAG}.mp4 ($(stat -c%s "$OUT/${TAG}.mp4") bytes)" >> $LOG
  else
    echo "[vid] $TAG NO MP4 FOUND" >> $LOG
  fi
  echo "[vid] $TAG done $(date -u +%H:%M:%S)" >> $LOG
}

echo "[vid] START $(date -u)" >> $LOG
shoot early     "$B/energy_abl/2026-09-06_07-59-52_wE0/model_0.pt"
shoot converged "$B/energy_abl/2026-09-06_23-10-57_hE4/model_5899.pt"
echo "[vid] ALL DONE $(date -u)" >> $LOG
