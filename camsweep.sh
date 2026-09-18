#!/bin/bash
# Camera sweep on ONE staircase. Short and low-res: this is for choosing a
# framing, not for the final clip.
#
# -Play- task is the point here: STAIR_TERRAINS_CFG_PLAY is num_rows=1,
# num_cols=1, so all 100 envs share a single inverted-pyramid staircase, which
# is the "many robots, one staircase" look.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
CK=$B/energy_abl/2026-09-06_23-10-57_hE4/model_5899.pt
OUT=vids/cam
LOG=$OUT/sweep.log
mkdir -p $OUT

export DOUBLEBEE_NO_DR=1
export DOUBLEBEE_NO_ARROWS=1
export DOUBLEBEE_REWARD_V2=1
export DOUBLEBEE_EPISODE_S=12
export DOUBLEBEE_SKY_RGB=0.05,0.06,0.12
export DOUBLEBEE_SKY_INTENSITY=300
export DOUBLEBEE_SUN_INTENSITY=6000

shoot () {   # tag  eye_x eye_y eye_z
  TAG=$1; EX=$2; EY=$3; EZ=$4
  echo "[cam] $TAG eye=$EX,$EY,$EZ start $(date -u +%H:%M:%S)" >> $LOG
  timeout 900 $IL -p scripts/co_rl/play.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
    --checkpoint "$CK" --num_envs 100 --headless \
    --video --video_length 60 --render_res 1280 720 \
    --cam_eye $EX $EY $EZ --cam_lookat 0 0 0 \
    >> $OUT/${TAG}.log 2>&1
  NEW=$(find logs -name "*.mp4" -newermt "-10 min" 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1)
  if [ -n "$NEW" ]; then
    cp "$NEW" "$OUT/${TAG}.mp4"; mv "$NEW" "${NEW}.used"
    echo "[cam] $TAG ok" >> $LOG
  else
    echo "[cam] $TAG NO MP4" >> $LOG
  fi
}

echo "[cam] START $(date -u)" >> $LOG
shoot a  6 -6  3
shoot b  4 -4  2
shoot c  9 -9  5
echo "[cam] ALL DONE $(date -u)" >> $LOG
