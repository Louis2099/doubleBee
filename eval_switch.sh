#!/bin/bash
# Evaluate the mode-switching baseline on the same protocol as every other arm:
# the last ten checkpoints, 200 episodes, staircases pinned at 3/4/5/6/7 cm.
# Output naming matches abl_h/climb_<tag>_h<HH>_<ckpt>.csv so fig_step_height.py
# picks it up with a one-line ARMS change.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
RUN="$1"            # training run directory under logs/co_rl/doublebee_velocity/tqc
TAG="${2:-sw}"

CKPTS=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
echo "[eval] run  $RUN"
echo "[eval] ckpts $(echo $CKPTS | tr '\n' ' ')"

for H in 03 04 05 06 07; do
  echo "@@@@@@@@@@ HEIGHT $H cm @@@@@@@@@@"
  for C in $CKPTS; do
    OUT="abl_h/climb_${TAG}_h${H}_${C}.csv"
    [ -s "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    $IL -p scripts/paper/eval_climb.py \
      --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo \
      --checkpoint "$RUN/model_${C}.pt" \
      --step-height "0.$H" --episodes 200 --num_envs 64 \
      --out "$OUT" 2>&1 | grep -aE "^wrote|Traceback|RuntimeError"
  done
  echo "---- $TAG done at $H cm ----"
done
echo "[eval] ALL DONE $(date)"
