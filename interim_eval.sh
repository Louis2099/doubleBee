#!/bin/bash
# Preliminary 6 cm read on the switched-thrust arms, fired mid-training so there
# is something to look at before the full run ends.
#
# 6 cm only, because that is the paper's headline cell: the geometric rolling
# limit h_max = r, where the learned policy is reported at 292 W against 391 W
# for the best fixed allocation. Ten checkpoints, 200 episodes, same protocol.
#
# Results go to abl_h_interim/ so they can never be mistaken for, or block, the
# final numbers in abl_h/. These checkpoints are NOT the final ones.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
L=logs/co_rl/doublebee_velocity/tqc
mkdir -p abl_h_interim

echo "[interim] waiting for both arms to reach iteration 4000, $(date -u)"
while true; do
  A=$(ls $L/2026-09-11_06-03-40_abl_sw/model_*.pt  2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  B=$(ls $L/2026-09-11_06-04-03_abl_swL/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  [ "${A:-0}" -ge 4000 ] && [ "${B:-0}" -ge 4000 ] && break
  # If training died, evaluate whatever exists rather than waiting forever.
  pgrep -f "SwitchThrust.*max_iterations 4000" >/dev/null 2>&1 || {
    echo "[interim] training no longer running; proceeding with $A / $B"; break; }
  sleep 300
done
echo "[interim] proceeding at $(date -u): sw=$A swL=$B"

for pair in "2026-09-11_06-03-40_abl_sw sw" "2026-09-11_06-04-03_abl_swL swL"; do
  set -- $pair
  D=$L/$1; TAG=$2
  echo "[interim] ===== $TAG ====="
  for C in $(ls $D/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10); do
    $IL -p scripts/paper/eval_climb.py \
      --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo \
      --checkpoint "$D/model_${C}.pt" \
      --step-height 0.06 --episodes 200 --num_envs 64 \
      --out "abl_h_interim/climb_${TAG}_h06_${C}.csv" 2>&1 \
      | grep -aE "^wrote|Traceback|RuntimeError"
  done
done
echo "[interim] DONE $(date -u)"
