#!/bin/bash
# Sim rollout with per-tick policy IO logging, for the thrust-modulation figure.
# total_thrust in this log is the simulator's ACHIEVED thrust (from joint speed),
# not the command -- which is the whole reason for going to sim.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
CK=$B/energy_abl/2026-09-06_23-10-57_hE4/model_5899.pt
OUT=simprof/policy_io_hE4.csv
mkdir -p simprof
export DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12
echo "[sim] start $(date -u)"
timeout 1500 $IL -p scripts/co_rl/play.py \
  --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc \
  --checkpoint "$CK" --num_envs 16 --headless \
  --log_policy_io --log_policy_io_path "$OUT" \
  > simprof/play.log 2>&1
echo "[sim] done $(date -u) rows=$(wc -l < $OUT 2>/dev/null || echo 0)"
