#!/bin/bash
# Quick answer to "do the switched arms climb at all": newest 3 finished
# checkpoints per arm, 3 and 4 cm, 200 episodes, each arm's OWN switch settings.
# Output to abl_quick/ so switch_driver3.sh (which writes abl_h/) is unaffected.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo
B=logs/co_rl/doublebee_velocity/tqc
mkdir -p abl_quick
TAG="$1"; RUN="$B/$2"; LOW="$3"
LOG=sweep_logs/switch/quick_${TAG}.log
# skip any checkpoint written in the last 2 minutes, it may still be flushing
CKPTS=$(find "$RUN" -maxdepth 1 -name "model_*.pt" -mmin +2 | sed 's/.*model_//;s/\.pt//' | sort -n | tail -3)
echo "[quick] $TAG ckpts: $(echo $CKPTS)  low=$LOW  $(date -u)" > $LOG
for H in 03 04; do
  for C in $CKPTS; do
    OUT="abl_quick/climb_${TAG}_h${H}_${C}.csv"
    DOUBLEBEE_SWITCH_LOW=$LOW DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_LATCH=3.0 \
    $IL -p scripts/paper/eval_climb.py --task $TASK \
      --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
      --episodes 200 --num_envs 64 --out "$OUT" 2>&1 \
      | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
  done
done
echo "[quick] $TAG DONE $(date -u)" >> $LOG
