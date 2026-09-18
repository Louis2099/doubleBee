#!/bin/bash
# Option C: put Fig. 4 and Fig. 5 on Table V's paired-seed protocol.
#
# Same scheme as optionB_driver.sh: evaluation k at height H uses seed
# 1000*H + k for EVERY arm, last 10 checkpoints, 200 episodes, 64 envs.
# Output goes into abl_seeded/ alongside the existing Table V arms so one
# summariser covers everything.
#
# Missing pieces only. abl_seeded already holds hE4, swA3, swB3, ct10 at 3-7 cm.
#   Fig. 5  hE0 hE2 hE6 hE8            at 6 cm          =  40 evals
#   Fig. 4  ct050 ctm05 ctm45          at 3-7 cm        = 150 evals
# 6 cm runs first everywhere so the headline is complete if time runs short.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
LOG=sweep_logs/switch/optionC.log
mkdir -p abl_seeded sweep_logs/switch

arm () {  # tag task run_dir envstring heights...
  TAG=$1; TASK=$2; RUN=$3; ENVS=$4; shift 4
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  echo "[C] $TAG ckpts $(echo $CK) env[$ENVS] $(date -u)" >> $LOG
  for H in "$@"; do
    k=0
    for C in $CK; do
      SEED=$((1000 * 10#$H + k))
      OUT="abl_seeded/climb_${TAG}_h${H}_k${k}_${C}.csv"
      if [ ! -s "$OUT" ]; then
        env $ENVS timeout 3000 $IL -p scripts/paper/eval_climb.py --task $TASK \
          --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
          --episodes 200 --num_envs 64 --seed $SEED --out "$OUT" 2>&1 \
          | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
      fi
      k=$((k + 1))
    done
    echo "[C] $TAG done at $H cm  $(date -u)" >> $LOG
  done
  echo "[C] $TAG ALL DONE  $(date -u)" >> $LOG
}

PLAY=Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo
CONST=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo
E=$B/energy_abl/2026-09-06_23-10-57

echo "[C] START $(date -u)" >> $LOG

# --- Phase 1: Fig. 5 energy weights at 6 cm (hE4 already in abl_seeded) ---
arm hE0 $PLAY "${E}_hE0" "DOUBLEBEE_NOOP=1" 06 &
arm hE2 $PLAY "${E}_hE2" "DOUBLEBEE_NOOP=1" 06 &
arm hE6 $PLAY "${E}_hE6" "DOUBLEBEE_NOOP=1" 06 &
arm hE8 $PLAY "${E}_hE8" "DOUBLEBEE_NOOP=1" 06 &
wait
echo "[C] PHASE 1 (Fig. 5, 6 cm) DONE $(date -u)" >> $LOG

# --- Phase 2: Fig. 4 fixed allocations, 6 cm first then the rest ---
arm ct050 $CONST "$(ls -dt $B/*_abl_ct050 | head -1)" "DOUBLEBEE_HOLD_ACTION=0.5"   06 07 05 04 03 &
arm ctm05 $CONST "$(ls -dt $B/*_abl_ctm05 | head -1)" "DOUBLEBEE_HOLD_ACTION=-0.05" 06 07 05 04 03 &
arm ctm45 $CONST "$(ls -dt $B/*_abl_ctm45 | head -1)" "DOUBLEBEE_HOLD_ACTION=-0.45" 06 07 05 04 03 &
wait
echo "[C] ALL DONE $(date -u)" >> $LOG
