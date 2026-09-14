#!/bin/bash
# Option B: matched, seeded evaluation of the baseline comparison.
#
# Every arm is evaluated on its last 10 checkpoints at 3-7 cm, 200 episodes,
# 64 envs, default play horizon. Evaluation k at height H uses seed 1000*H + k
# for EVERY arm, so all controllers face identical terrain patches and goal
# draws: the comparison is paired. Figs. 4/5 are untouched (they live in abl_h/).
# 6 and 7 cm first (the headline), then 5, 4, 3. The four arms run in parallel.
#
# Output: abl_seeded/climb_<tag>_h<HH>_k<k>_<ckpt>.csv
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
B=logs/co_rl/doublebee_velocity/tqc
LOG=sweep_logs/switch/optionB.log
mkdir -p abl_seeded

arm () {  # tag task run_dir envstring
  TAG=$1; TASK=$2; RUN=$3; ENVS=$4
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  echo "[B] $TAG ckpts $(echo $CK) env[$ENVS] $(date -u)" >> $LOG
  for H in 06 07 05 04 03; do
    k=0
    for C in $CK; do
      SEED=$((1000 * 10#$H + k))
      OUT="abl_seeded/climb_${TAG}_h${H}_k${k}_${C}.csv"
      if [ ! -s "$OUT" ]; then
        env $ENVS $IL -p scripts/paper/eval_climb.py --task $TASK \
          --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
          --episodes 200 --num_envs 64 --seed $SEED --out "$OUT" 2>&1 \
          | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
      fi
      k=$((k + 1))
    done
    echo "[B] $TAG done at $H cm  $(date -u)" >> $LOG
  done
  echo "[B] $TAG ALL HEIGHTS DONE  $(date -u)" >> $LOG
}

HE4=$B/energy_abl/2026-09-06_23-10-57_hE4
SWA=$(ls -dt $B/*_abl_swA3 | head -1)
SWB=$(ls -dt $B/*_abl_swB3 | head -1)
CT10=$(ls -dt $B/*_abl_ct10 | head -1)

arm hE4  Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo "$HE4" "DOUBLEBEE_NOOP=1" &
arm swA3 Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo "$SWA" \
    "DOUBLEBEE_SWITCH_LOW=-0.05 DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_THRESH=0.02 DOUBLEBEE_SWITCH_LATCH=3.0" &
arm swB3 Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo "$SWB" \
    "DOUBLEBEE_SWITCH_LOW=0.50 DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_THRESH=0.02 DOUBLEBEE_SWITCH_LATCH=3.0" &
arm ct10 Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo "$CT10" "DOUBLEBEE_HOLD_ACTION=1.0" &
wait
echo "[B] ALL DONE $(date -u)" >> $LOG
