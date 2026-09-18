#!/bin/bash
# Fixed-thrust arms retrained on hE4's EXACT recipe, queued behind the matched
# switched arms so those results are not delayed.
#
# Why: all four Figure 4 fixed-thrust arms (ct10/ct050/ctm05/ctm45) trained on
# the code-default recipe (20 s episodes, all randomisation ON, default reward
# weights), COLD, to 3999. hE4 trained on REWARD_V2 / NO_DR / 12 s, warm from
# gE0/model_1900, to 5899. This makes the fixed arms identical to hE4 except
# that thrust is held constant.
#
# Order: T/W 0.55 (hold 1.0, the best fixed allocation) and T/W 0.46 (hold 0.5)
# first, since they carry the Figure 4 frontier and the 6 cm comparison.
# T/W 0.31 (-0.05) and 0.23 (-0.45) second, only if time allows; they clear
# ~1 % at 6 cm and set the low end of the frontier.
#
# Then evaluates each finished pair at 3-7 cm, last 10 checkpoints, 200 eps,
# into abl_h/ as ctA10 / ctA050 / ctAm05 / ctAm45 (A = matched).
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
LOG=sweep_logs/switch/ct_matched.log
B=logs/co_rl/doublebee_velocity/tqc
TRAIN_TASK=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-v1-ppo
PLAY_TASK=Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo
WARM=$B/energy_abl/2026-09-06_19-25-46_gE0/model_1900.pt
mkdir -p abl_h

rundir () { ls -dt $B/*_abl_$1 2>/dev/null | head -1; }
done5899 () { D=$(rundir $1); [ -n "$D" ] && [ -s "$D/model_5899.pt" ]; }

launch () {  # tag hold
  echo "[ct] launching $1 hold=$2  $(date -u)" >> $LOG
  DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12 DOUBLEBEE_W_E=4.0 \
  DOUBLEBEE_RESUME_PATH=$WARM DOUBLEBEE_HOLD_ACTION=$2 DOUBLEBEE_RUN_NAME=abl_$1 \
  nohup setsid $IL -p scripts/co_rl/train.py --task $TRAIN_TASK --algo tqc \
    --num_envs 1024 --max_iterations 4000 --headless --seed 42 \
    > sweep_logs/switch/train_$1.log 2>&1 < /dev/null &
  echo "$1 hold=$2 recipe=hE4 warm=gE0/1900 $(date -u)" > sweep_logs/switch/$1_params.txt
  sleep 30
}

evaluate () {  # tag hold
  RUN=$(rundir $1)
  CK=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  for H in 03 04 05 06 07; do for C in $CK; do
    OUT="abl_h/climb_$1_h${H}_${C}.csv"
    [ -s "$OUT" ] && continue
    DOUBLEBEE_HOLD_ACTION=$2 $IL -p scripts/paper/eval_climb.py --task $PLAY_TASK \
      --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
      --episodes 200 --num_envs 64 --out "$OUT" 2>&1 \
      | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
  done; echo "[ct] $1 eval done at $H cm  $(date -u)" >> $LOG; done
}

echo "[ct] waiting for swA3 + swB3 to finish training  $(date -u)" >> $LOG
until done5899 swA3 && done5899 swB3; do sleep 300; done

# pair 1
launch ctA10 1.0
launch ctA050 0.5
until done5899 ctA10 && done5899 ctA050; do sleep 300; done
echo "[ct] pair 1 trained  $(date -u)" >> $LOG

# pair 2 trains while pair 1 evaluates
launch ctAm05 -0.05
launch ctAm45 -0.45
evaluate ctA10 1.0 & evaluate ctA050 0.5 & wait
echo "[ct] pair 1 EVAL DONE  $(date -u)" >> $LOG
until done5899 ctAm05 && done5899 ctAm45; do sleep 300; done
evaluate ctAm05 -0.05 & evaluate ctAm45 -0.45 & wait
echo "[ct] ALL DONE  $(date -u)" >> $LOG
