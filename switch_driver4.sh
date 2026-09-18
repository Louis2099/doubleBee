#!/bin/bash
# Overnight driver for the MATCHED switched arms (swA3, swB3; hE4 recipe).
#
# 1. waits for model_5899.pt on both
# 2. main evaluation: 3-7 cm, last 10 checkpoints, 200 episodes, into abl_h/,
#    both arms in parallel, each with its OWN switch parameters exported
# 3. test-time switch sweep on each arm's best 6 cm checkpoint (paper metric:
#    peak gain >= step height): latch {3,1.5,6} x thresh {0.02,0.01,0.04} x
#    high {1.0,0.75}, low held at the trained value, at 6 and 4 cm, into
#    abl_sweep/, both arms in parallel
#
# Evaluation deliberately does NOT export the training recipe variables
# (REWARD_V2 / NO_DR / EPISODE_S / W_E). Reward weights do not enter the eval
# metrics, the Play config has no randomisation either way, and leaving the
# horizon at its default keeps these cells comparable with hE4's and the fixed
# arms', which were evaluated the same way.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
LOG=sweep_logs/switch/driver4.log
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo
B=logs/co_rl/doublebee_velocity/tqc
mkdir -p abl_h abl_sweep

rundir () { ls -dt $B/*_abl_$1 2>/dev/null | head -1; }

eval_one () {  # run tag low ckpt height out [high thresh latch]
  DOUBLEBEE_SWITCH_LOW=$3 DOUBLEBEE_SWITCH_HIGH=${7:-1.0} \
  DOUBLEBEE_SWITCH_THRESH=${8:-0.02} DOUBLEBEE_SWITCH_LATCH=${9:-3.0} \
  $IL -p scripts/paper/eval_climb.py --task $TASK \
    --checkpoint "$1/model_$4.pt" --step-height "0.$5" \
    --episodes 200 --num_envs 64 --out "$6" 2>&1 \
    | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
}

main_eval () {  # tag low
  TAG=$1; LOW=$2; RUN=$(rundir $TAG)
  CKPTS=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -10)
  echo "[d4] $TAG main eval, run $RUN, low $LOW, ckpts $(echo $CKPTS)  $(date -u)" >> $LOG
  for H in 03 04 05 06 07; do
    for C in $CKPTS; do
      OUT="abl_h/climb_${TAG}_h${H}_${C}.csv"
      [ -s "$OUT" ] || eval_one "$RUN" "$TAG" "$LOW" "$C" "$H" "$OUT"
    done
    echo "[d4] $TAG main done at $H cm  $(date -u)" >> $LOG
  done
}

best_ckpt () {
  python3 - "$1" <<'PY'
import csv, glob, re, sys
best = None
for f in glob.glob("abl_h/climb_%s_h06_*.csv" % sys.argv[1]):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    c = sum(float(r["max_gain_m"]) >= 0.06 for r in rows) / len(rows)
    k = int(re.search(r"_(\d+)\.csv$", f).group(1))
    if best is None or (c, k) > best:
        best = (c, k)
print(best[1] if best else 5899)
PY
}

sweep () {  # tag low
  TAG=$1; LOW=$2; RUN=$(rundir $TAG); C=$(best_ckpt $TAG)
  echo "[d4] $TAG sweep on ckpt $C, low $LOW  $(date -u)" >> $LOG
  for H in 06 04; do
    for HI in 1.0 0.75; do for TH in 0.02 0.01 0.04; do for LA in 3.0 1.5 6.0; do
      OUT="abl_sweep/sw_${TAG}_h${H}_hi${HI}_th${TH}_la${LA}.csv"
      [ -s "$OUT" ] || eval_one "$RUN" "$TAG" "$LOW" "$C" "$H" "$OUT" "$HI" "$TH" "$LA"
    done; done; done
    echo "[d4] $TAG sweep done at $H cm  $(date -u)" >> $LOG
  done
}

echo "[d4] waiting for swA3 and swB3 model_5899  $(date -u)" >> $LOG
while :; do
  A=$(rundir swA3); Bd=$(rundir swB3)
  [ -n "$A" ] && [ -n "$Bd" ] && [ -s "$A/model_5899.pt" ] && [ -s "$Bd/model_5899.pt" ] && break
  sleep 300
done
sleep 60
main_eval swA3 -0.05 &  main_eval swB3 0.50 &  wait
echo "[d4] MAIN EVAL DONE  $(date -u)" >> $LOG
sweep swA3 -0.05 &  sweep swB3 0.50 &  wait
echo "[d4] ALL DONE  $(date -u)" >> $LOG
