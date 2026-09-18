#!/bin/bash
# Test-time sweep of the switch settings, giving the baseline its best result.
#
# Runs only after switch_driver3.sh reports ALL DONE, so it never shares the GPU
# with the main evaluation. For each arm it picks the checkpoint with the highest
# mean 6 cm clearance from driver3's CSVs, then re-evaluates that one checkpoint
# under 18 switch configurations at 4 and 6 cm.
#
# Varied: latch {1.5, 3.0, 6.0} s x threshold {0.01, 0.02, 0.04} m x high hold
# {0.75, 1.0}. Held at the TRAINED value: low hold (it sets the cruise regime the
# wheels and servos learned under, so moving it measures a different controller)
# and lookahead (already the 0.105 m maximum the 4x4 scan reaches). The trained
# configuration (latch 3.0, thresh 0.02, high 1.0) is one of the 18 cells, so it
# doubles as a consistency check against driver3.
#
# Output: abl_sweep/sw_<arm>_h<HH>_hi<H>_th<T>_la<L>.csv   (never abl_h/)
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
LOG=sweep_logs/switch/sweep.log
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo
B=logs/co_rl/doublebee_velocity/tqc
mkdir -p abl_sweep

echo "[sweep] waiting for driver3 ALL DONE  $(date -u)" >> $LOG
while ! grep -q "ALL DONE" sweep_logs/switch/driver3.log 2>/dev/null; do sleep 300; done
echo "[sweep] starting  $(date -u)" >> $LOG

best_ckpt () {   # tag -> checkpoint number with highest 6 cm clearance
  python3 - "$1" <<'PY'
import csv, glob, re, sys
best = None
for f in glob.glob("abl_h/climb_%s_h06_*.csv" % sys.argv[1]):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    # paper metric: peak gain >= step height (reproduces hE4's published cells)
    c = sum(float(r["max_gain_m"]) >= 0.06 for r in rows) / len(rows)
    k = int(re.search(r"_(\d+)\.csv$", f).group(1))
    if best is None or (c, k) > best:
        best = (c, k)
print(best[1] if best else 5899)
PY
}

sweep_arm () {   # dir tag low
  RUN="$1"; TAG="$2"; LOW="$3"
  C=$(best_ckpt "$TAG")
  echo "[sweep] $TAG best 6 cm checkpoint = $C, low held at $LOW  $(date -u)" >> $LOG
  for H in 06 04; do
    for HI in 1.0 0.75; do for TH in 0.02 0.01 0.04; do for LA in 3.0 1.5 6.0; do
      OUT="abl_sweep/sw_${TAG}_h${H}_hi${HI}_th${TH}_la${LA}.csv"
      if [ -s "$OUT" ]; then continue; fi
      DOUBLEBEE_SWITCH_LOW=$LOW DOUBLEBEE_SWITCH_HIGH=$HI \
      DOUBLEBEE_SWITCH_THRESH=$TH DOUBLEBEE_SWITCH_LATCH=$LA \
      $IL -p scripts/paper/eval_climb.py --task $TASK \
        --checkpoint "$RUN/model_${C}.pt" --step-height "0.$H" \
        --episodes 200 --num_envs 64 --out "$OUT" 2>&1 \
        | grep -aE '^wrote|Traceback|RuntimeError' >> $LOG
    done; done; done
    echo "[sweep] $TAG done at $H cm  $(date -u)" >> $LOG
  done
}

sweep_arm $B/2026-09-12_14-27-48_abl_swA2 swA2 -0.05
sweep_arm $B/2026-09-12_14-27-58_abl_swB2 swB2 0.50
echo "[sweep] ALL DONE  $(date -u)" >> $LOG
