#!/bin/bash
# Wait for both switched-thrust training arms to finish, sanity-check them, then
# evaluate both on the standard protocol. Detached so it survives ssh dropping.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
L=logs/co_rl/doublebee_velocity/tqc
SW=$L/2026-09-11_06-03-40_abl_sw
SWL=$L/2026-09-11_06-04-03_abl_swL

echo "[driver] waiting for training to finish, started $(date -u)"
while pgrep -f "SwitchThrust.*max_iterations 4000" >/dev/null 2>&1; do sleep 120; done
echo "[driver] training processes gone at $(date -u)"

for D in "$SW" "$SWL"; do
  N=$(ls "$D"/model_*.pt 2>/dev/null | wc -l)
  TOP=$(ls "$D"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  echo "[driver] $(basename $D): $N checkpoints, highest $TOP"
  if [ "${TOP:-0}" -lt 5000 ]; then
    echo "[driver] WARNING $(basename $D) stopped early at $TOP -- evaluating anyway"
  fi
done

# Duty cycle: the one number that says whether the switch actually engaged.
for f in train_sw train_swL; do
  echo "[driver] === duty trend, $f ==="
  grep -a "\[switch\]" sweep_logs/switch/$f.log | tail -8
done

echo "[driver] evaluating sw  $(date -u)"
./eval_switch.sh "$SW"  sw  > sweep_logs/switch/eval_sw.log  2>&1
echo "[driver] evaluating swL $(date -u)"
./eval_switch.sh "$SWL" swL > sweep_logs/switch/eval_swL.log 2>&1
echo "[driver] ALL DONE $(date -u)"
ls abl_h/climb_sw_*.csv abl_h/climb_swL_*.csv 2>/dev/null | wc -l
