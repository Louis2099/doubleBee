#!/bin/bash
# Evaluate every switched-thrust arm once its training run reaches model_5899.pt.
#
# Keyed on the RUN DIRECTORY, not on a process pattern: the first driver waited
# on `pgrep SwitchThrust`, which would have blocked on the corrected arms too
# and delayed everything. Each arm is independent here.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
L=logs/co_rl/doublebee_velocity/tqc

wait_and_eval () {   # $1 = run dir suffix to match, $2 = csv tag
  local D N
  echo "[driver2] $2: waiting for a run dir matching *$1"
  while :; do
    D=$(ls -dt $L/*"$1" 2>/dev/null | head -1)
    [ -n "${D:-}" ] && break
    sleep 60
  done
  echo "[driver2] $2: run dir $D"
  while :; do
    N=$(ls "$D"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    [ "${N:-0}" -ge 5899 ] && break
    # Training gone and not finished: evaluate what exists rather than hang.
    if ! pgrep -f "$(basename "$D")" >/dev/null 2>&1 && \
       ! pgrep -f "train.py" >/dev/null 2>&1; then
      echo "[driver2] $2: training gone at $N, evaluating anyway"; break
    fi
    sleep 180
  done
  echo "[driver2] $2: training done at $N, $(date -u). Evaluating."
  ./eval_switch.sh "$D" "$2" > sweep_logs/switch/eval_$2.log 2>&1
  echo "[driver2] $2: EVAL DONE $(date -u), $(ls abl_h/climb_$2_*.csv 2>/dev/null | wc -l) csvs"
}

# _abl_sw and _abl_swL were stopped at 17:15 UTC: low level below useful, see notes
wait_and_eval _abl_swA  swA
wait_and_eval _abl_swB  swB
echo "[driver2] ALL ARMS DONE $(date -u)"
