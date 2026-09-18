#!/bin/bash
set -u
SWEEP_PIDS="194720,194726,194974,194980,195219,195225,195470,195476,195698,195704"
IL=../../isaaclab/IsaacLab/isaaclab.sh
mkdir -p sweep_logs/smoke

echo "[queue] waiting for the w_E sweep (PIDs $SWEEP_PIDS)"
while ps -p "$SWEEP_PIDS" >/dev/null 2>&1; do sleep 120; done
echo "[queue] sweep finished at $(date)"

# ---- SMOKE TEST -------------------------------------------------------------
# The WheelsOnly / WheelsServos / PropellerOnly configs set action terms to None.
# That pattern exists in actions.py but was never registered, so it may never have
# been executed. 30 iterations each catches a config error in ~90 s instead of at
# hour six of an eight-run queue.
echo "[queue] smoke-testing all arms, 30 iterations each"
FAILED=""
for T in ConstThrust WheelsOnly WheelsServos PropellerOnly; do
  $IL -p scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-$T-v1-ppo --algo tqc \
    --num_envs 1024 --max_iterations 30 --headless --seed 42 \
    > sweep_logs/smoke/$T.log 2>&1 \
    && echo "[smoke]  OK   $T" || { echo "[smoke] FAIL $T"; FAILED="$FAILED $T"; }
done
for M in xy xyz xyzu; do
  DOUBLEBEE_SUCCESS_MODE=$M $IL -p scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --algo tqc \
    --num_envs 1024 --max_iterations 30 --headless --seed 42 \
    > sweep_logs/smoke/succ_$M.log 2>&1 \
    && echo "[smoke]  OK   succ_$M" || { echo "[smoke] FAIL succ_$M"; FAILED="$FAILED succ_$M"; }
done
if [ -n "$FAILED" ]; then
  echo "[queue] ABORTING. Failed:$FAILED"
  echo "[queue] see sweep_logs/smoke/*.log -- nothing long was started."
  exit 1
fi
echo "[queue] all arms start cleanly. beginning full runs at $(date)"

# ---- FULL RUNS --------------------------------------------------------------
for T in ConstThrust WheelsOnly WheelsServos PropellerOnly; do
  echo "[queue] $(date +%H:%M) abl_$T"
  DOUBLEBEE_RUN_NAME=abl_$T $IL -p scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-$T-v1-ppo --algo tqc \
    --num_envs 1024 --max_iterations 4000 --headless --seed 42 \
    > sweep_logs/abl_$T.log 2>&1
done
for M in all xy xyz xyzu; do
  echo "[queue] $(date +%H:%M) succ_$M"
  DOUBLEBEE_SUCCESS_MODE=$M DOUBLEBEE_RUN_NAME=succ_$M $IL -p scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --algo tqc \
    --num_envs 1024 --max_iterations 4000 --headless --seed 42 \
    > sweep_logs/succ_$M.log 2>&1
done
echo "[queue] all done at $(date)"
