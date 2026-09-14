#!/bin/bash
# Switched-thrust arms retrained on hE4's EXACT recipe, so the only difference
# from the learned policy is who controls thrust.
#
# Recovered 2026-09-13 from hE4's dumped params/env.yaml against the current
# code defaults. hE4 (and its gE0 warm-start parent) trained with:
#   DOUBLEBEE_REWARD_V2=1   alive 0.5, props_upright 2.0, thrust_up_at_step 2.0,
#                           vertical_support 1.5, recovery_under_lean 1.5,
#                           stalling 6.0   (code defaults are 2/5/5/3/6/1)
#   DOUBLEBEE_NO_DR=1       all seven randomisation terms off (incl. pushes and
#                           +-20 % thrust); code default has them ON
#   DOUBLEBEE_EPISODE_S=12  code default 20
#   DOUBLEBEE_W_E=4.0       code default 0.25
# The previous arms (swA2/swB2) and all ct* arms picked up the code defaults.
#
# Usage:
#   ./relaunch_matched.sh verify              1 iteration, 16 envs, then diff the
#                                             dumped env.yaml against hE4. Expect
#                                             ONLY the propeller action block.
#   ./relaunch_matched.sh launch LOW_A LOW_B  both arms, 4000 iters, in parallel
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
TASK=Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-v1-ppo
B=logs/co_rl/doublebee_velocity/tqc
HE4=$B/energy_abl/2026-09-06_23-10-57_hE4
WARM=$B/energy_abl/2026-09-06_19-25-46_gE0/model_1900.pt
export DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12 DOUBLEBEE_W_E=4.0
export DOUBLEBEE_RESUME_PATH=$WARM DOUBLEBEE_SWITCH_DIAG=1
export DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_LATCH=3.0

yaml_diff () {  # run_dir -> differences vs hE4, propeller action block excluded
  python3 - "$HE4/params/env.yaml" "$1/params/env.yaml" <<'PY'
import sys, yaml
class L(yaml.SafeLoader): pass
L.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, s, n:
    l.construct_sequence(n) if isinstance(n, yaml.SequenceNode) else
    (l.construct_mapping(n) if isinstance(n, yaml.MappingNode) else l.construct_scalar(n)))
a, b = (yaml.load(open(p), Loader=L) for p in sys.argv[1:3])
def walk(x, y, path):
    if path in ("actions.propeller_vel",):
        return
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            walk(x.get(k), y.get(k), (path + "." if path else "") + str(k))
    elif x != y:
        print("  DIFF %-60s hE4=%r  new=%r" % (path, x, y))
walk(a, b, "")
print("  (end of diff; anything above other than scene/seed noise is a mismatch)")
PY
}

case "${1:-}" in
verify)
  DOUBLEBEE_SWITCH_LOW=-0.05 DOUBLEBEE_RUN_NAME=verify_matched \
  $IL -p scripts/co_rl/train.py --task $TASK --algo tqc --num_envs 16 \
      --max_iterations 1 --headless --seed 42 > /tmp/verify_matched.log 2>&1
  D=$(ls -dt $B/*_verify_matched | head -1)
  echo "verify run dir: $D"
  yaml_diff "$D"
  ;;
launch)
  LA="$2"; LB="$3"
  mkdir -p sweep_logs/switch
  DOUBLEBEE_SWITCH_LOW=$LA DOUBLEBEE_RUN_NAME=abl_swA3 nohup setsid \
    $IL -p scripts/co_rl/train.py --task $TASK --algo tqc --num_envs 1024 \
    --max_iterations 4000 --headless --seed 42 \
    > sweep_logs/switch/train_swA3.log 2>&1 < /dev/null &
  sleep 20
  DOUBLEBEE_SWITCH_LOW=$LB DOUBLEBEE_RUN_NAME=abl_swB3 nohup setsid \
    $IL -p scripts/co_rl/train.py --task $TASK --algo tqc --num_envs 1024 \
    --max_iterations 4000 --headless --seed 42 \
    > sweep_logs/switch/train_swB3.log 2>&1 < /dev/null &
  sleep 5
  # Switch parameters are NOT in the dumped config; record them in the run dir.
  echo "swA3 low=$LA high=1.0 latch=3.0 thresh=0.02 lookahead=0.105 recipe=hE4 $(date -u)" > sweep_logs/switch/swA3_params.txt
  echo "swB3 low=$LB high=1.0 latch=3.0 thresh=0.02 lookahead=0.105 recipe=hE4 $(date -u)" > sweep_logs/switch/swB3_params.txt
  echo "launched swA3 (low $LA) and swB3 (low $LB)"
  ;;
*) echo "usage: $0 verify | launch LOW_A LOW_B"; exit 1 ;;
esac
