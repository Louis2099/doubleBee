#!/bin/bash
# From-scratch training with simulated actuators matched to hardware.
# Not for the paper. Recipe otherwise identical to hE4 (REWARD_V2, no DR, 12 s).
#
#   ./train_matched_actuators.sh verify "<prop env vars>"   1 iteration, 16 envs
#   ./train_matched_actuators.sh launch "<prop env vars>"   two arms in parallel:
#        actfix_wE4   energy weight 4.0 (paper weight)
#        actfix_wE025 energy weight 0.25 (the from-scratch base recipe), hedge
#        in case a strong energy penalty from step 0 suppresses thrust before the
#        policy learns to use it.
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
TASK="${TASK:-Isaac-Velocity-HybridStair-DoubleBee-v1-ppo}"
PROP="$2"
COMMON="DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12 \
DOUBLEBEE_SERVO_VEL_LIMIT=10.0 DOUBLEBEE_WHEEL_ARMATURE=0.0085 $PROP"
unset DOUBLEBEE_RESUME_PATH
mkdir -p sweep_logs/actfix

case "$1" in
verify)
  env $COMMON DOUBLEBEE_W_E=4.0 DOUBLEBEE_RUN_NAME=actfix_verify \
    $IL -p scripts/co_rl/train.py --task $TASK --algo tqc --num_envs 16 \
    --max_iterations 1 --headless --seed 42 > sweep_logs/actfix/verify.log 2>&1
  D=$(ls -dt logs/co_rl/doublebee_velocity/tqc/*_actfix_verify | head -1)
  echo "verify dir: $D"
  grep -aE "Traceback|Error:|resume|Loading model" sweep_logs/actfix/verify.log | grep -v Replication | head -5
  python3 - "$D/params/env.yaml" <<'PY'
import sys, yaml
class L(yaml.SafeLoader): pass
L.add_multi_constructor("tag:yaml.org,2002:python/", lambda l,s,n: l.construct_sequence(n) if isinstance(n,yaml.SequenceNode) else (l.construct_mapping(n) if isinstance(n,yaml.MappingNode) else l.construct_scalar(n)))
c=yaml.load(open(sys.argv[1]),Loader=L)
a=c["scene"]["robot"]["actuators"]
for k in ("wheels","propeller_servos","propellers"):
    v=a[k]; print(k, "effort", v["effort_limit"], "vel", v["velocity_limit"], "damping", v["damping"], "armature", v["armature"])
print("episode_length_s", c["episode_length_s"], "| energy weight", c["rewards"]["energy_consumption"]["weight"])
print("events on:", sorted(k for k,v in (c.get("events") or {}).items() if v))
PY
  grep -h "resume" "$D/params/agent.yaml"
  ;;
launch)
  for W in 4.0 0.25; do
    TAG=actfix_wE$(echo $W | tr -d .)
    env $COMMON DOUBLEBEE_W_E=$W DOUBLEBEE_RUN_NAME=$TAG nohup setsid \
      $IL -p scripts/co_rl/train.py --task $TASK --algo tqc --num_envs 1024 \
      --max_iterations 4000 --headless --seed 42 \
      > sweep_logs/actfix/train_$TAG.log 2>&1 < /dev/null &
    echo "$TAG W_E=$W $COMMON $(date -u)" > sweep_logs/actfix/${TAG}_params.txt
    sleep 20
  done
  echo "launched"
  ;;
*) echo "usage: $0 verify|launch \"<prop env vars>\""; exit 1 ;;
esac
