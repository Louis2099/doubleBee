#!/bin/bash
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
W40=$(ls -dt logs/co_rl/doublebee_velocity/tqc/*_actfix_wE40 | head -1)
export DOUBLEBEE_SERVO_VEL_LIMIT=10.0
export DOUBLEBEE_WHEEL_ARMATURE=0.0085
export DOUBLEBEE_PROP_EFFORT=25
export DOUBLEBEE_PROP_DAMPING=0.16
export DOUBLEBEE_PROP_ARMATURE=0.0016
export DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12 DOUBLEBEE_W_E=4.0
export DOUBLEBEE_HOLD_ACTION=-0.21
export DOUBLEBEE_RESUME_PATH=$W40/model_3000.pt
export DOUBLEBEE_RUN_NAME=actfix_fixed055
exec $IL -p scripts/co_rl/train.py \
  --task Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-v1-ppo --algo tqc \
  --num_envs 1024 --max_iterations 2000 --headless --seed 42
