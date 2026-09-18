#!/bin/bash
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
DOUBLEBEE_SWITCH_LOW=-0.05 DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_THRESH=0.04 DOUBLEBEE_SWITCH_LATCH=3.0 ../../isaaclab/IsaacLab/isaaclab.sh -p scripts/paper/eval_climb.py --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo   --checkpoint "logs/co_rl/doublebee_velocity/tqc/2026-09-13_06-00-57_abl_swA3/model_5700.pt" --step-height 0.06 --episodes 200 --num_envs 64   --out abl_confirm/repeat_swA3t04_h06_5700_run3.csv > /tmp/repeat_5700.log 2>&1
echo "exit $?" >> /tmp/repeat_5700.log
