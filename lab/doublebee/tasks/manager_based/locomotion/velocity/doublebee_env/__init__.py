# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DoubleBee robot environments for velocity control tasks."""

import gymnasium as gym

from . import agents
from .flat_env.stand_drive.flat_env_stand_drive_cfg import DoubleBeeFlatStandDriveCfg, DoubleBeeFlatStandDriveCfg_PLAY
from .velocity_env_cfg import DoubleBeeVelocityEnvCfg

##
# Register Gym environments.
##

# Register DoubleBee velocity control tasks
gym.register(
    id="Isaac-Velocity-Flat-DoubleBee-v1-ppo",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoubleBeeFlatStandDriveCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-DoubleBee-Play-v1-ppo",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoubleBeeFlatStandDriveCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
    },
)

__all__ = ["DoubleBeeVelocityEnvCfg"]
