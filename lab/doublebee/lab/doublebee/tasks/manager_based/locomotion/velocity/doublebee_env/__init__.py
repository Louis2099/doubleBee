# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DoubleBee robot environments for velocity control tasks."""

import gymnasium as gym

from . import agents, flat_env
from .velocity_env_cfg import DoubleBeeVelocityEnvCfg

##
# Register Gym environments.
##

# Register DoubleBee velocity control tasks (hybrid: propeller + wheel on staircase)
gym.register(
    id="Isaac-Velocity-HybridStair-DoubleBee-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.hybrid_stair.hybrid_stair_cfg.DoubleBeeHybridStairCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
    },
)

gym.register(
    id="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.hybrid_stair.hybrid_stair_cfg.DoubleBeeHybridStairCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
    },
)

__all__ = ["DoubleBeeVelocityEnvCfg"]
