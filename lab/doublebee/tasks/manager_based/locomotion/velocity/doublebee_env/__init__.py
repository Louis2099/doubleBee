# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DoubleBee robot environments for velocity control tasks."""

import gymnasium as gym

from . import agents
from .flat_env.hybrid_stair.hybrid_stair_cfg import DoubleBeeHybridStairCfg, DoubleBeeHybridStairCfg_PLAY
from .flat_env.inverted_pendulum import DoubleBeeInvertedPendulumCfg, DoubleBeeInvertedPendulumCfg_PLAY
from .velocity_env_cfg import DoubleBeeVelocityEnvCfg

##
# Register Gym environments.
##

# Register DoubleBee velocity control tasks (hybrid: propeller + wheel on staircase)
gym.register(
    # TODO: remove ppo from the naming
    id="Isaac-Velocity-HybridStair-DoubleBee-v1-ppo",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoubleBeeHybridStairCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
        "co_rl_tqc_cfg_entry_point": agents.co_rl_tqc_cfg.DoubleBeeCoRlTqcCfg,
        "co_rl_sac_cfg_entry_point": agents.co_rl_sac_cfg.DoubleBeeCoRlSacCfg,
    },
)

gym.register(
    id="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        # TODO: remove ppo from the naming
        "env_cfg_entry_point": DoubleBeeHybridStairCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
        "co_rl_tqc_cfg_entry_point": agents.co_rl_tqc_cfg.DoubleBeeCoRlTqcCfg,
        "co_rl_sac_cfg_entry_point": agents.co_rl_sac_cfg.DoubleBeeCoRlSacCfg,
    },
)

# Inverted-pendulum: same-level target, no height scan, no propeller actuation
gym.register(
    id="Isaac-Velocity-InvertedPendulum-DoubleBee-v1-ppo",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoubleBeeInvertedPendulumCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
        # 2026-08-26: added so the balance-only task can be run under TQC as a
        # capability check -- if TQC cannot learn wheels-only balance with this
        # observation set and actuator delay, no hybrid_stair reward tuning will
        # help, and that is a different fix.
        "co_rl_tqc_cfg_entry_point": agents.co_rl_tqc_cfg.DoubleBeeCoRlTqcCfg,
    },
)

gym.register(
    id="Isaac-Velocity-InvertedPendulum-DoubleBee-Play-v1-ppo",
    entry_point="lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env:ManagerBasedConstraintRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoubleBeeInvertedPendulumCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,
        "co_rl_tqc_cfg_entry_point": agents.co_rl_tqc_cfg.DoubleBeeCoRlTqcCfg,
    },
)

__all__ = ["DoubleBeeVelocityEnvCfg"]
