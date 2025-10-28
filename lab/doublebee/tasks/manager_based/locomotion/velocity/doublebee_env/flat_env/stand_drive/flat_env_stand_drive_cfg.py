# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.velocity_env_cfg import DoubleBeeVelocityEnvCfg


@configclass
class DoubleBeeRewardsCfg:
    """Reward configuration for DoubleBee stand and drive task."""

    # Tracking rewards
    tracking_lin_vel = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b[:, :2] - env.command_manager.get_command("base_velocity")[:, :2]), dim=1),
        weight=1.0,
    )
    """Linear velocity tracking reward."""

    tracking_ang_vel = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.root_ang_vel_b[:, 2:3] - env.command_manager.get_command("base_velocity")[:, 2:3]), dim=1),
        weight=0.5,
    )
    """Angular velocity tracking reward."""

    # Stability rewards
    upright = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]), dim=1),
        weight=0.5,
    )
    """Upright orientation reward (penalize tilting)."""

    # Energy efficiency
    energy = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1),
        weight=0.0001,
    )
    """Energy efficiency reward."""

    # Action smoothness
    action_rate = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1),
        weight=0.01,
    )
    """Action rate penalty."""


@configclass
class DoubleBeeFlatStandDriveCfg(DoubleBeeVelocityEnvCfg):
    """Configuration for DoubleBee flat terrain stand and drive environment."""

    rewards: DoubleBeeRewardsCfg = DoubleBeeRewardsCfg()

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        
        # Override scene settings
        self.scene.robot = DOUBLEBEE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Episode settings
        self.episode_length_s = 20.0
        self.decimation = 4
        
        # Simulation settings
        self.sim.dt = 0.005
        
        # Command ranges (conservative for initial training)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class DoubleBeeFlatStandDriveCfg_PLAY(DoubleBeeFlatStandDriveCfg):
    """Configuration for DoubleBee flat terrain play/evaluation."""

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        
        # Disable observation noise for evaluation
        if hasattr(self.observations, 'policy'):
            if hasattr(self.observations.policy, 'enable_corruption'):
                self.observations.policy.enable_corruption = False
        
        # Render settings
        self.sim.render_interval = self.decimation
        
        # More aggressive command ranges for play
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)