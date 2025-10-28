# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.velocity_env_cfg import DoubleBeeVelocityEnvCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import aerodynamics


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
class DoubleBeeEventsCfg:
    """Event configuration for DoubleBee stand and drive task."""

    # Apply propeller aerodynamics every physics step
    propeller_aerodynamics = EventTerm(
        func=aerodynamics.apply_propeller_aerodynamics,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # Run every step
        params={
            "propeller_joint_names": ("leftPropeller", "rightPropeller"),
            "propeller_body_names": ("leftPropeller", "rightPropeller"),
            "thrust_coefficient": 1e-4,  # Increased for testing! (was 0.1)
            "drag_coefficient": 1e-5,
            "max_thrust_per_propeller": 500.0,  # Increased max thrust
            "visualize": True,
            "visualize_scale": 0.05,
            # asset_cfg defaults to SceneEntityCfg("robot")
        },
    )

    # Reset robot state on reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class DoubleBeeFlatStandDriveCfg(DoubleBeeVelocityEnvCfg):
    """Configuration for DoubleBee flat terrain stand and drive environment."""

    rewards: DoubleBeeRewardsCfg = DoubleBeeRewardsCfg()
    events: DoubleBeeEventsCfg = DoubleBeeEventsCfg()

    # Provide (optional) task-specific constraint terms override if needed in future

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