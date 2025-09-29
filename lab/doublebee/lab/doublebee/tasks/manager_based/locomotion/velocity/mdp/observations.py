# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


@configclass
class ObservationsCfg:
    """Observation specifications for DoubleBee robot."""

    @configclass
    class PolicyCfg(ObsTerm):
        """Policy observations."""

        def __init__(self):
            super().__init__()

        # Joint states
        joint_pos = ObsTerm(
            func=lambda env: env.scene["robot"].data.joint_pos,
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.01, 0.01),
                add_noise_before_normalization=True,
            ),
        )
        """Joint positions."""

        joint_vel = ObsTerm(
            func=lambda env: env.scene["robot"].data.joint_vel,
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.1, 0.1),
                add_noise_before_normalization=True,
            ),
        )
        """Joint velocities."""

        # Base states
        base_lin_vel = ObsTerm(
            func=lambda env: env.scene["robot"].data.root_lin_vel_b,
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.1, 0.1),
                add_noise_before_normalization=True,
            ),
        )
        """Base linear velocity in base frame."""

        base_ang_vel = ObsTerm(
            func=lambda env: env.scene["robot"].data.root_ang_vel_b,
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.1, 0.1),
                add_noise_before_normalization=True,
            ),
        )
        """Base angular velocity in base frame."""

        # Command
        command = ObsTerm(
            func=lambda env: env.command_manager.get_command("base_velocity"),
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.1, 0.1),
                add_noise_before_normalization=True,
            ),
        )
        """Velocity command."""

        # Previous actions
        previous_actions = ObsTerm(
            func=lambda env: env.action_manager.get_term("base_actions").raw_actions,
            noise=ObsTerm.NoiseCfg(
                noise_func="uniform",
                noise_range=(-0.01, 0.01),
                add_noise_before_normalization=True,
            ),
        )
        """Previous actions."""

    # Policy observations
    policy: PolicyCfg = PolicyCfg()

    # Value observations (same as policy for simplicity)
    value: PolicyCfg = PolicyCfg()