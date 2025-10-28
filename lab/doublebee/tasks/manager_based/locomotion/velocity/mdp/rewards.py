# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


@configclass
class RewardsCfg:
    """Reward specifications for DoubleBee velocity tracking task."""

    # ========== Velocity Command Tracking Rewards ==========
    
    track_lin_vel_xy = RewTerm(
        func=lambda env: torch.exp(
            -torch.sum(
                torch.square(
                    env.scene["robot"].data.root_lin_vel_b[:, :2] 
                    - env.command_manager.get_command("base_velocity")[:, :2]
                ), 
                dim=1
            )
        ),
        weight=1.0,
    )
    """Horizontal linear velocity tracking (x, y). Exponential reward: exp(-||v_xy - v_cmd_xy||²)"""

    track_lin_vel_z = RewTerm(
        func=lambda env: torch.exp(
            -torch.square(
                env.scene["robot"].data.root_lin_vel_b[:, 2] 
                - env.command_manager.get_command("base_velocity")[:, 2]
            )
        ),
        weight=1.0,
    )
    """Vertical linear velocity tracking (z). Exponential reward: exp(-||v_z - v_cmd_z||²)"""

    track_ang_vel_z = RewTerm(
        func=lambda env: torch.exp(
            -torch.square(
                env.scene["robot"].data.root_ang_vel_b[:, 2] 
                - env.command_manager.get_command("base_velocity")[:, 3]
            )
        ),
        weight=0.5,
    )
    """Yaw angular velocity tracking. Exponential reward: exp(-||ω_z - ω_cmd_z||²)"""

    # ========== Efficiency Rewards ==========
    
    propeller_efficiency = RewTerm(
        func=lambda env: -torch.sum(
            torch.square(
                env.scene["robot"].data.joint_vel[:, 
                    [env.scene["robot"].joint_names.index("leftPropeller"),
                     env.scene["robot"].joint_names.index("rightPropeller")]
                ]
            ),
            dim=1
        ),
        weight=0.0001,
    )
    """Penalize excessive propeller speeds. Since thrust ∝ ω², high speeds are inefficient."""

    action_smoothness = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.action_manager.action), dim=1),
        weight=0.001,
    )
    """Penalize large action magnitudes to encourage smooth, energy-efficient control."""
