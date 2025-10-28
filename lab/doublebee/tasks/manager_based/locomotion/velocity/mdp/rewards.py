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
    """Reward specifications for DoubleBee robot."""

    # Tracking rewards
    tracking_lin_vel = RewTerm(
        func=lambda env: torch.exp(-torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b[:, :2] - env.command_manager.get_command("base_velocity")[:, :2]), dim=1)),
        weight=1.0,
    )
    """Linear velocity tracking reward."""

    # Stability rewards
    upright = RewTerm(
        func=lambda env: torch.exp(-torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b), dim=1)),
        weight=1.0,
    )
    """Upright orientation reward."""

    # Energy efficiency
    energy = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.joint_torques), dim=1),
        weight=0.01,
    )
    """Energy efficiency reward."""
