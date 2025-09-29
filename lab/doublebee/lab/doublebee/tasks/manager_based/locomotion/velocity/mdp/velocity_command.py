# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.utils import configclass


@configclass
class VelocityCommandCfg:
    """Velocity command specifications for DoubleBee robot."""

    # Base velocity command
    base_velocity = CommandTerm(
        func=lambda env: torch.zeros(env.num_envs, 3, device=env.device),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )
    """Base velocity command."""
