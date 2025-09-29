# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


@configclass
class TerminationsCfg:
    """Termination specifications for DoubleBee robot."""

    # Episode timeout
    time_out = DoneTerm(
        func=lambda env: env.episode_length_buf >= env.max_episode_length,
        mode="truncate",
    )
    """Episode timeout termination."""

    # Falling termination
    fall = DoneTerm(
        func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, 2] < -0.5,
        mode="terminate",
    )
    """Falling termination."""

    # Excessive tilt termination
    tilt = DoneTerm(
        func=lambda env: torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b), dim=1) > 0.5,
        mode="terminate",
    )
    """Excessive tilt termination."""
