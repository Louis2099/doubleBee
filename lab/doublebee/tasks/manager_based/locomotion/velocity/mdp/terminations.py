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

    time_out = DoneTerm(
        func=lambda env, env_ids=None: env.episode_length_buf >= env.max_episode_length,
        time_out=True,
    )
    """Episode timeout termination."""

    fall = DoneTerm(
        func=lambda env, env_ids=None: env.scene["robot"].data.root_lin_vel_b[:, 2] < -0.5,
    )
    """Falling termination."""

    tilt = DoneTerm(
        func=lambda env, env_ids=None: torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b), dim=1) > 0.5,
    )
    """Excessive tilt termination."""
