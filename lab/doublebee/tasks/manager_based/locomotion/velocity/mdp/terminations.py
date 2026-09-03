# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


def _goal_reached_done(env, env_ids=None):
    """Terminate the episode when the task is actually complete.

    2026-09-03: there was NO goal termination. TerminationsCfg carried only
    time_out, fall and tilt, so an episode continued after the robot arrived and
    the policy had no episode-level signal that arriving ends the task. In play
    the robot reaches the target and keeps going.

    This is a genuine terminal state, not a timeout, so time_out is left False
    and the value function does not bootstrap past it. The forfeited future
    reward is already compensated by terminal_goal_reached, whose lambda*(T - t)
    term exists precisely so that finishing early is not punished.

    Imported lazily: constraints.py pulls in env-side modules and a top-level
    import here would run at config-definition time.
    """
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import (
        goal_reached,
    )
    return goal_reached(env, distance_threshold=0.25).bool()


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

    goal_reached = DoneTerm(func=_goal_reached_done)
    """Task success termination. See _goal_reached_done."""
