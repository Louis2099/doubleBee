# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


@configclass
class ObservationsCfg:
    """Observation specifications for DoubleBee robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint states
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.15)

        # Base states
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Command
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Policy observations
    policy: PolicyCfg = PolicyCfg()

    # Value observations (same as policy for simplicity)
    value: PolicyCfg = PolicyCfg()