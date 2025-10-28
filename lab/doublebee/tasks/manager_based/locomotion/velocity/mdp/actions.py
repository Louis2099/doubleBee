# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for DoubleBee robot."""

    # Wheel velocity actions (for ground locomotion)
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftWheel", "rightWheel"],
        scale=200.0,
        use_default_offset=False,
        preserve_order=True,
    )

    # Propeller servo position actions (for propeller tilt control)
    propeller_servo_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leftPropellerServo", "rightPropellerServo"],
        scale=2.0,
        use_default_offset=False,
        preserve_order=True,
    )

    # Propeller velocity actions (for propeller rotation)
    propeller_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftPropeller", "rightPropeller"],
        scale=600.0,
        use_default_offset=False,
        preserve_order=True,
    )
