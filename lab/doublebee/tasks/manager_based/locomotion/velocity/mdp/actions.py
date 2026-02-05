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
    # Split into left and right to allow sign inversion for opposite directions
    # Right wheel is inverted (negative scale) so wheels can rotate in opposite directions
    wheel_vel_left = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftWheel"],
        # scale=250.0,
        scale=500.0,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_vel_right = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rightWheel"],
        # scale=-250.0,  # Negative scale to invert for opposite direction
        scale=-500.0,  # Negative scale to invert for opposite direction
        use_default_offset=False,
        preserve_order=True,
    )

    # Propeller servo position actions (for propeller tilt control)
    # Split into left and right to allow sign inversion for opposite directions
    # Right servo is inverted (negative scale) so servos can move in opposite directions
    propeller_servo_pos_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leftPropellerServo"],
        scale=2.0,
        use_default_offset=False,
        preserve_order=True,
    )
    propeller_servo_pos_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["rightPropellerServo"],
        scale=-2.0,  # Negative scale to invert for opposite direction
        use_default_offset=False,
        preserve_order=True,
    )

    # Propeller velocity actions (for propeller rotation)
    # Split into left and right to allow sign inversion for gyroscopic balance
    # Right propeller is inverted (negative scale) so propellers spin in opposite directions
    propeller_vel_left = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftPropeller"],
        # scale=500.0,
        scale=300.0,
        # scale=6.0,
        use_default_offset=False,
        preserve_order=True,
    )
    propeller_vel_right = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rightPropeller"],
        #scale=-500.0,  # Negative scale to invert for gyroscopic balance
        # scale=-6.0,
        scale=-300.0,
        use_default_offset=False,
        preserve_order=True,
    )
