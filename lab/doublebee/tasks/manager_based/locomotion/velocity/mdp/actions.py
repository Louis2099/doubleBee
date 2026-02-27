# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

# ±45° in rad, for servo position scale so that policy [-1, 1] → [-45°, 45°]
SERVO_POS_LIMIT_RAD = math.pi / 4  # 0.785 rad


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
    # Policy output [-1, 1] → position in [-scale, scale] rad. Scale = π/4 gives ±45°.
    # Right servo uses negative scale so both servos move in opposite directions.
    propeller_servo_pos_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leftPropellerServo"],
        scale=SERVO_POS_LIMIT_RAD,  # ±45° (π/4 rad)
        use_default_offset=False,
        preserve_order=True,
    )
    propeller_servo_pos_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["rightPropellerServo"],
        scale=-SERVO_POS_LIMIT_RAD,  # ±45°, inverted for opposite direction
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
        # scale=300.0, # for speed based thrust model
        scale = 100.0, # for PWM based thrust model, the actual scale is 2000, 10 is multiplied in aerodynamics.py to prevent large rotational forces
        # scale=6.0,
        use_default_offset=False,
        preserve_order=True,
    )
    propeller_vel_right = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rightPropeller"],
        #scale=-500.0,  # Negative scale to invert for gyroscopic balance
        # scale=-6.0,
        # scale=-300.0, # for speed based thrust model
        scale=-100.0, # for PWM based thrust model
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ActionsCfgWheelsServosOnly(ActionsCfg):
    """Action config with only wheels and propeller servos (no propeller velocity).

    Use for tasks where propellers are not used for thrust but servo tilt is still controlled.
    """

    propeller_vel_left = None
    propeller_vel_right = None


@configclass
class ActionsCfgWheelsOnly(ActionsCfg):
    """Action config with only wheel velocity (no servos, no propeller velocity).

    Use for inverted-pendulum or balance tasks where only ground locomotion is actuated;
    servos and propellers are fixed (e.g. servos at 0, propellers off).
    """

    propeller_servo_pos_left = None
    propeller_servo_pos_right = None
    propeller_vel_left = None
    propeller_vel_right = None