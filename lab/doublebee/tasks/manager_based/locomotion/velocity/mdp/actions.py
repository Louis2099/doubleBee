# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

# Servo position scale: policy [-1, 1] -> [-pi/2, +pi/2] rad.
SERVO_POS_LIMIT_RAD = math.pi / 2  # 1.57 rad


class TiedJointPositionAction(JointPositionAction):
    """Drive N joints from ONE action dimension.

    A normal JointPositionAction gives one action per joint, so listing two
    servos costs two action dims and lets the policy pose them independently.
    This term collapses that to a single dim: the one command is broadcast to
    every listed joint, and each joint still gets its own scale/offset.

    IMPORTANT: use an EQUAL scale across the joints if you want them tied to
    the same physical pose. A mirrored scale dict (+pi/2 / -pi/2) would drive
    them to equal-and-opposite angles, i.e. the arms permanently opposed --
    which on hardware means one propeller pointing at the ground.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # the base class sized this one-action-per-joint; collapse to a single dim
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # (N,1) broadcasts against (N,num_joints) scale/offset
        self._processed_actions = self._raw_actions * self._scale + self._offset


@configclass
class TiedJointPositionActionCfg(mdp.JointPositionActionCfg):
    class_type: type[ActionTerm] = TiedJointPositionAction


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
        scale = 500.0, # for PWM based thrust model, the actual scale is 2000, 10 is multiplied in aerodynamics.py to prevent large rotational forces
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
        scale=-500.0, # for PWM based thrust model
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ActionsCfg4D:
    """Reduced 4D action space for DoubleBee robot.
    
    This config eliminates redundant outputs by having only one action for servos
    and one for propellers. The environment will duplicate these with opposite signs.
    
    Action mapping:
    - [0]: left wheel velocity
    - [1]: right wheel velocity (negative scale)
    - [2]: servo position (duplicated to both servos with opposite signs)
    - [3]: propeller: [-1,1] -> [0,1] -> left [0,500] rad/s, right [0,-500] rad/s -> PWM 1000-2000 -> thrust
    """

    # Wheel velocity actions (still separate for differential drive).
    # scale 47 = ~2x the 23.6 rad/s actuator limit. Measured no-load ceiling is
    # 23.6 rad/s (db_wheels.py duty, 2026-08-20); 2x keeps the top half of the
    # action range saturated for easy exploration while the lower half stays
    # genuinely proportional. The old 500 saturated 95% of the range.
    wheel_vel_left = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftWheel"],
        scale=47.0,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_vel_right = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rightWheel"],
        scale=-47.0,  # Negative scale for opposite rotation
        use_default_offset=False,
        preserve_order=True,
    )

    # TIED servo action: ONE action dim drives BOTH servos to the SAME angle.
    # Note the scalar scale rather than the old mirrored dict -- see
    # TiedJointPositionAction. The two arms are physically symmetric, so a
    # single command is all the policy needs, and it removes the failure mode
    # where the policy poses them opposed and kills its own lift.
    propeller_servo_pos = TiedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["leftPropellerServo", "rightPropellerServo"],
        scale=SERVO_POS_LIMIT_RAD,
        use_default_offset=False,
        preserve_order=True,
    )

    # Propeller velocity, one action per propeller (still independent -- the
    # differential is the only roll authority available).
    #   processed = offset + scale*action, so this spans 0..250 rad/s per prop.
    #
    # HALVED from 250/250 (which spanned 0..500). Rationale: at 4.47 kg the
    # airframe has T/W 0.84 at full thrust, and every hardware run so far has
    # been flown with --prop_scale 0.3..0.6 because full authority flips the
    # robot. Capping thrust HERE instead means the policy learns a strategy that
    # works at the authority it will actually be given, rather than learning to
    # rely on thrust that gets attenuated away at deploy time. Deploy with
    # --prop_scale 1.0 to match.
    propeller_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftPropeller", "rightPropeller"],
        scale={"leftPropeller": 125.0, "rightPropeller": -125.0},
        offset={"leftPropeller": 125.0, "rightPropeller": -125.0},
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
class ActionsCfgWheelsServosOnly4D(ActionsCfg4D):
    """4D action config with only wheels and propeller servos (no propeller velocity)."""

    propeller_vel = None


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


@configclass
class ActionsCfgWheelsOnly4D(ActionsCfg4D):
    """4D action config with only wheels (no servos, no propeller velocity)."""

    propeller_servo_pos = None
    propeller_vel = None