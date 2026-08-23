# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

# Servo position scale: policy [-1, 1] -> [-SERVO_POS_LIMIT_RAD, +...] rad, where
# the angle is measured OFF VERTICAL (joint 0 = propeller thrust axis at world +Z,
# verified against doubleBee_modified.usd).
#
# Reduced pi/2 -> pi/6 on 2026-08-23. Why:
#
# The policy saturates this term. Measured on hardware (transfer_clamp.csv and
# transfer_smooth.csv): the tied servo action sat at mean +0.979, max +1.000 for
# essentially every live tick of both runs. It is not a transient -- it is where
# the policy lives, because reward_progress_to_target (weight 10.0) is earned by
# vectoring thrust forward and outbids reward_props_upright (5.0) +
# reward_vertical_thrust_support (3.0).
#
# At pi/2 a saturated command is 90 deg off vertical, so cos(theta) = 0.0 and the
# ENTIRE thrust vector goes horizontal at the propellers' 0.443 m lever arm above
# the wheel axle. That is the torque that put the robot on its face in 0.14 s.
#
# Reward re-tuning was the obvious fix and is the wrong one: it only makes
# saturation less attractive, it cannot make it safe. Shrinking the range makes
# the saturated case itself harmless --
#
#     pi/2 (90 deg): cos = 0.00   full thrust horizontal
#     pi/6 (30 deg): cos = 0.87   87% still vertical, worst case
#
# (The previous value also disagreed with its own comments, which claimed
# +/-45 deg / pi/4 -- the constant had been doubled and the comments left stale.)
#
# RAISED pi/6 -> pi/4 later the same day. pi/6 was too tight, for a reason that
# only became visible once the give-up behaviour was diagnosed:
#
# The servo angle is measured off the BODY's up axis, not the world's. So the
# angle needed to keep thrust vertical is roughly the body's own lean. This
# robot's median operating tilt in sim is 42 deg. At pi/6 the servo physically
# cannot re-aim thrust to vertical once the body passes 30 deg -- which is most
# of the time -- so any reward that asks the policy to "re-aim rather than shut
# down" is geometrically unsatisfiable, and it will shut down instead. That is
# precisely the failure this file is trying to remove.
#
#     pi/2 (90 deg): cos = 0.00  full thrust horizontal, 7.4 N*m flip torque
#     pi/4 (45 deg): cos = 0.71  covers the 42 deg median tilt, 5.3 N*m worst case
#     pi/6 (30 deg): cos = 0.87  safest saturated, but cannot correct normal lean
#
# pi/4 is the compromise: enough authority to hold thrust vertical across the
# attitude range the robot actually lives in, while a saturated command still
# leaves 71% of thrust vertical instead of 0%.
#
# DEPLOYMENT: db_inference.py must be told this number. JAIOut servo units are
# [-1, 1] over +/-pi/2 rad, so the hardware command is
#   servo = -(2/pi) * theta = -(2/pi) * SERVO_POS_LIMIT_RAD * action
# i.e. a factor of 1/2 at pi/4, NOT the bare -action[2] that pi/2 implied. Pass
# --sim_servo_limit_rad 0.7854 for checkpoints trained after this change.
SERVO_POS_LIMIT_RAD = math.pi / 4  # 0.785 rad = 45 deg off vertical

# Wheel velocity scale: policy [-1, 1] -> [-47, +47] rad/s at the joint.
#
# The two wheel joints are MIRRORED in the USD, so driving forward requires
# joint velocities of opposite sign. Every wheel action term therefore carries
# +WHEEL_VEL_LIMIT on the left and -WHEEL_VEL_LIMIT on the right. Verified
# against hardware on 2026-08-23: sim predicts obs_0 shares the left action's
# sign and obs_1 opposes the right action's, and the measured correlations were
# +0.58/-0.71 (transfer_clamp), +0.13/-0.83 (transfer_auth_no_prop) and
# +0.21/-0.38 (transfer_auth). Do NOT "simplify" this to a single sign.
WHEEL_VEL_LIMIT_RAD_S = 47.0


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


class TiedJointVelocityAction(JointVelocityAction):
    """Drive N joints from ONE action dimension (velocity version).

    Same collapse as TiedJointPositionAction, for velocity-controlled joints.

    NOTE THE OPPOSITE SCALE CONVENTION FROM THE SERVOS. The servos want an
    EQUAL scale across joints, because their joint axes agree and equal scales
    give the same physical pose. The WHEELS want a MIRRORED scale
    (+47 / -47), because their joint axes are mirrored in the USD and opposite
    joint velocities are what "both wheels rolling forward" means. Using an
    equal scale here would make one action drive the robot in a permanent
    pivot -- exactly the behaviour this term exists to remove.

    THE PER-JOINT SCALE GOES IN `tied_scale`, NOT `scale`. `scale` must stay a
    plain float. This is not stylistic -- a dict in `scale` crashes on the GPU.

    IsaacLab's JointAction.__init__ resolves a dict `scale` by allocating a
    (num_envs, action_dim) buffer and writing into it at the JOINT indices, here
    [0, 1]. This subclass reports action_dim == 1, so the buffer is one column
    wide and the write to column 1 trips a device-side
    "index out of bounds" assert in IndexKernel.cu. Because CUDA reports
    asynchronously, the traceback surfaces at whatever line syncs next rather
    than inside the base constructor, which makes it look like a bug in this
    file. Observed 2026-08-23 on the training box.

    TiedJointPositionAction escapes this only because it is handed a scalar.
    """

    def __init__(self, cfg, env):
        if isinstance(cfg.scale, dict):
            raise ValueError(
                "TiedJointVelocityAction: cfg.scale must be a float, not a dict -- "
                "a dict here is resolved by the base class against action_dim=1 "
                "and trips a CUDA index-out-of-bounds assert. Put the per-joint "
                "signed scale in cfg.tied_scale instead and leave scale=1.0.")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)

        names = self._resolved_joint_names()
        n = len(names)
        tied = torch.zeros(1, n, device=self.device)
        spec = getattr(cfg, "tied_scale", None)
        if spec is None:
            # no per-joint spec: fall back to the scalar the base class got
            tied[:] = float(cfg.scale)
        elif isinstance(spec, dict):
            missing = [j for j in names if j not in spec]
            if missing:
                raise ValueError(
                    "TiedJointVelocityAction: tied_scale has no entry for %s "
                    "(joints resolved in order %s). Every tied joint needs an "
                    "explicit scale -- the SIGN is what makes the wheels roll "
                    "together rather than pivot." % (missing, names))
            for k, j in enumerate(names):
                tied[0, k] = float(spec[j]) * float(cfg.scale)
        else:
            if len(spec) != n:
                raise ValueError(
                    "TiedJointVelocityAction: tied_scale has %d entries for %d "
                    "joints %s." % (len(spec), n, names))
            for k, v in enumerate(spec):
                tied[0, k] = float(v) * float(cfg.scale)
        self._tied_scale = tied
        # (N,num_joints), replacing the base class's (N,action_dim) buffer
        self._processed_actions = torch.zeros(self.num_envs, n, device=self.device)

        # process_actions applies scale only. The wheels need no offset (0 rad/s
        # is a legitimate command), so rather than carry an unused code path,
        # refuse a configured offset outright -- silently dropping one would be a
        # constant velocity bias that is very hard to see in a training curve.
        off = getattr(cfg, "offset", 0.0)
        if isinstance(off, dict) or float(off or 0.0) != 0.0:
            raise ValueError(
                "TiedJointVelocityAction does not apply cfg.offset (got %r). Add "
                "offset support here before configuring one." % (off,))

    def _resolved_joint_names(self):
        """Joint names in the order this term drives them.

        IsaacLab has moved this attribute around between releases, so try the
        known names before falling back to cfg (which is correct here only
        because preserve_order=True is set on the cfg).
        """
        for attr in ("_joint_names", "joint_names"):
            names = getattr(self, attr, None)
            if names:
                return list(names)
        return list(self.cfg.joint_names)

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # (N,1) broadcasts against (1,num_joints) -> (N,num_joints)
        torch.mul(self._raw_actions, self._tied_scale, out=self._processed_actions)


@configclass
class TiedJointVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type[ActionTerm] = TiedJointVelocityAction

    tied_scale: dict[str, float] | tuple[float, ...] | None = None
    """Per-joint signed multiplier, applied ON TOP of the scalar `scale`.

    Keep `scale` a float and put the signs here -- see TiedJointVelocityAction
    for why a dict in `scale` trips a CUDA index-out-of-bounds assert. For the
    mirrored wheels: scale=47.0, tied_scale={"leftWheel": 1.0, "rightWheel": -1.0}.
    """


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
    # Policy output [-1, 1] -> position in [-scale, +scale] rad, measured OFF
    # VERTICAL. See SERVO_POS_LIMIT_RAD for why that is pi/4 and not pi/2.
    # Right servo uses negative scale so both servos move in opposite directions.
    propeller_servo_pos_left = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leftPropellerServo"],
        scale=SERVO_POS_LIMIT_RAD,  # +/-45 deg (pi/4 rad)
        use_default_offset=False,
        preserve_order=True,
    )
    propeller_servo_pos_right = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["rightPropellerServo"],
        scale=-SERVO_POS_LIMIT_RAD,  # +/-45 deg, inverted for opposite direction
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
    
    Action mapping as of 2026-08-23 (4 actions, obs 36):
    - [0]: wheel velocity, TIED -- both wheels, same ground speed, no steering
    - [1]: servo position, TIED -- both servos, same angle off vertical
    - [2]: left propeller velocity   0..250 rad/s
    - [3]: right propeller velocity  0..250 rad/s (mirrored, counter-rotating)

    The propellers stay independent: their differential is the only roll
    authority left once the wheels are tied.

    History: 6 actions (independent servos) -> 5 (tied servos, 2026-08-21)
    -> 4 (tied wheels, 2026-08-23). db_inference.py auto-detects which from the
    checkpoint's output width; see LAYOUTS there.
    """

    # TIED wheel action: ONE action dim drives BOTH wheels at the SAME ground
    # speed. Added 2026-08-23 at the user's request, after four hardware runs.
    #
    # scale 47 = ~2x the 23.6 rad/s actuator limit. Measured no-load ceiling is
    # 23.6 rad/s (db_wheels.py duty, 2026-08-20); 2x keeps the top half of the
    # action range saturated for easy exploration while the lower half stays
    # genuinely proportional. The old 500 saturated 95% of the range.
    #
    # NOTE THE MIRRORED SCALE, opposite to the tied SERVOS -- see
    # TiedJointVelocityAction. The wheel joints are mirrored in the USD, so
    # +47/-47 is what "both wheels rolling forward together" means.
    #
    # WHY TIE THEM. The independent version commanded a saturated counter-
    # rotation in every hardware run, with the target dead ahead and the robot
    # upright:
    #   transfer_smooth.csv   a0 -0.986  a1 +0.388  ->  des_l +13.9  des_r -5.5
    #   transfer_props.csv    a0 -0.975  a1 +0.661
    #   transfer_auth.csv     a0 -0.815  a1 +0.418  corr(a0,a1) = -0.54
    # Every run that drove straight did so only because db_inference.py's
    # --max_wheel_diff clamp overrode the policy. Tying the wheels here makes
    # that clamp unnecessary rather than load-bearing.
    #
    # COST, stated plainly: this removes ALL yaw authority. The robot can no
    # longer steer, so it must be placed already pointing at the step. That is
    # acceptable for the climb task and not for navigation. To go back, restore
    # the two separate terms below and set --max_wheel_diff on the deploy side.
    # NOTE: scale stays a FLOAT and the mirroring lives in tied_scale. A dict in
    # `scale` is resolved by the base class against action_dim=1 and trips a CUDA
    # index-out-of-bounds assert -- see TiedJointVelocityAction.
    wheel_vel = TiedJointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftWheel", "rightWheel"],
        scale=WHEEL_VEL_LIMIT_RAD_S,
        tied_scale={"leftWheel": 1.0, "rightWheel": -1.0},
        use_default_offset=False,
        preserve_order=True,
    )
    # Pre-2026-08-23 independent wheels, kept for A/B:
    # wheel_vel_left = mdp.JointVelocityActionCfg(
    #     asset_name="robot", joint_names=["leftWheel"], scale=47.0,
    #     use_default_offset=False, preserve_order=True)
    # wheel_vel_right = mdp.JointVelocityActionCfg(
    #     asset_name="robot", joint_names=["rightWheel"], scale=-47.0,
    #     use_default_offset=False, preserve_order=True)

    # TIED servo action: ONE action dim drives BOTH servos to the SAME angle.
    # Note the scalar scale rather than the old mirrored dict -- see
    # TiedJointPositionAction. The two arms are physically symmetric, so a
    # single command is all the policy needs, and it removes the failure mode
    # where the policy poses them opposed and kills its own lift.
    #
    # Side benefit of the pi/4 limit: soft_joint_pos_limit_factor=0.8 against the
    # USD's +/-90 deg servo limits clamps the sim joint at 72 deg, while the real
    # servo travels the full 90 deg. At pi/2 the policy saturated into that gap,
    # so sim kept cos(72)=0.31 of its thrust vertical where hardware kept
    # cos(88)=0.03 -- sim never experienced the state the robot was actually in.
    # At 45 deg the command never reaches the 72 deg soft limit, so the gap closes.
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