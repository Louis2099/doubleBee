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
# TRIED 47 -> 23.6 on 2026-08-25 (the measured no-load ceiling) and REVERTED
# the same day. Two reasons:
#
# 1. penalize_action_rate acts on ACTION deltas, so halving the scale doubles
#    the action delta for the same physical command and QUADRUPLES its squared
#    penalty. Together with the weight going 0.1 -> 0.3 that is 12x the
#    smoothness pressure of the previous run -- enough to make the policy
#    sluggish and stop correcting, which is the opposite of the goal.
#
# 2. It may not even bind. Sim's [WHEEL] prints show target = +/-44.5 with
#    actual only +/-3.7 to 10 rad/s: the actuator saturates on effort_limit
#    long before the commanded target matters, so lowering the target from 47
#    to 23.6 changes the demand without changing what the wheel does.
#
# If the bang-bang survives the push DR and action_rate 0.3, this is worth
# testing ON ITS OWN with action_rate back at 0.1, so the two effects can be
# told apart.
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

        # OFFSET SUPPORT ADDED 2026-08-26, for the tied propeller action.
        #
        # The wheels need no offset (0 rad/s is a legitimate command) and the
        # original version refused one outright rather than carry a silent
        # code path. The propellers DO need one: they must map action [-1, 1]
        # onto [0, 375] rad/s, not [-375, +375], because negative propeller
        # speed is not a physically different command -- the aero model takes
        # |omega|, so a signed span would waste half the action range on a
        # mirror of itself.
        #
        # Resolved exactly like tied_scale: per-joint dict in the SAME joint
        # order, or a scalar applied to every joint. Kept out of cfg.scale for
        # the same CUDA reason documented above.
        # THE PER-JOINT OFFSET GOES IN `tied_offset`, NOT `offset` -- for exactly
        # the same reason tied_scale exists. IsaacLab's JointAction.__init__
        # resolves a dict `offset` against action_dim (which this class reports
        # as 1) and writes at joint index 1, which is out of bounds and takes
        # the process down with a device-side assert whose traceback points at
        # whatever line runs next. Learned the hard way 2026-08-26.
        raw_off = getattr(cfg, "offset", 0.0)
        if isinstance(raw_off, dict) or float(raw_off or 0.0) != 0.0:
            raise ValueError(
                "TiedJointVelocityAction: put the per-joint offset in "
                "cfg.tied_offset and leave cfg.offset at 0.0 (got %r). A dict "
                "in cfg.offset is resolved by the base class against "
                "action_dim=1 and crashes on the GPU." % (raw_off,))

        off = getattr(cfg, "tied_offset", None)
        if off is None:
            off = 0.0
        if isinstance(off, dict):
            missing = [nm for nm in names if nm not in off]
            if missing:
                raise ValueError(
                    "TiedJointVelocityAction: tied_offset has no entry for %s "
                    "(joints driven: %s)" % (missing, names))
            if len(off) != n:
                raise ValueError(
                    "TiedJointVelocityAction: tied_offset has %d entries for %d "
                    "joints" % (len(off), n))
            offs = torch.tensor([float(off[nm]) for nm in names],
                                device=self.device, dtype=torch.float32)
        else:
            offs = torch.full((n,), float(off or 0.0),
                              device=self.device, dtype=torch.float32)
        self._tied_offset = offs.unsqueeze(0)   # (1, n), broadcasts over envs

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
        self._processed_actions.add_(self._tied_offset)


@configclass
class TiedJointVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type[ActionTerm] = TiedJointVelocityAction

    tied_scale: dict[str, float] | tuple[float, ...] | None = None
    # Per-joint offset. MUST live here rather than in `offset`, which the base
    # class resolves against action_dim=1 -> out-of-bounds write on the GPU.
    tied_offset: dict[str, float] | tuple[float, ...] | None = None
    """Per-joint signed multiplier, applied ON TOP of the scalar `scale`.

    Keep `scale` a float and put the signs here -- see TiedJointVelocityAction
    for why a dict in `scale` trips a CUDA index-out-of-bounds assert. For the
    mirrored wheels: scale=47.0, tied_scale={"leftWheel": 1.0, "rightWheel": -1.0}.
    """



class CommonDiffJointVelocityAction(JointVelocityAction):
    """Drive TWO mirrored wheels from a (common, differential) pair.

    Added 2026-08-26 to give the robot heading authority without going back to
    independent left/right actions.

    WHY NOT LEFT/RIGHT. The policy's balance loop lives on ONE number -- how hard
    to drive both wheels together (measured corr(lean, wheel_action) = -0.50 in
    sim, -0.6..-0.8 on hardware). Splitting into left/right would force it to
    reconstruct that common mode from two coordinated outputs, and the
    convergence it took four days to reach is built on the single-action form.
    Here the balance loop keeps its own action untouched and steering is an
    ORTHOGONAL second action the policy can simply leave at zero.

    WHY IT IS NEEDED AT ALL. With wheels tied there is no yaw authority of any
    kind, so lateral drift is uncorrectable -- yet penalize_cross_track_error was
    charging -0.1576 for it, the LARGEST penalty in the set at iteration 2754.
    An unfixable penalty does not just waste reward, it distorts the value
    function everywhere. Either give the robot a rudder or stop the charge; this
    is the rudder.

    SIGN CONVENTION. The wheel joint axes are mirrored in the USD (leftWheel +X,
    rightWheel -X, verified from physics:localRot0), so:

        left_joint  = +common*scale + diff*diff_scale
        right_joint = -common*scale + diff*diff_scale

    Opposite joint signs = same physical direction = translation.
    Equal joint signs    = opposite physical direction = spin in place.

    diff_scale is deliberately much smaller than scale: steering is a trim, and
    a differential large enough to fight the balance loop would be a liability.
    """

    def __init__(self, cfg, env):
        if isinstance(cfg.scale, dict):
            raise ValueError(
                "CommonDiffJointVelocityAction: cfg.scale must be a float. A dict "
                "is resolved by the base class against action_dim and writes out "
                "of bounds on the GPU -- see TiedJointVelocityAction.")
        off = getattr(cfg, "offset", 0.0)
        if isinstance(off, dict) or float(off or 0.0) != 0.0:
            raise ValueError(
                "CommonDiffJointVelocityAction does not take an offset (got %r). "
                "Zero wheel velocity is a legitimate command." % (off,))
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

        names = self._resolved_joint_names()
        if len(names) != 2:
            raise ValueError(
                "CommonDiffJointVelocityAction drives exactly 2 joints, got %s"
                % (names,))
        self._names = names
        # +1 / -1 on the COMMON term, mirroring the USD axes. Same order as
        # joint_names, which preserve_order=True guarantees.
        cm = getattr(cfg, "common_sign", None) or {names[0]: 1.0, names[1]: -1.0}
        missing = [nm for nm in names if nm not in cm]
        if missing:
            raise ValueError(
                "CommonDiffJointVelocityAction: common_sign has no entry for %s "
                "(joints driven: %s)" % (missing, names))
        self._common = torch.tensor([[float(cm[nm]) * float(cfg.scale)
                                      for nm in names]], device=self.device)
        ds = float(getattr(cfg, "diff_scale", 0.0) or 0.0)
        self._diff = torch.full((1, 2), ds, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)

    def _resolved_joint_names(self):
        for attr in ("_joint_names", "joint_names"):
            v = getattr(self, attr, None)
            if v:
                return list(v)
        return list(self.cfg.joint_names)

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # (N,1)*(1,2) broadcasts to (N,2) for each term
        torch.mul(self._raw_actions[:, 0:1], self._common,
                  out=self._processed_actions)
        self._processed_actions.addcmul_(self._raw_actions[:, 1:2], self._diff)


@configclass
class CommonDiffJointVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type[ActionTerm] = CommonDiffJointVelocityAction

    # +/-1 per joint on the COMMON (translation) term. Defaults to
    # {first: +1, second: -1}, matching the mirrored USD wheel axes.
    common_sign: dict[str, float] | None = None
    # rad/s that |differential action| = 1 represents. Keep well below `scale`.
    diff_scale: float = 0.0


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
    """Reduced action space for DoubleBee robot.

    NAME IS HISTORICAL -- this is now THREE actions, not four. The class name is
    kept because it is imported in four places and a rename is churn with no
    benefit; the mapping below is the authority, not the name.

    This config eliminates redundant outputs by having only one action for servos
    and one for propellers. The environment will duplicate these with opposite signs.

    Action mapping as of 2026-08-26 (4 actions, obs 40):
    - [0]: wheel COMMON,       both wheels together -- translation, the balance loop
    - [1]: wheel DIFFERENTIAL, opposite directions -- yaw trim, diff_scale 8 rad/s
    - [2]: servo position,     TIED -- both servos, same angle off vertical
    - [3]: propeller velocity, TIED -- both props 0..375 rad/s, counter-rotating

    OBSERVATION SIZE: 40 = 2 wheel_vel + 2 servo_pos + 2 propeller_vel
                         + 3 lin_vel + 3 ang_vel + 3 gravity + 16 height_scan
                         + 2 contact + 3 command + 4 actions

    DEPLOYMENT. act_dim 4 now COLLIDES with the pre-2026-08-26 four-action layout
    {"wheel":(0,0),"servo":(1,1),"prop":(2,3)}, which pairs with obs 36. The two
    are told apart by obs_dim: 36 = legacy, 40 = this one. db_inference.py keys
    its layout on (act_dim, obs_dim) for exactly that reason.

    Superseded 2026-08-26 (kept for the record): the propellers used to stay
    independent because their differential is the only roll
    authority left once the wheels are tied.

    History: 6 actions (independent servos) -> 5 (tied servos, 2026-08-21)
    -> 4 (tied wheels, 2026-08-23). db_inference.py auto-detects which from the
    checkpoint's output width; see LAYOUTS there.
    """

    # TIED wheel action: ONE action dim drives BOTH wheels at the SAME ground
    # speed. Added 2026-08-23 at the user's request, after four hardware runs.
    #
    # scale 47. See WHEEL_VEL_LIMIT_RAD_S for why 23.6 was tried and reverted.
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
    # (COMMON, DIFFERENTIAL) 2026-08-26. Was TIED -- one action, zero steering.
    #
    # [0] common       both wheels together: translation. THE BALANCE LOOP.
    #                  Untouched from the tied version, same scale, so the
    #                  behaviour that converged is preserved exactly.
    # [1] differential opposite physical directions: yaw. A trim, not a drive.
    #
    # diff_scale 8.0 against scale 47.0 -- steering authority deliberately ~1/6
    # of translation. A differential large enough to fight the balance loop
    # would be a liability, and drift only needs correcting slowly.
    #
    # Added because with wheels tied the robot had NO yaw authority at all, so
    # lateral drift was uncorrectable -- while penalize_cross_track_error was
    # charging -0.1576 for it, the largest penalty in the set at iteration 2754.
    # An unfixable penalty distorts the value function everywhere.
    wheel_vel = CommonDiffJointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftWheel", "rightWheel"],
        scale=WHEEL_VEL_LIMIT_RAD_S,                    # must stay a float (CUDA)
        common_sign={"leftWheel": 1.0, "rightWheel": -1.0},
        diff_scale=8.0,
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
    #   processed = offset + scale*action, so this spans 0..375 rad/s per prop.
    #
    # SIZED FROM THE RIGHTING-TORQUE REQUIREMENT, 2026-08-23. At the confirmed
    # mass of 4.47 kg (W = 43.9 N) the gravity torque about the wheel axle is
    # 6.10*sin(theta) N*m, and the propellers act at a 0.443 m arm, so a servo at
    # the pi/4 limit produces T*sin(45deg)*0.443 of righting torque:
    #
    #   range      pwm    thrust   T/W    righting   max lean it can right
    #   0-250     1325    16.8 N   0.38     5.27      59.8 deg
    #   0-375     1488    25.8 N   0.59     8.07      >90  deg   <-- chosen
    #   0-500     1650    36.6 N   0.83    11.45      >90  deg
    #
    # 0-250 (the value staged on 2026-08-21, when the mass was believed to be
    # 2.76 kg) tops out at 59.8 deg -- BELOW the 70 deg termination threshold. It
    # left a dead band from 59.8 to 70 deg where the episode kept running but
    # recovery was physically impossible, which is the worst possible shape for
    # this reward set: the policy is asked to recover in a region where no action
    # can, so it learns to stop trying. That is the give-up behaviour.
    #
    # 0-375 covers the full pre-termination range with 1.4x margin at 70 deg,
    # while T/W 0.59 still leaves 41% of the weight on the wheels -- the props
    # stabilise and the wheels keep traction to drive, which is the intended
    # division of labour. 0-500 was rejected: T/W 0.83 approaches hover and
    # unloads the wheels (normal force -> 0), costing the traction the robot
    # needs to translate. See reward_vertical_thrust_support for that argument.
    #
    # For comparison the WHEELS supply 2 x 0.35 = 0.70 N*m, which rights 6.6 deg.
    # They are not the balance actuator on this machine and cannot be.
    #
    # DEPLOYMENT: db_inference.py --prop_rad_s_max 375 for checkpoints from here
    # on (was 250). Getting it wrong is a proportional thrust error at every
    # command. Deploy with --prop_scale 1.0 to match training authority.
    # TIED 2026-08-26. Was two independent dims (action space 4 -> 3).
    #
    # The stated reason for keeping them independent was that differential
    # thrust is the only roll authority. Measured, that authority is barely
    # used: the [BALANCE] probe reads roll sd 0.022 against pitch sd 0.354, 16x
    # smaller -- two laterally separated wheel contacts make this machine
    # statically stable in roll the way a car is. Pitch is the axis that kills
    # it, and pitch comes from the servos, not from a thrust differential.
    #
    # What it buys: one less exploration dimension on a task whose measured
    # bottleneck is discovery (episode length 30 steps == the passive fall
    # time), and thrust that is symmetric by construction.
    #
    # NOTE the evidence for asymmetry being harmful -- 162.9 vs -66.4 rad/s,
    # "sometimes only the left one fires" -- was all collected while the LEFT
    # PROPELLER JOINT WAS GEOMETRICALLY BROKEN (a spurious 80 deg rotation,
    # fixed in doubleBee_merged.usd the same day). The policy was treating them
    # as different actuators because they were. This tie is a deliberate choice
    # to constrain, not a conclusion from that data.
    #
    # tied_scale mirrors the sign so the pair counter-rotates, matching the old
    # per-joint offsets. That sign is bookkeeping only: aerodynamics.py takes
    # |omega| for thrust and applies zero reaction torque, so co- and
    # counter-rotation are indistinguishable in sim.
    propeller_vel = TiedJointVelocityActionCfg(
        asset_name="robot",
        joint_names=["leftPropeller", "rightPropeller"],
        scale=1.0,                                    # must stay a float (CUDA)
        tied_scale={"leftPropeller": 187.5, "rightPropeller": -187.5},
        tied_offset={"leftPropeller": 187.5, "rightPropeller": -187.5},
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