# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.velocity_env_cfg import DoubleBeeVelocityEnvCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import aerodynamics
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import events as mdp  # Use local events module instead of source
from isaaclab.envs.mdp import randomize_actuator_gains, randomize_rigid_body_mass, push_by_setting_velocity
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import RewardsCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.terrain_config.stair_config import StairConfigCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.velocity_command import TerrainTargetDirectionCommandCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import ActionsCfg4D
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import ActionsCfg4DConstantThrust
import os


# Note: Using RewardsCfg from mdp/rewards.py instead of local DoubleBeeRewardsCfg
# The local DoubleBeeRewardsCfg has been replaced with RewardsCfg which uses:
# - Exponential rewards (exp(-error²)) instead of quadratic (-error²)
# - Separate Z velocity tracking
# - Propeller-specific efficiency instead of total energy
# - Action magnitude penalty instead of action rate penalty
# - No upright reward (removed)
#
# --- How events are managed through the cfg (step by step) ---
# 1. This class (DoubleBeeEventsCfg) is assigned to the env config as events=DoubleBeeEventsCfg().
# 2. Each attribute (e.g. apply_wheel_friction, propeller_aerodynamics, reset_base) is an EventTerm
#    with func=..., mode="startup"|"reset"|"interval", and params={...}.
# 3. The env builds an EventManager from cfg.events; the manager groups terms by mode.
# 4. When the env runs:
#    - "startup": event_manager.apply(mode="startup") is called once in load_managers() after
#      the scene and managers are set up. Use for one-time setup (e.g. PhysX materials).
#    - "reset": event_manager.apply(mode="reset", env_ids=env_ids, ...) is called inside
#      _reset_idx(env_ids) for each batch of envs that are reset.
#    - "interval": event_manager.apply(mode="interval", dt=step_dt) is called every
#      simulation step after physics and reset handling.
# 5. The manager calls each term's func(env, env_ids, **params) (env_ids is None for startup).

@configclass
class DoubleBeeEventsCfg:
    """Event configuration for DoubleBee hybrid (propeller + wheel) staircase task."""

    # https://github.com/Louis2099/doubleBee/commit/eaa18f843379c086a4de37a52bbf24d6e2039bc6
    # https://github.com/Louis2099/doubleBee/commit/d264a864d8382b9910b8446cf217c52d0bad6120

    # One-time at spawn: assign PhysX material to wheel colliders so friction is correct
    # apply_wheel_friction = EventTerm(
    #     func=mdp.apply_wheel_physx_material,
    #     mode="startup",
    #     params={
    #         "robot_prim_path_template": "/World/envs/env_{}/Doublebee",
    #         "static_friction": 1.2,
    #         "dynamic_friction": 0.9,
    #         "restitution": 0.0,
    #         "friction_combine_mode": "multiply",
    #         "restitution_combine_mode": "multiply",
    #     },
    # )

    # Apply propeller aerodynamics every physics step
    propeller_aerodynamics = EventTerm(
        func=aerodynamics.apply_propeller_aerodynamics,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # Run every step
        params={
            "propeller_joint_names": ("leftPropeller", "rightPropeller"),
            "propeller_body_names": ("leftPropeller", "rightPropeller"),
            "thrust_coefficient": 1e-4,  # Kept for compatibility (unused in PWM model)
            "max_thrust_per_propeller": 500.0,  # Maximum thrust per propeller
            "visualize": False,  # carb.plugins not available in all Isaac builds; set True only when debugging
            "visualize_scale": 0.2,  # Increased scale for better visibility
            # asset_cfg defaults to SceneEntityCfg("robot")
        },
    )

    # Joint action logging disabled.
    # log_prop_servo_joint_state = EventTerm(
    #     func=joint_logging.log_propeller_servo_joint_state,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "log_path": "prop_servo_joint_log.csv",
    #         "log_interval_steps": 1,
    #         "env_ids_to_log": [0],
    #     },
    # )

    # Domain randomization: thrust output +/-20% per env per propeller (at reset).
    #
    # 2026-09-01: was commented out, so EVERY run to date trained with a perfect
    # thrust model. That is the single largest sim2real gap on this robot -- a
    # whole hardware session went into prop_map / prop_scale / prop_min_frac
    # precisely because the real thrust curve is not the modelled one.
    #
    # Gated on DOUBLEBEE_DR so the w_E ablation and the robustness run cannot
    # contaminate each other: the ablation needs all five runs identical except
    # w_E, and silently flipping DR under it would invalidate the table.
    #   DOUBLEBEE_DR unset/0 -> no DR   (use for the w_E sweep)
    #   DOUBLEBEE_DR=1       -> +/-20%  (use for the sim2real robustness run)
    sample_thrust_scale_dr = (
        EventTerm(
            func=aerodynamics.sample_thrust_scale_dr,
            mode="reset",
            params={"range_low": 0.8, "range_high": 1.2, "num_propellers": 2},
        )
        if os.environ.get("DOUBLEBEE_DR", "0") not in ("0", "", "false", "False")
        else None
    )

    # NOTE: Reset/spawn is controlled here. Position is sampled from terrain "init_pos" flat patches.
    # - pose_range: roll, pitch, yaw in rad. Only orientation is randomized (position from terrain).
    # - velocity_range: x, y, z in m/s (linear); roll, pitch, yaw in rad/s (angular). Sampled uniformly.
    # To randomize initial velocity and orientation, set non-zero (min, max) for the desired keys.
    reset_base = EventTerm(
        func=mdp.reset_root_state_climb_commit_mix,
        mode="reset",
        params={
            "pose_range": {
                # "roll": (0.0, 0.0),       # No roll randomization - perfectly upright
                # "pitch": (0.0, 0.0),      # No pitch randomization - perfectly level
                # +/-3 deg -> +/-20 deg, 2026-08-26. THE SERVOS HAD NOTHING TO LEARN.
                #
                # Resetting near 0 and then balancing under 2 deg means the
                # policy never spends time in the 20-60 deg band, and the servo
                # requirement there is a different problem entirely:
                #
                #   restoring = T*Lprop*sin(SERVO)   tipping = W*Lcom*sin(LEAN)
                #
                #   lean  2 deg -> servo needs  1.3 deg to break even
                #   lean 20 deg -> servo needs 13.4 deg
                #   lean 50 deg -> servo needs 31.0 deg
                #
                # At 2 deg a servo parked near zero is CORRECT and sufficient, so
                # that is what it learned. Measured on hardware (hw_v18.csv) the
                # robot reaches 50 deg within half a second of engaging while the
                # servo sits near 0 -- below break-even, contributing nothing,
                # exactly as trained.
                #
                # The capability is already there: at the 45 deg servo limit and
                # 10.6 N the restoring moment beats gravity at EVERY lean up to
                # 90 deg. The policy simply never had a reason to ask for it.
                # 20 -> 11 deg on 2026-08-26, after measuring what 20 did.
                # At +/-20 the run CONVERGED (alpha 0.0163 by iteration 747,
                # three times faster than the previous run reached 0.005) to a
                # policy that does not balance: episode length 32-43 against a
                # passive fall time of 31, corr(lean, wheel_action) = +0.011,
                # terrain_levels back to 0.0020. Starting every episode already
                # losing taught it that falling is normal.
                # 11 deg still needs 13 deg of servo to break even, so the servo
                # still has a job, without conceding the episode at t=0.
                # Back to +/-3 deg on 2026-08-26. +/-20 converged fast to a
                # non-balancing policy; +/-11 was untested and this run is
                # already carrying enough new variables. The servo-behaviour
                # argument for reset DR still stands, but it is not what the
                # paper needs in the next 19 days.
                # RESTORED (-0.2, 0.2) from model_3500_params/env.yaml.
                "roll": (-0.2, 0.2),      # +/-11 deg
                "pitch": (-0.2, 0.2),     # +/-11 deg
                "yaw" :(0.0, 0.0),
                # "yaw_noise": (0.0, 0.0),  # No yaw noise - perfect alignment toward target 
                "yaw_noise": (-0.05, 0.05), # ±6° yaw noise
            },
            "velocity_range": {
                "x": (0.0, 0.0),      # No initial linear velocity in X
                "y": (0.0, 0.0),      # No initial linear velocity in Y
                "z": (0.0, 0.0),      # No initial linear velocity in Z
                "roll": (0.0, 0.0),   # No initial angular velocity around X (roll rate)
                "pitch": (0.0, 0.0),  # No initial angular velocity around Y (pitch rate)
                "yaw": (0.0, 0.0),    # No initial angular velocity around Z (yaw rate - NOT SPINNING)
            },
            "align_axis": "x",  # Align on X axis (robot moves along Y axis),
            "frac_commit": 0.25,
            "commit_pitch": 0.05,
        },
    )

    # CRITICAL: Reset joints to default positions to prevent error accumulation
    # Without this, joints retain their previous state, causing PD controller to
    # try to move from reset position to previous target, accumulating error
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.0, 0.0),  # Reset to exact default positions (0.0 for all joints)
    #         "velocity_range": (0.0, 0.0),  # Reset to zero velocity
    #     },
    # )

    # randomize_robot_mass = EventTerm(
    #     func=randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.85, 1.15),  # ±15% scale
    #         "operation": "scale",
    #     },
    # )

    # randomize_robot_mass = EventTerm(
    #     func=randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),  # main body only, not .*
    #         "mass_distribution_params": (0.95, 1.05),  # ±5% scale
    #         "operation": "scale",
    #     },
    # )

    # randomize_com = EventTerm(
    #     func=mdp.randomize_com_positions,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
    #         "com_distribution_params": (-0.003, 0.003),  # ±1cm COM offset
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # ) # WASS 0.01

        # push_robot roll/pitch RAISED 0.05 -> 0.25 rad/s on 2026-08-25.
    #
    # At 0.05 the disturbance was invisible. With tau_eff ~102 ms the lean
    # grows as theta(t) = omega*tau*sinh(t/tau), so 0.05 rad/s reaches only
    # ~3 deg at 0.3 s -- far inside the ~23 deg the wheels alone can recover
    # (they deliver ~4.2 m/s^2 of contact-point acceleration, and holding a
    # lean needs g*tan(theta)).
    #
    # So the wheels were never challenged, thrust was never necessary, and
    # the policy learned WHEELS-ONLY locomotion. Confirmed in sim AND on
    # hardware: propeller actions average about -0.4 (a third throttle) and
    # total thrust sits below the 7.17 N static-stability threshold on most
    # ticks. That strategy cannot work on the real robot, where the moment
    # you let go it meets a disturbance the wheels cannot absorb.
    #
    # 0.25 rad/s reaches ~14 deg at 0.3 s -- past where the wheels are
    # comfortable, short of the ~0.39 rad/s that reaches 23 deg and makes a
    # fall unavoidable. Enough to make thrust genuinely useful without
    # making the task unlearnable.
    push_robot = EventTerm(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),   # was (8,15) — more frequent so policy sees many slow tilts
        params={
            "velocity_range": {
                "x": (-0.05, 0.05),    # keep gentle
                "y": (-0.05, 0.05),
                # add a small angular component to induce actual tilt, not just translation
                "roll": (-0.25, 0.25),  # see the note above → slow tilts the servo must respond to
                "pitch": (-0.25, 0.25),
            },
        },
    )

    randomize_joint_actuator_gains = EventTerm(
        func=randomize_actuator_gains,
        mode="startup",  # once at start, not every reset
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leftWheel", "rightWheel"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # randomize_servo_actuator_gains = EventTerm(
    #     func=randomize_actuator_gains,   # no mdp. prefix — imported from isaaclab.envs.mdp
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["leftPropellerServo", "rightPropellerServo"]),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # randomize_friction = EventTerm(
    #         func=mdp.randomize_rigid_body_material,
    #         mode="reset",
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", body_names=["leftWheel", "rightWheel"]),
    #             "static_friction_range": (0.8, 1.2),
    #             "dynamic_friction_range": (0.7, 1.0),
    #             "restitution_range": (0.0, 0.0),
    #             "num_buckets": 64,
    #             "make_consistent": True,  # keeps dynamic <= static
    #         },
    #     )

@configclass
class DoubleBeeEventsCfg_PLAY:
    """Event configuration for DoubleBee hybrid staircase task in play mode with aligned initialization."""

    # Same startup event as training so wheel PhysX material is applied
    # apply_wheel_friction = EventTerm(
    #     func=mdp.apply_wheel_physx_material,
    #     mode="startup",
    #     params={
    #         "robot_prim_path_template": "/World/envs/env_{}/Doublebee",
    #         "static_friction": 1.2,
    #         "dynamic_friction": 0.9,
    #         "restitution": 0.0,
    #         "friction_combine_mode": "multiply",
    #         "restitution_combine_mode": "multiply",
    #     },
    # )

    # Apply propeller aerodynamics every physics step
    propeller_aerodynamics = EventTerm(
        func=aerodynamics.apply_propeller_aerodynamics,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # Run every step
        params={
            "propeller_joint_names": ("leftPropeller", "rightPropeller"),
            "propeller_body_names": ("leftPropeller", "rightPropeller"),
            "thrust_coefficient": 1e-4,  # Kept for compatibility (unused in PWM model)
            "max_thrust_per_propeller": 500.0,  # Maximum thrust per propeller
            "visualize": False,  # carb.plugins not available in all Isaac builds; set True only when debugging
            "visualize_scale": 0.2,  # Increased scale for better visibility
            # asset_cfg defaults to SceneEntityCfg("robot")
        },
    )

    # NOTE: Reset robot state with aligned start/end positions for play mode
    # This ensures start and end points share the same X or Y coordinate, and robot faces the target
    reset_base = EventTerm(
        func=mdp.reset_root_state_climb_commit_mix,
        mode="reset",
        params={
            "pose_range": {
                "roll": (0.0, 0.0),       # No roll randomization - perfectly upright
                "pitch": (0.0, 0.0),      # No pitch randomization - perfectly level
                "yaw_noise": (0.0, 0.0),  # No yaw noise - perfect alignment toward target
            },
            "velocity_range": {
                "x": (0.0, 0.0),      # No initial linear velocity in X
                "y": (0.0, 0.0),      # No initial linear velocity in Y
                "z": (0.0, 0.0),      # No initial linear velocity in Z
                "roll": (0.0, 0.0),   # No initial angular velocity around X (roll rate)
                "pitch": (0.0, 0.0),  # No initial angular velocity around Y (pitch rate)
                "yaw": (0.0, 0.0),    # No initial angular velocity around Z (yaw rate - NOT SPINNING)
            },
            "align_axis": "x",  # Align on X axis (robot moves along Y axis),
            "frac_commit": 0.0,
            "commit_pitch": 0.0,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 2026-08-26: PROPELLERS START SPUN UP. Must run AFTER reset_robot_joints,
    # which zeroes every joint velocity. See mdp.events.reset_propeller_spin --
    # the short version is that measured episode length (30 steps) equals the
    # PASSIVE fall time (31 steps), so the policy is doing nothing, and from
    # dead props it has 0.6 s to discover "spin up AND point vertical" as a
    # conjunction. 80-160 rad/s = 6.4-11.5 N, straddling the 7.17 N static
    # stability threshold, so some episodes start already stable and some do
    # not -- which is the gradient we want.
    reset_propeller_spin = EventTerm(
        func=mdp.reset_propeller_spin,
        mode="reset",
        params={"speed_range": (80.0, 160.0)},
    )


@configclass
class DoubleBeeHybridStairCfg(DoubleBeeVelocityEnvCfg):
    """Configuration for DoubleBee hybrid (propeller + wheel) mode on staircase terrain.

    Uses propeller aerodynamics and staircase terrain for testing hybrid locomotion.
    """

    rewards: RewardsCfg = RewardsCfg()
    events: DoubleBeeEventsCfg = DoubleBeeEventsCfg()

    # 4D action space: wheels (2) + servo (1) + propeller (1). Servos and propellers active.
    actions: ActionsCfg4D = ActionsCfg4D()

    # Provide (optional) task-specific constraint terms override if needed in future

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        
        # Override scene settings - keep prim_path consistent with sensors
        # Use Doublebee (not Robot) to match the actual robot name
        self.scene.robot = DOUBLEBEE_CFG.replace(prim_path="{ENV_REGEX_NS}/Doublebee")
        
        # Override terrain to use staircase terrain
        stair_config = StairConfigCfg()
        self.scene.terrain = stair_config.stair_terrain
        print("[INFO] Using staircase terrain for DoubleBee environment.")
        
        # Override command to use TerrainTargetDirectionCommand for target-based navigation
        # This makes the robot follow terrain targets instead of random velocity commands
        self.commands.base_velocity = TerrainTargetDirectionCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),  # Not used, but required
            rel_standing_envs=0.0,
            debug_vis=False,
            ranges=TerrainTargetDirectionCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),  # Not used, but required
                lin_vel_y=(-1.0, 1.0),  # Not used, but required
                ang_vel_z=(-1.0, 1.0),  # Not used, but required
            ),
        )
        print("[INFO] Using TerrainTargetDirectionCommand - robot will follow terrain targets.")
        
        # Episode settings
        self.episode_length_s = 20.0
        self.decimation = 4
        
        # Simulation settings
        self.sim.dt = 0.005


@configclass
class DoubleBeeHybridStairCfg_PLAY(DoubleBeeHybridStairCfg):
    """Configuration for DoubleBee hybrid staircase play/evaluation."""

    # Override events to use aligned initialization
    events: DoubleBeeEventsCfg_PLAY = DoubleBeeEventsCfg_PLAY()

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()
        
        # Override terrain to use simplified PLAY terrain with only 2 gentle stairs
        from lab.doublebee.tasks.manager_based.locomotion.velocity.terrain_config.stair_config import StairConfigCfg_PLAY
        stair_config_play = StairConfigCfg_PLAY()
        self.scene.terrain = stair_config_play.stair_terrain
        print("[INFO] Using simplified stair terrain for play mode - only 2 gentle stairs, shorter distance.")
        
        # Disable observation noise for evaluation
        if hasattr(self.observations, 'policy'):
            if hasattr(self.observations.policy, 'enable_corruption'):
                self.observations.policy.enable_corruption = False
        
        # Render settings
        self.sim.render_interval = self.decimation
        
        # More aggressive command ranges for play
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        print("[INFO] Using aligned initialization for play mode - start/end points aligned, robot faces target.")


# python scripts/co_rl/train.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --algo ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2


@configclass
class DoubleBeeHybridStairConstantThrustCfg(DoubleBeeHybridStairCfg):
    """FIXED-ALLOCATION BASELINE. Identical to the full task except the
    propellers are pinned at a constant throttle.

    This is the non-modulating comparison the energy claim needs, and the answer
    to IROS R1's "how much better than a well-designed mode-switching
    controller?". Preferred over a hand-tuned decoupled controller because there
    is nothing to tune: wheels and servos are learned by the same algorithm, on
    the same task, against the same reward. The only difference is that thrust
    is held rather than modulated.

    Held at hold_action = +1.0 -> 17.3 N total, T/W 0.55, which is what the
    published decoupled controller uses (BB_HOV_DC = 1335 us). Sweep hold_action
    for a fixed-allocation curve to plot against the learned Pareto front.
    """

    actions: ActionsCfg4DConstantThrust = ActionsCfg4DConstantThrust()
