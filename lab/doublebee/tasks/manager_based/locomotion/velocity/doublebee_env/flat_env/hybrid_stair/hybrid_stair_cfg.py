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
from isaaclab.envs.mdp import randomize_actuator_gains
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import RewardsCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.terrain_config.stair_config import StairConfigCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.velocity_command import TerrainTargetDirectionCommandCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import ActionsCfgWheelsOnly4D


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

    # Domain randomization: thrust output ±20% per env per propeller (sampled at reset)
    # sample_thrust_scale_dr = EventTerm(
    #     func=aerodynamics.sample_thrust_scale_dr,
    #     mode="reset",
    #     params={"range_low": 0.8, "range_high": 1.2, "num_propellers": 2},
    # )

    # NOTE: Reset/spawn is controlled here. Position is sampled from terrain "init_pos" flat patches.
    # - pose_range: roll, pitch, yaw in rad. Only orientation is randomized (position from terrain).
    # - velocity_range: x, y, z in m/s (linear); roll, pitch, yaw in rad/s (angular). Sampled uniformly.
    # To randomize initial velocity and orientation, set non-zero (min, max) for the desired keys.
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain_aligned,
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
            "align_axis": "x",  # Align on X axis (robot moves along Y axis)
        },
    )

    # CRITICAL: Reset joints to default positions to prevent error accumulation
    # Without this, joints retain their previous state, causing PD controller to
    # try to move from reset position to previous target, accumulating error
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),  # Reset to exact default positions (0.0 for all joints)
            "velocity_range": (0.0, 0.0),  # Reset to zero velocity
        },
    )


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
        func=mdp.reset_root_state_from_terrain_aligned,
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
            "align_axis": "x",  # Align on X axis (robot moves along Y axis)
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


@configclass
class DoubleBeeHybridStairCfg(DoubleBeeVelocityEnvCfg):
    """Configuration for DoubleBee hybrid (propeller + wheel) mode on staircase terrain.

    Uses propeller aerodynamics and staircase terrain for testing hybrid locomotion.
    """

    rewards: RewardsCfg = RewardsCfg()
    events: DoubleBeeEventsCfg = DoubleBeeEventsCfg()

    # 4D wheels-only action space: differential wheel velocities only.
    actions: ActionsCfgWheelsOnly4D = ActionsCfgWheelsOnly4D()
    
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