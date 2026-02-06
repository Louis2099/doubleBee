# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inverted-pendulum experiment configuration.

Decoupled design:
- Same-level destination: terrain with init_pos and target flat_patches at the same Z (plane).
- No height-scanner: scene has no height_scanner; observations use ObservationsCfgNoHeightScan.
- No propeller or servo actuation: actions use ActionsCfgWheelsOnly (wheels only).

This file composes overrides on top of DoubleBeeFlatStandDriveCfg so the rest of the
pipeline (rewards, constraints, commands, events for reset/wheel friction) is reused.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.velocity_env_cfg import DoubleBeeVelocityEnvCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.flat_env.stand_drive.flat_env_stand_drive_cfg import (
    DoubleBeeFlatStandDriveCfg,
)
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import events as mdp
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.observations import ObservationsCfgNoHeightScan
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.actions import ActionsCfgWheelsOnly
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import RewardsCfgInvertedPendulum
from lab.doublebee.tasks.manager_based.locomotion.velocity.terrain_config.plane_same_level_config import (
    PlaneSameLevelConfigCfg,
    PLANE_SAME_LEVEL_TERRAINS_CFG,
)
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.velocity_command import TerrainTargetDirectionCommandCfg


@configclass
class DoubleBeeEventsCfgInvertedPendulum:
    """Events for inverted-pendulum: wheel friction at startup, reset base and joints. No propeller aerodynamics."""

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

    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
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
class DoubleBeeInvertedPendulumCfg(DoubleBeeFlatStandDriveCfg):
    """Configuration for inverted-pendulum experiment.

    - Destination at same level as robot (flat plane, init and target patches at same Z).
    - Height scanner disabled (no elevation map in observations).
    - Only wheels actuated (servos and propellers disabled; RewardsCfgInvertedPendulum drops propeller_efficiency).
    """

    # Override observations, actions, rewards, events at class level
    observations: ObservationsCfgNoHeightScan = ObservationsCfgNoHeightScan()
    actions: ActionsCfgWheelsOnly = ActionsCfgWheelsOnly()
    rewards: RewardsCfgInvertedPendulum = RewardsCfgInvertedPendulum()
    events: DoubleBeeEventsCfgInvertedPendulum = DoubleBeeEventsCfgInvertedPendulum()

    # CRITICAL: Override scene at class level by defining a nested SceneCfg
    # This is the proper Isaac Lab pattern - scene is built with correct config from start
    @configclass
    class SceneCfg(DoubleBeeFlatStandDriveCfg.SceneCfg):
        """Scene configuration for inverted pendulum - same-level terrain, no height scanner."""
        
        # Disable height scanner for this experiment
        height_scanner = None
        
        # Use same-level terrain (flat plane with init_pos and target at same Z)
        # Define TerrainImporterCfg directly to avoid instantiation issues
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=PLANE_SAME_LEVEL_TERRAINS_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
            debug_vis=False,
        )

    # Create scene instance with overridden SceneCfg
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self) -> None:
        # CRITICAL: Call GRANDPARENT's __post_init__ directly, skipping parent
        # Parent's __post_init__ modifies self.scene which invalidates PhysX views
        # Grandparent's __post_init__ only sets sim parameters (safe)
        DoubleBeeVelocityEnvCfg.__post_init__(self)

        # Override command configuration (doesn't affect scene building)
        self.commands.base_velocity = TerrainTargetDirectionCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            rel_standing_envs=0.0,
            debug_vis=False,
            ranges=TerrainTargetDirectionCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
            ),
        )

        # Episode and simulation settings
        self.episode_length_s = 20.0
        self.decimation = 4
        self.sim.dt = 0.005

        print("[INFO] Using DoubleBee inverted-pendulum config: same-level target, no height scan, wheels only.")


@configclass
class DoubleBeeEventsCfgInvertedPendulum_PLAY(DoubleBeeEventsCfgInvertedPendulum):
    """Play mode: aligned init/target, same event setup (no aero)."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain_aligned,
        mode="reset",
        params={
            "pose_range": {
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                # yaw computed to face target
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
            "align_axis": "x",
        },
    )


@configclass
class DoubleBeeInvertedPendulumCfg_PLAY(DoubleBeeInvertedPendulumCfg):
    """Play/evaluation config for inverted-pendulum."""

    events: DoubleBeeEventsCfgInvertedPendulum_PLAY = DoubleBeeEventsCfgInvertedPendulum_PLAY()

    def __post_init__(self) -> None:
        super().__post_init__()

        if hasattr(self.observations, "policy") and hasattr(self.observations.policy, "enable_corruption"):
            self.observations.policy.enable_corruption = False

        self.sim.render_interval = self.decimation
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
