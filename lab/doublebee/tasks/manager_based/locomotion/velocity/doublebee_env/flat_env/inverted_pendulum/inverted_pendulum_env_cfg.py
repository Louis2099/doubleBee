# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inverted-pendulum experiment configuration.

Decoupled design:
- Same-level destination: terrain with init_pos and target flat_patches at the same Z (plane).
- No height-scanner: scene has no height_scanner; observations use ObservationsCfgNoHeightScan.
- No propeller or servo actuation: actions use ActionsCfgWheelsOnly (wheels only).

This file composes overrides on top of DoubleBeeHybridStairCfg so the rest of the
pipeline (rewards, constraints, commands, events for reset/wheel friction) is reused.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.velocity_env_cfg import DoubleBeeVelocityEnvCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.flat_env.hybrid_stair.hybrid_stair_cfg import (
    DoubleBeeHybridStairCfg,
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
class DoubleBeeInvertedPendulumCfg(DoubleBeeHybridStairCfg):
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
    class SceneCfg(DoubleBeeHybridStairCfg.SceneCfg):
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

        # DIAGNOSTIC 2026-08-26: WHEEL DEAD TIME -> 0, task-local so run A is untouched.
        #
        # With the reward sign fixed (survival strictly profitable, net +0.05)
        # this task still could not stay up past ~1 s after 5,472 gradient steps
        # on a problem with a known-good hardware precedent. That rules out
        # reward shaping and points at the plant.
        #
        # CoM sits 0.1016 m above the axle -> tau = 102 ms, unstable pole
        # p = 9.83 rad/s. Stabilizing an unstable pole p through dead time theta
        # needs p*theta < 1 to be possible at all and p*theta < 0.3 to be usable:
        #
        #     min_delay=2  ->  40 ms  ->  p*theta = 0.39
        #     mean         ->  60 ms  ->  p*theta = 0.59
        #     max_delay=4  ->  80 ms  ->  p*theta = 0.79
        #
        # and the policy has to cope with the worst case in the randomized range.
        # Every setting is past the usable limit.
        #
        # This is a DIAGNOSTIC, not a deployment config -- the real robot has
        # real dead time. If balance appears at 0 and not at 2-4, the delay is
        # the wall, and the answer is not "train harder": propellers held
        # world-vertical remove the unstable pole entirely (they turn the
        # diverging pendulum into a 0.8-1.4 s oscillator), which would make
        # prop-assist mandatory rather than optional.
        self.scene.robot.actuators["wheels"].min_delay = 0
        self.scene.robot.actuators["wheels"].max_delay = 0

        print("[INFO] Using DoubleBee inverted-pendulum config: same-level target, no height scan, wheels only.")
        print("[INFO] DIAGNOSTIC: wheel actuator delay forced to 0 (was 2-4 steps / 40-80 ms).")


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
