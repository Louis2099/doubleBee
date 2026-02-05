# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.assets import AssetBaseCfg
from lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env_cfg import (
    ManagerBasedConstraintRLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import (
    ActionsCfg as DoubleBeeActionsCfg,
    CurriculumCfg as DoubleBeeCurriculumCfg,
    ObservationsCfg as DoubleBeeObservationsCfg,
    RewardsCfg as DoubleBeeRewardsCfg,
    TerminationsCfg as DoubleBeeTerminationsCfg,
)
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.velocity_command import DoubleBeeVelocityCommandCfg
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import constraints as mdp_constraints
from lab.doublebee.isaaclab.isaaclab.managers import ConstraintTermCfg as ConstrTerm


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = DoubleBeeVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=DoubleBeeVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            lin_vel_z=(-1.0, 1.0),  # Vertical velocity for drone
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class DoubleBeeVelocityEnvCfg(ManagerBasedConstraintRLEnvCfg):
    """Configuration for the DoubleBee velocity control environment."""

    episode_length_s: float = 20.0
    decimation: int = 4

    class SceneCfg(InteractiveSceneCfg):
        """Scene configuration."""

        # Terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # Robot
        # Use {ENV_REGEX_NS} placeholder which gets replaced with /World/envs/env_0, /World/envs/env_1, etc.
        # This matches the pattern used by sensors
        robot = DOUBLEBEE_CFG.replace(prim_path="{ENV_REGEX_NS}/Doublebee")
        
        # Sensors
        # Height scanner for 6x6 elevation map around the robot
        # Note: Using robot root directly since USD structure may not have a "base" prim
        height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Doublebee",  # Attach to robot root (articulation root)
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # Cast rays from 20m above
            attach_yaw_only=True,  # Only follow yaw rotation, not roll/pitch
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.07,  # 7cm spacing between rays
                size=[0.35, 0.35]  # 35cm x 35cm square → 6x6 grid (36 rays)
            ),
            debug_vis=False,  # Disable visualization to avoid headless mode issues
            mesh_prim_paths=["/World/ground"],  # Raycast against terrain
        )
        
        # Contact sensor for wheel-ground contact detection
        # Use /.* pattern to match all child bodies under Doublebee (like Flamingo uses /Robot/.*)
        # The contact sensor will find all rigid bodies with contact reporter API under this path
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Doublebee/.*",  # Match all child bodies (wheels, propellers, etc.)
            history_length=3,  # Keep last 3 timesteps of contact data
            track_air_time=True,  # Track how long wheels have been in air
        )
        
        # Lighting (to make robot visible)
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=3000.0,
            ),
        )
        
        dome_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.8, 0.8, 1.0),
                intensity=1000.0,
            ),
        )

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DoubleBeeObservationsCfg = DoubleBeeObservationsCfg()
    actions: DoubleBeeActionsCfg = DoubleBeeActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: DoubleBeeRewardsCfg = DoubleBeeRewardsCfg()
    terminations: DoubleBeeTerminationsCfg = DoubleBeeTerminationsCfg()
    curriculum: DoubleBeeCurriculumCfg = DoubleBeeCurriculumCfg()
    
    # Constraints manager configuration (required by ManagerBasedConstraintRLEnv)
    @configclass
    class ConstraintsCfg:
        """Constraint specifications for DoubleBee robot.
        
        Note: ManagerBasedConstraintRLEnv uses constraints, not terminations!
        Terminations are converted to constraints here.
        """
        
        time_out = ConstrTerm(
            func=lambda env: env.episode_length_buf >= env.max_episode_length,
            time_out="truncate",  # This is a timeout, not a hard constraint
        )
        """Episode timeout constraint."""
        
        propeller_collision = ConstrTerm(
            func=mdp_constraints.propeller_collision,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "threshold": 1.0,  # Force threshold in Newtons
            },
            time_out="terminate",  # Hard termination when propeller collides
        )
        """Propeller collision constraint - terminates if propellers collide with obstacles."""
        
        goal_reached = ConstrTerm(
            func=mdp_constraints.goal_reached,
            params={
                "distance_threshold": 0.2,  # Distance in meters to consider goal reached
            },
            time_out="terminate",  # Hard termination when goal is reached
        )
        """Goal reached constraint - terminates when robot reaches the target goal."""
        
        robot_out_of_bounds = ConstrTerm(
            func=mdp_constraints.robot_out_of_bounds,
            params={
                "max_height": 3.0,  # Maximum allowed height in meters
                "max_xy_distance": 6.0,  # Maximum allowed XY distance from env origin in meters
            },
            time_out="terminate",  # Hard termination when robot is thrown out of scene
        )
        """Robot out of bounds constraint - terminates if robot height > 3m or XY distance > 6m from origin."""
        
        # fall = ConstrTerm(
        #     func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, 2] < -1.0,
        #     time_out="terminate",  # This terminates the episode
        # )
        # """Falling constraint - robot falling too fast."""
        
        # tilt = ConstrTerm(
        #     func=lambda env: torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b), dim=1) > 0.5,
        #     time_out="terminate",  # This terminates the episode
        # )
        # """Excessive tilt constraint - robot tilted too much."""

    constraints: ConstraintsCfg = ConstraintsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = self.decimation
        
        # Debug: Print episode length calculation
        from math import ceil
        max_ep_len = ceil(self.episode_length_s / (self.decimation * self.sim.dt))
        print(f"[DEBUG DoubleBeeVelocityEnvCfg] Episode length calculation:")
        print(f"  episode_length_s = {self.episode_length_s}s")
        print(f"  decimation = {self.decimation}")
        print(f"  sim.dt = {self.sim.dt}s")
        print(f"  step_dt = {self.decimation * self.sim.dt}s")
        print(f"  max_episode_length = {max_ep_len} steps")
        print(f"  Expected episode duration: {max_ep_len * self.decimation * self.sim.dt}s")
