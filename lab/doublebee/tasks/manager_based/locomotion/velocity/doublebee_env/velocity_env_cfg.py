# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env_cfg import (
    ManagerBasedConstraintRLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
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
        """Minimal constraints configuration placeholder.
        Add constraint terms here as needed (e.g., timeouts, safety limits).
        """
        pass

    constraints: ConstraintsCfg = ConstraintsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
