# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
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
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class DoubleBeeVelocityEnvCfg(ManagerBasedRLEnvCfg):
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
        robot = DOUBLEBEE_CFG.replace(prim_path="/World/envs/env_.*/Doublebee")
        
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

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
