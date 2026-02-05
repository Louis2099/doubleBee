# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


@configclass
class RoughConfigCfg:
    """Rough terrain configuration for DoubleBee robot."""

    # Rough terrain
    rough_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="rough",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )
    """Rough terrain configuration."""
