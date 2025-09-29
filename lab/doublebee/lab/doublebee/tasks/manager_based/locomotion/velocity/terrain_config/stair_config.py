# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class StairConfigCfg:
    """Stair terrain configuration for DoubleBee robot."""

    # Stair terrain
    stair_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="stair",
        collision_group=-1,
        physics_material=TerrainImporterCfg.PhysicsMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    """Stair terrain configuration."""
