# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for flat plane terrain with init and target patches at the same level.

Used for inverted-pendulum or same-level navigation: destination is on the same
plane as the robot (no stairs / elevation). Requires terrain_type='generator' so
that flat_patches are available for spawn and target.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


# Flat plane with init_pos and target at the same Z (same-level)
# Use pyramid stairs with zero step height so we get a flat surface and flat_patch_sampling
PLANE_SAME_LEVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 0.0),
    use_cache=True,
    sub_terrains={
        "flat_plane": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            step_height_range=(0.0, 0.0),  # Flat
            step_width=0.4,
            platform_width=4.0,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,
                    x_range=(-1.0, 1.0),
                    y_range=(-1.0, 1.0),
                    z_range=(0.0, 0.0),  # Same level
                    max_height_diff=0.05,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,
                    x_range=(-4.0, 4.0),
                    y_range=(-4.0, 4.0),
                    z_range=(0.0, 0.0),  # Same level as init
                    max_height_diff=0.05,
                ),
            },
        ),
    },
)


@configclass
class PlaneSameLevelConfigCfg:
    """Flat plane terrain with init and target at the same level (for inverted-pendulum, etc.)."""

    plane_same_level_terrain = TerrainImporterCfg(
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
