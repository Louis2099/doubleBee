# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for stair terrain."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


# Terrain generator configuration with stairs
# Single patch with one stair pyramid terrain
STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.01, 0.7),
    use_cache=True,
    sub_terrains={
        "hf_pyramid_stair_inv": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            step_height_range=(0.01, 0.18),
            step_width=0.4,
            platform_width=4.0,     # ⟵ was 2.5; larger flat bottom area
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,    # keep as-is; now fits more comfortably on the larger platform
                    x_range=(-1.0, 1.0), # centered search stays well inside the new 4 m platform
                    y_range=(-1.0, 1.0),
                    z_range=(-1.0, 1.0),
                    max_height_diff=0.15,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,
                    x_range=(-5.0, 5.0),
                    y_range=(-5.0, 5.0),
                    z_range=(0.0, 1.0),
                    max_height_diff=0.15,
                ),
            },
        ),
    },
)
"""Stair terrain generator configuration."""


@configclass
class StairConfigCfg:
    """Stair terrain configuration for DoubleBee robot."""

    # Stair terrain using generator
    stair_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )
    """Stair terrain configuration."""
