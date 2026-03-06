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
"""Stair terrain generator configuration for training."""


STAIR_TERRAINS_CFG_PLAY = TerrainGeneratorCfg(
    seed=42,
    size=(6.0, 6.0),  # Smaller terrain for simpler task
    border_width=1.5,
    num_rows=1,
    num_cols=1,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 0.2),  # Much lower difficulty
    use_cache=False,  # Disable cache for play mode to allow easier regeneration
    sub_terrains={
        "hf_pyramid_stair_inv": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            step_height_range=(0.05, 0.08),  # Only 5-8cm steps (gentle stairs)
            step_width=0.5,  # Wider steps for easier climbing
            platform_width=2.5,  # Smaller platform since terrain is smaller
            border_width=0.5,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=5,  # Fewer patches needed for simpler terrain
                    patch_radius=0.2,  # Slightly larger patches for easier spawning
                    x_range=(-1.0, 1.0),  # Narrower range to keep robot on platform
                    y_range=(-1.0, 1.0),
                    z_range=(-1.0, 1.0),
                    max_height_diff=0.15,  # Flatter spawn area
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,  # Fewer target options
                    patch_radius=0.2,
                    x_range=(-4.0, 4.0),  # Shorter distance to target
                    y_range=(-4.0, 4.0),
                    z_range=(0.0, 0.3),  # Target on stairs (~2 steps up: 2 * 0.08 = 0.16m)
                    max_height_diff=0.1,
                ),
            },
        ),
    },
)
"""Simplified stair terrain for PLAY mode: only 2 gentle stairs, shorter distance to target."""





@configclass
class StairConfigCfg:
    """Stair terrain configuration for DoubleBee robot (training mode)."""

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
    """Stair terrain configuration for training."""


@configclass
class StairConfigCfg_PLAY:
    """Simplified stair terrain configuration for DoubleBee robot (play/eval mode)."""

    # Simplified stair terrain with only 2 gentle stairs
    stair_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAINS_CFG_PLAY,
        max_init_terrain_level=0,  # No curriculum in play mode
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )
    """Simplified stair terrain configuration for play/evaluation."""
