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
    # num_rows=1,
    # num_cols=1,
    num_rows=5,        # 5 difficulty levels
    num_cols=5,        # 5 variations per level
    curriculum=True,   # difficulty increases by row
    # difficulty_range=(0.0, 1.0),
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    # difficulty_range=(0.01, 0.7),
    difficulty_range=(0.0, 1.0),
    # use_cache=True,
    use_cache=False,
    sub_terrains={
        "hf_pyramid_stair_inv": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            # step_height_range=(0.01, 0.18),
            # step_height_range=(0.01, 0.12),
            # step_height_range=(0.03, 0.12),
            # step_height_range=(0.03, 0.09),
            step_height_range=(0.03, 0.09),   # narrower range centered around 5-7cm
            step_width=0.4,
            platform_width=3.0,     # ⟵ was 2.5; larger flat bottom area
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=8,
                    patch_radius=0.05,    # keep as-is; now fits more comfortably on the larger platform
                    # x_range=(-1.0, 1.0), # centered search stays well inside the new 4 m platform
                    # y_range=(-1.0, 1.0),
                    # z_range=(-1.0, 1.0),
                    x_range=(-0.5, 0.5),
                    y_range=(-0.5, 0.5),
                    z_range=(-0.5, 0.5),
                    max_height_diff=0.15,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.1,
                    # x_range=(-5.0, 5.0),
                    # y_range=(-5.0, 5.0),
                    # z_range=(0.0, 1.0),
                    x_range=(-1.0, 1.0),  # narrow - same corridor as spawn
                    # y_range=(0.5, 4.0),   # wider — from close (0.5m) to far (4m)
                    # z_range=(0.0, 0.20),  # wider — from flat (0.0) to multi-step (0.20)# CHANGED: was (0.03, 0.20) — now includes flat-ground targets too
                    y_range=(1.5, 3.2),
                    z_range=(0.03, 0.20),
                    max_height_diff=0.25,
                ),
            },
        ),
    },
)
"""Stair terrain generator configuration for training."""
STAIR_TERRAINS_CFG_PLAY = TerrainGeneratorCfg(
    seed=42,
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    curriculum=True,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    # formula used, step_height = step_min + difficulty * (step_max - step_min)
    # difficulty_range=(0.10, 0.24), # 4cm
    difficulty_range=(0.26, 0.40), # 5cm
    # difficulty_range=(0.43, 0.57), # 6cm
    # difficulty_range=(0.60, 0.74), # 7cm
    use_cache=False,
    sub_terrains={
        "hf_pyramid_stair_inv": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            # 2026-08-28: WAS (0.03, 0.09) UNDER A COMMENT CLAIMING IT MATCHED
            # TRAINING. It did not. Training is (0.04, 0.07); this range is twice
            # as wide and runs past training on BOTH ends, so play could generate
            # a 9 cm step -- 29% taller than anything any checkpoint has trained
            # on -- and a 3 cm one below the range too.
            #
            # This makes play misleading in a specific, asymmetric way: a policy
            # that actually attempts the climb meets an out-of-range riser and
            # fails, while a policy that never engages steps is untouched by it.
            # Observed 2026-08-28: RUN 1 (terrain_levels 1.51, success 0.36)
            # looked far worse in play than RUN 2 (terrain_levels 0.02, success
            # 0.12), the opposite of what every training metric said.
            step_height_range=(0.03, 0.09),  # now genuinely matches training
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=8,
                    patch_radius=0.05,
                    x_range=(-0.5, 0.5),
                    y_range=(-0.5, 0.5),
                    z_range=(-0.5, 0.5),
                    max_height_diff=0.15,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.1,
                    x_range=(-1.0, 1.0),
                    y_range=(1.5, 3.2),   # match training
                    z_range=(0.03, 0.20), # match training
                    max_height_diff=0.25,
                ),
            },
        ),
    },
)
# STAIR_TERRAINS_CFG_PLAY = TerrainGeneratorCfg(
#     seed=42,
#     size=(10.0, 10.0),
#     border_width=2.0,
#     # num_rows=1,
#     # num_cols=1,
#     num_rows=5,        # 5 difficulty levels
#     num_cols=5,        # 5 variations per level
#     curriculum=True,   # difficulty increases by row
#     # difficulty_range=(0.0, 1.0),
#     color_scheme="random",
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.5,
#     # difficulty_range=(0.05, 0.051),
#     # difficulty_range=(0.95, 1.0),
#     difficulty_range=(0.25, 0.40),   # instead of (0.95, 1.0)
#     # use_cache=True,
#     use_cache=False,
#     sub_terrains={
#         "hf_pyramid_stair_inv": terrain_gen.HfPyramidStairsTerrainCfg(
#             inverted=True,
#             proportion=1.0,
#             # step_height_range=(0.04, 0.041),
#             # step_height_range=(0.05,0.051),
#             step_height_range=(0.06, 0.061),
#             # step_height_range=(0.07, 0.071),
#             # step_height_range=(0.08, 0.081),
#             # step_height_range=(0.09, 0.091),
#             # step_height_range=(0.1, 0.101),
#             step_width=0.4,
#             platform_width=3.0,
#             border_width=1.0,
#             flat_patch_sampling={
#                 "init_pos": FlatPatchSamplingCfg(
#                     num_patches=8,
#                     patch_radius=0.05,    # keep as-is; now fits more comfortably on the larger platform
#                     # x_range=(-1.0, 1.0), # centered search stays well inside the new 4 m platform
#                     # y_range=(-1.0, 1.0),
#                     # z_range=(-1.0, 1.0),
#                     x_range=(-0.5, 0.5),
#                     y_range=(-0.5, 0.5),
#                     z_range=(-0.5, 0.5),
#                     max_height_diff=0.15,
#                 ),
#                 "target": FlatPatchSamplingCfg(
#                     num_patches=5,
#                     patch_radius=0.1,
#                     x_range=(-1.0, 1.0),

#                     # y_range=(2.5, 4.0), ### 4cm  (3-4 steps: 12-16cm)
#                     # z_range=(0.11, 0.17),

#                     # y_range=(3.0, 4.0), ### 5cm  (3-4 steps: 15-20cm)
#                     # z_range=(0.03, 0.6),
#                     # y_range=(1.5, 3.0), ### 5cm  (3-4 steps: 15-20cm)
#                     # z_range=(0.08, 0.14),   # <-- FIX: was (0.03, 0.6), way too wide/tall

#                     y_range=(3.0, 5.5), ### 6cm  (3 steps: 18cm; range catches 2-3 steps)
#                     z_range=(0.18, 0.30),

#                     # y_range=(2.5, 4.0), ### 7cm  (2-3 steps: 14-21cm)
#                     # z_range=(0.13, 0.22),

#                     # y_range=(2.0, 3.5), ### 8cm  (2-3 steps: 16-24cm)
#                     # z_range=(0.15, 0.25),

#                     # y_range=(2.0, 3.5), ### 9cm  (2 steps: 18cm)
#                     # z_range=(0.16, 0.20),

#                     # y_range=(2.0, 3.5), ### 10cm (2 steps: 20cm)
#                     # z_range=(0.18, 0.22),

#                     max_height_diff=0.25,
#                 ),
#             },
#         ),
#     },
# )
"""Simplified stair terrain for PLAY mode: only 2 gentle stairs, shorter distance to target."""





@configclass
class StairConfigCfg:
    """Stair terrain configuration for DoubleBee robot (training mode)."""

    # Stair terrain using generator
    stair_terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAINS_CFG,
        max_init_terrain_level=4,
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
        max_init_terrain_level=1,
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
