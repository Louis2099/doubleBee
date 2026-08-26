# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os as _os
import torch
# from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import goal_reached

# def terrain_levels_goal(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
#     asset = env.scene[asset_cfg.name]
#     terrain = env.scene.terrain

#     reached = goal_reached(env, distance_threshold=0.25)[env_ids].bool()

#     if not hasattr(terrain, "_success_streak"):
#         terrain._success_streak = torch.zeros(terrain.terrain_levels.shape[0], device=env.device)
#     terrain._success_streak[env_ids] = torch.where(
#         reached, terrain._success_streak[env_ids] + 1,
#         torch.zeros_like(terrain._success_streak[env_ids]),
#     )

#     cmd_term = env.command_manager._terms.get("base_velocity")
#     if cmd_term is not None and hasattr(cmd_term, "current_targets_w"):
#         target_xy = cmd_term.current_targets_w[env_ids, :2]
#         robot_xy = asset.data.root_pos_w[env_ids, :2]
#         dist_to_target = torch.norm(robot_xy - target_xy, dim=1)
#     else:
#         dist_to_target = torch.zeros(len(env_ids), device=env.device)

#     move_up = terrain._success_streak[env_ids] >= 5    # was 5 — stronger mastery bar

#     if not hasattr(terrain, "_failure_streak"):
#         terrain._failure_streak = torch.zeros(terrain.terrain_levels.shape[0], device=env.device)
#     terrain._failure_streak[env_ids] = torch.where(
#         reached, torch.zeros_like(terrain._failure_streak[env_ids]),
#         terrain._failure_streak[env_ids] + 1,
#     )
#     move_down = ((dist_to_target > 4.0) & ~reached) | (terrain._failure_streak[env_ids] >= 10)
#     terrain._failure_streak[env_ids] = torch.where(
#         move_down, torch.zeros_like(terrain._failure_streak[env_ids]),
#         terrain._failure_streak[env_ids],
#     )

#     MAX_LEVEL = 4
#     MIN_LEVEL = 0 # whatever level but should be 0 for play
#     # clamp ALL envs BEFORE updating origins
#     terrain.terrain_levels.clamp_(min=MIN_LEVEL, max=MAX_LEVEL)
#     terrain.update_env_origins(env_ids, move_up, move_down)
#     # clamp again in case update pushed any up, then force origins to match
#     terrain.terrain_levels.clamp_(min=MIN_LEVEL, max=MAX_LEVEL)
#     # re-point origins for the clamped envs to the capped row
#     terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

#     print(f"Terrain level histogram: {torch.bincount(terrain.terrain_levels.long())}")

#     return torch.mean(terrain.terrain_levels.float())

# @configclass
# class CurriculumCfg:
#     """Curriculum specifications for DoubleBee robot."""

#     terrain_levels = CurrTerm(func=terrain_levels_goal)

def terrain_levels_goal(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain

    # --- strict goal_reached (with height check) for failure tracking only ---
    reached = goal_reached(env, distance_threshold=0.25)[env_ids].bool()

    # --- XY-only proximity for curriculum PROMOTION ---
    # decoupled from strict goal_reached so height check doesn't collapse curriculum
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is not None and hasattr(cmd_term, "current_targets_w"):
        target_xy = cmd_term.current_targets_w[env_ids, :2]
        robot_xy = asset.data.root_pos_w[env_ids, :2]
        dist_to_target = torch.norm(robot_xy - target_xy, dim=1)
        reached_for_curriculum = dist_to_target < 0.4
    else:
        dist_to_target = torch.zeros(len(env_ids), device=env.device)
        reached_for_curriculum = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)

    # success streak uses XY proximity — curriculum stays stable
    if not hasattr(terrain, "_success_streak"):
        terrain._success_streak = torch.zeros(terrain.terrain_levels.shape[0], device=env.device)
    terrain._success_streak[env_ids] = torch.where(
        reached_for_curriculum,
        terrain._success_streak[env_ids] + 1,
        torch.zeros_like(terrain._success_streak[env_ids]),
    )

    # 10 -> 3 on 2026-08-26. THIS IS WHY NOTHING EVER CLIMBED.
    #
    # Ten CONSECUTIVE successes by the SAME env, at the measured success rate
    # of 0.186, is p^10 = 5e-8 -- roughly 20 million episodes per promotion.
    # Meanwhile demotion fired on a SINGLE episode ending >4 m from target,
    # which any episode that tips early always does. Promotion impossible,
    # demotion near-certain: terrain_levels sat at 0.0000 through 2754
    # iterations and 67M timesteps, so the policy has literally never seen a
    # step. At 3 the expectation is ~155 episodes, which is reachable.
    move_up = terrain._success_streak[env_ids] >= 3

    # failure streak uses strict goal_reached — demotes on genuine failure
    if not hasattr(terrain, "_failure_streak"):
        terrain._failure_streak = torch.zeros(terrain.terrain_levels.shape[0], device=env.device)
    terrain._failure_streak[env_ids] = torch.where(
        reached,  # strict: only reset failure streak on real height+upright success
        torch.zeros_like(terrain._failure_streak[env_ids]),
        terrain._failure_streak[env_ids] + 1,
    )
    # DEMOTION NOW REQUIRES A STREAK TOO. The old form demoted on any single
    # episode ending >4 m out, with no streak required, while promotion needed
    # ten in a row. That asymmetry is what collapsed the curriculum: one early
    # tip undid whatever progress had been made. Failure streak lowered 10 -> 5
    # so genuine incapability still demotes, just not on one bad sample.
    move_down = terrain._failure_streak[env_ids] >= 5
    terrain._failure_streak[env_ids] = torch.where(
        move_down,
        torch.zeros_like(terrain._failure_streak[env_ids]),
        terrain._failure_streak[env_ids],
    )

    MAX_LEVEL = 4
    MIN_LEVEL = 0
    terrain.terrain_levels.clamp_(min=MIN_LEVEL, max=MAX_LEVEL)
    terrain.update_env_origins(env_ids, move_up, move_down)
    terrain.terrain_levels.clamp_(min=MIN_LEVEL, max=MAX_LEVEL)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    # 2026-08-26: gated. bincount + print forces a GPU->CPU sync on EVERY reset
    # event -- the logs showed ~10 of these per training iteration, i.e. a stall
    # every couple of collection steps. Set DOUBLEBEE_DEBUG_TERRAIN=1 to restore.
    if _os.environ.get("DOUBLEBEE_DEBUG_TERRAIN"):
        print(f"Terrain level histogram: {torch.bincount(terrain.terrain_levels.long())}")

    return torch.mean(terrain.terrain_levels.float())


@configclass
class CurriculumCfg:
    """Curriculum specifications for DoubleBee robot."""

    terrain_levels = CurrTerm(func=terrain_levels_goal)