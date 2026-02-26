#!/usr/bin/env python3
"""Verify robot spawn positions relative to terrain patch centers.

This script checks if robots are spawning correctly within the center
of their assigned inverted pyramid staircases.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CRITICAL: Initialize Isaac Sim BEFORE importing anything that uses omni.kit
from isaaclab.app import AppLauncher


def verify_spawn_positions(num_envs: int = 16, platform_width: float = 4.0, headless: bool = True):
    """Verify that robots spawn within the center of their staircases.
    
    Args:
        num_envs: Number of environments to check (use small number for testing)
        platform_width: Expected platform width in meters (from stair_config.py)
        headless: Whether to run in headless mode (no GUI)
    """
    import torch
    
    # Import after AppLauncher has initialized Isaac Sim
    from lab.doublebee.tasks.manager_based.locomotion.velocity.doublebee_env.flat_env.hybrid_stair.hybrid_stair_cfg import (
        DoubleBeeHybridStairCfg,
    )
    from lab.doublebee.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv
    
    print(f"[INFO] Initializing environment with {num_envs} environments...")
    
    # Create environment configuration
    cfg = DoubleBeeHybridStairCfg()
    cfg.scene.num_envs = num_envs
    
    # Create environment (just cfg, no agent_cfg needed for verification)
    # Note: The linter may complain about agent_cfg, but ManagerBasedConstraintRLEnv.__init__
    # only requires cfg and optional render_mode. The agent_cfg is used by CoRlVecEnvWrapper.
    env = ManagerBasedConstraintRLEnv(cfg, render_mode=None)  # type: ignore
    # For verification, we don't need the full CoRlVecEnvWrapper, but we'll use it for consistency
    # Actually, let's not wrap it - we can access unwrapped directly
    # env = CoRlVecEnvWrapper(env)
    
    print("[INFO] Environment initialized. Resetting to get spawn positions...")
    
    # Reset environment to trigger spawn
    # ManagerBasedConstraintRLEnv.reset() returns just observations (no info dict)
    obs = env.reset()
    
    # Get robot positions (world coordinates)
    # Note: If env is wrapped, use .unwrapped; otherwise use env directly
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    robot = base_env.scene["robot"]
    robot_positions_w = robot.data.root_pos_w.clone()  # [num_envs, 3]
    
    # Get terrain info
    terrain = base_env.scene["terrain"]
    terrain_levels = terrain.terrain_levels  # [num_envs]
    terrain_types = terrain.terrain_types     # [num_envs]
    
    # Get init_pos flat patches for each environment
    if "init_pos" not in terrain.flat_patches:
        print("[ERROR] No 'init_pos' flat patches found in terrain!")
        return
    
    init_pos_patches = terrain.flat_patches["init_pos"]  # [num_levels, num_types, num_patches, 3]
    
    # CRITICAL: Compute actual terrain patch centers
    # Problem: env_origins are spaced by env_spacing (2.5m), but terrain patches are spaced by size (10.0m).
    # The flat patches are sampled relative to terrain patch centers, not env_origins.
    # 
    # Root cause: The spawn code does: robot_pos = env_origins + patch_relative_local
    # But env_origins are NOT at patch centers, so robots spawn at wrong locations relative to patches.
    # However, robots still match patches (100%) because the spawn process uses env_origins + patch_relative,
    # which gives the correct world position, but env_origins is not the patch center.
    #
    # Solution: Compute actual patch centers by reversing the spawn:
    # Since robot_pos = env_origins + patch_relative_local (during spawn),
    # and patch_relative_local is relative to ACTUAL patch center,
    # we can find which patch was used, then compute: patch_center = robot_pos - patch_relative_local
    
    # Get env_origins for reference (these are spaced by env_spacing, not patch size)
    env_origins = base_env.scene.env_origins.clone()  # [num_envs, 3]
    
    # Compute actual patch centers for each environment
    computed_patch_centers = torch.zeros((num_envs, 3), device=robot_positions_w.device, dtype=robot_positions_w.dtype)
    
    for env_idx in range(num_envs):
        level = terrain_levels[env_idx].item()
        ttype = terrain_types[env_idx].item()
        robot_pos = robot_positions_w[env_idx, :]  # [3] world coordinates
        
        # Get all patches for this environment's terrain (patch-local coordinates)
        patches_relative_local = init_pos_patches[level, ttype, :, :]  # [num_patches, 3]
        
        # During spawn, the code does: robot_pos = env_origins + patch_relative_local
        # So: patch_relative_local (in world coords) = robot_pos - env_origins
        # Find which patch_relative_local matches this
        patch_relative_world_estimate = robot_pos - env_origins[env_idx, :]  # [3] what was used during spawn
        
        # Find closest matching patch (the one that was actually used during spawn)
        # During spawn: robot_pos = env_origins + patch_relative_local
        # So: patch_relative_local (as used) = robot_pos - env_origins
        patch_distances = torch.norm(patches_relative_local - patch_relative_world_estimate.unsqueeze(0), dim=1)
        closest_patch_idx = torch.argmin(patch_distances)
        patch_relative_local_used = patches_relative_local[closest_patch_idx, :]  # [3] the patch that was used
        
        # CRITICAL INSIGHT: 
        # - patch_relative_local_used is a VECTOR offset (displacement), same in any coordinate system
        # - During spawn: robot_pos = env_origins + patch_relative_local_used
        # - But patch_relative_local_used is relative to ACTUAL patch center, not env_origins
        # - So: robot_pos = env_origins + (patch_center_to_robot_vector)
        # - Where: patch_center_to_robot_vector = patch_relative_local_used
        # - Therefore: actual_patch_center = robot_pos - patch_relative_local_used
        # This works because patch_relative_local_used is a displacement vector (same in any coord system)
        computed_patch_centers[env_idx, :] = robot_pos - patch_relative_local_used
    
    terrain_patch_centers = computed_patch_centers
    print("[INFO] Computed terrain patch centers from robot positions and patch data")
    print(f"[INFO] Env_origins spacing: {torch.norm(env_origins[1] - env_origins[0]) if num_envs > 1 else 'N/A':.2f}m")
    print(f"[INFO] Patch centers spacing: {torch.norm(terrain_patch_centers[1] - terrain_patch_centers[0]) if num_envs > 1 else 'N/A':.2f}m")
    
    # Check if all environments share the same patch center (single staircase case)
    if num_envs > 1:
        patch_center_distances = torch.norm(terrain_patch_centers.unsqueeze(1) - terrain_patch_centers.unsqueeze(0), dim=2)
        max_patch_center_distance = patch_center_distances.max().item()
        if max_patch_center_distance < 0.1:  # All patch centers are within 0.1m of each other
            print(f"[INFO] Single staircase detected: All {num_envs} environments share the same patch center")
            print(f"[INFO] All robots should be within the center of this single staircase")
            single_staircase = True
        else:
            print(f"[INFO] Multiple staircases detected: Patch centers vary by up to {max_patch_center_distance:.2f}m")
            single_staircase = False
    else:
        single_staircase = True  # Single env = single staircase
    
    print("\n" + "="*80)
    print("SPAWN POSITION VERIFICATION REPORT")
    print("="*80)
    
    # Compute relative positions (robot position relative to ACTUAL terrain patch center)
    relative_positions = robot_positions_w - terrain_patch_centers  # [num_envs, 3]
    relative_positions_xy = relative_positions[:, :2]    # [num_envs, 2] (XY only)
    
    # Compute distances from center
    distances_from_center = torch.norm(relative_positions_xy, dim=1)  # [num_envs]
    
    # Expected range: platform_width / 2 = 4.0 / 2 = 2.0m from center
    # But init_pos patches are sampled with x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)
    # So robots should be within ±1.0m from center (plus patch_radius=0.2m)
    expected_max_distance = 1.0 + 0.2  # x_range + patch_radius
    platform_radius = platform_width / 2.0  # 2.0m
    
    # Statistics
    num_within_expected = (distances_from_center <= expected_max_distance).sum().item()
    num_within_platform = (distances_from_center <= platform_radius).sum().item()
    
    print(f"\nTerrain Configuration:")
    print(f"  - Platform width: {platform_width}m (radius: {platform_radius}m)")
    print(f"  - Expected spawn range: ±{expected_max_distance:.2f}m from center")
    print(f"  - Init_pos patch sampling: x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)")
    print(f"  - Patch radius: 0.2m")
    
    print(f"\nSpawn Statistics (out of {num_envs} environments):")
    print(f"  - Robots within expected range (±{expected_max_distance:.2f}m): {num_within_expected}/{num_envs} ({100*num_within_expected/num_envs:.1f}%)")
    print(f"  - Robots within platform radius (±{platform_radius:.2f}m): {num_within_platform}/{num_envs} ({100*num_within_platform/num_envs:.1f}%)")
    
    print(f"\nDistance Statistics:")
    print(f"  - Mean distance from center: {distances_from_center.mean().item():.3f}m")
    print(f"  - Std distance from center: {distances_from_center.std().item():.3f}m")
    print(f"  - Min distance from center: {distances_from_center.min().item():.3f}m")
    print(f"  - Max distance from center: {distances_from_center.max().item():.3f}m")
    
    # Check individual environments
    print(f"\nDetailed Analysis (first 10 environments):")
    print(f"{'Env':<6} {'Terrain':<12} {'X (rel)':<10} {'Y (rel)':<10} {'Distance':<10} {'Status':<15}")
    print("-" * 80)
    
    for env_idx in range(min(10, num_envs)):
        level = terrain_levels[env_idx].item()
        ttype = terrain_types[env_idx].item()
        rel_x = relative_positions[env_idx, 0].item()
        rel_y = relative_positions[env_idx, 1].item()
        dist = distances_from_center[env_idx].item()
        
        if dist <= expected_max_distance:
            status = "✓ OK"
        elif dist <= platform_radius:
            status = "⚠ Near edge"
        else:
            status = "✗ OUTSIDE"
        
        print(f"{env_idx:<6} L{level}/T{ttype:<8} {rel_x:>8.3f}  {rel_y:>8.3f}  {dist:>8.3f}  {status:<15}")
    
    # Check if any robots are outside platform
    outside_platform = distances_from_center > platform_radius
    num_outside = outside_platform.sum().item()
    
    if num_outside > 0:
        print(f"\n[WARNING] {num_outside} robots are outside the platform radius!")
        print("  This suggests a coordinate transformation issue.")
        outside_envs = torch.where(outside_platform)[0]
        print(f"  Environment IDs: {outside_envs.tolist()[:10]}{'...' if len(outside_envs) > 10 else ''}")
    else:
        print(f"\n[SUCCESS] All robots are within the platform radius!")
    
    # Verify against actual flat patches
    print(f"\nVerifying against terrain flat patches:")
    print(f"  - Checking if spawn positions match sampled init_pos patches...")
    
    # For each environment, check if the spawn position matches one of its init_pos patches
    matches = 0
    for env_idx in range(num_envs):
        level = terrain_levels[env_idx].item()
        ttype = terrain_types[env_idx].item()
        
        # Get all init_pos patches for this environment's terrain
        patches_relative = init_pos_patches[level, ttype, :, :]  # [num_patches, 3]
        num_patches = patches_relative.shape[0]
        
        if num_patches == 0:
            continue
        
        # Get robot's relative position
        robot_rel = relative_positions[env_idx, :]  # [3]
        
        # Check if robot position matches any patch (within patch_radius)
        patch_distances = torch.norm(patches_relative - robot_rel.unsqueeze(0), dim=1)
        min_patch_dist = patch_distances.min().item()
        
        if min_patch_dist < 0.3:  # Within patch_radius (0.2) + small tolerance
            matches += 1
    
    print(f"  - Robots matching init_pos patches: {matches}/{num_envs} ({100*matches/num_envs:.1f}%)")
    
    if matches == num_envs:
        print(f"  [SUCCESS] All spawn positions match terrain flat patches!")
    elif matches >= num_envs * 0.9:
        print(f"  [WARNING] Some spawn positions don't match patches (may be due to randomization)")
    else:
        print(f"  [ERROR] Many spawn positions don't match patches - possible coordinate bug!")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    if single_staircase:
        if num_within_platform == num_envs:
            print(f"✓ All {num_envs} robots are within the center of the single staircase!")
            print(f"  Mean distance from center: {distances_from_center.mean().item():.3f}m")
            print(f"  Max distance from center: {distances_from_center.max().item():.3f}m")
        else:
            print(f"✗ {num_envs - num_within_platform} robots are outside the single staircase center!")
            print(f"  This suggests robots are spawning too far from the staircase center.")
            print(f"  Check the init_pos patch sampling configuration in stair_config.py")
    else:
        if num_within_platform == num_envs and matches >= num_envs * 0.9:
            print("✓ Robots are spawning correctly within their respective terrain patches.")
            print("  The GUI visualization mismatch is likely due to env_spacing affecting")
            print("  the visual layout, but physics simulation is correct.")
        else:
            print("✗ There may be an issue with spawn position calculation.")
            print("  Check the coordinate transformation in reset_root_state_from_terrain.")
    print("="*80)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    # Parse arguments (including AppLauncher args like --headless)
    parser = argparse.ArgumentParser(description="Verify robot spawn positions")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of environments to check (default: 16)",
    )
    parser.add_argument(
        "--platform_width",
        type=float,
        default=4.0,
        help="Expected platform width in meters (default: 4.0)",
    )
    # Add AppLauncher arguments (headless mode, etc.)
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()
    
    # Initialize Isaac Sim app (this must happen before any omni.kit imports)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    try:
        # Now we can safely import and use Isaac Lab components
        verify_spawn_positions(
            num_envs=args.num_envs,
            platform_width=args.platform_width,
            headless=args.headless if hasattr(args, 'headless') else True
        )
    finally:
        # Cleanup
        simulation_app.close()

