# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def propeller_collision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Constraint that terminates if propellers collide with obstacles.
    
    Checks if either left or right propeller has contact force above threshold.
    This indicates the propeller has collided with an obstacle (terrain, wall, etc.).
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (should be "contact_forces")
        threshold: Force threshold in Newtons to consider as collision (default: 1.0N)
    
    Returns:
        Binary collision indicator per environment. Shape: (num_envs,)
        - 1.0 = propeller collision detected (terminate episode)
        - 0.0 = no collision (continue episode)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces on all bodies
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # Get the articulation to find propeller body indices
    asset: Articulation = env.scene["robot"]
    
    # Find left and right propeller body indices
    left_propeller_ids = asset.find_bodies("leftPropeller")
    right_propeller_ids = asset.find_bodies("rightPropeller")
    
    # Check if propellers were found
    if len(left_propeller_ids) == 0 or len(right_propeller_ids) == 0:
        # Propellers not found, return no collision
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    left_propeller_id = left_propeller_ids[0]
    right_propeller_id = right_propeller_ids[0]
    
    # Extract forces for each propeller
    # Shape: (num_envs, history_length, 3) for each propeller
    left_propeller_forces = net_contact_forces[:, :, left_propeller_id, :]
    right_propeller_forces = net_contact_forces[:, :, right_propeller_id, :]
    
    # Compute force magnitudes: sqrt(fx^2 + fy^2 + fz^2)
    # Shape: (num_envs, history_length) for each propeller
    left_force_mags = torch.norm(left_propeller_forces, dim=-1)
    right_force_mags = torch.norm(right_propeller_forces, dim=-1)
    
    # Get maximum force over history for each propeller
    # Shape: (num_envs,) for each propeller
    # Use keepdim=False to ensure 1D output
    left_max_force = torch.max(left_force_mags, dim=1, keepdim=False)[0]
    right_max_force = torch.max(right_force_mags, dim=1, keepdim=False)[0]
    
    # Ensure 1D shape: (num_envs,) not (num_envs, 1)
    if left_max_force.dim() > 1:
        left_max_force = left_max_force.squeeze()
    if right_max_force.dim() > 1:
        right_max_force = right_max_force.squeeze()
    
    # Check if either propeller has collision (force > threshold)
    left_collision = (left_max_force > threshold).float()
    right_collision = (right_max_force > threshold).float()
    
    # Ensure 1D shape for collision tensors
    if left_collision.dim() > 1:
        left_collision = left_collision.squeeze()
    if right_collision.dim() > 1:
        right_collision = right_collision.squeeze()
    
    # Return 1.0 if ANY propeller collides, 0.0 otherwise
    # Use torch.maximum for element-wise max
    collision = torch.maximum(left_collision, right_collision)
    
    # Final safety check: ensure output is 1D: (num_envs,) not (num_envs, 1)
    if collision.dim() > 1:
        collision = collision.squeeze()
    elif collision.dim() == 0:
        # Handle scalar case (shouldn't happen, but be safe)
        collision = collision.unsqueeze(0)
    
    return collision


def goal_reached(
    env: ManagerBasedEnv,
    distance_threshold: float = 0.2,
) -> torch.Tensor:
    """Constraint that terminates if robot reaches the goal target.
    
    Checks if the robot's XY position is within distance_threshold of the current target.
    This indicates the robot has successfully reached the goal.
    
    Args:
        env: The environment instance
        distance_threshold: Maximum distance in meters to consider as "reached" (default: 0.5m)
    
    Returns:
        Binary goal reached indicator per environment. Shape: (num_envs,)
        - 1.0 = goal reached (terminate episode)
        - 0.0 = goal not reached (continue episode)
    """
    robot = env.scene["robot"]
    
    # Get robot base position in world frame (XY only)
    robot_pos_w = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # Get the command term to access its selected target
    cmd_manager = env.command_manager
    if "base_velocity" not in cmd_manager._terms:
        # Command not found, return no goal reached
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    command_term = cmd_manager._terms["base_velocity"]
    
    # Check if this is TerrainTargetDirectionCommand with current_targets_w
    if not hasattr(command_term, "current_targets_w"):
        # Not using terrain target command, fall back to finding nearest target
        terrain = env.scene["terrain"]
        if "target" not in terrain.flat_patches:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        
        target_patches = terrain.flat_patches["target"]
        terrain_levels = terrain.terrain_levels
        terrain_types = terrain.terrain_types
        env_origins = env.scene.env_origins
        
        level_indices = terrain_levels
        type_indices = terrain_types
        targets_relative = target_patches[level_indices, type_indices, :, :]
        targets_world = targets_relative + env_origins.unsqueeze(1)
        targets_xy = targets_world[:, :, :2]
        
        robot_pos_xy_expanded = robot_pos_w.unsqueeze(1)
        distances = torch.norm(targets_xy - robot_pos_xy_expanded, dim=2)
        min_distances = torch.min(distances, dim=1)[0]
    else:
        # Use the command's selected target (aligned with command)
        current_targets_w = command_term.current_targets_w  # [num_envs, 3]
        current_targets_xy = current_targets_w[:, :2]  # [num_envs, 2]
        
        # Compute distance from robot to command's selected target
        distances_xy = robot_pos_w - current_targets_xy  # [num_envs, 2]
        min_distances = torch.norm(distances_xy, dim=1)  # [num_envs]
    
    # Check if robot is within threshold of goal
    goal_reached = (min_distances <= distance_threshold).float()
    
    # Ensure 1D shape: (num_envs,) not (num_envs, 1)
    if goal_reached.dim() > 1:
        goal_reached = goal_reached.squeeze()
    elif goal_reached.dim() == 0:
        # Handle scalar case (shouldn't happen, but be safe)
        goal_reached = goal_reached.unsqueeze(0)
    
    return goal_reached


def robot_out_of_bounds(
    env: ManagerBasedEnv,
    max_height: float = 3.0,
    max_xy_distance: float = 6.0,
) -> torch.Tensor:
    """Constraint that terminates if robot is thrown away from the scene.
    
    Checks if the robot:
    1. Height (Z position) exceeds max_height (default: 3.0m)
    2. XY distance from environment origin exceeds max_xy_distance (default: 6.0m)
    
    This constraint is useful to terminate episodes when the robot is thrown too high
    or too far from its starting position, which typically indicates a failure state.
    
    Args:
        env: The environment instance
        max_height: Maximum allowed height in meters (default: 3.0m)
        max_xy_distance: Maximum allowed XY distance from env origin in meters (default: 6.0m)
    
    Returns:
        Binary out-of-bounds indicator per environment. Shape: (num_envs,)
        - 1.0 = robot is out of bounds (terminate episode)
        - 0.0 = robot is within bounds (continue episode)
    """
    robot = env.scene["robot"]
    
    # Get robot base position in world frame
    robot_pos_w = robot.data.root_pos_w  # [num_envs, 3]
    
    # Check height constraint: Z > max_height
    height_violation = (robot_pos_w[:, 2] > max_height).float()  # [num_envs]
    
    # Check XY distance constraint: distance from env origin > max_xy_distance
    # Get environment origins (center of each environment's terrain)
    env_origins = env.scene.env_origins  # [num_envs, 3]
    
    # Compute XY position relative to environment origin
    robot_pos_xy = robot_pos_w[:, :2]  # [num_envs, 2]
    env_origin_xy = env_origins[:, :2]  # [num_envs, 2]
    relative_pos_xy = robot_pos_xy - env_origin_xy  # [num_envs, 2]
    
    # Compute XY distance from origin
    xy_distance = torch.norm(relative_pos_xy, dim=1)  # [num_envs]
    
    # Check if XY distance exceeds threshold
    xy_violation = (xy_distance > max_xy_distance).float()  # [num_envs]
    
    # Robot is out of bounds if EITHER height or XY distance is violated
    out_of_bounds = torch.maximum(height_violation, xy_violation)  # [num_envs]
    
    # Ensure 1D shape: (num_envs,) not (num_envs, 1)
    if out_of_bounds.dim() > 1:
        out_of_bounds = out_of_bounds.squeeze()
    elif out_of_bounds.dim() == 0:
        # Handle scalar case (shouldn't happen, but be safe)
        out_of_bounds = out_of_bounds.unsqueeze(0)
    
    return out_of_bounds

