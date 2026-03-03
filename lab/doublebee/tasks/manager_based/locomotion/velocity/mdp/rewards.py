# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


def velocity_direction_alignment(env) -> torch.Tensor:
    """Reward for aligning robot's XY velocity direction with command's XY velocity direction.
    
    Uses cosine similarity (dot product of normalized vectors) to measure alignment,
    scaled by the robot's actual velocity magnitude relative to the command magnitude.
    Returns values in range [-1, 1], where:
    - 1 = perfectly aligned AND moving at or above commanded speed
    - -1 = opposite direction AND moving at or above commanded speed
    - Values scale down toward 0 when velocity magnitude is low
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled alignment reward per environment [num_envs]
    """
    robot = env.scene["robot"]
    cmd_manager = env.command_manager
    
    # Get current robot velocity in body frame (XY components)
    robot_vel_xy = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]
    
    # Get command velocity (XY components)
    vel_cmd = cmd_manager.get_command("base_velocity")  # [num_envs, 4]
    cmd_vel_xy = vel_cmd[:, :2]  # [num_envs, 2]
    
    # Compute magnitudes
    robot_vel_mag = torch.norm(robot_vel_xy, dim=1)  # [num_envs]
    cmd_vel_mag = torch.norm(cmd_vel_xy, dim=1)  # [num_envs]
    
    # Normalize vectors (handle zero velocity case)
    robot_vel_norm = robot_vel_xy / (robot_vel_mag.unsqueeze(1) + 1e-6)  # [num_envs, 2]
    cmd_vel_norm = cmd_vel_xy / (cmd_vel_mag.unsqueeze(1) + 1e-6)  # [num_envs, 2]
    
    # Compute cosine similarity (dot product of normalized vectors)
    # This gives alignment in range [-1, 1]
    alignment = torch.sum(robot_vel_norm * cmd_vel_norm, dim=1)  # [num_envs]
    
    # Clamp alignment to [-1, 1] for numerical stability
    alignment = torch.clamp(alignment, min=-0.5, max=1.0) #NOTE: -0.5 to tolerate some misalignment
    
    # Scale by velocity magnitude: use ratio of actual velocity to a reference velocity
    # Since cmd_vel is normalized (magnitude = 1.0), we use a fixed reference velocity (e.g., 2.0 m/s)
    # to tolerate higher velocities and scale the reward appropriately
    reference_velocity = 2.0  # Reference velocity in m/s to tolerate higher speeds
    velocity_scale = robot_vel_mag / reference_velocity  # [num_envs]
    velocity_scale = torch.clamp(velocity_scale, min=0.0, max=1.0)  # Cap at 1.0 to keep reward in [-1, 1]
    
    # Scale alignment by velocity magnitude factor
    # Result: alignment * velocity_scale is in [-1, 1] range
    scaled_alignment = alignment * velocity_scale
    
    return scaled_alignment


def reach_terrain_target(env) -> torch.Tensor:
    """Reward for reaching terrain target positions.
    
    Uses the same target that the command has selected for the current episode.
    Computes distance from robot base to the command's current target.
    Uses exponential reward: exp(-distance² / scale²) to encourage getting closer.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Target reaching reward per environment [num_envs]
    """
    robot = env.scene["robot"]
    
    # Get robot base position in world frame (XY only)
    robot_pos_w = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # Get the command term to access its selected target
    cmd_manager = env.command_manager
    if "base_velocity" not in cmd_manager._terms:
        # Command not found, return zero reward
        return torch.zeros(robot.num_instances, device=robot.device)
    
    command_term = cmd_manager._terms["base_velocity"]
    
    # Check if this is TerrainTargetDirectionCommand with current_targets_w
    if not hasattr(command_term, "current_targets_w"):
        # Not using terrain target command, fall back to finding nearest target
        terrain = env.scene["terrain"]
        if "target" not in terrain.flat_patches:
            return torch.zeros(robot.num_instances, device=robot.device)
        
        target_patches = terrain.flat_patches["target"]
        terrain_levels = terrain.terrain_levels
        terrain_types = terrain.terrain_types
        env_origins = env.scene.env_origins
        
        level_indices = terrain_levels
        type_indices = terrain_types
        num_patches = target_patches.shape[2]
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
    
    # Exponential reward: exp(-distance² / scale²)
    # Scale = 2.0 means reward drops to ~0.6 at 2m, ~0.13 at 4m
    scale = 2.0
    rewards = torch.exp(-(min_distances ** 2) / (scale ** 2))
    
    return rewards


def terminal_reward_goal_reached(env) -> torch.Tensor:
    """Terminal reward for successfully reaching the goal.
    
    Returns a positive reward only when the robot reaches the goal (goal_reached constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to goal completion.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal reward per environment [num_envs]
        - Positive value (e.g., 10.0) if goal reached
        - 0.0 otherwise
    """
    # Import constraint function to check if goal is reached
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import goal_reached
    
    # Check if goal is reached (constraint is active)
    goal_reached_mask = goal_reached(env, distance_threshold=0.2)  # [num_envs]
    
    # Return positive reward only for environments where goal is reached
    reward_value = 10.0  # Positive terminal reward
    rewards = goal_reached_mask * reward_value
    
    return rewards


def terminal_reward_propeller_collision(env) -> torch.Tensor:
    """Terminal reward (penalty) for propeller collision.
    
    Returns a negative reward only when propellers collide (propeller_collision constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to collision.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal penalty per environment [num_envs]
        - Negative value (e.g., -10.0) if propeller collision occurred
        - 0.0 otherwise
    """
    # Import constraint function to check if collision occurred
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import propeller_collision
    from isaaclab.managers import SceneEntityCfg
    
    # Check if propeller collision occurred (constraint is active)
    collision_mask = propeller_collision(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces"),
        threshold=1.0
    )  # [num_envs]
    
    # Return negative reward only for environments where collision occurred
    penalty_value = -10.0  # Negative terminal reward (penalty)
    rewards = collision_mask * penalty_value
    
    return rewards


def terminal_reward_robot_out_of_bounds(env) -> torch.Tensor:
    """Terminal reward (penalty) for robot being thrown out of bounds.
    
    Returns a negative reward only when robot is out of bounds (robot_out_of_bounds constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to being out of bounds.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal penalty per environment [num_envs]
        - Negative value (e.g., -10.0) if robot is out of bounds
        - 0.0 otherwise
    """
    # Import constraint function to check if robot is out of bounds
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import robot_out_of_bounds
    
    # Check if robot is out of bounds (constraint is active)
    out_of_bounds_mask = robot_out_of_bounds(
        env,
        max_height=3.0,
        max_xy_distance=6.0
    )  # [num_envs]
    
    # Return negative reward only for environments where robot is out of bounds
    penalty_value = -10.0  # Negative terminal reward (penalty)
    rewards = out_of_bounds_mask * penalty_value
    
    return rewards


def penalize_propeller_efficiency(env) -> torch.Tensor:
    """Penalty for excessive propeller speeds to encourage efficiency.
    
    Computes a penalty based on propeller joint velocities.
    Scales the penalty to [-1, 0] range using e^(-x) - 1 transformation.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for higher propeller speeds
        - Values closer to 0 for low/no propeller speeds
    """
    robot = env.scene["robot"]
    
    # Get propeller joint velocities
    try:
        left_propeller_idx = robot.joint_names.index("leftPropeller")
        right_propeller_idx = robot.joint_names.index("rightPropeller")
        
        propeller_velocities = robot.data.joint_vel[:, [left_propeller_idx, right_propeller_idx]]  # [num_envs, 2]
        
        # Compute raw penalty magnitude (sum of squared velocities)
        raw_penalty_magnitude = torch.sum(torch.square(propeller_velocities), dim=1)  # [num_envs]
        
        # Scale to [-1, 0] using e^(-x) - 1 transformation
        scaled_penalty = torch.exp(-raw_penalty_magnitude/500) - 1.0  # [num_envs]
        
        return scaled_penalty
    except (ValueError, IndexError):
        # If propeller joints not found, return zero penalty
        return torch.zeros(robot.num_instances, device=robot.device)


def penalize_facing_direction_mismatch(env) -> torch.Tensor:
    """Penalty for mismatch between robot facing direction and target direction.
    
    Reads the angle error directly from vel_command_b[:, 2], which represents the normalized
    angle error between robot's facing direction and the direction to target.
    The angle error is normalized to [-1, 1] range (where 0 = aligned, ±1 = max error).
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for large angle errors (robot not facing target)
        - Values closer to 0 for small/no angle errors (robot facing target)
    """
    cmd_manager = env.command_manager
    
    # Get velocity command which contains angle error in vel_command_b[:, 2]
    vel_cmd = cmd_manager.get_command("base_velocity")  # [num_envs, 3] or [num_envs, 4]
    
    # Extract angle error (normalized to [-1, 1] range)
    # This is the mismatch between robot facing direction and target direction
    if vel_cmd.shape[1] >= 3:
        angle_error_normalized = vel_cmd[:, 2]  # [num_envs] - normalized angle error
    else:
        # Fallback: return zero penalty if command doesn't have ang_vel_z
        robot = env.scene["robot"]
        return torch.zeros(robot.num_instances, device=robot.device)
    
    # Use absolute value to get magnitude of angle error (range: [0, 1])
    angle_error_magnitude = -torch.abs(angle_error_normalized)  # [num_envs]
    
    
    return angle_error_magnitude


def penalize_tilt_angle(env) -> torch.Tensor:
    """Penalty for excessive tilt angle (roll/pitch) to keep robot upright.
    
    Penalizes deviation from upright orientation. Uses projected gravity to measure tilt.
    Scales the penalty to [-1, 0] range using e^(-x) - 1 transformation.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for large tilt angles
        - Values closer to 0 when robot is upright
    """
    robot = env.scene["robot"]
    
    # Get projected gravity in body frame [num_envs, 3]
    # Projected gravity = rotation_matrix^T * [0, 0, -1]
    # When robot is perfectly upright: projected_gravity = [0, 0, -1]
    # When tilted, X and Y components are non-zero
    projected_gravity = robot.data.projected_gravity_b  # [num_envs, 3]
    
    # Extract tilt components (X and Y components indicate roll/pitch)
    # Z component indicates how upright the robot is
    tilt_x = torch.abs(projected_gravity[:, 0])  # [num_envs] - related to pitch
    tilt_y = torch.abs(projected_gravity[:, 1])  # [num_envs] - related to roll
    
    # Weights for each component (can be adjusted based on importance)
    # Both roll and pitch are important for stability
    weight_x = 3.0  # Pitch-like tilt
    weight_y = 7.0  # Roll-like tilt
    
    # Compute weighted sum of squared tilt components
    # Using squared values to strongly penalize large tilts
    weighted_sum = (
        weight_x * torch.sqrt(tilt_x) +
        weight_y * torch.sqrt(tilt_y)
    )  # [num_envs]
    
    # Scale to [-1, 0] using e^(-x) - 1 transformation
    scaled_penalty = torch.exp(-weighted_sum) - 1.0  # [num_envs]
    
    return scaled_penalty


def penalize_excessive_linear_speed(env, speed_threshold: float = 3.0) -> torch.Tensor:
    """Penalty for excessive linear speed above a threshold.
    
    This prevents the robot from moving dangerously fast. Penalty is only active
    when speed exceeds the threshold, and increases quadratically with excess speed.
    
    Args:
        env: The environment instance
        speed_threshold: Speed threshold in m/s above which penalty is applied (default 3.0 m/s)
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - 0 when speed <= threshold
        - Increasingly negative as speed exceeds threshold
    """
    robot = env.scene["robot"]
    
    # Get linear velocity in world frame [num_envs, 3]
    lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]
    
    # Compute speed magnitude (3D Euclidean norm)
    speed = torch.norm(lin_vel, dim=1)  # [num_envs]
    
    # Compute excess speed (only positive when above threshold)
    excess_speed = torch.clamp(speed - speed_threshold, min=0.0)  # [num_envs]
    
    # Quadratic penalty on excess speed, scaled to [-1, 0]
    # Using e^(-x) - 1 for consistency with other penalties
    # Scale factor makes penalty reach ~-0.63 at 1 m/s excess, ~-0.95 at 3 m/s excess
    penalty_magnitude = (excess_speed ** 2)  # [num_envs]
    scaled_penalty = torch.exp(-penalty_magnitude) - 1.0  # [num_envs]
    
    return scaled_penalty


def penalize_propeller_on_flat_ground(env, flatness_threshold: float = 0.03) -> torch.Tensor:
    """Penalty for using propellers on flat ground where wheels should suffice.
    
    Determines ground flatness from height scanner variance. If ground is flat
    (std dev < threshold) and propellers are being used, applies a penalty
    proportional to propeller usage.
    
    Args:
        env: The environment instance
        flatness_threshold: Standard deviation threshold (m) below which ground is considered flat.
                          Default 0.03m means if terrain height variation < 3cm, it's flat.
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 when ground is flat AND propellers are used heavily
        - Values closer to 0 when ground is not flat OR propellers not used
    """
    robot = env.scene["robot"]
    
    try:
        # Get height scanner data - use try-except since "in" operator doesn't work with InteractiveScene
        height_scanner = env.scene["height_scanner"]
        height_data = height_scanner.data.ray_hits_w  # [num_envs, num_rays, 3] - world positions
        
        # Extract Z (height) component
        heights = height_data[..., 2]  # [num_envs, num_rays]
        
        # Compute standard deviation of heights for each environment
        # Low std dev = flat ground, high std dev = uneven terrain
        height_std = torch.std(heights, dim=1)  # [num_envs]
        
        # Determine if ground is flat (1 = flat, 0 = not flat)
        is_flat = (height_std < flatness_threshold).float()  # [num_envs]
        
        # Get propeller usage (joint velocities)
        left_propeller_idx = robot.joint_names.index("leftPropeller")
        right_propeller_idx = robot.joint_names.index("rightPropeller")
        
        propeller_velocities = robot.data.joint_vel[:, [left_propeller_idx, right_propeller_idx]]  # [num_envs, 2]
        
        # Compute propeller usage magnitude (sum of absolute velocities)
        propeller_usage = torch.sum(torch.abs(propeller_velocities), dim=1)  # [num_envs]
        
        # Normalize propeller usage to [0, 1] range using tanh
        # High propeller speeds -> closer to 1, low speeds -> closer to 0
        normalized_usage = torch.tanh(propeller_usage / 200.0)  # [num_envs], scale by 200 for typical velocities
        
        # Penalty = is_flat * normalized_usage, scaled to [-1, 0]
        # Only penalize when BOTH ground is flat AND propellers are used
        penalty = -is_flat * normalized_usage  # [num_envs]
        
        return penalty
        
    except (ValueError, IndexError, KeyError):
        # If height_scanner or propeller joints not found, return zero penalty
        return torch.zeros(robot.num_instances, device=robot.device)


def penalize_energy_consumption(env) -> torch.Tensor:
    """Penalty for total energy consumption from propellers and wheels.
    
    Computes energy consumption by:
    1. Converting propeller joint velocities (rad/s) to equivalent PWM
    2. Using PWM-to-Power model to get propeller power (W)
    3. Converting wheel joint velocities (rad/s) to RPM
    4. Using RPM-to-Power model to get wheel power (W)
    5. Summing total power and multiplying by simulation dt to get energy per step (J)
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for high energy consumption
        - Values closer to 0 for low energy consumption
    """
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.thrust_energy_model import (
        pwm_to_thrust,
        rpm_to_power,
    )
    import numpy as np
    
    robot = env.scene["robot"]
    
    try:
        # Get simulation timestep
        dt = env.step_dt  # Time per step in seconds
        
        # ========== Propeller Energy Consumption ==========
        # Get propeller joint velocities (rad/s)
        left_propeller_idx = robot.joint_names.index("leftPropeller")
        right_propeller_idx = robot.joint_names.index("rightPropeller")
        propeller_vels = robot.data.joint_vel[:, [left_propeller_idx, right_propeller_idx]]  # [num_envs, 2]
        
        # Convert rad/s to RPM: RPM = (rad/s) * (60 / 2π)
        propeller_rpm = propeller_vels * (60.0 / (2.0 * np.pi))  # [num_envs, 2]
        
        # Convert RPM to equivalent PWM (approximate mapping)
        # Typical PWM range: 1000-2000, typical RPM range: 0-10000
        # Using linear approximation: PWM = 1000 + (RPM / 10000) * 1000
        # Clamp to valid PWM range [1000, 2000]
        propeller_pwm = 1000.0 + (torch.abs(propeller_rpm) / 10000.0) * 1000.0  # [num_envs, 2]
        propeller_pwm = torch.clamp(propeller_pwm, min=1000.0, max=2000.0)
        
        # Convert PWM to power (W) using the energy model
        # Process each propeller separately and convert to numpy for the model
        propeller_power_left = torch.tensor(
            [pwm_to_thrust(pwm.item(), target="power") for pwm in propeller_pwm[:, 0]],
            device=robot.device,
            dtype=torch.float32,
        )  # [num_envs]
        propeller_power_right = torch.tensor(
            [pwm_to_thrust(pwm.item(), target="power") for pwm in propeller_pwm[:, 1]],
            device=robot.device,
            dtype=torch.float32,
        )  # [num_envs]
        
        total_propeller_power = propeller_power_left + propeller_power_right  # [num_envs] in Watts
        
        # ========== Wheel Energy Consumption ==========
        # Get wheel joint velocities (rad/s)
        left_wheel_idx = robot.joint_names.index("leftWheel")
        right_wheel_idx = robot.joint_names.index("rightWheel")
        wheel_vels = robot.data.joint_vel[:, [left_wheel_idx, right_wheel_idx]]  # [num_envs, 2]
        
        # Convert rad/s to RPM: RPM = (rad/s) * (60 / 2π)
        wheel_rpm = torch.abs(wheel_vels) * (60.0 / (2.0 * np.pi))  # [num_envs, 2]
        
        # Convert RPM to power (W) using the RPM-to-Power model
        # Note: The model expects RPM in range [0, 300] based on the CSV data
        wheel_rpm_clamped = torch.clamp(wheel_rpm, min=0.0, max=300.0)
        
        wheel_power_left = torch.tensor(
            [rpm_to_power(rpm.item()) for rpm in wheel_rpm_clamped[:, 0]],
            device=robot.device,
            dtype=torch.float32,
        )  # [num_envs]
        wheel_power_right = torch.tensor(
            [rpm_to_power(rpm.item()) for rpm in wheel_rpm_clamped[:, 1]],
            device=robot.device,
            dtype=torch.float32,
        )  # [num_envs]
        
        total_wheel_power = wheel_power_left + wheel_power_right  # [num_envs] in Watts
        
        # ========== Total Energy Consumption ==========
        # Total power = propeller power + wheel power
        total_power = total_propeller_power + total_wheel_power  # [num_envs] in Watts
        
        # Energy per step = power * dt (Joules)
        energy_per_step = total_power * dt  # [num_envs] in Joules
        
        # Scale to [-1, 0] using exponential transformation
        # Scale factor: typical energy per step might be 0-50 Joules
        # exp(-energy/scale) - 1 maps [0, inf] to [-1, 0]
        # Using scale=20 means: 0J -> 0, 20J -> -0.63, 40J -> -0.86, 60J -> -0.95
        scale = 20.0
        scaled_penalty = torch.exp(-energy_per_step / scale) - 1.0  # [num_envs]
        
        return scaled_penalty
        
    except (ValueError, IndexError, KeyError) as e:
        # If joints not found or energy model fails, return zero penalty
        return torch.zeros(robot.num_instances, device=robot.device)


@configclass
class RewardsCfg:
    """Reward specifications for DoubleBee velocity tracking task."""

    # ========== Velocity Command Tracking Rewards ==========
    
    # track_lin_vel_xy = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.sum(
    #             torch.square(
    #                 env.scene["robot"].data.root_lin_vel_b[:, :2] 
    #                 - env.command_manager.get_command("base_velocity")[:, :2]
    #             ), 
    #             dim=1
    #         )
    #     ),
    #     weight=1.0,
    # )
    # """Horizontal linear velocity tracking (x, y). Exponential reward: exp(-||v_xy - v_cmd_xy||²)"""
    

    # track_lin_vel_z = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.square(
    #             env.scene["robot"].data.root_lin_vel_b[:, 2] 
    #             - env.command_manager.get_command("base_velocity")[:, 2]
    #         )
    #     ),
    #     weight=1.0,
    # )
    # """Vertical linear velocity tracking (z). Exponential reward: exp(-||v_z - v_cmd_z||²)"""

    # track_ang_vel_z = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.square(
    #             env.scene["robot"].data.root_ang_vel_b[:, 2] 
    #             - env.command_manager.get_command("base_velocity")[:, 3]
    #         )
    #     ),
    #     weight=0.5,
    # )
    # """Yaw angular velocity tracking. Exponential reward: exp(-||ω_z - ω_cmd_z||²)"""

    # ========== Locomotion Direction Rewards ==========
    
    velocity_direction_alignment = RewTerm(
        func=velocity_direction_alignment,
        weight=0.2,
    )
    """Reward for aligning robot's XY velocity direction with command's XY velocity direction.
    Uses cosine similarity: reward = dot(normalize(v_robot_xy), normalize(v_cmd_xy)).
    Range: [-1, 1] where 1 = perfectly aligned, -1 = opposite direction."""
    
    # ========== Target Reaching Rewards ==========
    
    reach_terrain_target = RewTerm(
        func=reach_terrain_target,
        weight=1.0,
    )
    """Reward for reaching terrain target positions.
    Computes distance to nearest target patch from terrain.flat_patches['target'].
    Uses exponential reward: exp(-distance² / scale²) with scale=2.0m."""

    #========== Efficiency Rewards ==========
    
    propeller_efficiency = RewTerm(
        func=penalize_propeller_efficiency,
        weight=0.01,
    )
    """Penalty for excessive propeller speeds to encourage efficiency.
    Computes penalty based on propeller joint velocities, scaled to [-1, 0] using e^(-x) - 1.
    Since thrust ∝ ω², high speeds are inefficient."""
    
    energy_consumption = RewTerm(
        func=penalize_energy_consumption,
        weight=0.05,
    )
    """Penalty for total energy consumption from propellers and wheels.
    Uses PWM-to-Power model for propellers and RPM-to-Power model for wheels.
    Computes total power (W) and multiplies by dt to get energy per step (J).
    Scaled to [-1, 0] using exponential transformation with scale=20J."""

    # propeller_on_flat_ground = RewTerm(
    #     func=penalize_propeller_on_flat_ground,
    #     weight=10.0,
    # )
    """Penalty for using propellers on flat ground where wheels should suffice.
    Uses height scanner to detect flat terrain (std dev < 0.05m).
    Penalizes propeller usage when ground is flat, encouraging wheel-only locomotion on even terrain."""

    # ========== Stability Rewards ==========
    
    penalize_facing_mismatch = RewTerm(
        func=penalize_facing_direction_mismatch,
        weight=0.3,
    )
    """Penalty for mismatch between robot facing direction and target direction.
    Reads angle error directly from vel_command_b[:, 2] (normalized angle error in [-1, 1]).
    Penalizes when robot is not facing the target, scaled to [-1, 0] using e^(-x) - 1."""
    
    penalize_rotation = RewTerm(
        func=penalize_tilt_angle,
        weight=0.3,
    )
    """Penalty for excessive tilt angle (roll/pitch deviation from upright).
    Uses projected gravity to measure tilt. Strongly penalizes large roll/pitch angles
    to encourage upright, stable posture."""
    
    penalize_high_speed = RewTerm(
        func=penalize_excessive_linear_speed,
        weight=0.1,
    )
    """Penalty for excessive linear speed above 3 m/s threshold.
    Prevents dangerous high-speed movement. Only active when speed exceeds threshold."""

    
    
    
    # ========== Terminal Rewards ==========
    
    terminal_goal_reached = RewTerm(
        func=terminal_reward_goal_reached,
        weight=1.0,
    )
    """Terminal reward for successfully reaching the goal.
    Returns +10.0 when robot reaches the goal (episode ends due to goal_reached constraint).
    This is a positive terminal reward that encourages task completion."""
    
    terminal_propeller_collision = RewTerm(
        func=terminal_reward_propeller_collision,
        weight=1.0,
    )
    """Terminal reward (penalty) for propeller collision.
    Returns -10.0 when propellers collide (episode ends due to propeller_collision constraint).
    This is a negative terminal reward that penalizes unsafe behavior."""
    
    terminal_robot_out_of_bounds = RewTerm(
        func=terminal_reward_robot_out_of_bounds,
        weight=1.0,
    )
    """Terminal reward (penalty) for robot being thrown out of bounds.
    Returns -10.0 when robot height > 3m or XY distance > 6m from origin (episode ends due to robot_out_of_bounds constraint).
    This is a negative terminal reward that penalizes when the robot is thrown away from the scene."""

    

    # action_smoothness = RewTerm(
    #     func=lambda env: -torch.sum(torch.square(env.action_manager.action), dim=1),
    #     weight=0.001,
    # )
    # """Penalize large action magnitudes to encourage smooth, energy-efficient control."""


@configclass
class RewardsCfgInvertedPendulum(RewardsCfg):
    """Reward config for inverted-pendulum (wheels-only, same-level target).

    Deprecates propeller-related shaping rewards; terminal and task rewards unchanged.
    - propeller_efficiency: removed (no propeller actuation).
    - energy_consumption: removed (simplified to wheels-only, no hybrid energy tracking).
    - propeller_on_flat_ground: removed (no propeller actuation).
    - terminal_propeller_collision: kept (still penalize if propellers touch obstacles).
    """

    propeller_efficiency = None
    energy_consumption = None
    propeller_on_flat_ground = None
