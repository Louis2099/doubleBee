# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# class DoubleBeeVelocityCommand(UniformVelocityCommand):
#     def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)
#         # Command: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z]
#         # Override base class's 3-element buffer with 4-element buffer for DoubleBee
#         print(f"[DEBUG DoubleBeeVelocityCommand.__init__] BEFORE override: vel_command_b.shape = {self.vel_command_b.shape}")
#         self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
#         print(f"[DEBUG DoubleBeeVelocityCommand.__init__] AFTER override: vel_command_b.shape = {self.vel_command_b.shape}")
#         # Initialize with randomized commands instead of zeros
#         all_env_ids = torch.arange(self.num_envs, device=self.device)
#         self._resample_command(all_env_ids)
#         print(f"[DEBUG DoubleBeeVelocityCommand.__init__] AFTER resample: vel_command_b[0] = {self.vel_command_b[0]}")
#         print(f"[DEBUG DoubleBeeVelocityCommand.__init__] time_left[0] = {self.time_left[0]}")

#     @property
#     def command(self) -> torch.Tensor:
#         # Debug: Print commands periodically
#         if not hasattr(self, '_step_counter'):
#             self._step_counter = 0
#         self._step_counter += 1
#         if self._step_counter % 500 == 0:
#             print(f"[DEBUG command property] Step {self._step_counter}: vel_command_b[0] = {self.vel_command_b[0]}, time_left[0] = {self.time_left[0]:.2f}")
#         return self.vel_command_b

#     def _resample_command(self, env_ids):
#         # print(f"[DEBUG _resample_command] Called for {len(env_ids)} envs. Ranges: x={self.cfg.ranges.lin_vel_x}, y={self.cfg.ranges.lin_vel_y}, z={self.cfg.ranges.lin_vel_z}, yaw={self.cfg.ranges.ang_vel_z}")
#         r = torch.empty(len(env_ids), device=self.device)
#         self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
#         self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
#         self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_vel_z)
#         self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.ang_vel_z)
#         if len(env_ids) > 0:
#             print(f"[DEBUG _resample_command] AFTER resample: vel_command_b[{env_ids[0]}] = {self.vel_command_b[env_ids[0]]}")

#     def _update_command(self):
#         """Post-processes the velocity command.
        
#         This is called EVERY step by command_manager.compute().
#         Do NOT handle resets here - that's done by command_manager.reset() 
#         which is called automatically when episodes end.
        
#         For DoubleBee:
#         - No standing environments (rel_standing_envs = 0.0)
#         - No heading-based control
#         So no post-processing needed.
#         """
#         pass


class TerrainTargetDirectionCommand(UniformVelocityCommand):
    """Command generator that generates normalized XY velocity direction toward terrain targets.
    
    This command samples target positions from terrain.flat_patches["target"] and computes
    the normalized direction vector from robot base to target. The direction is updated
    every step in _update_command() to always point toward the current target.
    
    Command format: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z]
    - lin_vel_x, lin_vel_y: Normalized direction vector (magnitude = 1.0) in body frame
    - lin_vel_z: Set to 0.0 (no vertical velocity command)
    - ang_vel_z: Set to 0.0 (no angular velocity command)
    """
    
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the terrain-based direction command generator.
        
        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        
        # IMPORTANT: Keep command buffer as 3D to match original training: [lin_vel_x, lin_vel_y, ang_vel_z]
        # The original UniformVelocityCommand uses 3D format, so we must match it for checkpoint compatibility
        # Do NOT override to 4D - this would change observation space from 62 to 63 dimensions
        # self.vel_command_b is already 3D from parent class (UniformVelocityCommand)
        
        # Get terrain asset
        self.terrain = env.scene["terrain"]
        
        # Check if terrain has target patches
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "TerrainTargetDirectionCommand requires terrain.flat_patches['target']. "
                f"Found: {list(self.terrain.flat_patches.keys())}"
            )
        
        # Store target patches reference
        self.target_patches = self.terrain.flat_patches["target"]  # [terrain_levels, terrain_types, num_patches, 3]
        
        # Store current target positions for each environment (in world frame)
        # Shape: [num_envs, 3] - will be set in _resample_command
        self.current_targets_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize targets for all environments
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(all_env_ids.tolist())
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new target positions from terrain for specified environments.
        
        Args:
            env_ids: Environment indices to resample targets for.
        """
        if len(env_ids) == 0:
            return
        
        terrain_levels = self.terrain.terrain_levels  # [num_envs]
        terrain_types = self.terrain.terrain_types     # [num_envs]
        env_origins = self._env.scene.env_origins      # [num_envs, 3]
        
        # For each environment, sample a random target
        for env_idx in env_ids:
            # NOTE: In play mode with aligned initialization, the reset function
            # (reset_root_state_from_terrain_aligned) will set the target AFTER
            # this resample function runs. So we always resample here, and the
            # aligned reset function will overwrite it with an aligned target.
            # This ensures aligned initialization works correctly.
            
            level = terrain_levels[env_idx].item()
            ttype = terrain_types[env_idx].item()
            
            # Get target patches for this environment's terrain type
            targets = self.target_patches[level, ttype, :, :]  # [num_patches, 3]
            num_patches = targets.shape[0]
            
            if num_patches == 0:
                # No targets available, set to zero
                self.current_targets_w[env_idx, :] = 0.0
                continue
            
            # Randomly select one target patch
            patch_idx = torch.randint(0, num_patches, (1,), device=self.device).item()
            target_relative = targets[patch_idx, :]  # [3]
            
            # Transform to world coordinates
            target_world = target_relative + env_origins[env_idx, :]
            
            # NOTE: WORKAROUND: Add height offset to account for terrain step heights
            # Flat patches are stored with Z=0 relative to terrain base, but they may be on steps
            # at various heights. Since we can't query the actual terrain height here, we use
            # a fixed offset that approximates the average step height.
            # Note: This only affects visualization - rewards/constraints use XY only, so Z doesn't matter
            # For inverted pyramid with step_height_range=(0.01, 0.18), average step height is ~0.1m
            # Using 0.5m offset to account for multiple steps (up to ~0.72m for 4 steps)
            target_world[2] += 0.3  # Add 50cm offset to approximate step height
            
            self.current_targets_w[env_idx, :] = target_world
    
    def _update_command(self):
        """Update velocity command every step to point toward current target.
        
        This is called EVERY step by command_manager.compute().
        Computes normalized direction from robot base to target in body frame.
        """
        robot = self.robot
        num_envs = self.num_envs
        
        # Get robot base position in world frame
        robot_pos_w = robot.data.root_pos_w  # [num_envs, 3]
        
        # Compute direction vector from robot to target (in world frame)
        direction_w = self.current_targets_w - robot_pos_w  # [num_envs, 3]
        direction_xy_w = direction_w[:, :2]  # [num_envs, 2] - XY only
        
        # Compute distance
        distance_xy = torch.norm(direction_xy_w, dim=1)  # [num_envs]
        
        # Normalize direction (handle zero distance case)
        direction_xy_norm_w = direction_xy_w / (distance_xy.unsqueeze(1) + 1e-6)  # [num_envs, 2]
        
        # Transform direction from world frame to body frame
        # Get robot orientation (quaternion)
        robot_quat_w = robot.data.root_quat_w  # [num_envs, 4] (w, x, y, z)
        
        # Convert quaternion to rotation matrix (2D rotation for XY plane)
        # Extract yaw angle from quaternion
        # For quaternion (w, x, y, z), yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
        w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # [num_envs]
        
        # Create 2D rotation matrix to transform from world to body frame
        cos_yaw = torch.cos(yaw)  # [num_envs]
        sin_yaw = torch.sin(yaw)  # [num_envs]
        
        # Rotation matrix: R_body^world = [[cos, sin], [-sin, cos]]
        # To transform from world to body: v_body = R_body^world @ v_world
        direction_x_body = cos_yaw * direction_xy_norm_w[:, 0] + sin_yaw * direction_xy_norm_w[:, 1]
        direction_y_body = -sin_yaw * direction_xy_norm_w[:, 0] + cos_yaw * direction_xy_norm_w[:, 1]
        
        # Calculate angular velocity command based on angle error between robot facing direction and target direction
        # Robot's facing direction in body frame is [0, 1] (always along +Y axis)
        # Target direction in body frame is [direction_x_body, direction_y_body]
        # Angle error = atan2(direction_x_body, direction_y_body)
        # This gives the angle the robot needs to rotate to face the target (in radians, range: [-π, π])
        # atan2(x, y) gives angle from +Y axis to vector [x, y]
        angle_error = torch.atan2(direction_x_body, direction_y_body)  # [num_envs]
        
        # Normalize angle error to [-1, 1] range accounting for robot's symmetric structure
        # Since the robot is symmetric, facing the target backwards (abs(theta)=π) is equivalent to facing forward (theta=0)
        # Error should be: 0 at theta=0 or abs(theta)=π, and ±1 at abs(theta)=π/2
        # Using sin(angle_error) achieves this: sin(0)=0, sin(π/2)=1, sin(π)=0, sin(-π/2)=-1, sin(-π)=0
        ang_vel_z_command = torch.sin(angle_error)  # [num_envs]
        
        # Set normalized direction as XY velocity command
        # Use 3D format to match original training: [lin_vel_x, lin_vel_y, ang_vel_z]
        self.vel_command_b[:, 0] = direction_x_body  # lin_vel_x in body frame
        self.vel_command_b[:, 1] = direction_y_body  # lin_vel_y in body frame
        self.vel_command_b[:, 2] = ang_vel_z_command  # ang_vel_z: normalized angle error to face target
        # Note: lin_vel_z is not included in 3D format (original training didn't use it)
        
        # NOTE: Target is NOT resampled here. It stays fixed for the entire episode.
        # Resampling only happens in _resample_command() which is automatically called
        # by command_manager when environments reset (episode ends).


@configclass
class TerrainTargetDirectionCommandCfg(UniformVelocityCommandCfg):
    """Configuration for terrain target direction command."""
    
    class_type: type = TerrainTargetDirectionCommand
    
    # Note: ranges are not used for this command type, but required by base class
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
    )
    resampling_time_range: tuple[float, float] = (20.0, 20.0)  # Not used, but required
    rel_standing_envs: float = 0.0
    debug_vis: bool = False


@configclass
class DoubleBeeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for DoubleBee velocity command with vertical velocity support."""
    
    # Note: DoubleBeeVelocityCommand is currently commented out above
    # class_type: type = DoubleBeeVelocityCommand
    class_type: type = UniformVelocityCommand  # Placeholder until DoubleBeeVelocityCommand is uncommented
    
    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for velocity commands including vertical velocity."""
        
        lin_vel_z: tuple[float, float] = (-1.0, 1.0)
        """Range for the linear velocity in z-direction (vertical velocity for drone)."""
    
    ranges: Ranges = Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        lin_vel_z=(-1.0, 1.0),  # Vertical velocity for drone
        ang_vel_z=(-1.0, 1.0),
    )
    # resampling_time_range: tuple[float, float] = (6.0, 8.0)
    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    rel_standing_envs: float = 0.0
    debug_vis: bool = False
