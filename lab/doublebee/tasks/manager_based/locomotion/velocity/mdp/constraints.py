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

import os as _os
_DEBUG_GOAL = bool(_os.environ.get("DOUBLEBEE_DEBUG_GOAL"))
# Success-criterion ablation counters: ON by default (cheap, no syncs in the
# hot path). Set DOUBLEBEE_SUCCESS_ABLATION=0 to silence.
_SUCCESS_ABLATION = _os.environ.get("DOUBLEBEE_SUCCESS_ABLATION", "1") not in ("0", "", "false", "False")
_SUCCESS_EVERY = int(_os.environ.get("DOUBLEBEE_SUCCESS_EVERY", 2000))
# Which criteria GATE the terminal reward. Default "all" reproduces the
# behaviour of every run to date; the other three exist for the ablation.
_SUCCESS_MODE = _os.environ.get("DOUBLEBEE_SUCCESS_MODE", "all").lower()
if _SUCCESS_MODE not in ("xy", "xyz", "xyzu", "all"):
    raise ValueError("DOUBLEBEE_SUCCESS_MODE must be xy|xyz|xyzu|all, got %r" % _SUCCESS_MODE)
if _SUCCESS_MODE != "all":
    print("[SUCCESS ABLATION] terminal reward gated on %r, NOT the usual four criteria"
          % _SUCCESS_MODE, flush=True)

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

# def propeller_collision(
#     env: ManagerBasedEnv,
#     sensor_cfg: SceneEntityCfg,
#     threshold: float = 150.0,
# ) -> torch.Tensor:
#     """Terminate if a propeller sustains real contact with an obstacle.

#     Uses MEAN force over the history window (not max), because PhysX reports
#     huge single-substep contact spikes (500-6800N) for even trivial grazes.
#     Those spikes are solver artifacts, not real collisions. The MEAN reflects
#     sustained pressure — a prop genuinely jammed against a riser shows high
#     sustained force, while a brief graze averages low.

#     Args:
#         threshold: sustained force (N) over history to count as a real collision.
#                    ~40N given observed ~10N baseline during normal motion.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history  # (envs, hist, bodies, 3)

#     asset: Articulation = env.scene["robot"]
#     left_ids = asset.find_bodies("leftPropeller")[0]
#     right_ids = asset.find_bodies("rightPropeller")[0]

#     if len(left_ids) == 0 or len(right_ids) == 0:
#         return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

#     left_id = left_ids[0]
#     right_id = right_ids[0]

#     # force magnitude per history step: (envs, hist)
#     left_mags = torch.norm(net_contact_forces[:, :, left_id, :], dim=-1)
#     right_mags = torch.norm(net_contact_forces[:, :, right_id, :], dim=-1)

#     # SUSTAINED force = mean over history (ignores single-substep spikes)
#     left_sustained = left_mags.mean(dim=1)   # (envs,)
#     right_sustained = right_mags.mean(dim=1)  # (envs,)

#     # optional peak guard: also terminate on a truly enormous strike
#     # (real violent ram, not a graze). Set high so grazes never hit it.
#     # PEAK_GUARD = 3000.0
#     # left_peak = left_mags.max(dim=1)[0]
#     # right_peak = right_mags.max(dim=1)[0]

#     # left_collision = ((left_sustained > threshold) | (left_peak > PEAK_GUARD)).float()
#     # right_collision = ((right_sustained > threshold) | (right_peak > PEAK_GUARD)).float()
#     left_collision = (left_sustained > threshold).float()
#     right_collision = (right_sustained > threshold).float()

#     collision = torch.maximum(left_collision, right_collision)

#     if collision.dim() > 1:
#         collision = collision.squeeze()
#     elif collision.dim() == 0:
#         collision = collision.unsqueeze(0)
    
#     # print("LEFT SUSTAINED MAX: ", left_sustained.max().item())
#     # print("RIGHT SUSTAINED MAX: ", right_sustained.max().item())

#     # one-time symmetry check
#     if not hasattr(env, "_sym_check"):
#         asset = env.scene["robot"]
#         lid = asset.find_bodies("leftPropeller")[0][0]
#         rid = asset.find_bodies("rightPropeller")[0][0]
#         # positions relative to base, env 0
#         base_pos = asset.data.root_pos_w[0]
#         lpos = asset.data.body_pos_w[0, lid] - base_pos
#         rpos = asset.data.body_pos_w[0, rid] - base_pos
#         # print(f"[SYM] left_rel={lpos.tolist()}", flush=True)
#         # print(f"[SYM] right_rel={rpos.tolist()}", flush=True)
#         # masses
#         # print(f"[SYM] left_mass={asset.data.default_mass[0, lid].item():.4f} right_mass={asset.data.default_mass[0, rid].item():.4f}", flush=True)
#         env._sym_check = True
        
#     return collision

# def goal_reached(
#     env: ManagerBasedEnv,
#     distance_threshold: float = 0.25,
# ) -> torch.Tensor:
#     """Constraint that terminates if robot reaches the goal target.
    
#     Checks if the robot's XY position is within distance_threshold of the current target.
#     This indicates the robot has successfully reached the goal.
    
#     Args:
#         env: The environment instance
#         distance_threshold: Maximum distance in meters to consider as "reached" (default: 0.5m)
    
#     Returns:
#         Binary goal reached indicator per environment. Shape: (num_envs,)
#         - 1.0 = goal reached (terminate episode)
#         - 0.0 = goal not reached (continue episode)
#     """
#     robot = env.scene["robot"]
    
#     # Get robot base position in world frame (XY only)
#     robot_pos_w = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    
#     # Get the command term to access its selected target
#     cmd_manager = env.command_manager
#     if "base_velocity" not in cmd_manager._terms:
#         # Command not found, return no goal reached
#         return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
#     command_term = cmd_manager._terms["base_velocity"]
    
#     # Check if this is TerrainTargetDirectionCommand with current_targets_w
#     if not hasattr(command_term, "current_targets_w"):
#         # Not using terrain target command, fall back to finding nearest target
#         terrain = env.scene["terrain"]
#         if "target" not in terrain.flat_patches:
#             return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        
#         target_patches = terrain.flat_patches["target"]
#         terrain_levels = terrain.terrain_levels
#         terrain_types = terrain.terrain_types
#         env_origins = env.scene.env_origins
        
#         level_indices = terrain_levels
#         type_indices = terrain_types
#         targets_relative = target_patches[level_indices, type_indices, :, :]
#         targets_world = targets_relative + env_origins.unsqueeze(1)
#         targets_xy = targets_world[:, :, :2]
        
#         robot_pos_xy_expanded = robot_pos_w.unsqueeze(1)
#         distances = torch.norm(targets_xy - robot_pos_xy_expanded, dim=2)
#         min_distances = torch.min(distances, dim=1)[0]
#     else:
#         # Use the command's selected target (aligned with command)
#         current_targets_w = command_term.current_targets_w  # [num_envs, 3]
#         current_targets_xy = current_targets_w[:, :2]  # [num_envs, 2]
        
#         # Compute distance from robot to command's selected target
#         distances_xy = robot_pos_w - current_targets_xy  # [num_envs, 2]
#         min_distances = torch.norm(distances_xy, dim=1)  # [num_envs]
    
#     # Check if robot is within threshold of goal
#     goal_reached = (min_distances <= distance_threshold).float()
    
#     # Ensure 1D shape: (num_envs,) not (num_envs, 1)
#     if goal_reached.dim() > 1:
#         goal_reached = goal_reached.squeeze()
#     elif goal_reached.dim() == 0:
#         # Handle scalar case (shouldn't happen, but be safe)
#         goal_reached = goal_reached.unsqueeze(0)
    
#     # print(f"[GOAL] robot_w={robot_pos_w[0].tolist()} target={current_targets_xy[0].tolist()} "
#         #   f"env_origin={env.scene.env_origins[0,:2].tolist()} dist={min_distances[0].item():.2f}", flush=True)
    
#     # in goal_reached:
#     # print(f"[GOAL] robot={robot_pos_w[0].tolist()} "
#         # f"cmd_target={current_targets_xy[0].tolist()} "
#         # f"buffer_target={env._aligned_targets_buffer[0,:2].tolist()} "
#         # f"dist={min_distances[0].item():.2f}", flush=True)
    
#     return goal_reached

def goal_reached(
    env: ManagerBasedEnv,
    distance_threshold: float = 0.25,
    upright_threshold: float = 0.15,
    ang_vel_threshold: float = 3.5,
) -> torch.Tensor:
    """Constraint that terminates if robot reaches the goal target.

    Checks THREE things now, not just XY distance: (1) close to target,
    (2) upright (not tipped over), (3) settled (not mid-tumble). This
    prevents counting a stumble-that-ends-up-close as a "success" — the
    metric was previously blind to HOW the robot arrived, only WHERE.

    Args:
        env: The environment instance
        distance_threshold: Maximum distance in meters to consider as "reached"
        upright_threshold: Max allowed tilt-from-vertical (via projected gravity)
        ang_vel_threshold: Max allowed angular velocity magnitude (rad/s) — must be settled

    Returns:
        Binary goal reached indicator per environment. Shape: (num_envs,)
    """
    robot = env.scene["robot"]

    robot_pos_w = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    cmd_manager = env.command_manager
    if "base_velocity" not in cmd_manager._terms:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command_term = cmd_manager._terms["base_velocity"]

    if not hasattr(command_term, "current_targets_w"):
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
        current_targets_w = command_term.current_targets_w
        current_targets_xy = current_targets_w[:, :2]

        distances_xy = robot_pos_w - current_targets_xy
        min_distances = torch.norm(distances_xy, dim=1)

    # --- existing XY check ---
    close_enough = min_distances <= distance_threshold

    # --- upright check ---
    uprightness = -robot.data.projected_gravity_b[:, 2]
    is_upright = uprightness > (1.0 - upright_threshold)

    # --- settled check ---
    ang_vel_mag = torch.norm(robot.data.root_ang_vel_w, dim=1)
    is_settled = ang_vel_mag < ang_vel_threshold

    # --- height check ---
    robot_z = robot.data.root_pos_w[:, 2]
    if hasattr(command_term, "current_targets_w"):
        target_z = command_term.current_targets_w[:, 2]
        height_diff = target_z - robot_z
        at_height = height_diff < 0.15
    else:
        # no target Z available — skip height check
        at_height = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    # DOUBLEBEE_SUCCESS_MODE selects WHICH criteria gate the terminal reward.
    # This is a training-signal ablation, not a reporting one: the conjunction
    # decides which terminal states the policy is paid for, so a looser gate
    # teaches it to finish in attitudes the hardware cannot survive. The
    # counters below always score all four subsets regardless of the mode, so a
    # run trained under one gate is still measured under every gate.
    #   xy    proximity only              (the naive criterion)
    #   xyz   + elevation                 (climbing required)
    #   xyzu  + uprightness               (no rate check)
    #   all   + settled body rate         (ours, the default)
    if _SUCCESS_MODE == "xy":
        _gate = close_enough
    elif _SUCCESS_MODE == "xyz":
        _gate = close_enough & at_height
    elif _SUCCESS_MODE == "xyzu":
        _gate = close_enough & at_height & is_upright
    else:
        _gate = close_enough & is_upright & is_settled & at_height
    goal_reached = _gate.float()

    # ---- success-criterion ablation counters (2026-09-02) -------------------
    # The four criteria gate the TERMINAL REWARD, so they are a training signal
    # and not only a reporting convention: a looser criterion pays the policy
    # for terminal states the hardware cannot survive. This accumulates, for
    # the same trajectories, how often each nested subset would have declared
    # success -- which gives the inflation factor alpha directly -- plus the
    # attitude at which the XY-only criterion would have fired, which is the
    # sim2real argument in physical units.
    #
    # Everything stays on the GPU; the only .item() calls happen at print time,
    # once every DOUBLEBEE_SUCCESS_EVERY calls (default 2000 ~= every 80
    # iterations at 24 calls/iteration). Cost per call is a handful of reduces.
    if _SUCCESS_ABLATION:
        if not hasattr(env, "_succ_abl"):
            env._succ_abl = {
                "n": 0,
                "xy": torch.zeros((), device=env.device),
                "xyz": torch.zeros((), device=env.device),
                "xyzu": torch.zeros((), device=env.device),
                "all": torch.zeros((), device=env.device),
                # attitude at which XY-only would have fired but all-four did not
                "lean_sum": torch.zeros((), device=env.device),
                "rate_sum": torch.zeros((), device=env.device),
                "loose_n": torch.zeros((), device=env.device),
            }
        _a = env._succ_abl
        _xy = close_enough
        _xyz = close_enough & at_height
        _xyzu = _xyz & is_upright
        _all = _xyzu & is_settled
        _a["xy"] += _xy.sum()
        _a["xyz"] += _xyz.sum()
        _a["xyzu"] += _xyzu.sum()
        _a["all"] += _all.sum()
        # states the loose criterion would have accepted and the strict one rejects
        _loose = _xy & (~_all)
        _lean = torch.rad2deg(torch.arccos(torch.clamp(uprightness, -1.0, 1.0)))
        _a["lean_sum"] += (_lean * _loose).sum()
        _a["rate_sum"] += (ang_vel_mag * _loose).sum()
        _a["loose_n"] += _loose.sum()
        _a["n"] += 1
        if _a["n"] % _SUCCESS_EVERY == 0:
            xy = _a["xy"].item(); allf = _a["all"].item()
            ln = _a["loose_n"].item()
            print("[SUCCESS ABLATION] hits: XY %d | +Z %d | +upright %d | +settled(all four) %d"
                  "  ->  alpha = XY/all = %s"
                  % (xy, _a["xyz"].item(), _a["xyzu"].item(), allf,
                     ("%.2f" % (xy / allf)) if allf > 0 else "n/a (no strict successes yet)"),
                  flush=True)
            print("[SUCCESS ABLATION] states XY-only accepts but all-four rejects: "
                  "n=%d  mean lean %.1f deg  mean body rate %.2f rad/s"
                  % (ln, (_a["lean_sum"].item() / ln) if ln > 0 else float("nan"),
                     (_a["rate_sum"].item() / ln) if ln > 0 else float("nan")),
                  flush=True)
    # -------------------------------------------------------------------------

    if goal_reached.dim() > 1:
        goal_reached = goal_reached.squeeze()
    elif goal_reached.dim() == 0:
        goal_reached = goal_reached.unsqueeze(0)

    # add inside goal_reached, after computing all the check tensors, before the return
    # 2026-08-31: gated behind DOUBLEBEE_DEBUG_GOAL. `close_enough[0].item()` is a
    # GPU sync on EVERY call, and goal_reached is called from the reward term, the
    # constraint manager and the curriculum -- so several syncs per step, plus 7
    # more and a flushed stdout write on every step env0 sits near the target.
    # With success now ~14% that fires constantly. Same gating pattern as
    # DOUBLEBEE_DEBUG_RESET in off_policy_runner.
    if _DEBUG_GOAL and close_enough[0].item():
        print(f"[NEAR GOAL env0] close={close_enough[0].item()} "
            f"upright={is_upright[0].item()} "
            f"settled={is_settled[0].item()} "
            f"height_ok={at_height[0].item()} "
            f"height_diff={height_diff[0].item():.3f} "
            f"ang_vel={torch.norm(robot.data.root_ang_vel_w[0]).item():.3f} "
            f"uprightness={(-robot.data.projected_gravity_b[0, 2]).item():.3f}", flush=True)

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

