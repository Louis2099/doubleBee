# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab.utils.math import quat_apply 

# top of rewards.py, after imports
import json, os as _os

def _load_poly(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    import torch
    return torch.tensor(d["coeffs"], dtype=torch.float32), d["is_exponential"]

_MDPDIR = _os.path.dirname(__file__)
_PWM_POWER_COEFFS, _PWM_POWER_IS_EXP = _load_poly(
    _os.path.join(_MDPDIR, "pwm2power_params.json")
)
_RPM_POWER_COEFFS, _RPM_POWER_IS_EXP = _load_poly(
    _os.path.join(_MDPDIR, "rpm2power_params.json")
)

def _torch_polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Horner's method — works on any shape x, coeffs on same device."""
    out = torch.zeros_like(x)
    for c in coeffs:
        out = out * x + c
    return out

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

def reward_stable_after_climb(env) -> torch.Tensor:
    """Reward being UPRIGHT and SETTLED after gaining height. Targets the
    brute-force-then-topple failure: the robot boosts up the step but can't catch
    itself. This rewards 'you climbed AND you're level AND you're not tumbling',
    so lurch-and-fall gets nothing and controlled climb-and-settle gets rewarded."""
    robot = env.scene["robot"]

    # height gained above spawn
    spawn_z = env.scene.env_origins[:, 2]
    height_above = robot.data.root_pos_w[:, 2] - spawn_z
    climbed = torch.clamp(height_above / 0.04, 0.0, 1.0)  # ramps 0->1 over first 4cm

    # upright: projected gravity Z near -1 means body is level
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)  # 1 = level, 0 = on its side

    # settled: low angular velocity (not tumbling/toppling)
    ang_vel_mag = torch.norm(robot.data.root_ang_vel_w, dim=1)
    settled = torch.exp(-ang_vel_mag / 2.0)  # 1 when still, decays as it spins

    # reward only when all three: climbed, upright, settled
    return climbed * upright * settled

def reward_climb_progress(env) -> torch.Tensor:
    """Reward GAINING HEIGHT while at a step. Technique-agnostic: doesn't prescribe
    pitch/prop/servo — rewards the OUTCOME (going up at a step) so the policy can
    discover how. Safety (no face-plant, no collision) is handled by separate penalties."""
    robot = env.scene["robot"]

    # --- rising: upward velocity of the body ---
    vz = robot.data.root_lin_vel_w[:, 2]
    # 2026-08-28: LOWER BOUND 0.0 -> -1.0. THIS TERM WAS A BOUNCE FARM.
    #
    # Clamped at 0 the up-stroke paid and the down-stroke was FREE, and the only
    # gate is "is a step nearby" -- which stays true while hovering beside one.
    # So oscillating vertically next to a step collected reward on every rise and
    # lost nothing on every fall, without ever going anywhere.
    #
    # Observed in play on RUN 2: climbs well, but "goes up and down around the
    # target rather than to it", with terrain_levels stuck at 0.02 and success at
    # 0.12 -- it has the motor skill and was spending it farming. The play log
    # backs the motor skill up: thrust z_frac 0.967 (min 0.951), servo controlled
    # to -0.107..+0.253 rad, wheel tracking 0.45, lean settling, prop action
    # modulating 0.013-0.85.
    #
    # Symmetric clamp makes a bounce net ~zero, so only NET height gain pays.
    # Safe on a staircase, where legitimate motion is monotonically upward.
    rising = torch.clamp(vz / 0.2, -1.0, 1.0)  # -1 to 1; a bounce now nets zero

    # --- step-ahead gate: is there terrain higher than the robot nearby (a step)? ---
    step_ahead = torch.ones(robot.num_instances, device=robot.device)
    try:
        hs = env.scene["height_scanner"]
        ray_z = torch.nan_to_num(
            hs.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0
        )  # [num_envs, num_rays] world heights
        ground_z = ray_z.median(dim=1)[0]              # flat-majority reference
        max_ahead = ray_z.max(dim=1)[0]                # highest scanned point
        step_ahead = torch.clamp((max_ahead - ground_z) / 0.04, 0.0, 1.0)  # 0-1, sat at 4cm step
    except (KeyError, ValueError, IndexError):
        pass  # no scanner -> gate stays 1 (fallback)
    
    try:
        sj = robot.joint_names.index("leftPropellerServo")  # or whatever the servo joint is named
        servo_pos = robot.data.joint_pos[:, sj]
        if not hasattr(env, "_servo_dbg"): env._servo_dbg = 0
        env._servo_dbg += 1
        if env._servo_dbg % 50 == 0:
            print(f"[SERVO] pos(rad)={servo_pos[0].item():.3f} min={servo_pos.min().item():.3f} max={servo_pos.max().item():.3f}", flush=True)
    except (ValueError, IndexError):
        pass

        # ---- TEMP: log thrust world direction during climb ----
    try:
        lp = robot.body_names.index("leftPropeller")
        rp = robot.body_names.index("rightPropeller")
        prop_ids = torch.tensor([lp, rp], device=robot.device)
        prop_quat = robot.data.body_quat_w[:, prop_ids, :]  # [n,2,4]

        # thrust is along prop local +Z (matches apply_propeller_aerodynamics)
        thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
        thrust_local[:, :, 2] = 1.0
        thrust_world = quat_apply(prop_quat, thrust_local)  # [n,2,3]
        z_frac = thrust_world[:, :, 2].mean(dim=1)  # world-Z fraction, 1=up 0=horiz -1=down

        if not hasattr(env, "_thrust_dbg"):
            env._thrust_dbg = 0
        env._thrust_dbg += 1
        if env._thrust_dbg % 50 == 0:
            print(f"[THRUST DIR] env0 z_frac={z_frac[0].item():.3f} "
                  f"mean={z_frac.mean().item():.3f} "
                  f"min={z_frac.min().item():.3f} max={z_frac.max().item():.3f}", flush=True)
    except (ValueError, IndexError):
        pass
    # ---- END TEMP ----

    # ---- TEMP 2026-08-26: BALANCE PROBE ----------------------------------
    # Does the wheel command respond to lean AT ALL? Everything else has been
    # eliminated (optimizer, propeller actuator, reward sign, reward shaping,
    # wheel wiring, body inertia, and actuator dead time -- p*theta 0.79 -> 0.00
    # changed nothing). If the policy is not using its own attitude observation,
    # no reward or plant change can matter.
    #
    # Correlations are taken ACROSS ENVS at one step, which is the right axis
    # here: 1024 robots at 1024 different lean angles, each choosing a wheel
    # command. A balancing policy MUST show a strong lean->command relationship.
    try:
        if not hasattr(env, "_bal_dbg"):
            env._bal_dbg = 0
            env._sact_prev = None
            env._sact_ac_sum = 0.0
            env._sact_ac_n = 0
        env._bal_dbg += 1

        # TEMPORAL lag-1 autocorrelation of the servo action, accumulated EVERY
        # tick. A value near -1 means the policy alternates its servo command
        # every step -- the failure mode measured on hardware (hw_v17.csv,
        # corr(servo_pos, servo_act) = -1.000).
        #
        # 2026-08-27: THIS WAS READING THE WRONG ACTION INDEX.
        #
        # Hardcoded [:, 1] was the servo under the PRE-2026-08-26 layout
        # {wheel:0, servo:1, prop:2,3}. ActionsCfg4D now maps
        # {wheel_common:0, wheel_diff:1, servo:2, prop:3}, so index 1 has been
        # the WHEEL DIFFERENTIAL since the CommonDiff change. Every
        # corr(spos,sact) / corr(lean,sact) / lag1_ac printed since then
        # correlated servo POSITION against a WHEEL command -- they say nothing
        # about the servo loop. Resolve by term name so a future remap cannot
        # silently break it again.
        if not hasattr(env, "_servo_act_idx"):
            _idx, _found = 0, None
            for _name, _term in env.action_manager._terms.items():
                if _name == "propeller_servo_pos":
                    _found = _idx
                    break
                _idx += _term.action_dim
            env._servo_act_idx = _found
        _si = env._servo_act_idx
        _sa = (env.action_manager.action[:, _si] if _si is not None
               else torch.zeros(env.num_envs, device=env.device))
        if env._sact_prev is not None:
            _a = _sa - _sa.mean()
            _b = env._sact_prev - env._sact_prev.mean()
            _d = _a.norm() * _b.norm()
            if _d > 1e-8:
                env._sact_ac_sum += float((_a @ _b / _d).item())
                env._sact_ac_n += 1
        env._sact_prev = _sa.clone()

        if env._bal_dbg % 50 == 0:
            def _corr(a, b):
                a = a.float(); b = b.float()
                a = a - a.mean(); b = b - b.mean()
                d = a.norm() * b.norm()
                return (a @ b / d).item() if d > 1e-8 else float("nan")

            # SERVO CHANNEL added 2026-08-26. Deployment has now guessed three
            # times at how to reconstruct servo_pos for hardware (pure delay ->
            # 4.2 Hz limit cycle; raw command -> corr(pos,act) = -1.000 exactly,
            # a one-tick flip-flop; first-order lag -> untested). These three
            # numbers are the ground truth those guesses should be matched to:
            #   corr(servo_pos, servo_act)  is the loop gain the policy applies
            #   corr(lean,      servo_act)  is whether it aims at the fall AT ALL
            #   ac1                         is whether SIM's own servo alternates
            sj_l = robot.joint_names.index("leftPropellerServo")
            svp = robot.data.joint_pos[:, sj_l]
            sva = _sa   # resolved by term name above, NOT hardcoded index 1
            # lag-1 correlation must be TEMPORAL (this tick vs last tick), not
            # across envs -- env i against env i+1 measures nothing. Accumulated
            # every tick below and averaged over the 50-tick print interval.
            ac1 = env._sact_ac_sum / max(env._sact_ac_n, 1)

            lean = robot.data.projected_gravity_b[:, 1]   # fwd/back (sim forward = +Y)
            roll = robot.data.projected_gravity_b[:, 0]
            rate = robot.data.root_ang_vel_b[:, 0]        # pitch rate about the axle
            act = env.action_manager.action
            wcmd = act[:, 0]                              # tied wheel action (index 0 in both cfgs)
            wj = robot.joint_names.index("leftWheel")
            wvel = robot.data.joint_vel[:, wj]

            print(
                "[SERVOLOOP] corr(spos,sact)=%+.3f corr(lean,sact)=%+.3f "
                "lag1_ac=%+.3f | spos mean=%+.3f sd=%.3f min=%+.3f max=%+.3f | "
                "sact mean=%+.3f sd=%.3f"
                % (_corr(svp, sva), _corr(lean, sva), ac1,
                   svp.mean().item(), svp.std().item(),
                   svp.min().item(), svp.max().item(),
                   sva.mean().item(), sva.std().item()),
                flush=True,
            )
            # SERVO ASYMMETRY, added 2026-08-27 to test a hardware observation:
            # "servos only help the lean when falling BACK, never when falling
            # FORWARD, because they are being used to move forward -- and
            # forward is the most common failure."
            #
            # The two uses demand OPPOSITE tilts. Propulsion needs horizontal
            # thrust pointing forward, which at a 0.4476 m arm above the axle
            # torques the body further forward. Recovery from a forward fall
            # needs the opposite. So a policy paid for progress (weight 10.0)
            # learns the propulsion tilt and is then holding exactly the wrong
            # servo angle at the moment a forward fall starts. Backward falls
            # get recovery for free, because there the propulsion tilt IS the
            # restoring tilt -- which is why the asymmetry shows up as "helps
            # one way only" rather than "never helps".
            #
            # lean = projected_gravity_b[:, 1]; rewards.py uses
            # pitch_signed = -that as "positive when pitched FORWARD", so here
            # lean < 0 is FORWARD lean and lean > 0 is BACKWARD lean.
            #
            # Read it as: if the servo is doing balance work, |corr| should be
            # comparable in both bins and the SIGNS SHOULD MATCH (same restoring
            # convention either way). A strong correlation in the back bin and
            # ~0 or opposite-signed in the fwd bin confirms the observation.
            _fwd = lean < -0.02
            _bck = lean > 0.02
            _nf, _nb = int(_fwd.sum()), int(_bck.sum())
            _cf = _corr(lean[_fwd], sva[_fwd]) if _nf > 8 else float("nan")
            _cb = _corr(lean[_bck], sva[_bck]) if _nb > 8 else float("nan")
            print(
                "[SERVOASYM] FWD-lean n=%4d corr(lean,sact)=%+.3f sact=%+.3f | "
                "BACK-lean n=%4d corr(lean,sact)=%+.3f sact=%+.3f"
                % (_nf, _cf,
                   sva[_fwd].mean().item() if _nf > 0 else float("nan"),
                   _nb, _cb,
                   sva[_bck].mean().item() if _nb > 0 else float("nan")),
                flush=True,
            )

            print(
                "[BALANCE] corr(lean,wcmd)=%+.3f corr(rate,wcmd)=%+.3f "
                "corr(lean,wvel)=%+.3f corr(wcmd,wvel)=%+.3f | "
                "lean sd=%.3f roll sd=%.3f wcmd mean=%+.3f sd=%.3f "
                "wvel mean=%+.1f sd=%.1f"
                % (_corr(lean, wcmd), _corr(rate, wcmd), _corr(lean, wvel),
                   _corr(wcmd, wvel), lean.std().item(), roll.std().item(),
                   wcmd.mean().item(), wcmd.std().item(),
                   wvel.mean().item(), wvel.std().item()),
                flush=True,
            )
            env._sact_ac_sum = 0.0
            env._sact_ac_n = 0
    except Exception as _e:
        # NEVER swallow this one silently -- a bare `except: pass` is why the
        # first version of this probe printed nothing at all and looked like it
        # had not been synced.
        if not hasattr(env, "_bal_err"):
            env._bal_err = 0
        env._bal_err += 1
        if env._bal_err <= 3:
            print("[BALANCE] PROBE FAILED: %s: %s" % (type(_e).__name__, _e), flush=True)
    # ---- END BALANCE PROBE ----------------------------------------------

    # reward: going UP while AT a step. Technique-agnostic.
    return rising * step_ahead

def reward_thrust_up_at_step(env) -> torch.Tensor:
    robot = env.scene["robot"]
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac = torch.clamp(thrust_world[:, :, 2].mean(dim=1), 0.0, 1.0)
    z_frac = z_frac ** 2

    # step gate only — NO prop_active term, so it doesn't reward spinning props FAST
    step_ahead = torch.zeros(robot.num_instances, device=robot.device)
    try:
        hs = env.scene["height_scanner"]
        ray_z = torch.nan_to_num(hs.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        step_ahead = torch.clamp((ray_z.max(dim=1)[0] - ray_z.median(dim=1)[0]) / 0.04, 0.0, 1.0)
    except (KeyError, ValueError, IndexError):
        pass
    return z_frac * step_ahead   # reward pointing up at a step, regardless of prop speed

def reward_progress_to_target(env):
    robot = env.scene["robot"]
    cmd = env.command_manager._terms.get("base_velocity")
    if cmd is None or not hasattr(cmd, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)
    dist = torch.norm(robot.data.root_pos_w[:, :2] - cmd.current_targets_w[:, :2], dim=1)
    if not hasattr(env, "_pd") or env._pd.shape != dist.shape:
        env._pd = dist.clone(); return torch.zeros_like(dist)
    prog = env._pd - dist; env._pd = dist.clone()
    return torch.clamp(prog, min=0.0) * 10.0

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
        # if torch.rand(1).item() < 0.01:
        #     print(f"[REACH] min_dist range: {min_distances.min().item():.2f} to {min_distances.max().item():.2f}")
    
    # Exponential reward: exp(-distance² / scale²)
    # Scale = 2.0 means reward drops to ~0.6 at 2m, ~0.13 at 4m
    scale = 1.5
    rewards = torch.exp(-(min_distances ** 2) / (scale ** 2))

    # NEW: scale proximity reward by height-match to target.
    # Parking at the base earns 50% (still pulls robot in for navigation),
    # being at target elevation earns 100% — makes climbing the better play.
    # if hasattr(command_term, "current_targets_w"):
    #     robot_z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    #     target_z = command_term.current_targets_w[:, 2]
    #     height_match = torch.exp(-torch.abs(target_z - robot_z) / 0.1)  # 1 at target height, →0 at base
    #     height_factor = 0.5 + 0.5 * height_match  # floor 0.5, max 1.0
    #     rewards = rewards * height_factor
    if hasattr(command_term, "current_targets_w"):
        robot_z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        target_z = command_term.current_targets_w[:, 2]
        height_match = torch.exp(-torch.abs(target_z - robot_z) / 0.1)
        # height_factor = 0.5 + 0.5 * height_match
        height_factor = 0.2 + 0.8 * height_match
        rewards = rewards * height_factor

        # print(f"[FRAME] robot_z={robot.data.root_pos_w[0,2].item():.2f} "
        #   f"target_z={command_term.current_targets_w[0,2].item():.2f} "
        #   f"env_origin_z={env.scene.env_origins[0,2].item():.2f}", flush=True)

    return rewards

def reward_prop_catch_when_falling(env) -> torch.Tensor:
    """Reward using upward prop thrust to arrest a downward fall. Fires when the robot
    is moving DOWN (vz < 0) and rewards props pointed up + spinning — teaching the
    'catch reflex' to save itself from slowly falling."""
    robot = env.scene["robot"]

    # falling: downward velocity (only active when descending)
    vz = robot.data.root_lin_vel_w[:, 2]
    falling = torch.clamp(-vz / 0.2, 0.0, 1.0)  # 0 when rising/level, →1 when falling fast

    # props pointed up
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac = torch.clamp(thrust_world[:, :, 2].mean(dim=1), 0.0, 1.0)

    # props spinning (producing thrust)
    lpj = robot.joint_names.index("leftPropeller")
    rpj = robot.joint_names.index("rightPropeller")
    prop_speed = torch.clamp(robot.data.joint_vel[:, [lpj, rpj]].abs().mean(dim=1) / 200.0, 0.0, 1.0)

    # reward: falling AND props up AND spinning = catching itself
    return falling * z_frac * prop_speed

def penalize_cross_track_error(env) -> torch.Tensor:
    """Penalize lateral deviation from the straight line spawn→target."""
    robot = env.scene["robot"]
    cmd = env.command_manager._terms.get("base_velocity")
    if cmd is None or not hasattr(cmd, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)

    target_xy = cmd.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    x_deviation = torch.abs(robot_xy[:, 0] - target_xy[:, 0])

    # ---- TEMP FRAME DEBUG ----
    # if not hasattr(env, "_ct_dbg"):
    #     env._ct_dbg = 0
    # env._ct_dbg += 1
    # if env._ct_dbg % 30 == 0:
    #     env_origin = env.scene.env_origins[0]
    #     print(f"[CROSSTRACK] robot_xy={robot_xy[0].tolist()} "
    #           f"target_xy={target_xy[0].tolist()} "
    #           f"env_origin_xy={env_origin[:2].tolist()} "
    #           f"x_dev={x_deviation[0].item():.3f} "
    #           f"y_dev={(robot_xy[0,1]-target_xy[0,1]).item():.3f}", flush=True)
    # ---- END TEMP ----

    return -x_deviation

def terminal_reward_goal_reached(env, alive_weight: float = 0.0,
                                 terminal_weight: float = 10.0,
                                 reward_value: float = 100.0) -> torch.Tensor:
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
    goal_reached_mask = goal_reached(env, distance_threshold=0.25)  # [num_envs]

    rewards = goal_reached_mask * float(reward_value)

    # ---- FORFEITED-INCOME COMPENSATION, added 2026-08-27 -------------------
    #
    # Reaching the goal ENDS the episode, which cuts off every per-step reward
    # the policy would have collected for the rest of it. reward_alive_upright
    # pays `alive_weight` per step while upright, so finishing at step t costs
    # the policy alive_weight * (T - t) and gains it only the terminal bonus.
    #
    # Measured, over a 1000-step cap with alive_weight 2.0 and terminal 10.0:
    #     episode length 175  -> forfeits 1650, gains 10
    #     episode length 369  -> forfeits 1262, gains 10
    # Farming strictly dominates AT EVERY EPISODE LENGTH.
    #
    # RUN 1 (2026-08-27_02-28) still went for the goal only because it could not
    # actually survive to the cap -- episodes ended by tilt at ~175 steps, so the
    # alive income was not reachable. RUN 2 gained enough balance (action history
    # against 300 ms of wheel lag) to reach 369 steps with 33% timeouts, at which
    # point the exploit became available and it took it: success 0.34 -> 0.00,
    # terrain 0.58 -> 0.00, and velocity_direction_alignment went NEGATIVE, i.e.
    # it drove away from the goal deliberately.
    #
    # The improvement unlocked the bug. Lowering alive_weight only moves the
    # threshold -- a better policy crosses it again. Paying the forfeited income
    # AS PART OF the terminal bonus removes the trap instead: finishing early and
    # surviving to the cap become worth the same, so the goal bonus is pure
    # profit on top and the policy finishes AS FAST AS IT CAN.
    #
    # alive_weight and terminal_weight MUST match RewardsCfg. They are passed in
    # rather than read from the reward manager because the manager's term-config
    # accessor is not stable across IsaacLab versions; the assert below catches
    # the drift that duplication invites.
    if alive_weight > 0.0:
        assert terminal_weight > 0.0, "terminal_weight must be positive"
        steps_left = torch.clamp(
            env.max_episode_length - env.episode_length_buf.float(), min=0.0)
        # expressed in units of THIS term, since RewTerm multiplies by
        # terminal_weight afterwards
        comp = (alive_weight / terminal_weight) * steps_left
        rewards = rewards + goal_reached_mask * comp

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
        threshold=1.0 # was 1.0
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
    cmd_manager = env.command_manager
    # vel_cmd = cmd_manager.get_command("base_velocity")
    
    # print(f"[FACING] cmd shape={vel_cmd.shape} cmd[0]={vel_cmd[0].tolist()}", flush=True)

    # if vel_cmd.shape[1] >= 3:
    #     angle_error_normalized = vel_cmd[:, 2]
    # else:
    #     robot = env.scene["robot"]
    #     return torch.zeros(robot.num_instances, device=robot.device)
    # return -torch.abs(angle_error_normalized)

    robot = env.scene["robot"]
    cmd_term = cmd_manager._terms["base_velocity"]
    target_xy = cmd_term.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    to_target = target_xy - robot_xy
    desired_yaw = torch.atan2(to_target[:, 0], to_target[:, 1])  # +Y facing convention
    # robot's actual yaw from quat
    q = robot.data.root_quat_w
    robot_yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2))
    angle_error = torch.atan2(torch.sin(desired_yaw - robot_yaw), torch.cos(desired_yaw - robot_yaw))
    
    deadzone = 0.3 # ~17° free
    penalty = torch.clamp(torch.abs(angle_error) - deadzone, min=0.0)
    
    # print(f"[FACING] angle_error[0]={angle_error[0].item():.2f} rad", flush=True)

    return -penalty

def penalize_tilt_angle(env) -> torch.Tensor:
    robot = env.scene["robot"]
    projected_gravity = robot.data.projected_gravity_b
    tilt_x = torch.abs(projected_gravity[:, 0])
    tilt_y = torch.abs(projected_gravity[:, 1])
    weighted_sum = 3.0 * torch.sqrt(tilt_x) + 7.0 * torch.sqrt(tilt_y)
    scaled_penalty = torch.exp(-weighted_sum) - 1.0

    # Reduce tilt penalty near stairs — robot needs to pitch to climb
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is not None and hasattr(cmd_term, "current_targets_w"):
        target_xy = cmd_term.current_targets_w[:, :2]
        robot_xy = robot.data.root_pos_w[:, :2]
        dist = torch.norm(robot_xy - target_xy, dim=1)
        near_target = torch.exp(-dist / 1.5)
        scaled_penalty = scaled_penalty * (1.0 - 0.85 * near_target)

    return scaled_penalty

def penalize_excessive_linear_speed(env, speed_threshold: float = 1.0) -> torch.Tensor: # was 3.0
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
    """Vectorized energy penalty from empirically-fit actuator power models.
    
    Propeller: joint vel (rad/s) -> PWM -> Power (W) via degree-4 poly fit to bench data.
    Wheel: joint vel (rad/s) -> RPM -> Power (W) via degree-4 poly fit to motor data.
    No Python loops — pure tensor ops on GPU.
    """
    robot = env.scene["robot"]
    device = robot.device

    try:
        dt = env.step_dt

        pwm_coeffs = _PWM_POWER_COEFFS.to(device)
        rpm_coeffs = _RPM_POWER_COEFFS.to(device)

        # ===== Propeller power =====
        left_p = robot.joint_names.index("leftPropeller")
        right_p = robot.joint_names.index("rightPropeller")
        prop_vels = robot.data.joint_vel[:, [left_p, right_p]]      # [num_envs, 2]

        prop_pwm = 1000.0 + (torch.abs(prop_vels) / 500.0) * 650.0
        prop_pwm = torch.clamp(prop_pwm, 1000.0, 2000.0)

        prop_power = _torch_polyval(pwm_coeffs, prop_pwm)           # [num_envs, 2]
        if _PWM_POWER_IS_EXP:
            prop_power = torch.exp(prop_power)
        prop_power = torch.clamp(prop_power, min=0.0)
        total_prop_power = prop_power.sum(dim=1)                     # [num_envs]

        # ===== Wheel power =====
        left_w = robot.joint_names.index("leftWheel")
        right_w = robot.joint_names.index("rightWheel")
        wheel_vels = robot.data.joint_vel[:, [left_w, right_w]]     # [num_envs, 2]

        wheel_rpm = torch.abs(wheel_vels) * (60.0 / (2.0 * torch.pi))
        wheel_rpm = torch.clamp(wheel_rpm, 0.0, 300.0)

        wheel_power = _torch_polyval(rpm_coeffs, wheel_rpm)         # [num_envs, 2]
        if _RPM_POWER_IS_EXP:
            wheel_power = torch.exp(wheel_power)
        wheel_power = torch.clamp(wheel_power, min=0.0)
        total_wheel_power = wheel_power.sum(dim=1)                   # [num_envs]

        # ===== Total energy per step =====
        total_power = total_prop_power + total_wheel_power           # [num_envs] Watts
        energy_per_step = total_power * dt                           # [num_envs] Joules

        # exp(-E/20) - 1: 0J->0, 20J->-0.63, 60J->-0.95
        return torch.exp(-energy_per_step / 20.0) - 1.0

    except (ValueError, IndexError, KeyError):
        return torch.zeros(robot.num_instances, device=robot.device)

def penalize_stalling_near_target(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is None or not hasattr(cmd_term, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)
    target_xy = cmd_term.current_targets_w[:, :2]
    robot_xy = (robot.data.root_pos_w[:, :2])
    dist = torch.norm(robot_xy - target_xy, dim=1)
    near_target = (dist < 0.5).float() # was 1.5
    robot_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    stalling = torch.exp(-robot_speed / 0.3)  # ~1 when nearly stationary
    return -near_target * stalling

def reward_stopping_at_target(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is None or not hasattr(cmd_term, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)
    target_xy = cmd_term.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    dist = torch.norm(robot_xy - target_xy, dim=1)
    # Only very close to target (0.5m)
    very_near = torch.exp(-dist / 0.3)
    # Reward LOW speed when near target
    robot_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    stopped = torch.exp(-robot_speed / 0.2)  # 1 when stopped, 0 when moving
    return very_near * stopped

def penalize_prolonged_no_progress(env, window: int = 100) -> torch.Tensor:
    """Penalize going a long stretch (window steps) without meaningful XY or
    height progress — directly targets the 'stuck, wandering, times out'
    failure mode that plain proximity-based stalling doesn't catch."""
    robot = env.scene["robot"]
    cmd = env.command_manager._terms.get("base_velocity")
    if cmd is None or not hasattr(cmd, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)

    target_xy = cmd.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    dist = torch.norm(robot_xy - target_xy, dim=1)
    height = robot.data.root_pos_w[:, 2]

    if not hasattr(env, "_stall_dist_ref") or not hasattr(env, "_stall_height_ref") or not hasattr(env, "_stall_counter"):
        env._stall_dist_ref = dist.clone()
        env._stall_height_ref = height.clone()
        env._stall_counter = torch.zeros(robot.num_instances, device=robot.device)

    made_progress = (env._stall_dist_ref - dist > 0.05) | (height - env._stall_height_ref > 0.02)

    env._stall_counter = torch.where(
        made_progress, torch.zeros_like(env._stall_counter), env._stall_counter + 1
    )
    # reset reference points whenever progress happens
    env._stall_dist_ref = torch.where(made_progress, dist, env._stall_dist_ref)
    env._stall_height_ref = torch.where(made_progress, height, env._stall_height_ref)

    prolonged_stall = (env._stall_counter > window).float()
    return -prolonged_stall

def reward_forward_progress(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is None or not hasattr(cmd_term, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)
    
    target_xy = cmd_term.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    dist_now = torch.norm(robot_xy - target_xy, dim=1)
    
    if not hasattr(env, "_prev_dist_to_target"):
        env._prev_dist_to_target = dist_now.clone()
    
    progress = env._prev_dist_to_target - dist_now
    env._prev_dist_to_target = dist_now.clone()
    
    # Reward getting closer, penalize moving away (asymmetric)
    return torch.clamp(progress, min=0.0)  # was min=0.5

def penalize_time(env) -> torch.Tensor:
    # Small penalty every step — makes parking expensive over time
    return -torch.ones(env.num_envs, device=env.device) * 0.01

def reward_prop_assisted_climb(env) -> torch.Tensor:
    """Reward prop-assist ONLY when climbing a real step ahead. At T/W~1.2 props
    assist (unload + tip), not lift. Reward props engaged + moderate pitch + rising,
    gated to when terrain ahead is higher than the robot (a step), so it doesn't
    learn to pitch on flat ground."""
    robot = env.scene["robot"]

    # prop engagement
    try:
        lpj = robot.joint_names.index("leftPropeller")
        rpj = robot.joint_names.index("rightPropeller")
        prop_active = torch.tanh(robot.data.joint_vel[:, [lpj, rpj]].abs().mean(dim=1) / 200.0)
    except ValueError:
        prop_active = torch.ones(robot.num_instances, device=robot.device)

    # moderate pitch, gaussian peak ~0.25 rad (no reward for face-plant pitch)
    # pitch = robot.data.projected_gravity_b[:, 1].abs()
    # pitching = torch.exp(-((pitch - 0.25) ** 2) / (2 * 0.10 ** 2))
    # forward pitch only (negative Y = nose-down toward step, from your spawn data)
    pitch_signed = -robot.data.projected_gravity_b[:, 1]   # positive when pitched FORWARD
    pitch_fwd = torch.clamp(pitch_signed, min=0.0)          # 0 if leaning back
    pitching = torch.exp(-((pitch_fwd - 0.25) ** 2) / (2 * 0.10 ** 2)) * (pitch_fwd > 0.05).float()

    # ---- TEMP AXIS CHECK ----
    if not hasattr(env, "_pitch_dbg"):
        env._pitch_dbg_count = 0
        env._pitch_dbg = True
    env._pitch_dbg_count = getattr(env, "_pitch_dbg_count", 0) + 1
    if env._pitch_dbg_count % 20 == 0:  # print every 20 steps to avoid spam
        pg = robot.data.projected_gravity_b[0]  # env 0, all 3 components
        print(f"[PITCH AXIS] proj_grav=[{pg[0]:.3f}, {pg[1]:.3f}, {pg[2]:.3f}]", flush=True)
    # ---- END TEMP ----

    # rising
    vz = robot.data.root_lin_vel_w[:, 2]
    # rising = torch.clamp(vz / 0.2, 0.0, 1.0)
    rising = torch.clamp((vz + 0.1) / 0.3, 0.0, 1.0)  # small credit even at vz~0, full at 0.2

    # step-ahead gate via 6x6 height grid
    step_ahead = torch.ones(robot.num_instances, device=robot.device)
    try:
        hs = env.scene["height_scanner"]
        ray_z = torch.nan_to_num(hs.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)  # [envs,36]
        ground_z = ray_z.median(dim=1)[0]              # flat-majority reference
        max_ahead = ray_z.max(dim=1)[0]                # highest scanned point
        step_ahead = torch.clamp((max_ahead - ground_z) / 0.04, 0.0, 1.0)  # 0-1, sat at 4cm
    except (KeyError, ValueError, IndexError):
        pass  # no scanner -> gate stays 1 (fallback)

    return prop_active * pitching * rising * step_ahead

# def penalize_excessive_pitch(env, max_pitch: float = 0.4) -> torch.Tensor:
#     """Penalize body pitching too far forward (face-plant), EVERYWHERE — flat or step."""
#     robot = env.scene["robot"]
#     pitch = robot.data.projected_gravity_b[:, 1].abs()
#     return -torch.clamp(pitch - max_pitch, min=0.0)

def penalize_not_upright(env, upright_tol: float = 0.002) -> torch.Tensor:
    """Penalize body tilt from upright in ANY direction (forward/back/roll).

    upright_tol is a DEADZONE in units of (1 - cos(theta)): tilts below it are
    free. Reduced 0.08 -> 0.002 on 2026-08-25, i.e. from 23.1 deg to 3.6 deg.

    Why. 0.08 was chosen to leave room for climbing pitch, which was harmless
    while the body could not rotate at all (see the USD inertia bug). Once the
    robot could actually fall, it became the reason it could not learn not to:

        passes  6.6 deg (limit of wheel authority) at t=0.27s -> penalty 0.0000
        passes 23.1 deg (deadzone edge)            at t=0.45s -> penalty 0.0008
        terminates at 70 deg                       at t=0.60s
        measured mean episode length               0.56 s

    The whole recoverable window sat inside the deadzone, so the first signal
    that anything was wrong arrived with 0.11 s left, in a state no action could
    fix. Measured consequence: 195 iterations and 5.4M timesteps with mean
    episode length flat at 28.0 -> 28.3 steps, every reward term unchanged to
    four decimals, and terrain_levels collapsing 0.359 -> 0.000. No gradient.

    At 0.002 the gradient starts at 3.6 deg (t~0.13 s), inside the window where
    the wheels still have authority, so early corrective action is rewarded.
    Climbing pitch is still cheap: at 40 deg the penalty is ~1.03 against
    reward_progress_to_target's 10.0."""
    robot = env.scene["robot"]
    uprightness = -robot.data.projected_gravity_b[:, 2]  # 1=upright, <1 tilted
    tilt = torch.clamp(1.0 - uprightness, min=0.0)
    # deadzone: no penalty until tilt exceeds upright_tol (3.6 deg at 0.002 -- see docstring)
    return -torch.clamp(tilt - upright_tol, min=0.0)

def reward_alive_upright(env, tol: float = 0.5) -> torch.Tensor:
    """Constant per-step bonus while the robot is still standing.

    Added 2026-08-26 for the wheels-only balance task. The hybrid reward set has
    no survival term at all -- staying alive pays only through forfeited future
    reward, which works ONLY while net per-step return is positive.

    In the pendulum run it was not. Measured at iteration 853:

        positives +0.1090   negatives -0.1838   NET -0.0748 per step

    and tilt carries no terminal penalty, so ending the episode was the highest
    return action available. The policy found it: episode length fell 66.57 ->
    53.43 while mean reward ROSE -2.65 -> -1.20, with alpha collapsed to 0.009.
    Shorter episodes cannot raise return unless per-step return is negative.

    tol=0.5 is uprightness (-projected_gravity_b[:, 2]) = cos(60 deg), inside the
    70 deg termination, so the bonus stops before the episode does and falling is
    never worth more than standing.
    """
    robot = env.scene["robot"]
    uprightness = -robot.data.projected_gravity_b[:, 2]
    return (uprightness > tol).float()


def reward_climb_transition(env) -> torch.Tensor:
    """Dense reward concentrated at the pitch-onto-step moment:
    near target + gaining height + props active + upright.
    This is the 0.3s window where prop-assist must engage."""
    robot = env.scene["robot"]
    cmd_term = env.command_manager._terms.get("base_velocity")
    if cmd_term is None or not hasattr(cmd_term, "current_targets_w"):
        return torch.zeros(robot.num_instances, device=robot.device)

    spawn_z = env.scene.env_origins[:, 2]
    height_above = robot.data.root_pos_w[:, 2] - spawn_z

    # in the climb band: off the ground but not yet at full target height
    in_climb_band = ((height_above > 0.02) & (height_above < 0.15)).float()

    # near the target XY
    target_xy = cmd_term.current_targets_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    dist = torch.norm(robot_xy - target_xy, dim=1)
    near = torch.exp(-dist / 1.0)

    # props active
    try:
        lp = robot.joint_names.index("leftPropeller")
        rp = robot.joint_names.index("rightPropeller")
        prop = torch.tanh(robot.data.joint_vel[:, [lp, rp]].abs().mean(dim=1) / 200.0)
    except (ValueError, IndexError):
        prop = torch.zeros_like(dist)

    # upright (not toppling)
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=0.0)

    # rising
    rising = torch.clamp(robot.data.root_lin_vel_w[:, 2], min=0.0)

    return in_climb_band * near * prop * upright * rising

def penalize_thrust_forward_at_step(env) -> torch.Tensor:
    """Penalize thrust pointing forward/horizontal ONLY when at a step (climbing context).
    On flat ground, props can point however they help navigation — not penalized.
    At a step, forward thrust (shoving into the riser) is penalized, forcing props up to climb."""
    robot = env.scene["robot"]
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac = thrust_world[:, :, 2].mean(dim=1)

    lpj = robot.joint_names.index("leftPropeller")
    rpj = robot.joint_names.index("rightPropeller")
    prop_active = torch.tanh(robot.data.joint_vel[:, [lpj, rpj]].abs().mean(dim=1) / 200.0)

    # step-ahead gate: only force props-up when there's a step to climb
    step_ahead = torch.zeros(robot.num_instances, device=robot.device)
    try:
        hs = env.scene["height_scanner"]
        ray_z = torch.nan_to_num(hs.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        step_ahead = torch.clamp((ray_z.max(dim=1)[0] - ray_z.median(dim=1)[0]) / 0.04, 0.0, 1.0)
    except (KeyError, ValueError, IndexError):
        pass

    forward_shortfall = torch.clamp(1.0 - z_frac, 0.0, 2.0)
    # penalize forward thrust ONLY when props active AND at a step
    return -(forward_shortfall * prop_active * step_ahead)

def penalize_action_rate(env) -> torch.Tensor:
    """Penalize CHANGE in action between steps (not magnitude). Forces smooth, gradual
    control instead of bang-bang max-thrust jerks that topple the robot when props fire."""
    if not hasattr(env, "_prev_action") or env._prev_action.shape[0] != env.action_manager.action.shape[0]:
        env._prev_action = env.action_manager.action.clone()
        return torch.zeros(env.action_manager.action.shape[0], device=env.action_manager.action.device)
    rate = torch.sum(torch.square(env.action_manager.action - env._prev_action), dim=1)
    env._prev_action = env.action_manager.action.clone()
    return -rate

def reward_props_upright(env) -> torch.Tensor:
    """Small reward for props pointing upward at any time — not just at steps.
    Encourages the policy to keep props in an upright position generally."""
    robot = env.scene["robot"]
    try:
        lp = robot.body_names.index("leftPropeller")
        rp = robot.body_names.index("rightPropeller")
        prop_ids = torch.tensor([lp, rp], device=robot.device)
        prop_quat = robot.data.body_quat_w[:, prop_ids, :]
        thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
        thrust_local[:, :, 2] = 1.0
        from isaaclab.utils.math import quat_apply
        thrust_world = quat_apply(prop_quat, thrust_local)
        z_frac = torch.clamp(thrust_world[:, :, 2].mean(dim=1), 0.0, 1.0)

        # 2026-08-26: GATE ON ACTUAL PROP SPEED.
        #
        # This term carries the largest weight in the whole reward set (5.0) and
        # until now read propeller ORIENTATION only -- it never touched joint_vel.
        # So the policy could collect the single biggest propeller reward in the
        # environment by pointing the props up and never spinning them, which is
        # exactly what it did: props_upright was the largest positive term in the
        # log (0.0430) while vertical_thrust_support sat at 0.0154 and every
        # prop-related term drifted DOWN as alpha fell (0.85 -> 0.49).
        #
        # Reference speed 120 rad/s = 9.5 N total = 1.33x the 7.17 N static
        # stability threshold (props at 0.4476 m vs CoM at 0.1016 m), and well
        # inside the 158 rad/s the actuator can hold at damping 0.015 -- so the
        # gate saturates somewhere the robot can actually reach and hold.
        #
        # 0.25 floor: keeps a direction-only gradient alive when the props are
        # stopped, otherwise a policy with props off gets no signal about WHICH
        # WAY to point them and can never discover the pose in the first place.
        lpj = robot.joint_names.index("leftPropeller")
        rpj = robot.joint_names.index("rightPropeller")
        spin = torch.clamp(
            robot.data.joint_vel[:, [lpj, rpj]].abs().mean(dim=1) / 120.0, 0.0, 1.0
        )
        return (z_frac ** 2) * (0.25 + 0.75 * spin)
    except (ValueError, IndexError):
        return torch.zeros(robot.num_instances, device=robot.device)

def reward_vertical_thrust_support(env, target_frac: float = 0.7) -> torch.Tensor:
    """Reward the VERTICAL component of total thrust, as a fraction of weight.

    Why this exists (added 2026-08-21 after the hardware divergence):

    The propellers sit ~443 mm above the wheel axle and the CoM ~139 mm above
    it, so this machine is a SHORT inverted pendulum -- natural fall time
    constant tau = sqrt(L/g) ~= 119 ms. The real actuator chain (MAVLink -> ESC
    -> propeller spin-up) carries 40-100 ms of delay. A plant with tau = 119 ms
    and 70 ms of delay is only marginally stabilisable, which is exactly what
    hardware showed: the policy recovered 69 deg of lean, overshot through
    vertical, and diverged.

    Upward thrust offsetting fraction f of weight reduces effective gravity to
    (1-f)g, stretching tau by 1/sqrt(1-f):

        f = 0.0 -> 119 ms      f = 0.7 -> 217 ms      f = 0.84 -> 297 ms (T/W max)

    So sustained vertical thrust is not a luxury on this airframe -- it is what
    makes the plant controllable at the delays it actually has. reward_props_upright
    only rewards the props POINTING up; this rewards them pointing up AND
    actually pushing, which is the part that buys phase margin.

    Rewards approach to target_frac and does NOT reward beyond it: overshooting
    toward T/W 1.0 unloads the wheels (normal force -> 0) and costs traction,
    which the robot still needs for translation. 0.7 leaves ~30% of weight on
    the wheels while more than doubling the time constant.

    NOTE this fights `energy_consumption` (weight 0.25) by design. Stability
    first; the energy story is only meaningful for a robot that stays upright.
    """
    robot = env.scene["robot"]
    try:
        lp = robot.body_names.index("leftPropeller")
        rp = robot.body_names.index("rightPropeller")
        prop_ids = torch.tensor([lp, rp], device=robot.device)

        # thrust direction: each propeller's local +z in world frame
        prop_quat = robot.data.body_quat_w[:, prop_ids, :]
        thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
        thrust_local[:, :, 2] = 1.0
        from isaaclab.utils.math import quat_apply
        thrust_dir_w = quat_apply(prop_quat, thrust_local)

        # thrust magnitude proxy: |propeller joint velocity| / max, per propeller.
        # The aerodynamic model is monotonic in |omega| so this is a faithful
        # ordering even though it is not newtons.
        joint_ids = [robot.joint_names.index(n) for n in ("leftPropeller", "rightPropeller")]
        omega = robot.data.joint_vel[:, joint_ids].abs()
        omega_max = 375.0                                    # matches propeller_vel action span
        mag = torch.clamp(omega / omega_max, 0.0, 1.0)

        # vertical support fraction: sum over both props of mag * (thrust z-component)
        vert = (mag * torch.clamp(thrust_dir_w[:, :, 2], 0.0, 1.0)).sum(dim=1) * 0.5

        # saturating reward: climbs to target_frac, flat beyond it
        return torch.clamp(vert / target_frac, 0.0, 1.0)
    except (ValueError, IndexError):
        return torch.zeros(robot.num_instances, device=robot.device)


def reward_thrust_up_when_pitched(env) -> torch.Tensor:
    """Reward props pointing UP scaled by how much the robot is pitched forward.
    The more it pitches (about to fall forward), the more it's rewarded for having
    props up to catch it. Teaches props to react to the forward pitch."""
    robot = env.scene["robot"]
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac = torch.clamp(thrust_world[:, :, 2].mean(dim=1), 0.0, 1.0)

    # forward pitch magnitude (Y axis for +Y robot)
    pitch_fwd = torch.clamp(robot.data.projected_gravity_b[:, 1].abs(), 0.0, 1.0)

    # reward props-up SCALED by forward pitch: pitched forward + props up = high reward
    return z_frac * pitch_fwd

def penalize_thrust_pointing_down(env) -> torch.Tensor:
    """Penalize thrust pointing DOWN (negative world-Z). The servo over-tilts past
    vertical, swinging thrust past 'up' into pointing down/back (z_frac seen at -0.41),
    which shoves the robot into/down the step. This penalizes that overshoot so the
    policy keeps thrust at or above horizontal, not past vertical into down-pointing."""
    robot = env.scene["robot"]
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac_raw = thrust_world[:, :, 2].mean(dim=1)  # NOT clamped — can be negative

    # penalty grows as thrust points down (z_frac negative)
    penalty_down = torch.clamp(-z_frac_raw, min=0.0)  # 0 if up/horizontal, >0 if pointing down

    # NO prop_active MULTIPLIER. This is the fix for the give-up behaviour, and
    # it is the single most important line in this file.
    #
    # Until 2026-08-23 this returned -(penalty_down * prop_active), where
    # prop_active was tanh(|prop_speed|/200). That made the penalty proportional
    # to how hard the propellers were spinning, so the cheapest way to escape it
    # was to STOP THE PROPELLERS. At weight 3.0 that is exactly what the policy
    # learned, and it is fatal on hardware.
    #
    # Measured, transfer_clamped_props.csv 2026-08-23:
    #     t=0.4s  lean 30 deg  thrust 18.8 N  vertical support 62%  a3 +0.93
    #     t=0.7s  lean 50 deg  thrust  4.6 N  vertical support 16%  a3 -1.00
    #     t=0.8s  lean 71 deg  ... a3 = a4 = -1.00, held for the next 3.8 s
    # It threw away 62% vertical support -- the reward_vertical_thrust_support
    # target, achieved -- at the exact moment it needed it, because past ~50 deg
    # of lean the thrust axis passes horizontal, z_frac goes negative, and
    # shutting down zeroed a 3.0-weighted penalty. The robot then lay on its
    # frame with the propellers off and could not get up.
    #
    # Direction-only, the incentive is right: shutting down no longer escapes
    # the penalty, and the only way to reduce it is to RE-AIM the props with the
    # servo. That is a real, reachable action now that SERVO_POS_LIMIT_RAD is
    # pi/4 (45 deg) rather than pi/6 -- see the note there; at 30 deg the
    # re-aiming this asks for would have been geometrically impossible and the
    # policy would have shut down again for lack of an alternative.
    return -penalty_down

def reward_thrust_recovery_under_lean(env, lean_onset: float = 0.05) -> torch.Tensor:
    """Pay for VERTICAL thrust in proportion to how far the robot is leaning.

    Added 2026-08-23. This is the counterpart to removing prop_active from
    penalize_thrust_pointing_down: that stopped paying the policy to give up,
    and this pays it to fight.

    Why the existing terms leave a gap:

    * reward_vertical_thrust_support is attitude-blind. It rewards the same
      vertical thrust at 5 deg of lean as at 50, so it says nothing about the
      case that actually matters.
    * reward_prop_catch_when_falling gates on root_lin_vel_w[2] < 0, i.e. the
      CoM descending. A robot tipping about its wheel axle is ROTATING, not
      descending -- vz stays near zero for the first ~0.3 s, which is the whole
      recovery window. Measured on hardware the term would have been ~0
      throughout the tip. At weight 0.3 it was never going to matter anyway.

    So nothing in the reward set asked the policy to push harder as it went
    over. This does: the weight ramps with lean, so the incentive is strongest
    exactly where the last four hardware runs died.

    Args:
        lean_onset: uprightness deficit below which this pays nothing, so
            ordinary climbing pitch is not rewarded as if it were a recovery.
            0.15 corresponds to roughly 32 deg of lean.
    """
    robot = env.scene["robot"]
    try:
        lp = robot.body_names.index("leftPropeller")
        rp = robot.body_names.index("rightPropeller")
        prop_ids = torch.tensor([lp, rp], device=robot.device)

        # HOW FAR OFF UPRIGHT -- measured as sin(theta), not 1-cos(theta).
        #
        # 1-cos is quadratic near zero, so with the old onset of 0.15 this term
        # was EXACTLY ZERO across 5-20 deg, which is the entire band the robot
        # operates in. It only woke up past 32 deg, by which point the wheels
        # (good to ~23 deg) are already beyond recovering and thrust cannot
        # catch it either. The reward meant to teach "use the props to counter
        # lean" therefore never fired during normal operation, and the policy
        # learned wheels-only locomotion -- confirmed in sim and on hardware,
        # where propeller actions average about -0.4, a third throttle.
        #
        # sin(theta) is the horizontal component of projected gravity, and it is
        # also exactly the term in the gravity torque m*g*L*sin(theta) that the
        # props have to fight. So urgency now tracks the actual disturbance:
        #     5 deg -> 0.08    15 deg -> 0.46    23 deg -> 0.76    30+ -> 1.0
        # Saturating at 30 deg keeps the gradient concentrated in the band where
        # thrust can still save it rather than rewarding heroics at 60 deg.
        lean_sin = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
        # POSITION-ONLY, as in model_3500_params/env.yaml. A rate term
        # (urgency = max(position, rate)) was added on 2026-08-26 and is a good
        # idea, but it was NOT in the run that works. Re-add it as a deliberate
        # experiment against this baseline, not as part of it.
        urgency = torch.clamp((lean_sin - lean_onset) / (0.5 - lean_onset), 0.0, 1.0)

        # vertical component of thrust actually being produced
        prop_quat = robot.data.body_quat_w[:, prop_ids, :]
        thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
        thrust_local[:, :, 2] = 1.0
        thrust_dir_w = quat_apply(prop_quat, thrust_local)

        joint_ids = [robot.joint_names.index(n) for n in ("leftPropeller", "rightPropeller")]
        omega = robot.data.joint_vel[:, joint_ids].abs()
        mag = torch.clamp(omega / 375.0, 0.0, 1.0)   # matches propeller_vel action span
        vert = (mag * torch.clamp(thrust_dir_w[:, :, 2], 0.0, 1.0)).sum(dim=1) * 0.5

        return urgency * vert
    except (ValueError, IndexError):
        return torch.zeros(robot.num_instances, device=robot.device)


def reward_prop_thrust_when_climbing(env) -> torch.Tensor:
    """Reward actually PRODUCING upward thrust during a climb — not just pointing props up.
    The props point up but idle; this rewards spinning them to produce real upward force
    at a step, so the assist actually happens."""
    robot = env.scene["robot"]
    lpj = robot.joint_names.index("leftPropeller")
    rpj = robot.joint_names.index("rightPropeller")
    # prop speed (higher = more thrust)
    prop_speed = robot.data.joint_vel[:, [lpj, rpj]].abs().mean(dim=1)
    prop_speed_norm = torch.clamp(prop_speed / 200.0, 0.0, 1.0)  # normalized

    # thrust direction up
    lp = robot.body_names.index("leftPropeller")
    rp = robot.body_names.index("rightPropeller")
    prop_ids = torch.tensor([lp, rp], device=robot.device)
    prop_quat = robot.data.body_quat_w[:, prop_ids, :]
    thrust_local = torch.zeros(robot.num_instances, 2, 3, device=robot.device)
    thrust_local[:, :, 2] = 1.0
    thrust_world = quat_apply(prop_quat, thrust_local)
    z_frac = torch.clamp(thrust_world[:, :, 2].mean(dim=1), 0.0, 1.0)

    # at a step
    step_ahead = torch.ones(robot.num_instances, device=robot.device)
    try:
        hs = env.scene["height_scanner"]
        ray_z = torch.nan_to_num(hs.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        step_ahead = torch.clamp((ray_z.max(dim=1)[0] - ray_z.median(dim=1)[0]) / 0.04, 0.0, 1.0)
    except (KeyError, ValueError, IndexError):
        pass

    # reward: spinning props HARD, pointed up, at a step = real upward assist
    return prop_speed_norm * z_frac * step_ahead

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
        weight=0.25, # WASS 0.1
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
        weight=0.5,
    )
    """Penalty for mismatch between robot facing direction and target direction.
    Reads angle error directly from vel_command_b[:, 2] (normalized angle error in [-1, 1]).
    Penalizes when robot is not facing the target, scaled to [-1, 0] using e^(-x) - 1."""
    
    penalize_rotation = RewTerm(
        func=penalize_tilt_angle,
        weight=1.0,
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
    
    # alive_weight/terminal_weight MUST mirror reward_alive_upright's weight and
    # this term's weight. See terminal_reward_goal_reached for why: without the
    # compensation, finishing the task costs the policy more than it gains, and a
    # policy that balances well enough will farm the clock instead (measured,
    # RUN 2: success 0.34 -> 0.00 purely from getting BETTER at balancing).
    terminal_goal_reached = RewTerm(
        func=terminal_reward_goal_reached,
        params={"alive_weight": 2.0, "terminal_weight": 10.0},
        weight=10.0, # was 1.0
    )
    """Terminal reward for successfully reaching the goal.
    Returns +10.0 when robot reaches the goal (episode ends due to goal_reached constraint).
    This is a positive terminal reward that encourages task completion."""
    
    terminal_propeller_collision = RewTerm(
        func=terminal_reward_propeller_collision,
        weight=3.0, # WASS 2.0
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

    

    action_smoothness = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.action_manager.action), dim=1),
        weight=0.005,
    )

    # The jerkiness — if you want to address it, action_smoothness (magnitude penalty) is the wrong tool; you'd need an action-rate penalty (penalize action - last_action). But don't add that now — it's another term in an already-overloaded run.
    """Penalize large action magnitudes to encourage smooth, energy-efficient control."""
    
    penalize_stalling_near_target = RewTerm(
        func=penalize_stalling_near_target,
        weight=1.0,
    )
    
    penalize_time = RewTerm(
        func=penalize_time,
        weight=1.5,
    )

    penalize_thrust_pointing_down = RewTerm(
        func=penalize_thrust_pointing_down,
        weight=3.0, # was 2.0
    )
    
    penalize_thrust_forward_at_step =RewTerm(
        func=penalize_thrust_forward_at_step,
        weight=1.0, # was 0.5
    )

    penalize_prolonged_no_progress = RewTerm(
        func=penalize_prolonged_no_progress,
        weight=0.5,
    )

    penalize_not_upright = RewTerm(
        func=penalize_not_upright,
        weight=4.5, # WASS 2.5
    )

    # Raised 0.1 -> 0.3 on 2026-08-25.
    #
    # The wheels can slew at ~58 rad/s^2 (bench) and the policy was commanding
    # full-scale reversals: at wheel_scale 1.0 a single action step of 1.0 is
    # 47 rad/s, i.e. 2350 rad/s^2 demanded. The measured speed then chases a
    # command that has already flipped, which is what the 260 ms
    # command-to-response lag in hw_v8.csv actually is -- saturation, not delay.
    #
    # 0.1 was too weak to bite: logged at 0.0170 against
    # reward_progress_to_target's 0.1496, about 11%. At 0.3 it is ~34%, enough
    # to make reversals the wheels can genuinely execute cheaper than ones they
    # cannot. Lower it if the policy goes sluggish and stops correcting.
    # 2026-08-26: back to 0.1. The 0.3 raise above was made for HARDWARE
    # smoothness, and it was decided against a run whose optimizer was crippled
    # (batch_size 256 -> replay ratio 0.125), so it never got a fair read.
    #
    # With the optimizer verifiably learning (critic loss 0.11 -> 0.28, alpha
    # 0.85 -> 0.54 over 2,300 grad steps), the environment still did not move at
    # all, and this was the single largest penalty in the log:
    #
    #     penalize_action_rate   -0.0261     <- taxes corrective action
    #     penalize_not_upright   -0.0194     <- taxes falling over
    #
    # i.e. the policy paid more to move its actuators than to tip past 70 deg.
    # Balancing a 102 ms pendulum at 50 Hz REQUIRES large per-step corrections.
    # See the "lower it if the policy goes sluggish" note above -- that is the
    # observed symptom: 89% of episodes end in tilt at 0.62 s, flat for 96 iters.
    penalize_action_rate = RewTerm(
        func=penalize_action_rate,
        weight=0.1,
    )
    
    penalize_cross_track_error = RewTerm(
        func=penalize_cross_track_error,
        weight=0.5,
    )

    reward_climb_progress = RewTerm(
        func=reward_climb_progress,
        weight=2.0, # WASS 0.5
    )

    reward_forward_progress = RewTerm(
        func=reward_forward_progress,
        weight=2.0,
    )

    reward_progress_to_target = RewTerm(
        func=reward_progress_to_target,
        weight=10.0,
    )

    reward_stable_after_climb = RewTerm(
        func=reward_stable_after_climb,
        weight=2.0,
    )

    reward_thrust_up_at_step = RewTerm(
        func=reward_thrust_up_at_step,
        weight=5.0, # WASS 3.0
    )

    # 5.0 -> 2.0 on 2026-08-26. THE PROPELLER TERMS HAD TAKEN OVER THE REWARD.
    #
    # Measured at iteration 358, summed over the whole set:
    #     propeller rewards   +0.2643  (6 terms)
    #     task rewards        +0.1405  (7 terms)
    # The policy was being paid 88% MORE for propeller posture than for doing
    # the task, and props_upright at 0.1239 was the single largest term in the
    # entire reward -- larger than progress_to_target and alive_upright combined.
    #
    # It was raised to 5.0 when the props genuinely were not pointing up
    # (z_frac 0.41). They are now: measured z_frac mean 0.85-0.86, max 1.000,
    # after the left-propeller frame fix. The reason for 5.0 no longer holds.
    # RESTORED 5.0 (model_3500_params/env.yaml).

    # 2026-08-27: HALVED. THESE THREE ARE POSITION-INDEPENDENT INCOME.
    #
    # Observed in play on model_3500: the robot "climbs really nice but in
    # random places and circles around", i.e. the locomotion is solved and the
    # NAVIGATION is not. Measured reward split at iteration 3622:
    #     prop posture (6 terms)   0.2708
    #     task         (7 terms)   0.1393     ratio 1.94:1
    #
    # props_upright, vertical_thrust_support and thrust_recovery_under_lean can
    # ALL be earned standing still, circling, or anywhere on the map -- they are
    # functions of attitude and thrust only. reward_progress_to_target requires
    # being in a PARTICULAR PLACE. When the position-independent income is 2x
    # the position-dependent income, circling while holding a good prop pose is
    # strictly better than driving to the goal. That is the observed behaviour.
    #
    # Halving these three brings prop posture to ~0.135 against task 0.139, so
    # going somewhere finally outbids posing. NOT cut further: the pose still has
    # to be worth holding, and thrust is what makes the machine stable at all.
    #
    # THIS DOES NOT ENDANGER THE HARDWARE "PROPS PINNED UP" REQUIREMENT.
    # At deployment, --servo_attitude_hold OVERWRITES the policy's servo action
    # outright (db_inference.py: action[layout["servo"][0]] = act_units), so
    # props-up on the real robot is guaranteed by the PID hold, not by this
    # reward. These weights govern sim behaviour only.

    # 2026-08-28: RESTORED TO FULL (5.0/3.0/6.0) FOR THE "full props + slow servo"
    # RUN. This is the combination neither RUN 1 nor RUN 2 tested.
    #
    # The two runs each got one variable right and one wrong:
    #                props   servo vel_limit   terrain_levels  success  play
    #     RUN 1      full        10.0              1.51         0.356   servo THRASHING
    #     RUN 2      halved       2.0              0.04         0.077   clean, but stuck
    #
    # Full prop rewards are what drive terrain progress -- RUN 1 promoted to 1.51
    # with them, RUN 2 sat at 0.04 without. reward_props_upright buys WORLD-VERTICAL
    # thrust, which is the restoring moment that makes this machine stable at all
    # (T*L_prop*sin(th) against gravity's W*L_com*sin(th), L_prop/L_com = 4.4).
    # Halving it on 2026-08-27 was my call, aimed at the circling, and it did NOT
    # fix navigation: RUN 2 has the halved weights and WORSE terrain progress.
    #
    # servo velocity_limit stays at 2.0 (see doublebee_v1.py). RUN 1's 10.0 hands
    # an untrained servo head 5x the authority and its play shows exactly that:
    # thrust z_frac 0.827 with min 0.175, servo swinging -0.767..+0.791 rad,
    # wheel tracking 0.14, lean diverging 6.8 -> 15.6 deg. RUN 2 at 2.0 reads
    # z_frac 0.967 (min 0.951), servo -0.107..+0.253, tracking 0.45.
    #
    # MUST BE A FRESH RUN, not a resume: the point is to train the servo head
    # against 2.0 from the start rather than inherit one shaped by 10.0.
    #
    # The circling/farming is attacked with POSITION-DEPENDENT terms instead of by
    # weakening posture: terminal_goal_reached compensation (measured 1.0132 in
    # RUN 1, second-largest term in the set) and the reward_climb_progress bounce
    # clamp. Measured farming split in RUN 2 at iteration 4787:
    #     position-independent 0.8626 : goal-directed 0.3226  = 2.7 : 1
    # climb_progress was only 0.0136 of that, so the clamp alone was never going
    # to be enough -- it is kept because it is correct, not because it is decisive.
    reward_props_upright = RewTerm(
        func=reward_props_upright,
        weight=5.0,
    )

    # Added 2026-08-21. Rewards props pointing up *and pushing*, which is what
    # actually offsets gravity and stretches the pendulum time constant from
    # 119 ms to ~217 ms -- the margin the real 40-100 ms actuator delay needs.
    # See reward_vertical_thrust_support for the derivation.
    # target_frac 0.7 -> 0.35 on 2026-08-26. THIS IS WHAT RESTORES ENERGY
    # MODULATION, which is the paper's central claim.
    #
    # The term saturates at omega = target_frac * 375. At 0.70 that is 262 rad/s
    # = 17.6 N = 2.46x the 7.17 N static-stability threshold, so the reward kept
    # paying long past the point where extra thrust buys any stability. Measured
    # against the energy penalty over the same interval:
    #
    #     marginal thrust reward, action 0 -> 1   +1.47
    #     marginal energy penalty, same move      -0.0335
    #
    # 44:1. The optimal propeller action was therefore PINNED AT +1, and the play
    # log confirms it: raw_action = 0.99 on every tick. A policy that never
    # modulates thrust cannot demonstrate energy-aware actuation -- the IROS
    # submission's headline was that it modulated over 1145-1435 us.
    #
    # 0.35 saturates at 131 rad/s = 10.1 N = 1.41x threshold. Comfortable margin
    # for stability, and every rad/s beyond it earns NOTHING while still costing
    # energy, so spinning down becomes strictly better whenever stability allows.
    #
    # Sizing the thrust reward to the STABILITY THRESHOLD rather than to an
    # arbitrary fraction is the principled version of this: thrust is worth
    # paying for exactly up to the point it changes whether the robot can stand.
    # weight 3.0 -> 1.5, COMPENSATING THE target_frac CHANGE.
    #
    # The term is clamp(vert / target_frac), so halving target_frac (0.7 -> 0.35)
    # does two things, and only one was intended: it moves WHERE the reward
    # saturates (intended -- see above), and it DOUBLES the reward everywhere
    # below saturation (not intended). Measured, this term went 0.0155 -> 0.0820
    # across the recent changes, becoming the second-largest in the set.
    #
    # Halving the weight keeps the earlier saturation -- which is what produces
    # thrust modulation -- while restoring the term's original magnitude, so it
    # no longer crowds out progress_to_target and alive_upright.
    # RESTORED weight 3.0 (model_3500_params/env.yaml). target_frac 0.35 was
    # ALREADY 0.35 in that run -- it is not part of the revert.
    reward_vertical_thrust_support = RewTerm(
        func=reward_vertical_thrust_support,
        weight=3.0,  # full; see the note above reward_props_upright
        params={"target_frac": 0.35},
    )

    # Added 2026-08-23. Pays for vertical thrust IN PROPORTION TO LEAN, which
    # nothing else in this set does -- see reward_thrust_recovery_under_lean for
    # why reward_vertical_thrust_support (attitude-blind) and
    # reward_prop_catch_when_falling (gated on vz, which stays ~0 during a tip)
    # both leave the recovery window unrewarded.
    #
    # Weight 6.0 is deliberately larger than penalize_thrust_pointing_down (3.0).
    # Past ~50 deg of lean the thrust axis passes horizontal and that penalty
    # goes positive, so the two terms are in direct competition exactly where
    # recovery has to happen. 6.0 vs 3.0 means fighting always beats quitting.
    # If the policy starts hanging at high lean to farm this, lower it -- but the
    # failure mode it is fixing (props off, robot down, 3.8 s and counting) is
    # far worse than the one it risks.
    reward_thrust_recovery_under_lean = RewTerm(
        func=reward_thrust_recovery_under_lean,
        weight=6.0,  # full; see the note above reward_props_upright
        params={"lean_onset": 0.05},
    )

    # reward_climb_transition = RewTerm(
    #     func=reward_climb_transition,
    #     weight=1.5,
    # )

    # reward_thrust_up_when_pitched = RewTerm(
    #     func=reward_thrust_up_when_pitched,
    #     weight=1.5,
    # )    

    reward_prop_thrust_when_climbing = RewTerm(
        func=reward_prop_thrust_when_climbing,
        weight=2.5,
    )


    reward_prop_catch_when_falling = RewTerm(
        func=reward_prop_catch_when_falling,
        weight=0.3,
    )

    # 2026-08-26: SURVIVAL BONUS ON THE MAIN TASK TOO.
    #
    # There was no term anywhere that paid simply for still being upright.
    # Staying alive paid only through forfeited future reward, which is an
    # indirect and weak gradient -- and it competes with reward_progress_to_target
    # (weight 10.0, the largest in the set), which is earned by DRIVING FORWARD,
    # the one thing guaranteed to topple an inverted pendulum. A random policy
    # can collect progress in 0.6 s; it cannot collect balance in 0.6 s. So
    # gradient descent takes the reachable one.
    #
    # Weight 2.0 against progress_to_target's 10.0: at ~0.9 of steps upright
    # this pays ~1.8/step, which is comparable to what a short forward dash
    # earns, so surviving is no longer strictly dominated. Deliberately NOT
    # larger -- overpaying survival produces a robot that stands still, and
    # penalize_prolonged_no_progress is only weight 0.5.
    # 2.0 -> 0.25 on 2026-08-27. THIS TERM CREATED A SURVIVAL-FARMING OPTIMUM.
    #
    # It pays EVERY STEP while upright; terminal_goal_reached pays ONCE (weight
    # 10.0) and ENDS the episode, truncating that income. So over a 1000-step
    # episode, surviving is worth 2000 against 10 for finishing -- reaching the
    # goal costs the policy 200x what it gains.
    #
    # Measured across two runs that differ mainly in how well they balance:
    #     RUN 1  ep_len 175  success 0.34  terrain 0.58  alive 0.35
    #     RUN 2  ep_len 369  success 0.00  terrain 0.00  alive 0.95
    # RUN 1 never balanced well enough to cross the threshold, so it still
    # finished. RUN 2 got BETTER at balancing, crossed it, and locked into
    # running out the clock -- 33% timeout, 25% out of bounds, and
    # velocity_direction_alignment went POSITIVE to NEGATIVE, i.e. it started
    # driving away from the goal on purpose.
    #
    # 0.25 keeps the term's original purpose (make survival strictly profitable
    # so falling is never the best move) without letting it outbid the task.
    # RESTORED 2.0 (model_3500_params/env.yaml). NOTE: this is the weight that
    # produced RUN 2's survival-farming failure -- but RUN 1 used it and still
    # reached 34% success, so it is NOT the thing to change while reproducing.
    reward_alive_upright = RewTerm(
        func=reward_alive_upright,
        weight=2.0,
        params={"tol": 0.5},
    )

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

    # 2026-08-26: NAVIGATION PENALTIES OFF FOR THE BALANCE-ONLY TASK.
    #
    # This config inherited the full hybrid navigation reward set, so a task
    # meant to test whether the robot can STAND was also being charged for
    # tracking a line and holding a heading -- and charged more than it could
    # earn. Measured at iteration 853, the four biggest costs were all
    # navigation and none were balance:
    #
    #     cross_track_error       -0.0672
    #     rotation                -0.0333
    #     facing_mismatch         -0.0326
    #     prolonged_no_progress   -0.0229
    #
    # Net per-step return was -0.0748, which makes falling over the optimal
    # policy. Removing these three brings the balance of terms back positive.
    penalize_cross_track_error = None
    penalize_facing_mismatch = None
    penalize_prolonged_no_progress = None

    # Explicit survival bonus. Without it a robot that balances in place but
    # does not travel still nets negative, because most of the remaining
    # positive reward (progress_to_target) requires moving toward the goal.
    # ~0.9 of steps upright x weight 0.15 = ~+0.135, against ~-0.061 of
    # remaining penalties, so standing still is strictly profitable.
    reward_alive_upright = RewTerm(
        func=reward_alive_upright,
        weight=0.15,
        params={"tol": 0.5},
    )
