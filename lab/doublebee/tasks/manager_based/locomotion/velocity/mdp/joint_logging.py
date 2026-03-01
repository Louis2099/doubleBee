# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Log target and actual joint state (position, velocity) for propellers and propeller servos."""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Joint names to log (props: velocity-controlled; servos: position-controlled)
PROP_JOINT_NAMES = ("leftPropeller", "rightPropeller")
SERVO_JOINT_NAMES = ("leftPropellerServo", "rightPropellerServo")
ALL_JOINT_NAMES = PROP_JOINT_NAMES + SERVO_JOINT_NAMES

# Module-level step counter (incremented each time the logger runs)
_log_step = 0
_file_handle = None


def log_propeller_servo_joint_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    log_path: str = "prop_servo_joint_log.csv",
    log_interval_steps: int = 1,
    env_ids_to_log: Sequence[int] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Append target and actual joint pos/vel for propellers and propeller_servos to a CSV file.

    Called every simulation step (or every log_interval_steps). Logs one row per
    (step, env_id, joint_name). Columns: step, env_id, joint_name, pos_target, vel_target,
    pos_actual, vel_actual.

    Args:
        env: ManagerBasedEnv (provided by event manager).
        env_ids: Not used for interval events; we use env_ids_to_log instead.
        log_path: Path to CSV file (default: prop_servo_joint_log.csv in cwd).
        log_interval_steps: Log every N calls (1 = every step).
        env_ids_to_log: Which env indices to log (default: [0]).
        asset_cfg: Robot asset (default: SceneEntityCfg("robot")).
    """
    global _log_step, _file_handle

    if env_ids_to_log is None:
        env_ids_to_log = [0]

    robot: Articulation = env.scene[asset_cfg.name]
    joint_names = list(robot.joint_names)

    # Resolve joint indices
    joint_indices = []
    for name in ALL_JOINT_NAMES:
        if name in joint_names:
            joint_indices.append((name, joint_names.index(name)))
    if not joint_indices:
        return

    _log_step += 1
    if _log_step % log_interval_steps != 0:
        return

    pos_target = robot.data.joint_pos_target.cpu().numpy()
    vel_target = robot.data.joint_vel_target.cpu().numpy()
    pos_actual = robot.data.joint_pos.cpu().numpy()
    vel_actual = robot.data.joint_vel.cpu().numpy()

    need_header = False
    if _file_handle is None:
        dirname = os.path.dirname(log_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        _file_handle = open(log_path, "w", encoding="utf-8")
        need_header = True

    if need_header:
        _file_handle.write("step,env_id,joint_name,pos_target,vel_target,pos_actual,vel_actual\n")

    for eid in env_ids_to_log:
        if eid >= env.num_envs:
            continue
        for name, jidx in joint_indices:
            pt = pos_target[eid, jidx]
            vt = vel_target[eid, jidx]
            pa = pos_actual[eid, jidx]
            va = vel_actual[eid, jidx]
            _file_handle.write(f"{_log_step},{eid},{name},{pt:.6f},{vt:.6f},{pa:.6f},{va:.6f}\n")

    _file_handle.flush()


def close_propeller_servo_joint_log():
    """Close the log file if open. Call at end of run to ensure data is written."""
    global _file_handle
    if _file_handle is not None:
        _file_handle.close()
        _file_handle = None
