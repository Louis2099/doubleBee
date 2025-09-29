# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class LiftMaskCfg:
    """Contact sensor configuration for DoubleBee robot."""

    # Contact sensor for wheels
    wheel_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Doublebee/leftWheel",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )
    """Left wheel contact sensor."""

    # Contact sensor for right wheel
    right_wheel_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Doublebee/rightWheel",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )
    """Right wheel contact sensor."""
