# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass


@configclass
class LiftMask(ContactSensor):
    """Contact sensor for DoubleBee robot."""

    def __init__(self, cfg: object, env: object):
        super().__init__(cfg, env)
        self._lift_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def compute(self) -> torch.Tensor:
        """Compute lift mask based on contact forces."""
        # Check if wheels are in contact with ground
        contact_forces = self.data.net_forces_w_history[0]
        self._lift_mask = contact_forces[:, 2] > 0.1  # Z-axis force threshold
        return self._lift_mask
