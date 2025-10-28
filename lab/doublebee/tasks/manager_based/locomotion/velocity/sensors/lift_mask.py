# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from isaaclab.sensors.ray_caster import RayCaster


@dataclass
class LiftMaskData:
    """Data container for the lift mask sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame. Shape is (N, 3)."""
    
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame. Shape is (N, 4)."""
    
    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame. Shape is (N, B, 3)."""
    
    mask: torch.Tensor = None
    """The mask for the lift sensor. Shape is (N, T, 1)."""
    
    mask_history: torch.Tensor | None = None
    """The mask history. Shape is (N, T, 1), where T is the configured history length."""


class LiftMask(RayCaster):
    """Ray-cast based lift mask sensor for DoubleBee robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = LiftMaskData()
        self._height_map_w = int(round(self.cfg.pattern_cfg.size[0] / self.cfg.pattern_cfg.resolution) + 1)
        self._height_map_h = int(round(self.cfg.pattern_cfg.size[1] / self.cfg.pattern_cfg.resolution) + 1)
        self._last_zero_index = round((self._height_map_h - getattr(self.cfg, 'last_zero_num', 3)) / 2)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Lift-mask @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
        )

    @property
    def data(self) -> LiftMaskData:
        """Returns the data container for the sensor."""
        return self._data

    def _update_buffers_impl(self, env_ids: torch.Tensor):
        """Compute lift mask based on ray-cast data."""
        # Call parent to update ray-cast data
        super()._update_buffers_impl(env_ids)
        
        # Compute mask (simple implementation - can be enhanced)
        if self.data.ray_hits_w is not None:
            # Check if any rays hit something close to the ground
            z_hits = self.data.ray_hits_w[env_ids, :, 2]
            self._data.mask = (z_hits < 0.05).any(dim=1, keepdim=True).float()
