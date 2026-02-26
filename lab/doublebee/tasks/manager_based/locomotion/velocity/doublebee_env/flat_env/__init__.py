# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain environments for DoubleBee robot."""

from .hybrid_stair.hybrid_stair_cfg import DoubleBeeHybridStairCfg, DoubleBeeHybridStairCfg_PLAY
from .inverted_pendulum import DoubleBeeInvertedPendulumCfg, DoubleBeeInvertedPendulumCfg_PLAY

__all__ = [
    "DoubleBeeHybridStairCfg",
    "DoubleBeeHybridStairCfg_PLAY",
    "DoubleBeeInvertedPendulumCfg",
    "DoubleBeeInvertedPendulumCfg_PLAY",
]
