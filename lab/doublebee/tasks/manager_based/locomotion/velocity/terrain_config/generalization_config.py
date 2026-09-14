# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unseen terrains for the zero-shot generalization evaluation.

NOT sim-to-sim. Same engine, same contact model, same robot. What changes is
only the GEOMETRY the policy meets, none of which appears anywhere in training:
training is inverted pyramid STAIRS, 4-7 cm discrete steps, ascending.

Three environments, each a single structural departure so a failure is
attributable:

    rough      no steps at all, continuous small-scale irregularity
    slope_up   continuous incline, ascending, no discrete step to key off
    slope_down descending, which the policy has never done in any form

Spawn/target sampling mirrors STAIR_TERRAINS_CFG_PLAY: `init_pos` near the
centre, `target` out at y 1.5-3.2. The z_range on `target` is what sets the
direction of travel, so it is the one field to check if a terrain misbehaves.
Ranges here are deliberately generous: too tight and the generator raises
"Failed to find valid patches" rather than silently degrading.
"""

from __future__ import annotations

import os

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.sim as sim_utils


# Spawn spread, for RENDERS only. Every environment draws its start from a
# 1 m box at the tile centre with 8 patches to share, so a multi-robot capture
# has them stacked on top of each other. Widening it for a screenshot must not
# silently change the configuration the numbers were measured under, so it is
# an override with the measured values as defaults.
#   DOUBLEBEE_SPAWN_SPREAD=1.2 DOUBLEBEE_SPAWN_PATCHES=32
# The central platform is platform_width/2 across, so do not exceed ~1.4 or
# spawns land on the stairs themselves.
_SPREAD = float(os.environ.get("DOUBLEBEE_SPAWN_SPREAD", 0.5))
_PATCHES = int(os.environ.get("DOUBLEBEE_SPAWN_PATCHES", 8))


def _init_patch(max_dh=0.03):
    # Spread LATERALLY (x) only, never along the travel axis (y).
    #
    # Targets are sampled at positive y, so a robot spawned at negative y must
    # cross the whole central platform before it can do the thing under test.
    # On the descending pyramid that was worse still: a square +-1.2 box has
    # corners at 1.70 m against a 1.5 m platform radius, so some environments
    # spawned a step BELOW the peak and had to climb up, traverse the top, and
    # only then descend. Keeping y tight fixes both, and lateral spread is what
    # a multi-robot render wants anyway: a row abreast, not a queue.
    # patch_radius 0.05 with max_height_diff 0.15 accepted a 5 cm disc that
    # tolerated a 15 cm height change, so a STEP EDGE qualified as flat and
    # robots spawned half on and half off the top of the pyramid. A 0.20 m disc
    # that must be flat to 0.03 m forces the spawn onto genuine plateau.
    #
    # SMOOTH terrains pass a looser max_dh. They have no step edge to guard
    # against, and on a 0.21-0.26 ramp a 0.20 m patch changes height by 4 cm,
    # so the strict tolerance admits ONLY the small central platform and every
    # robot spawns on top of every other one however wide SPAWN_SPREAD is set.
    return FlatPatchSamplingCfg(
        num_patches=_PATCHES, patch_radius=0.20,
        x_range=(-_SPREAD, _SPREAD), y_range=(-0.3, 0.3),
        z_range=(-0.6, 0.6), max_height_diff=max_dh,
    )


def _target_patch(z_range, y_range=(1.5, 3.2)):
    return FlatPatchSamplingCfg(
        num_patches=max(5, _PATCHES // 2), patch_radius=0.1,
        x_range=(-1.0, 1.0), y_range=y_range, z_range=z_range,
        max_height_diff=0.30,
    )


def _gen(sub, **kw):
    base = dict(
        seed=42, size=(10.0, 10.0), border_width=2.0,
        num_rows=1, num_cols=1, curriculum=True,
        # "none" lets the flat visual_material decide the look; "random" bakes
        # per-tile tints that fight it. "height" colours by ELEVATION, which is
        # the one setting that makes a smooth ramp legible: a featureless matte
        # surface under dome light has nothing to shade, so the slope is
        # invisible, but a height gradient shows it directly. It overrides the
        # flat colour, so use it per terrain rather than globally.
        color_scheme=os.environ.get("DOUBLEBEE_TERRAIN_COLOR_SCHEME", "none"),
        # 0.05 not 0.1: at 10 cm cells a shallow ramp tessellates visibly.
        horizontal_scale=0.05, vertical_scale=0.005,
        # THE STAIR CONFIG USES 0.1 AND THAT IS RIGHT FOR STAIRS -- it forces
        # step faces vertical. It is WRONG here. Our slope_range is 0.10-0.25,
        # so with a 0.1 threshold EVERY part of the ramp exceeds it and the
        # library rewrites the whole incline into vertical walls: the "slope"
        # renders and behaves as a staircase. 0.75 is the library default and
        # leaves a 0.10-0.25 ramp untouched.
        slope_threshold=0.75,
        use_cache=False,
    )
    base.update(kw)
    return TerrainGeneratorCfg(sub_terrains=sub, **base)


# ---- alpha: irregular ground, no steps ------------------------------------
GEN_ROUGH_CFG = _gen(
    {
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            # +-4 cm of noise: comparable in amplitude to a 4-7 cm step but
            # with no edge to key a climb off, and it never repeats.
            noise_range=(0.01, 0.04),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(),
                "target": _target_patch((-0.25, 0.25)),
            },
        ),
    },
    difficulty_range=(0.4, 0.6),
)

# ---- beta: continuous ascent ----------------------------------------------
# inverted=True puts the LOW point at the centre, so the robot spawns at the
# bottom and the target out at y 1.5-3.2 is uphill. Same direction of travel as
# the training stairs, but with the discrete step removed.
GEN_SLOPE_UP_CFG = _gen(
    {
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            inverted=True,
            slope_range=(0.15, 0.30),   # about 8.5 to 17 degrees
            # 2026-09-10: was 2.0, which leaves the central metre FLAT. The
            # robot tilts out around 1.2 m of travel, so it never reached the
            # incline and every episode reported zero height gain. Measured
            # from the mesh: with platform 2.0 the profile is z=-0.415 at r=0.3
            # and -0.411 at r=1.4, i.e. no slope at all where the robot is.
            platform_width=1.0,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(max_dh=0.12),
                # 2026-09-10: tried pushing the goal further up the ramp
                # (y 2.0-3.6 and y 2.6-4.2). It buys nothing. Median gain stays
                # at 0.110-0.113 m in every variant because the ceiling is the
                # POLICY tilting on a continuous incline at about 11 cm, not
                # where the goal sits. Moving it out only removed the
                # goal-reached terminations. Kept at the original placement.
                "target": _target_patch((0.03, 0.80)),
            },
        ),
    },
    difficulty_range=(0.4, 0.7),
    # A height field IS a staircase: the tread it presents is
    # horizontal_scale * slope, so smoothness is set by CELL SIZE, not by how
    # steep the ramp is. At the original 5 cm cells even a shallow ramp showed
    # 20-35 mm steps. At 2.5 cm cells a 0.30 slope shows 7.5 mm, against a
    # 60 mm wheel radius -- so the incline can be steep enough to SEE and still
    # be smooth. A ramp nobody can tell is a ramp fails as a figure.
    horizontal_scale=0.025,
    vertical_scale=0.0025,
)

# ---- gamma: continuous descent --------------------------------------------
# inverted=False puts the HIGH point at the centre, so the robot spawns on top
# and the target is downhill. Descent is a different control problem: the
# policy must brake a wheeled balancer rather than drive it up, and it has
# never seen it.
GEN_SLOPE_DOWN_CFG = _gen(
    {
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            inverted=False,
            slope_range=(0.15, 0.30),
            platform_width=1.0,   # see the note on slope_up
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(max_dh=0.12),
                "target": _target_patch((-0.80, -0.03)),
            },
        ),
    },
    difficulty_range=(0.4, 0.7),
    # A height field IS a staircase: the tread it presents is
    # horizontal_scale * slope, so smoothness is set by CELL SIZE, not by how
    # steep the ramp is. At the original 5 cm cells even a shallow ramp showed
    # 20-35 mm steps. At 2.5 cm cells a 0.30 slope shows 7.5 mm, against a
    # 60 mm wheel radius -- so the incline can be steep enough to SEE and still
    # be smooth. A ramp nobody can tell is a ramp fails as a figure.
    horizontal_scale=0.025,
    vertical_scale=0.0025,
)


# Terrain colour, LINEAR rgb as PreviewSurfaceCfg expects.
#
# Two things make this unintuitive. First the display is gamma-encoded, so
# linear 0.09 shows as about 0.33. Second, and more important: the dome light
# contributes an ambient floor that diffuse_color does NOT scale, which is why
# (0,0,0) still renders mid-grey. Everything you set is ADDED on top of that
# grey, so candidates that differ by 0.04 are indistinguishable, and the way to
# get a dark tint is to zero the other two channels rather than to lower all
# three. (0, 0, 0.10) reads as dark blue; (0.09, 0.13, 0.22) reads as grey.
#
# Override without editing: DOUBLEBEE_TERRAIN_RGB=0,0,0.10
_TERRAIN_RGB = tuple(
    float(v) for v in os.environ.get("DOUBLEBEE_TERRAIN_RGB", "0,0,0").split(",")
)


# ---- delta: descending discrete steps -------------------------------------
# The sharpest test in the set. Training is ASCENDING 4-7 cm steps and nothing
# else; here inverted=False puts the high point at the centre, so the robot
# spawns on top and the target out at y 1.5-3.2 is below it. Braking a
# two-wheeled balancer down an edge is a different control problem from driving
# up into one, and the policy has never done it. Step heights are held at the
# training range so the ONLY thing that changed is the direction.
GEN_STAIR_DOWN_CFG = _gen(
    {
        "stair_down": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=False,
            proportion=1.0,
            step_height_range=(0.04, 0.07),   # the training range exactly
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(),
                # Targets pushed well down the staircase. At -0.03 m the
                # 0.15 m goal tolerance is satisfied ABOVE the spawn, so the
                # episode could end without descending at all; -0.25 m is the
                # shallowest target that still requires a real descent, and
                # the y range is widened so patches that deep exist.
                "target": _target_patch((-0.50, -0.25), y_range=(1.5, 4.5)),
            },
        ),
    },
    difficulty_range=(0.4, 0.7),
    # Steps want vertical faces, so unlike the ramps this one keeps the stair
    # config's threshold rather than the smooth-ramp 0.75.
    slope_threshold=0.1,
)

# ---- epsilon: undulating ground -------------------------------------------
# Continuous periodic pitch disturbance with no edge anywhere: neither a step to
# key off nor a steady incline to lean into. Amplitude is kept near the step
# heights the policy knows so the disturbance is comparable in size, not in
# shape.
GEN_WAVE_CFG = _gen(
    {
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.03, 0.08),
            num_waves=4,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(max_dh=0.12),
                "target": _target_patch((-0.20, 0.20)),
            },
        ),
    },
    difficulty_range=(0.4, 0.7),
    horizontal_scale=0.025,
    vertical_scale=0.0025,
    slope_threshold=0.75,
)


# ---- narrow tread: the same climb with half the recovery distance ---------
# Training uses a 0.40 m step width throughout. This halves it to 0.20 m and
# changes nothing else: same ascending direction, same step-height range, same
# spawn and target sampling. Tread width is the distance the robot has to
# settle between consecutive steps, which is the variable the hardware
# campaign turned on -- the first step was solved long before the second, on
# 0.40 m treads. It tests the climbing mechanism itself rather than tolerance
# to unfamiliar ground.
GEN_NARROW_CFG = _gen(
    {
        # Key contains "stair" so eval_climb.py --step-height can find it;
        # its lookup is next(k for k in sub_terrains if "stair" in k).
        "narrow_stair": terrain_gen.HfPyramidStairsTerrainCfg(
            inverted=True,
            proportion=1.0,
            # Pinned at 6 cm, the operating point used everywhere else in the
            # paper, so tread width is compared at a familiar step height
            # rather than at the 5.2-6.1 cm the difficulty range produced.
            step_height_range=(0.06, 0.06),
            # DOUBLEBEE_STEP_WIDTH lets the SAME task run the trained 0.40 m
            # as a matched control, so tread width is the only variable that
            # differs between the two conditions.
            step_width=float(os.environ.get("DOUBLEBEE_STEP_WIDTH", 0.20)),
            platform_width=3.0,
            border_width=1.0,
            flat_patch_sampling={
                "init_pos": _init_patch(),
                # The goal-reached check carries a 0.15 m elevation tolerance
                # (DOUBLEBEE_GOAL_DZ), so a target at 0.20 m is satisfied from
                # 0.05 m and the episode can end with a lean rather than a
                # climb. Targets are placed at 0.25-0.50 m, which is 0.10 m
                # clear of the tolerance at its lowest, and the y range is
                # widened so the 0.40 m control can still find valid patches
                # at that elevation.
                "target": _target_patch((0.25, 0.50), y_range=(1.5, 4.5)),
            },
        ),
    },
    difficulty_range=(0.4, 0.7),
    slope_threshold=0.1,   # steps want vertical faces, as in training
)


def _importer(gen):
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=gen,
        max_init_terrain_level=1,
        collision_group=-1,
        # Dark blue-black reads well in renders and against the pale robot.
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=_TERRAIN_RGB, roughness=1.0, metallic=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )


ROUGH_TERRAIN = _importer(GEN_ROUGH_CFG)
STAIR_DOWN_TERRAIN = _importer(GEN_STAIR_DOWN_CFG)
NARROW_TERRAIN = _importer(GEN_NARROW_CFG)
WAVE_TERRAIN = _importer(GEN_WAVE_CFG)
SLOPE_UP_TERRAIN = _importer(GEN_SLOPE_UP_CFG)
SLOPE_DOWN_TERRAIN = _importer(GEN_SLOPE_DOWN_CFG)
