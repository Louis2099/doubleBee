# Target Position Z-Coordinate Issue - Why Targets Appear at Z=0

## Problem Statement

Target positions sampled from terrain flat patches always appear at **Z = 0.0 m** (ground level), even though they should be at higher levels of the staircase (e.g., Z = 1.0 m for upper steps). The targets appear to be **under the physical terrain surface** instead of on top of it.

## Step-by-Step Analysis

### Step 1: Terrain Generation and Flat Patch Sampling

**Configuration** (`stair_config.py`):
```python
"target": FlatPatchSamplingCfg(
    num_patches=5,
    patch_radius=0.2,
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    z_range=(0.0, 1.0),  # Search range: 0m to 1m above terrain base
    max_height_diff=0.15,
),
```

**What happens:**
1. The terrain generator (`HfPyramidStairsTerrainCfg`) creates an inverted pyramid staircase
2. The generator searches for **flat surfaces** on the terrain geometry
3. It finds patches on:
   - Bottom platform (Z ≈ 0.0 m)
   - Step platforms at various heights (Z ≈ 0.18 m, 0.36 m, 0.54 m, etc.)
   - Top platform (Z ≈ 0.72 m - 1.0 m, depending on step count)

### Step 2: Flat Patch Coordinate System

**Critical Understanding:**
- Flat patches are sampled **on the actual terrain surface** (correct!)
- But their **Z coordinates are stored relative to the terrain base** (Z = 0), not as absolute world heights
- The terrain generator normalizes all patch Z coordinates to be relative to the terrain patch center/base

**Example:**
- A patch on a step at **physical height Z = 0.5 m** (above world origin)
- Is stored as **Z = 0.0 m** (relative to terrain base)
- Because the terrain base is at Z = 0.0 m

### Step 3: Environment Origins

**`env_origins` Definition:**
```python
env_origins = env.scene.env_origins  # [num_envs, 3]
# Shape: [num_envs, 3] where each row is [x, y, z]
# Z coordinate is always 0.0 (terrain base level)
```

**What `env_origins` represents:**
- The **base/origin** of each environment's terrain patch
- Z coordinate is **always 0.0 m** (ground level)
- This is the reference point for all terrain-relative coordinates

### Step 4: Target Position Calculation

**Code** (`velocity_command.py:147-148`):
```python
target_relative = targets[patch_idx, :]  # [3] - Patch coordinates relative to terrain base
target_world = target_relative + env_origins[env_idx, :]  # Transform to world coordinates
```

**The Problem:**
```python
# Example: Patch on step at physical height 0.5m
target_relative = [2.0, 1.0, 0.0]  # Z=0.0 (relative to terrain base)
env_origins[env_idx] = [10.0, 5.0, 0.0]  # Z=0.0 (terrain base)

target_world = [2.0, 1.0, 0.0] + [10.0, 5.0, 0.0]
            = [12.0, 6.0, 0.0]  # ❌ Z is still 0.0!
```

**Result:** All targets end up at **Z = 0.0 m** regardless of which step they're on!

### Step 5: Why This Happens

**Root Cause:**
1. **Terrain generator stores patches with Z=0 relative to base** - This is by design for consistency
2. **`env_origins` Z coordinate is always 0.0** - It represents the terrain base, not the patch surface
3. **No height information is preserved** - The actual physical height of the step is lost when patches are stored

**The terrain generator assumes:**
- Patches are on flat surfaces (correct)
- But their Z coordinates are normalized to terrain base (Z=0)
- The actual surface height must be queried from the terrain geometry separately

## Why Patches Are Stored at Z=0

**Design Decision:**
- Flat patches are stored as **relative coordinates** to enable:
  - Consistent coordinate system across different terrain types
  - Easy transformation using `env_origins`
  - Reusability across different terrain configurations

**Trade-off:**
- ✅ Easy coordinate transformation
- ❌ Loses actual surface height information

## Solution

To fix this, we need to **query the actual terrain height** at the target XY position, rather than using the stored Z coordinate.

### Option 1: Raycast to Get Actual Terrain Height

```python
# In velocity_command.py, after computing target_world XY:
target_xy = target_world[:, :2]  # [num_envs, 2]

# Raycast downward from above to find actual terrain surface
ray_start = torch.cat([target_xy, torch.ones(num_envs, 1) * 2.0], dim=1)  # Start 2m above
ray_end = torch.cat([target_xy, torch.ones(num_envs, 1) * -1.0], dim=1)  # End 1m below

# Perform raycast against terrain
hit_points = terrain.raycast(ray_start, ray_end)  # Returns hit Z coordinates
target_world[:, 2] = hit_points[:, 2]  # Use actual terrain height
```

### Option 2: Use Terrain Height Map

If the terrain provides a height map, query it directly:
```python
target_world[:, 2] = terrain.get_height_at(target_world[:, :2])
```

### Option 3: Store Absolute Heights in Patches (Requires Terrain Generator Changes)

Modify the terrain generator to store absolute heights instead of relative Z=0, but this would require changes to Isaac Lab's core terrain generation code.

## Current Workaround

The visualization in `play.py` adds an offset:
```python
target_pos[0, 2] += 0.4  # 40cm above ground
```

This makes the target **visible**, but it's still **not at the correct step height**. The target should be at the actual step surface height, not a fixed offset above ground.

## Summary

**Why targets are at Z=0:**
1. ✅ Flat patches are correctly sampled on terrain surfaces (steps)
2. ❌ But their Z coordinates are stored as **relative to terrain base (Z=0)**
3. ❌ `env_origins` Z is always **0.0** (terrain base)
4. ❌ `target_world = target_relative + env_origins` → Z always becomes **0.0**
5. ❌ **No actual terrain height information is preserved** in the patch coordinates

**The fix requires:**
- Querying the actual terrain geometry to get the real surface height at the target XY position
- This cannot be done with just the stored patch coordinates - we need to raycast or query a height map

