# DoubleBee Robot Visibility Issue - Solutions

The robot exists in the scene (confirmed by debug output) but is **invisible in the viewport**. This is a common USD issue.

## Confirmed Information
- ✅ Robot IS spawned (prim exists at `/World/envs/env_0/Robot`)
- ✅ Robot position is correct: `[0, 0, 0.5]`
- ✅ Robot is an `Articulation` with proper data
- ❌ Robot is NOT visible in Isaac Sim viewport

## Possible Causes & Solutions

### Solution 1: Check USD File Visibility (Most Likely)
The USD file itself might have visibility set to "invisible" for all prims.

**To fix in Isaac Sim GUI:**
1. Open Isaac Sim
2. File → Open → Browse to: `/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd`
3. In the Stage panel (left side), select the root prim or any geometry prim
4. In the Property panel (right side), find "Visibility" attribute
5. If it says "invisible", change it to "inherited" or "visible"
6. Do this for ALL geometry prims (bodies, links, meshes)
7. File → Save

**To fix via Python (inside Isaac Sim):**
```python
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd")
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        print(f"Made visible: {prim.GetPath()}")
stage.Save()
print("✓ Saved USD with all prims visible")
```

### Solution 2: Check for Missing Geometry
The USD might only have collision shapes but no visual meshes.

**To check:**
1. Open the USD in Isaac Sim
2. Look for prims with type "Mesh" or "Geom"
3. If you only see "CollisionAPI" or physics shapes, you need to add visual meshes

### Solution 3: Scale Issue
The robot might be too small (microscopic) or too large.

**To check/fix in doublebee_v1.py:**
```python
spawn=sim_utils.UsdFileCfg(
    usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee.usd",
    scale=(10.0, 10.0, 10.0),  # Try increasing scale
    visible=True,
)
```

### Solution 4: Camera Position
The camera might be far from the robot or facing the wrong direction.

**To fix:**
1. In Isaac Sim viewport, press `F` key while robot prim is selected (Frame Selected)
2. Or manually move camera closer to origin (0, 0, 0.5)
3. Or use Viewport → Camera → Create → Camera and position it at `(2, 2, 2)` looking at `(0, 0, 0.5)`

### Solution 5: Rendering Issues
Materials or shaders might be missing/incompatible.

**Quick test:**
Add a simple colored material to the USD in Isaac Sim:
1. Select a geometry prim
2. Right-click → Create → Material → Preview Surface
3. Set Diffuse Color to something bright (red, green, etc.)
4. Apply the material

## Immediate Action - Manual Test in Isaac Sim

1. **Open Isaac Sim standalone** (not via your script)
2. **File → Open** your USD: `/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd`
3. **Check what you see:**
   - Nothing? → Visibility issue (Solution 1)
   - Bounding box only? → Missing visual geometry (Solution 2)
   - Very tiny/huge? → Scale issue (Solution 3)
   - Can see it? → Camera/rendering issue in your script (Solution 4/5)

4. **If you CAN see it in standalone Isaac Sim:**
   - The USD is fine
   - Problem is with how IsaacLab spawns it
   - Try adding `debug_vis=True` to the ArticulationCfg

5. **If you CANNOT see it in standalone Isaac Sim:**
   - The USD file needs fixing
   - Follow Solution 1 to make all prims visible
   - Resave the USD

## Quick USD Visibility Fix Script

Run this in standalone Isaac Sim Script Editor:

```python
from pxr import Usd, UsdGeom
import omni.usd

# Get current stage or open the USD
usd_path = "/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd"
stage = omni.usd.get_context().get_stage()  # If already open
# OR
# stage = Usd.Stage.Open(usd_path)  # If opening fresh

invisible_count = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        vis = imageable.ComputeVisibility()
        if vis == UsdGeom.Tokens.invisible:
            imageable.MakeVisible()
            invisible_count += 1
            print(f"Fixed: {prim.GetPath()}")

print(f"\nFixed {invisible_count} invisible prims")
print("Now save the file: File → Save")
```

## Report Back
After trying the manual test in standalone Isaac Sim, let me know:
1. Can you see the robot when opening the USD directly?
2. What prims are in the USD (look at Stage panel)?
3. What are their visibility settings?
4. Are there visual meshes or only collision shapes?
