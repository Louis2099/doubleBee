"""
Script to check USD robot visibility issues and provide diagnostics.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check robot visibility in Isaac Sim")
parser.add_argument("--usd_path", type=str, default="lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    # Import after app launch
    from pxr import Usd, UsdGeom, Gf, Sdf
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.utils.stage import get_current_stage
    
    print("\n" + "="*80)
    print("DOUBLEBEE ROBOT USD DIAGNOSTICS")
    print("="*80 + "\n")
    
    # Get the full path to USD
    usd_full_path = PROJECT_ROOT / args.usd_path
    print(f"[1] USD File Path: {usd_full_path}")
    print(f"    Exists: {usd_full_path.exists()}\n")
    
    if not usd_full_path.exists():
        print("[ERROR] USD file not found!")
        sys.exit(1)
    
    # Open the USD stage
    print(f"[2] Opening USD file...")
    usd_stage = Usd.Stage.Open(str(usd_full_path))
    
    if not usd_stage:
        print("[ERROR] Failed to open USD stage!")
        sys.exit(1)
    
    print(f"    ✓ Stage opened successfully\n")
    
    # Check for meshes and geometry
    print(f"[3] Checking for geometry...")
    meshes = []
    geom_prims = []
    
    for prim in usd_stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        elif prim.IsA(UsdGeom.Gprim):
            geom_prims.append(prim)
    
    print(f"    Meshes found: {len(meshes)}")
    print(f"    Other geometry prims: {len(geom_prims)}")
    
    if meshes:
        print(f"\n    Mesh details:")
        for i, mesh_prim in enumerate(meshes[:5]):  # Show first 5
            mesh = UsdGeom.Mesh(mesh_prim)
            print(f"      [{i}] {mesh_prim.GetPath()}")
            
            # Check visibility
            vis_attr = mesh.GetVisibilityAttr()
            if vis_attr:
                vis = vis_attr.Get()
                print(f"          Visibility: {vis}")
            
            # Check points
            points_attr = mesh.GetPointsAttr()
            if points_attr:
                points = points_attr.Get()
                if points:
                    print(f"          Vertices: {len(points)}")
    
    # Check bounding box
    print(f"\n[4] Computing bounding box...")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    root_prim = usd_stage.GetDefaultPrim() or usd_stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    bounds = bbox.ComputeAlignedRange()
    
    print(f"    Min: {bounds.GetMin()}")
    print(f"    Max: {bounds.GetMax()}")
    print(f"    Size: {bounds.GetSize()}")
    
    size = bounds.GetSize()
    max_dim = max(size[0], size[1], size[2])
    print(f"    Max dimension: {max_dim:.4f} units")
    
    if max_dim < 0.01:
        print(f"    ⚠ WARNING: Robot appears very small (< 1cm)!")
    elif max_dim > 100:
        print(f"    ⚠ WARNING: Robot appears very large (> 100 units)!")
    else:
        print(f"    ✓ Robot size looks reasonable")
    
    # Now spawn it in Isaac Sim
    print(f"\n[5] Spawning robot in Isaac Sim...")
    stage = get_current_stage()
    
    robot_prim_path = "/World/DoubleBee"
    prim_utils.delete_prim(robot_prim_path)  # Clean up if exists
    
    # Create reference to USD
    robot_prim = prim_utils.create_prim(
        robot_prim_path,
        "Xform",
        usd_path=str(usd_full_path),
    )
    
    if robot_prim:
        print(f"    ✓ Robot spawned at {robot_prim_path}")
        
        # Set visibility explicitly
        imageable = UsdGeom.Imageable(robot_prim)
        if imageable:
            imageable.MakeVisible()
            print(f"    ✓ Set robot visibility to 'visible'")
        
        # Check all child prims
        all_prims = list(Usd.PrimRange(robot_prim))
        print(f"    Total child prims: {len(all_prims)}")
        
        invisible_count = 0
        for prim in all_prims:
            if prim.IsA(UsdGeom.Imageable):
                img = UsdGeom.Imageable(prim)
                vis = img.ComputeVisibility()
                if vis == UsdGeom.Tokens.invisible:
                    invisible_count += 1
                    # Force make visible
                    img.MakeVisible()
        
        if invisible_count > 0:
            print(f"    ⚠ Found {invisible_count} invisible child prims - forcing visibility")
        
        # Set transform to reasonable position
        xform = UsdGeom.Xformable(robot_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0, 0, 0.5))
        print(f"    ✓ Set robot position to (0, 0, 0.5)")
        
        print(f"\n[6] Robot spawned successfully!")
        print(f"    Look for the robot at world origin (0, 0, 0.5)")
        print(f"    Press Play to start simulation")
        print(f"    Press Ctrl+C to exit")
        
        # Keep app running
        while simulation_app.is_running():
            simulation_app.update()
    else:
        print(f"    [ERROR] Failed to spawn robot!")
        
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
