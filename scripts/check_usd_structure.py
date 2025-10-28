"""Check USD structure and fix common issues."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default="lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

try:
    from pxr import Usd, UsdGeom, Sdf
    
    usd_path = str(PROJECT_ROOT / args.usd)
    print(f"\n{'='*80}")
    print(f"USD STRUCTURE CHECK")
    print(f"{'='*80}")
    print(f"File: {usd_path}\n")
    
    stage = Usd.Stage.Open(usd_path)
    
    # Check default prim
    default_prim = stage.GetDefaultPrim()
    print(f"Default Prim: {default_prim.GetPath() if default_prim else 'NOT SET'}")
    
    if not default_prim:
        print(f"  ⚠ WARNING: No default prim! This can cause rendering issues.")
        # Find root prim
        root_prims = [p for p in stage.GetPseudoRoot().GetChildren()]
        if root_prims:
            print(f"  Available root prims: {[str(p.GetPath()) for p in root_prims]}")
            print(f"  → Setting default prim to: {root_prims[0].GetPath()}")
            stage.SetDefaultPrim(root_prims[0])
            stage.Save()
            print(f"  ✓ Fixed and saved!")
    
    # List all prims
    print(f"\nPrim Structure:")
    for prim in stage.Traverse():
        indent = "  " * (len(str(prim.GetPath()).split('/')) - 2)
        prim_type = prim.GetTypeName()
        is_mesh = "📦 MESH" if prim.IsA(UsdGeom.Mesh) else ""
        vis = ""
        if prim.IsA(UsdGeom.Imageable):
            v = UsdGeom.Imageable(prim).ComputeVisibility()
            vis = f"[{v}]"
        print(f"{indent}{prim.GetPath()} ({prim_type}) {is_mesh} {vis}")
    
    # Check for meshes
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    print(f"\nTotal Meshes: {len(meshes)}")
    
    if len(meshes) == 0:
        print(f"  ⚠ WARNING: No visual meshes found!")
        print(f"  → Robot may only have collision geometry")
    
    print(f"\n{'='*80}\n")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    app.close()
