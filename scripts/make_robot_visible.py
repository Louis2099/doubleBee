"""
Fix DoubleBee USD visibility by making all prims visible.
This script launches Isaac Sim and modifies the USD file.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Fix USD visibility")
parser.add_argument(
    "--usd_path",
    type=str,
    default="lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd",
    help="Path to USD file to fix"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    # Now we can import pxr
    from pxr import Usd, UsdGeom
    
    # Get full path
    usd_full_path = PROJECT_ROOT / args.usd_path
    print(f"\n{'='*80}", flush=True)
    print(f"FIXING USD VISIBILITY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"USD Path: {usd_full_path}", flush=True)
    
    if not usd_full_path.exists():
        print(f"[ERROR] USD file not found!", flush=True)
        sys.exit(1)
    
    # Open the USD
    print(f"\nOpening USD stage...", flush=True)
    stage = Usd.Stage.Open(str(usd_full_path))
    
    if not stage:
        print(f"[ERROR] Failed to open USD stage!", flush=True)
        sys.exit(1)
    
    print(f"✓ Stage opened successfully", flush=True)
    
    # Make all prims visible
    print(f"\nMaking all prims visible...", flush=True)
    visible_count = 0
    invisible_count = 0
    
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            vis = imageable.ComputeVisibility()
            
            if vis == UsdGeom.Tokens.invisible:
                imageable.MakeVisible()
                invisible_count += 1
                print(f"  Fixed invisible: {prim.GetPath()}", flush=True)
            else:
                visible_count += 1
    
    print(f"\nSummary:", flush=True)
    print(f"  Already visible: {visible_count}", flush=True)
    print(f"  Fixed (was invisible): {invisible_count}", flush=True)
    
    # Save the USD
    if invisible_count > 0:
        print(f"\nSaving changes to USD file...", flush=True)
        stage.Save()
        print(f"✓ USD file saved with all prims visible!", flush=True)
        print(f"\nYou can now run your test script - robot should be visible.", flush=True)
    else:
        print(f"\n✓ All prims were already visible - no changes needed.", flush=True)
    
    print(f"{'='*80}\n", flush=True)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
