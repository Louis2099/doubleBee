from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd")
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        print(f"Made visible: {prim.GetPath()}")
stage.Save()
print("✓ Saved USD with all prims visible")