#!/usr/bin/env python3
"""
从USD文件中读取几何尺寸信息
使用方法: python inspect_usd_geometry.py <usd_file_path>
"""

import sys
from pxr import Usd, UsdGeom, Gf
import numpy as np

def inspect_usd_file(usd_path):
    """检查USD文件中的几何信息"""
    print(f"\n{'='*60}")
    print(f"检查 USD 文件: {usd_path}")
    print(f"{'='*60}\n")

    # 打开USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"错误: 无法打开USD文件 {usd_path}")
        return

    # 遍历所有prim
    for prim in stage.Traverse():
        # 获取prim路径和类型
        prim_path = prim.GetPath()
        prim_type = prim.GetTypeName()

        # 只处理几何体
        if prim_type in ["Mesh", "Cylinder", "Cube", "Sphere", "Capsule", "Cone"]:
            print(f"\n发现几何体: {prim_path}")
            print(f"  类型: {prim_type}")

            # 获取变换信息
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                # 获取局部变换矩阵
                local_transform = xformable.GetLocalTransformation()
                translation = local_transform.ExtractTranslation()
                print(f"  位置: ({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}) 米")

            # 根据几何类型提取尺寸
            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get()
                    if points:
                        points_array = np.array(points)
                        min_bounds = points_array.min(axis=0)
                        max_bounds = points_array.max(axis=0)
                        size = max_bounds - min_bounds
                        print(f"  边界框尺寸 (米):")
                        print(f"    X: {size[0]:.6f} ({size[0]*1000:.2f} mm)")
                        print(f"    Y: {size[1]:.6f} ({size[1]*1000:.2f} mm)")
                        print(f"    Z: {size[2]:.6f} ({size[2]*1000:.2f} mm)")
                        print(f"  中心点: ({(min_bounds[0]+max_bounds[0])/2:.6f}, "
                              f"{(min_bounds[1]+max_bounds[1])/2:.6f}, "
                              f"{(min_bounds[2]+max_bounds[2])/2:.6f}) 米")

            elif prim_type == "Cylinder":
                cylinder = UsdGeom.Cylinder(prim)
                radius = cylinder.GetRadiusAttr().Get()
                height = cylinder.GetHeightAttr().Get()
                axis = cylinder.GetAxisAttr().Get()
                print(f"  圆柱体参数:")
                print(f"    半径: {radius:.6f} 米 ({radius*1000:.2f} mm)")
                print(f"    直径: {radius*2:.6f} 米 ({radius*2*1000:.2f} mm)")
                print(f"    高度: {height:.6f} 米 ({height*1000:.2f} mm)")
                print(f"    轴向: {axis}")

            elif prim_type == "Cube":
                cube = UsdGeom.Cube(prim)
                size = cube.GetSizeAttr().Get()
                print(f"  立方体边长: {size:.6f} 米 ({size*1000:.2f} mm)")

            elif prim_type == "Sphere":
                sphere = UsdGeom.Sphere(prim)
                radius = sphere.GetRadiusAttr().Get()
                print(f"  球体半径: {radius:.6f} 米 ({radius*1000:.2f} mm)")
                print(f"  球体直径: {radius*2:.6f} 米 ({radius*2*1000:.2f} mm)")

            elif prim_type == "Capsule":
                capsule = UsdGeom.Capsule(prim)
                radius = capsule.GetRadiusAttr().Get()
                height = capsule.GetHeightAttr().Get()
                axis = capsule.GetAxisAttr().Get()
                print(f"  胶囊体参数:")
                print(f"    半径: {radius:.6f} 米 ({radius*1000:.2f} mm)")
                print(f"    直径: {radius*2:.6f} 米 ({radius*2*1000:.2f} mm)")
                print(f"    高度: {height:.6f} 米 ({height*1000:.2f} mm)")
                print(f"    轴向: {axis}")

            # 检查物理材质属性
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                if mass:
                    print(f"  质量: {mass:.6f} kg ({mass*1000:.2f} g)")

            # 检查碰撞体
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                print(f"  [碰撞体已启用]")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python inspect_usd_geometry.py <usd_file_path>")
        print("\n示例:")
        print("  python inspect_usd_geometry.py assets/circle/circle_peg_I.usd")
        sys.exit(1)

    usd_path = sys.argv[1]
    inspect_usd_file(usd_path)
