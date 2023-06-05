import open3d as o3d
import numpy as np

# --------------------------- 加载点云 ---------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
print("原始点云：", pcd)
# ==============================================================

# ------------------------- 1.从点云中构建Octree --------------------------
print('octree 分割')
octree = o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd, size_expand=0.01)
print("->正在可视化Octree...")
o3d.visualization.draw_geometries([octree])

# ------------------------- 2.从体素栅格中构建Octree -------------------------
print('体素化')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)
print("体素：", voxel_grid)
print("正在可视化体素...")
o3d.visualization.draw_geometries([voxel_grid])

print('Octree 分割')
octree = o3d.geometry.Octree(max_depth=4)
octree.create_from_voxel_grid(voxel_grid)
print("Octree：", octree)
print("正在可视化Octree...")
o3d.visualization.draw_geometries([octree])
