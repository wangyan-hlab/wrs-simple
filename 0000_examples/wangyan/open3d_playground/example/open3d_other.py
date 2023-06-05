import open3d as o3d
import numpy as np

# ---------------------- 定义点云体素化函数 ----------------------
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

print("->正在加载点云1... ")
pcd1 = o3d.io.read_point_cloud("../pcd/ism_test_horse.pcd")
pcd1.paint_uniform_color([1, 0, 0])
print(pcd1)
print("->正在加载点云2...")
pcd2 = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
pcd2.paint_uniform_color([0, 1, 0])
print(pcd2)
# -------------------------- 计算点云间的距离 --------------------------
pcd1.translate((50, 0, 0))
print("->正在点云1每一点到点云2的最近距离...")
dists = pcd1.compute_point_cloud_distance(pcd2)
dists = np.asarray(dists)
print("->正在打印前10个点...")
print(dists[:10])

print("->正在提取距离大于5的点")
ind = np.where(dists > 40)[0]
pcd3 = pcd1.select_by_index(ind)
pcd3.paint_uniform_color([0, 0, 1])
print(pcd3)
o3d.visualization.draw_geometries([pcd2, pcd3])

# -------------------------- 计算点云最小包围盒 --------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
pcd.paint_uniform_color([1, 0, 0])

print("->正在计算点云轴向最小包围盒...")
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
print("->正在计算点云最小包围盒...")
obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd, aabb, obb])

# -------------------------- 计算点云凸包 --------------------------
print("->正在计算点云凸包...")
hull, _ = pcd.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcd, hull_ls])

# --------------------------- 体素化点云 -------------------------
print('执行体素化点云')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=3)
print("正在可视化体素...")
o3d.visualization.draw_geometries([voxel_grid])

# ------------------------- 点云体素化 --------------------------
print("->正在进行点云体素化...")
_relative_path = "../pcd/man_hand.ply"  # 设置相对路径
N = 2000        # 将点划分为N个体素
pcd = get_mesh(_relative_path).sample_points_poisson_disk(N)

# fit to unit cube
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
print("体素下采样点云：", pcd)
print("正在可视化体素下采样点云...")
o3d.visualization.draw_geometries([pcd])

print('执行体素化点云')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
print("正在可视化体素...")
o3d.visualization.draw_geometries([voxel_grid])

# ------------------------- 计算点云质心 -------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/tutorials/ism_test_horse.pcd")
print(pcd)
print(f'pcd质心：{pcd.get_center()}')

# ------------------------- 根据索引提取点云 -------------------------
print("->正在根据索引提取点云...")
idx = list(range(2000))    # 生成 从0到19999的列表

# 索引对应的点云（内点）
inlier_pcd = pcd.select_by_index(idx)
inlier_pcd.paint_uniform_color([1, 0, 0])
print("内点点云：", inlier_pcd)

# 索引外的点云（外点）
outlier_pcd = pcd.select_by_index(idx, invert=True)     # 对索引取反
outlier_pcd.paint_uniform_color([0, 1, 0])
print("外点点云：", outlier_pcd)

o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])
