import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
print(pcd)
# --------------------------- 1.DBSCAN 聚类分割 ---------------------------
print("->正在DBSCAN聚类...")
eps = 10             # 同一聚类中最大点间距
min_points = 100     # 有效聚类的最小点数
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

# --------------------------- 2.RANSAC 平面分割 ---------------------------
print("->正在RANSAC平面分割...")
distance_threshold = 10    # 内点到平面模型的最大距离
ransac_n = 10                # 用于拟合平面的采样点数
num_iterations = 1000       # 最大迭代次数

# 返回模型系数plane_model和内点索引inliers，并赋值
plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

# 输出平面方程
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 平面内点点云
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1.0])
print(inlier_cloud)

# 平面外点点云
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1.0, 0, 0])
print(outlier_cloud)

# 可视化平面分割结果
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# --------------------------- 3.正在剔除隐藏点 ---------------------------
print("->正在剔除隐藏点...")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
print("定义隐藏点去除的参数")
camera = [0, 0, diameter]       # 视点位置
radius = diameter * 100         # 噪声点云半径,The radius of the sperical projection
_, pt_map = pcd.hidden_point_removal(camera, radius)    # 获取视点位置能看到的所有点的索引 pt_map

# 可视点点云
pcd_visible = pcd.select_by_index(pt_map)
pcd_visible.paint_uniform_color([0, 0, 1])	# 可视点为蓝色
print("->可视点个数为：", pcd_visible)
# 隐藏点点云
pcd_hidden = pcd.select_by_index(pt_map, invert = True)
pcd_hidden.paint_uniform_color([1, 0, 0])	# 隐藏点为红色
print("->隐藏点个数为：", pcd_hidden)
print("->正在可视化可视点和隐藏点点云...")
o3d.visualization.draw_geometries([pcd_visible, pcd_hidden])
