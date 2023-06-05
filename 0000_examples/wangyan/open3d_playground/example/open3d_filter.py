import open3d as o3d
import numpy as np

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
print(pcd)

print("->正在可视化原始点云")
o3d.visualization.draw_geometries([pcd])

# --------------------------- 1.体素下采样 ---------------------------
# -> 体素下采样使用规则体素栅格从输入点云创建均匀下采样点云。该算法分两步操作：
# -> 将点云进行进行体素划分
# -> 对所有非空体素，取体素内点云的质心作为该体素的点的位置。

print("->正在体素下采样...")
voxel_size = 0.5
downpcd = pcd.voxel_down_sample(voxel_size)
print(downpcd)

print("->正在可视化下采样点云")
o3d.visualization.draw_geometries([downpcd])

# ------------------------- 2.统计滤波 --------------------------
# -> statistical_outlier_removal 会移除距离相邻点更远的点。它需要两个输入参数：
# -> num_neighbors，指定为了计算给定点的平均距离，需要考虑多少个邻居。即K邻域的点数
# -> std_ratio，允许根据点云平均距离的标准偏差设置阈值水平。该数值越低，滤除的点数就越多

print("->正在进行统计滤波...")
num_neighbors = 20 # K邻域点的个数
std_ratio = 2.0 # 标准差乘数
# 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
sor_pcd.paint_uniform_color([0, 0, 1])
print("统计滤波后的点云：", sor_pcd)
sor_pcd.paint_uniform_color([0, 0, 1])
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind, invert=True)
print("噪声点云：", sor_noise_pcd)
sor_noise_pcd.paint_uniform_color([1, 0, 0])

print("->可视化统计滤波后的点云和噪声点云")
o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])

# ------------------------- 3.半径滤波 --------------------------
print("->正在进行半径滤波...")
num_points = 100  # 邻域球内的最少点数，低于该值的点为噪声点
radius = 5    # 邻域半径大小
# 执行半径滤波，返回滤波后的点云sor_pcd和对应的索引ind
sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
sor_pcd.paint_uniform_color([0, 0, 1])
print("半径滤波后的点云：", sor_pcd)
sor_pcd.paint_uniform_color([0, 0, 1])
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", sor_noise_pcd)
sor_noise_pcd.paint_uniform_color([1, 0, 0])

print("->可视化半径滤波后的点云和噪声点云")
o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
