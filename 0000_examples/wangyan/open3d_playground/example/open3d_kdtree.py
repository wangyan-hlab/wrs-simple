import open3d as o3d
import numpy as np

print("-> Loading pcd1 ...")
pcd1 = o3d.io.read_point_cloud("../pcd/ism_test_horse.pcd")
print(pcd1)

print("-> Loading pcd2 ...")
pcd2 = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
print(pcd2)
# 将点云设置为灰色
pcd2.paint_uniform_color([0.8, 0.8, 0.8])
# 建立KDTree
pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
# 将第1500个点设置为绿色
pcd2.colors[1500] = [0, 0, 0]

# 法线估计
radius = 20  # 搜索半径
max_nn = 200     # 邻域内用于估算法线的最大点数
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计

# 使用K近邻，将第1500个点最近的2000个点设置为蓝色
print("使用K近邻，将第1500个点最近的2000个点设置为蓝色")
k = 2000
[num_k, idx_k, _] = pcd2_tree.search_knn_vector_3d(pcd2.points[1500], k)
np.asarray(pcd2.colors)[idx_k[1:], :] = [0, 0, 1]    # 跳过最近邻点（查询点本身）进行赋色
print("k邻域内的点数为：", num_k)

# 使用半径R近邻，将第500个点半径（0.02）范围内的点设置为红色
print("使用半径R近邻，将第1500个点半径50范围内的点设置为红色")
radius = 50   # 设置半径大小
[num_radius, idx_radius, _] = pcd2_tree.search_radius_vector_3d(pcd2.points[1500], radius)   # 返回邻域点的个数和索引
np.asarray(pcd2.colors)[idx_radius[1:], :] = [1, 0, 0]  # 跳过最近邻点（查询点本身）进行赋色
print("半径r邻域内的点数为：", num_radius)

# 使用混合邻域，将半径R邻域内不超过max_num个点设置为绿色
print("使用混合邻域，将第1500个点半径20邻域内不超过max_num个点设置为绿色")
radius = 20
max_num = 200   # 半径R邻域内最大点数
[num_hybrid, idx_hybrid, _] = pcd2_tree.search_hybrid_vector_3d(pcd2.points[1500], radius, max_num)
np.asarray(pcd2.colors)[idx_hybrid[1:], :] = [0, 1, 0]  # 跳过最近邻点（查询点本身）进行赋色
print("混合邻域内的点数为：", num_hybrid)

print("-> Visualizing 2 pcd ...")
o3d.visualization.draw_geometries(geometry_list=[pcd2],
                                  window_name='Open3D',
                                  width=960, height=540,
                                  left=30, top=30,
                                  point_show_normal=False,
                                  mesh_show_wireframe=False,
                                  mesh_show_back_face=False)

# print("-> Saving pcd")
# o3d.io.write_point_cloud("write.ocd", pcd, True)
# print(pcd)
