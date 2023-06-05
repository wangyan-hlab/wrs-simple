import open3d as o3d
import numpy as np

# ---------------------- 定义点云体素化函数 ----------------------
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

# --------------------------- 加载点云 ---------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_michael.pcd")
print("原始点云：", pcd)
# ==============================================================

# ------------------------- 1.Alpha shapes -----------------------
alpha = 8
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
print("Mesh：", mesh)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# =============================================================

# ------------------------- 2.Ball pivoting --------------------------
print("->Ball pivoting...")
_relative_path = "../pcd/man_hand.ply"  # 设置相对路径
N = 2000                       # 将点划分为N个体素
pcd = get_mesh(_relative_path).sample_points_poisson_disk(N)
o3d.visualization.draw_geometries([pcd])

radii = [5, 10, 20, 30]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])

# ------------------------- 3.Poisson surface reconstruction -------------------------
# -> 直接读取点云的方法
print('run Poisson surface reconstruction')
radius = 10   # 搜索半径
max_nn = 200  # 邻域内用于估算法线的最大点数
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
print(mesh)
o3d.visualization.draw_geometries([mesh])

# -> mesh方法
print("->Poisson surface reconstruction...")
_relative_path = "../pcd/man_hand.ply"  # 设置相对路径
N = 2000                        # 将点划分为N个体素
pcd = get_mesh(_relative_path).sample_points_poisson_disk(N)
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 使现有法线无效

# 法线估计
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
pcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# 泊松重建
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
print(mesh)
o3d.visualization.draw_geometries([mesh])
