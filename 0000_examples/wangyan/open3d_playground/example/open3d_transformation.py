import copy  # 点云深拷贝
import open3d as o3d
import numpy as np

# -------------------------- 加载点云 ------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../pcd/ism_test_horse.pcd")
print(pcd)
print(f'pcd质心：{pcd.get_center()}')
# ===========================================================

# -------------------------- 点云平移 ------------------------
print("\n->沿X轴平移0.2m")
pcd_tx = copy.deepcopy(pcd).translate((15, 0, 0), relative=False)
pcd_tx.paint_uniform_color([1, 0, 0])
print(pcd_tx)
print(f'pcd_tx质心：{pcd_tx.get_center()}')

print("\n->沿Y轴平移0.2m")
pcd_ty = copy.deepcopy(pcd_tx).translate((0, 15, 0))
pcd_ty.paint_uniform_color([0, 1, 0])
print(pcd_ty)
print(f'pcd_ty质心：{pcd_ty.get_center()}')

print("\n->沿X轴平移-0.2m，再沿Y轴平移0.2m")
pcd_txy = copy.deepcopy(pcd).translate((-15, 15, 0))
pcd_txy.paint_uniform_color([0, 0, 1])
print(pcd_txy)
print('pcd_txy质心：', pcd_txy.get_center())

o3d.visualization.draw_geometries([pcd, pcd_tx, pcd_ty, pcd_txy])
# ===========================================================

# -------------------------- 点云旋转 ------------------------
print("\n->采用欧拉角进行点云旋转")
pcd_EulerAngle = copy.deepcopy(pcd)
R1 = pcd.get_rotation_matrix_from_xyz((0, np.pi/2, 0))  # 从欧拉角 Euler angles 中转换
# R1 = pcd.get_rotation_matrix_from_axis_angle((0, np.pi/2, 0))  # 从轴角表示法 Axis-angle representation 中转换
# R1 = pcd.get_rotation_matrix_from_quaternion((1, 0, np.pi/2, 0))  # 从四元数 Quaternions 中转换
print("旋转矩阵：\n",R1)
# pcd_EulerAngle.rotate(R1)    # 不指定旋转中心
pcd_EulerAngle.rotate(R1, center=(5, 5, 5))    # 指定旋转中心
pcd_EulerAngle.paint_uniform_color([0, 0, 1])
print("\n->pcd_EulerAngle质心：", pcd_EulerAngle.get_center())

o3d.visualization.draw_geometries([pcd, pcd_EulerAngle])
# ===========================================================

# -------------------------- 点云缩放 ------------------------
print("\n->点云缩放")
pcd_scale1 = copy.deepcopy(pcd)
pcd_scale1.scale(1.5,center=pcd.get_center())
pcd_scale1.paint_uniform_color([0,0,1])
print("->pcd_scale1质心：",pcd_scale1.get_center()) # 缩放前后质心不变

pcd_scale2 = copy.deepcopy(pcd)
pcd_scale2.scale(0.5,center=(1,1,1))	# 自定义缩放后的质心
pcd_scale2.paint_uniform_color([0,1,0])
print("->pcd_scale2质心：",pcd_scale2.get_center())

o3d.visualization.draw_geometries([pcd, pcd_scale1,pcd_scale2])
# ===========================================================

# -------------------------- transform ------------------------
print("\n->点云的一般变换")
pcd_T = copy.deepcopy(pcd)
T = np.eye(4)
T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi/6,np.pi/4,0))   # 旋转矩阵
T[0, 3] = 20.0    # 平移向量的dx
T[1, 3] = 30.0    # 平移向量的dy
print("\n->变换矩阵：\n",T)
pcd_T.transform(T)
pcd_T.paint_uniform_color([0, 0, 1])
print("\n->pcd_scale1质心：", pcd_T.get_center())

o3d.visualization.draw_geometries([pcd, pcd_T])
# ===========================================================
