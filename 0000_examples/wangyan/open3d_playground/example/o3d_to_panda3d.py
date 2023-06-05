import copy
import open3d as o3d
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.data_adapter as da

# ---------------------- 定义点云体素化函数 ----------------------
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

if __name__ == '__main__':

    base = wd.World(cam_pos=[-1, -3, 1], lookat_pos=[0, 0, 0], w=960, h=720, backgroundcolor=[.8, .8, .8, .5])
    gm.gen_frame().attach_to(base)

    _relative_path = "../pcd/bunny.ply"  # 设置相对路径
    N = 2000       # 将点划分为N个体素
    pcd1 = get_mesh(_relative_path).sample_points_poisson_disk(N)
    # -> Poisson surface reconstruction
    print('run Poisson surface reconstruction')
    radius = 0.1      # 搜索半径
    max_nn = 50   #邻域内用于估算法线的最大点数
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 执行法线估计
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as hand:
        mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1, depth=16)
    print("Mesh：", mesh1)
    hand_cm1 = cm.CollisionModel(mesh1)
    hand_cm1.set_scale([1, 1, 1])
    hand_cm1.set_rgba([1, 0, 0, 1])
    hand_cm1.set_pos(np.array([0, 0, -0.1]))
    hand_cm1.attach_to(base)

    hand_cm2 = cm.CollisionModel(mesh1)
    hand_cm2.set_scale([1, 1, 1])
    hand_cm2.set_rgba([0, 0, 1, 1])
    hand_cm2.set_pos(np.array([0, 0, 0.1]))
    hand_cm2.set_rpy(np.pi/2, 0, 0)
    hand_cm2.show_cdprimit()
    hand_cm2.attach_to(base)

    base.run()
