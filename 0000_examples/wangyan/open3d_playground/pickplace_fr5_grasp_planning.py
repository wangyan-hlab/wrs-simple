import open3d as o3d
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import grasping.annotation.utils as gau
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as hnd

def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # object
    _relative_path = "pcd/dragon.ply"  # 设置相对路径
    N = 2000  # 将点划分为N个体素
    pcd1 = get_mesh(_relative_path).sample_points_poisson_disk(N)
    # -> Poisson surface reconstruction
    print('run Poisson surface reconstruction')
    radius = 0.1  # 搜索半径
    max_nn = 50  # 邻域内用于估算法线的最大点数
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 执行法线估计
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as hand:
        mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1, depth=6)
    print("Mesh：", mesh1)
    obj = cm.CollisionModel(mesh1)
    obj_name = 'dragon'
    obj.set_scale([.5, .5, .5])
    obj.set_rgba([1, 0, 0, 1])
    obj.attach_to(base)
    # hnd_s
    gripper_s = hnd.Robotiq85()
    grasp_info_list = gpa.plan_grasps(gripper_s, obj,
                                      openning_direction='loc_y',
                                      rotation_interval=np.radians(5),
                                      max_samples=100,
                                      min_dist_between_sampled_contact_points=0.02)
    # grasp_info_list = gau.define_grasp_with_rotation(gripper_s,
    #                                                  obj,
    #                                                  gl_jaw_center_pos=np.array([0,0,0.005]),
    #                                                  gl_jaw_center_z=np.array([0,0,-1]),
    #                                                  gl_jaw_center_y=np.array([0,1,0]),
    #                                                  jaw_width=0.015,
    #                                                  gl_rotation_ax=np.array([0,0,1]),
    #                                                  rotation_interval=np.radians(5))
    gpa.write_pickle_file(obj_name, grasp_info_list, './', 'fr5_'+obj_name+'.pickle')

    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, gl_jaw_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gripper_s.jaw_to(jaw_width)
        gripper_s.fix_to(hnd_pos, hnd_rotmat)
        gripper_s.gen_meshmodel(rgba=[.1,.1,.1,.2]).attach_to(base)
    base.run()
