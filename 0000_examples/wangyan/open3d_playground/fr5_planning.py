import math
import time
import open3d as o3d
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.fr5.fr5 as fr5
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[-2, -3, 1], lookat_pos=[0, 0, 0.5], w=960, h=720, backgroundcolor=[.8, .8, .8, .5])
    gm.gen_frame().attach_to(base)
    # object
    _relative_path = "pcd/man_hand.ply"  # 设置相对路径
    N = 2000  # 将点划分为N个体素
    pcd1 = get_mesh(_relative_path).sample_points_poisson_disk(N)
    # -> Poisson surface reconstruction
    print('run Poisson surface reconstruction')
    radius = 10  # 搜索半径
    max_nn = 200  # 邻域内用于估算法线的最大点数
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 执行法线估计
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as hand:
        mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1, depth=6)
    print("Mesh：", mesh1)
    hand_cm1 = cm.CollisionModel(mesh1)
    hand_cm1.set_scale([.001, .001, .001])
    hand_cm1.set_rgba([1, 0, 0, 1])
    hand_cm1.set_pos(np.array([0.1, -0.35, 0.75]))
    hand_cm1.set_rpy(np.pi/2, 0, 0)
    hand_cm1.attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, arm_jacobian_offset=np.array([0, 0, .145]), hnd_attached=True)
    # robot_s = fr5.FR5_robot(enable_cc=True)
    robot_s.fix_to(pos=[0, 0, 0], rotmat=rm.rotmat_from_euler(0, 0, 0))

    if robot_s.hnd_attached:
        robot_s.jaw_to(jawwidth=0.01)

    start_conf = np.radians([120, -120, 70, 0, 0, 0])
    goal_conf = np.radians([0, -110, 80, -80, -70, 20])
    robot_s.fk(component_name, start_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, 0.5]).attach_to(base)
    robot_s.fk(component_name, goal_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 1, 0, 0.5]).attach_to(base)
    # planner
    time_start = time.time()
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name=component_name,
                             start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=[hand_cm1],
                             ext_dist=0.1,
                             max_time=300)
    time_end = time.time()
    print("Planning time = ", time_end-time_start)

    print(path)
    # for pose in path:
    #     # print(pose)
    #     robot_s.fk(component_name, pose)
    #     robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    #     robot_meshmodel.attach_to(base)
    #     robot_stickmodel = robot_s.gen_stickmodel()
    #     robot_stickmodel.attach_to(base)

    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            genSphere(robot.get_gl_tcp(component_name)[0], radius=0.01, rgba=[1, 1, 0, 1])
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again
    rbtmnp = [None]
    motioncounter = [0]
    taskMgr.doMethodLater(0.07, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], appendTask=True)
    base.setFrameRateMeter(True)
    base.run()
