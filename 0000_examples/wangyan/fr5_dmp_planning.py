import copy
import math
import time

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.fr5.fr5 as fr5
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from dmp.dmp import DMPs_cartesian
from dmp.obstacle_superquadric import Obstacle_Dynamic as sq_dyn

def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[-2, -3, 1], lookat_pos=[0, 0, 0.5], w=960, h=720, backgroundcolor=[.8, .8, .8, .5])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, arm_jacobian_offset=np.array([0, 0, .165]), hnd_attached=True)
    robot_s.fix_to(pos=[0, 0, 0], rotmat=rm.rotmat_from_euler(0, 0, 0))

    if robot_s.hnd_attached:
        robot_s.jaw_to(jawwidth=0.001)

    start_conf = np.radians([0, -80, -120, -140, -90, 0])
    robot_s.fk(component_name, start_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, 0.3]).attach_to(base)
    start_pos = robot_s.get_gl_tcp(component_name)[0]
    temp_start_pos = copy.deepcopy(start_pos)
    start_orn = robot_s.get_gl_tcp(component_name)[1]
    t_des = np.linspace(0.0, 0.7, 50)
    gamma = np.transpose([t_des * np.cos(np.pi * t_des),
                          t_des * np.sin(np.pi * t_des)])
    pos_path = np.hstack((gamma, np.zeros((t_des.shape[0], 1)))) + start_pos

    org_path = [start_conf]
    for p in pos_path:
        genSphere(p, radius=0.002)
        conf = robot_s.get_jnt_values(component_name)
        orn = start_orn
        next_conf = robot_s.ik(tgt_pos=p, tgt_rotmat=orn, seed_jnt_values=conf,
                               local_minima="accept", max_niter=2000)
        robot_s.fk(component_name, next_conf)
        org_path.append(next_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 1, 0, 0.3]).attach_to(base)

    # ----- DMP Planning ----- #
    K = 1050.0
    dt = 0.005
    tol = 0.01
    MP = DMPs_cartesian(n_dmps=2, K=K, dt=dt, tol=tol)
    MP.imitate_path(t_des=t_des, x_des=gamma)
    # MP.x_goal += np.array([0.05, 0.0])      # a new goal
    gamma_track, _, _, _ = MP.rollout()
    t_des = np.linspace(0.0, 0.7, np.shape(gamma_track)[0])

    # Obstacle entity
    obj = cm.CollisionModel("../objects/ellipse.stl")
    obj_center = np.array([0.2, 0.3, -0.25])
    bounding = 0.0
    obj_axis = np.array([0.2, 0.3])
    scaling = np.array([0.2, 0.1])
    obj_axis = np.multiply(obj_axis, scaling) + np.ones(2)*bounding
    obj.set_pos(temp_start_pos + obj_center)
    obj.set_rgba([.4, .4, 1.0, 1.0])
    obj.set_scale([scaling[0], scaling[1], 0.5])
    obj.attach_to(base)

    obj_1 = cm.CollisionModel("../objects/ellipse.stl")
    obj_center_1 = np.array([0.0, 0.57, -0.25])
    bounding = 0.0
    obj_axis_1 = np.array([0.2, 0.3])
    scaling = np.array([0.3, 0.3])
    obj_axis_1 = np.multiply(obj_axis_1, scaling) + np.ones(2) * bounding
    obj_1.set_pos(temp_start_pos + obj_center_1)
    obj_1.set_rgba([1.0, .31, .31, 1.0])
    obj_1.set_scale([scaling[0], scaling[1], 0.5])
    obj_1.attach_to(base)
    # base.run()

    # Obstacle definition (2D)
    lmbda = 10.0
    beta = 2.0
    eta = 0.5
    obst_stat_vol = sq_dyn(center=obj_center[:2], axis=obj_axis, lmbda=lmbda, beta=beta, eta=eta)
    obst_stat_vol_1 = sq_dyn(center=obj_center_1[:2], axis=obj_axis_1[:2], lmbda=lmbda, beta=beta, eta=eta)

    def ext_f_vd(x, v):
        return obst_stat_vol.gen_external_force(x, v) + obst_stat_vol_1.gen_external_force(x, v)

    # Rollout with obstacles
    MP.reset_state()
    x_track_dynamic_volume = np.array([MP.x])
    ddx_track_dynamic_volume = np.array([MP.ddx])
    while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
        MP.step(external_force=ext_f_vd)
        x_track_dynamic_volume = np.append(x_track_dynamic_volume, np.array([MP.x]), axis=0)
        ddx_track_dynamic_volume = np.append(ddx_track_dynamic_volume, np.array([MP.ddx]), axis=0)

    dynamic_pos = np.hstack((x_track_dynamic_volume,
                            np.zeros((x_track_dynamic_volume.shape[0], 1)))) + start_pos

    robot_s.fk(component_name, start_conf)
    dmp_path = [start_conf]
    for p in dynamic_pos:
        genSphere(p, radius=0.002, rgba=[0, 1, 0, 1])
        conf = robot_s.get_jnt_values(component_name)
        orn = start_orn
        next_conf = robot_s.ik(tgt_pos=p, tgt_rotmat=orn, seed_jnt_values=conf)
        robot_s.fk(component_name, next_conf)
        dmp_path.append(next_conf)
    print("org_path_len = ", len(org_path))
    # Planning
    path = []
    rrtc_planner = rrtc.RRTConnect(robot_s)
    for i in range(len(dmp_path)-1):
        part_path = rrtc_planner.plan(component_name=component_name,
                                      start_conf=dmp_path[i],
                                      goal_conf=dmp_path[i+1],
                                      obstacle_list=[obj, obj_1],
                                      ext_dist=0.1,
                                      max_time=1000)
        for pp in part_path:
            path.append(pp)

    # # ----- Directly planning----- #
    # part_path = rrtc_planner.plan(component_name=component_name,
    #                          start_conf=org_path[0],
    #                          goal_conf=org_path[18],
    #                          obstacle_list=[obj],
    #                          ext_dist=0.005,
    #                          max_time=1000)
    # for pp in part_path:
    #     path.append(pp)
    # part_path = rrtc_planner.plan(component_name=component_name,
    #                               start_conf=org_path[18],
    #                               goal_conf=org_path[34],
    #                               obstacle_list=[obj],
    #                               ext_dist=0.005,
    #                               max_time=1000)
    # for pp in part_path:
    #     path.append(pp)
    # part_path = rrtc_planner.plan(component_name=component_name,
    #                               start_conf=org_path[34],
    #                               goal_conf=org_path[-1],
    #                               obstacle_list=[obj],
    #                               ext_dist=0.005,
    #                               max_time=1000)
    # for pp in part_path:
    #     path.append(pp)


    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            # genSphere(robot.get_gl_tcp(component_name)[0], radius=0.003, rgba=[1, 1, 0, 1])
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
