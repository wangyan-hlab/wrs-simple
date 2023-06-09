import math
import time

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.fr5_link_ext.fr5_link_ext as fr5
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[-2, -3, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
    gm.gen_frame().attach_to(base)
    # object
    obj = cm.CollisionModel("../objects/bunnysim.stl")
    obj.set_pos(np.array([0.3, -0.25, 0.554]))
    obj.set_rpy(0, 0, 0)
    # obj.set_scale([4, 4, 4])
    obj.set_rgba([.1, .2, .8, 1])
    obj.attach_to(base)
    obj1 = cm.CollisionModel("../objects/bunnysim.stl")
    obj1.set_pos(np.array([-0.26, -0.32, 0.57]))
    obj1.set_rpy(0, 0, math.pi)
    # obj1.set_scale([2, 2, 2])
    obj1.set_rgba([.5, .9, .1, 1])
    obj1.attach_to(base)
    # robot_s
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True)
    robot_s.fix_to(pos=[0,0,0], rotmat=rm.rotmat_from_euler(0,0,0))
    start_conf = np.radians([120,-120,120,0,0,0,10])
    goal_conf = np.radians([0,-110,80,-80,-70,20,0])
    robot_s.fk(component_name, start_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.5]).attach_to(base)
    robot_s.fk(component_name, goal_conf)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0,1,0,0.5]).attach_to(base)
    # planner
    time_start = time.time()
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name=component_name,
                             start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=[obj, obj1],
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
