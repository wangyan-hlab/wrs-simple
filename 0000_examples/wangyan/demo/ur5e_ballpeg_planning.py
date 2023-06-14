import os
import math
import time
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xa
import robot_sim.robots.ur5e.ur5e as ur5e
import visualization.panda.world as wd
import modeling.geometric_model as gm
from motion.probabilistic import rrt_connect as rrtc

def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = ur5e.ROBOT(enable_cc=True, peg_attached=False)
    robot_s.gen_meshmodel().attach_to(base)
    robot_s.show_cdprimit()
    # base.run()

    start_conf = np.deg2rad([0,-90,60,0,0,0])
    goal_conf = np.deg2rad([60,-90,0,0,0,0])
    robot_s.fk(component_name, goal_conf)
    robot_s.gen_meshmodel().attach_to(base)

    # planner
    time_start = time.time()
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name=component_name,
                                start_conf=start_conf,
                                goal_conf=goal_conf,
                                obstacle_list=[],
                                ext_dist=0.1,
                                max_time=300)
    time_end = time.time()
    print("Planning time = ", time_end-time_start)
    print(len(path), path)

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
    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], appendTask=True)
    base.setFrameRateMeter(True)
    base.run()
    