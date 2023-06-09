import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.fr5.fr5 as fr5
import visualization.panda.world as wd
import modeling.geometric_model as gm


def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)


if __name__ == '__main__':

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    # robot_s1
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, peg_attached=True)
    robot_s.fix_to(pos=[0, -1, 0], rotmat=rm.rotmat_from_euler(0,0,0))
    homeconf = np.radians([0, -60, 80, 15, 0, 15])
    robot_s.fk(component_name=component_name, jnt_values=homeconf)

    # xyz null space planning
    path = []
    for t in range(0, 500, 1):
        print("-------- timestep = ", t, " --------")
        fr5_jacob = robot_s.jacobian()
        # print(fr5_jacob)
        fr5_ns = rm.null_space(fr5_jacob[:3, :])
        # print(fr5_ns)
        cur_jnt_values = robot_s.get_jnt_values()
        cur_jnt_values += np.ravel(fr5_ns[:, 0]) * .01
        robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        if t % 20 == 0:
            path.append(cur_jnt_values)
            robot_s.gen_meshmodel(rgba=[1, 0, 0, .1]).attach_to(base)


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
    ########################################################################################################
    # robot_s2
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, peg_attached=False)
    robot_s.fix_to(pos=[0, 0, 0], rotmat=rm.rotmat_from_euler(0,0,0))
    homeconf = np.radians([0, -60, 80, 15, 0, 15])
    robot_s.fk(component_name=component_name, jnt_values=homeconf)

    # xyz null space planning
    path = []
    for t in range(0, 500, 1):
        print("-------- timestep = ", t, " --------")
        fr5_jacob = robot_s.jacobian()
        # print(fr5_jacob)
        fr5_ns = rm.null_space(fr5_jacob[:3, :])
        # print(fr5_ns)
        cur_jnt_values = robot_s.get_jnt_values()
        cur_jnt_values += np.ravel(fr5_ns[:, 1]) * .01
        robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        if t % 20 == 0:
            path.append(cur_jnt_values)
            robot_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)


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
    ########################################################################################################
    # robot_s3
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True,  peg_attached=False)
    robot_s.fix_to(pos=[0, 1, 0], rotmat=rm.rotmat_from_euler(0,0,0))
    homeconf = np.radians([0, -60, 80, 15, 0, 15])
    robot_s.fk(component_name=component_name, jnt_values=homeconf)

    # xyz null space planning
    path = []
    for t in range(0, 500, 1):
        print("-------- timestep = ", t, " --------")
        fr5_jacob = robot_s.jacobian()
        # print(fr5_jacob)
        fr5_ns = rm.null_space(fr5_jacob[:3, :])
        # print(fr5_ns)
        cur_jnt_values = robot_s.get_jnt_values()
        cur_jnt_values += np.ravel(fr5_ns[:, 2]) * .01
        robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        if t % 20 == 0:
            path.append(cur_jnt_values)
            robot_s.gen_meshmodel(rgba=[0, 0, 1, .1]).attach_to(base)


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
