"""
========================
Test Movement Primitives
========================

This example tests the movement_primitives package from https://github.com/dfki-ric/movement_primitives.

A trajectory is created manually, imitated with a Cartesian DMP, converted
to a joint trajectory by inverse kinematics, and executed with a 6-axis robot.
"""
print(__doc__)

import numpy as np
from visualization.panda import world as wd
from modeling import geometric_model as gm
from robot_sim.robots.fr5 import fr5 as fr5
from basis import robot_math as rm
from movement_primitives.dmp import CartesianDMP

if __name__ == '__main__':

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, peg_attached=False)

    # Start
    start_conf = robot_s.get_jnt_values()
    start_pos, start_rotmat = robot_s.get_gl_tcp()
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, 0.2]).attach_to(base)

    # Goal
    goal_conf = np.radians([30, -45, 30, 0, 50, 20])
    robot_s.fk(component_name=component_name, jnt_values=goal_conf)
    goal_pos, goal_rotmat = robot_s.get_gl_tcp()
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 0, 1, 0.2]).attach_to(base)

    # Interplate
    pos_list, rotmat_list = rm.interplate_pos_rotmat(start_pos, start_rotmat, goal_pos, goal_rotmat)
    n_steps = len(pos_list)
    dt = 0.01
    execution_time = (n_steps - 1) * dt
    T = np.linspace(0, execution_time, n_steps)
    Y = np.empty((n_steps, 7))
    for i, (pos, rotmat) in enumerate(zip(pos_list, rotmat_list)):
        rot_q = rm.quaternion_from_matrix(rotmat)   # quaternion = (w, x, y, z)
        Y[i] = (np.concatenate((pos, rot_q)))
    # print(len(Y), "path_to_imitate = ", Y)

    # DMP
    dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)
    dmp.imitate(T, Y)

    new_goal_pos = Y[-1][:3] + np.array([0, 0, -0.05])
    new_goal_rotq = rm.quaternion_from_axangle(np.pi/6, axis=[0, 1, 0])
    new_goal = np.concatenate((new_goal_pos, new_goal_rotq))
    dmp.goal_y = new_goal
    _, Y = dmp.open_loop()
    # print(len(Y), "imitated_path = ", Y)

    path = []
    for pt in Y:
        pos = pt[:3]
        rotmat = rm.rotmat_from_quaternion(pt[3:])[:3, :3]
        path.append(robot_s.ik(component_name=component_name, tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=start_conf))
        gm.gen_frame(pos=pos, rotmat=rotmat, length=0.05).attach_to(base)
        gm.gen_sphere(pos=pos, radius=0.005, rgba=[1, 1, 0, 1]).attach_to(base)

    robot_s.fk(component_name=component_name, jnt_values=path[-1])
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 1, 0, 0.2]).attach_to(base)

    # Animation
    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
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


