"""
This example tests the DMP reproducing a trajectory along a curve surface recorded by ROS.
"""
print(__doc__)

import math
import numpy as np
import basis.robot_math as rm
from visualization.panda import world as wd
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.ur5e_ballpeg import ur5e_ballpeg as ur5e
from motion.probabilistic import rrt_connect as rrtc
from movement_primitives.dmp import CartesianDMP
import pickle

def genSphere(pos, radius=0.005, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[2, 0, 1.3], lookat_pos=[1, 0, 1.2], w=960, h=720, backgroundcolor=[.7, .7, .7, 1])
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=True)
    robot_s.fix_to(pos=np.array([.11, .0, .69]), rotmat=rm.rotmat_from_euler(0, 0, -1.5707))
    start_jnt = np.radians([90.91, -97.36, 113.41, -106.04, -90.0, 0.89])
    robot_s.fk(component_name=component_name, jnt_values=start_jnt)
    robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, .3])
    robot_meshmodel.attach_to(base)

    table = cm.CollisionModel("../objects/task_table.stl")
    table.set_pos(np.array([.8, .0, 0.95]))
    table.set_rgba([.6, .6, .6, 1.0])
    table.attach_to(base)

    curve_surface = cm.CollisionModel("../objects/curve_surface.stl")
    curve_surface.set_pos(np.array([0.55, 0.02, 0.975]))
    curve_surface.set_rpy(0, 0, np.pi/2)
    curve_surface.set_rgba([.69, .56, .44, 1.0])
    curve_surface.attach_to(base)

    old_path = pickle.load(open("./trajectory.pickle", 'rb'))
    new_path = []
    for index, p in enumerate(old_path):
        if index % 10 == 0:
            new_path.append(old_path[index])

    n_steps = len(new_path)
    dt = 0.01
    execution_time = (n_steps - 1) * dt
    T = np.linspace(0, execution_time, n_steps)
    Y = np.empty((n_steps, 7))

    for i, jnt_value in enumerate(new_path):
        robot_s.fk(component_name='arm', jnt_values=jnt_value)
        pos = robot_s.get_gl_tcp(component_name)[0]
        rotmat = robot_s.get_gl_tcp(component_name)[1]
        rot_q = rm.quaternion_from_matrix(rotmat)
        Y[i] = (np.concatenate((pos, rot_q)))
        genSphere(pos, radius=0.002, rgba=[1, 0, 0, 1])

    # DMP
    dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)
    dmp.imitate(T, Y)
    new_start_pos = Y[0][:3] + np.array(([0.08, 0, 0]))
    new_start_rotq = Y[0][3:]
    new_start = np.concatenate((new_start_pos, new_start_rotq))
    new_goal_pos = Y[-1][:3] + np.array(([0.05, 0, 0]))
    new_goal_rotq = Y[-1][3:]
    new_goal = np.concatenate((new_goal_pos, new_goal_rotq))
    dmp.start_y = new_start
    dmp.goal_y = new_goal
    _, Y = dmp.open_loop()

    dmp_path = []
    for pt in Y:
        pos = pt[:3]
        rotmat = rm.rotmat_from_quaternion(pt[3:])[:3, :3]
        dmp_path.append(
            robot_s.ik(component_name=component_name,
                       tgt_pos=pos,
                       tgt_rotmat=rotmat,
                       seed_jnt_values=start_jnt)
        )

        gm.gen_sphere(pos=pos, radius=0.002, rgba=[0, 1, 0, 1]).attach_to(base)

    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            # genSphere(robot.get_gl_tcp(component_name)[0], radius=0.002, rgba=[1, 0, 1, 1])
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again


    rbtmnp = [None]
    motioncounter = [0]
    taskMgr.doMethodLater(0.07, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot_s, dmp_path, component_name], appendTask=True)
    base.setFrameRateMeter(True)
    base.run()
