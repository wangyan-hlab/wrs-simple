import numpy as np
import basis.robot_math as rm
import robot_sim.robots.fr5_link_ext.fr5_link_ext as fr5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import argparse

def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)


if __name__ == '__main__':

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    # robot_s1
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True)
    homeconf = np.radians([60,-40,80,-80,-70,80,90])
    # robot_s.gen_meshmodel(rgba=[0,1,0,.4]).attach_to(base)
    robot_s.fk(component_name=component_name, jnt_values=homeconf)
    # robot_s.gen_meshmodel().attach_to(base)

    # null space planning
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="all")  # mode = "xyz", "rpy", or "all"
    args = parser.parse_args()
    mode = args.mode

    path = []
    for t in range(0, 1000, 1):
        print("-------- timestep = ", t, " --------")
        fr5_jacob = robot_s.jacobian()
        print(fr5_jacob)
        if mode == "all":
            fr5_ns = rm.null_space(fr5_jacob)
        elif mode == "xyz":
            fr5_ns = rm.null_space(fr5_jacob[:3, :])
        elif mode == "rpy":
            fr5_ns = rm.null_space(fr5_jacob[3:, :])
        else:
            raise ValueError("Unexpected MODE!")
        print(fr5_ns)
        cur_jnt_values = robot_s.get_jnt_values()
        cur_jnt_values += np.ravel(fr5_ns[:, 0]) * .01
        robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        if t % 10 == 0:
            path.append(cur_jnt_values)
            # robot_s.gen_meshmodel(rgba=[1, 0, 0, .1]).attach_to(base)

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
