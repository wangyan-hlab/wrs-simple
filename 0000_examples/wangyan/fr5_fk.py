import math
import numpy as np
from visualization.panda import world as wd
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.fr5 import fr5 as fr5
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm

def genSphere(pos, radius=0.005, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = fr5.FR5_robot(enable_cc=True, peg_attached=False)
    robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_meshmodel.attach_to(base)
    # jnt limits-
    #           |- jnt1:[-175, 175], jnt2:[-265, 85], jnt3:[-160, 160],
    #           |- jnt4:[-265, 85], jnt5:[-175, 175], jnt6:[-175, 175]
    limits = [[-175, 175], [-265, 85], [-160, 160], [-265, 85], [-175, 175], [-175, 175]]
    jnt1, jnt2, jnt3, jnt4, jnt5, jnt6 = 0, 0, 0, 0, 0, 0
    interval = 45
    for jnt1 in range(limits[0][0], limits[0][1], interval):
        for jnt2 in range(limits[1][0], limits[1][1], interval):
            for jnt3 in range(limits[2][0], limits[2][1], interval):
                for jnt4 in range(limits[3][0], limits[3][1], interval):
                    # for jnt5 in range(limits[4][0], limits[4][1], interval):

                        goal_conf = np.array([jnt1, jnt2, jnt3, jnt4, jnt5, jnt6])*math.pi/180
                        robot_s.fk(component_name, goal_conf)
                        if not robot_s.is_collided():
                            genSphere(robot_s.get_gl_tcp(component_name)[0])
                            # robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
                            # robot_meshmodel.attach_to(base)

    base.run()
