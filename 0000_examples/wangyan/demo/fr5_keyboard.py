import math
import time
import keyboard
import numpy as np
from visualization.panda import world as wd
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.fr5 import fr5 as fr5
from basis import robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
component_name = 'arm'
init_conf = np.array([0,-60,-130,10,10,0])*math.pi/180
robot_s = fr5.FR5_robot(homeconf=init_conf, peg_attached=True)
# robot_s.jaw_to(hnd_name="hnd", jaw_width=0.03)
# current_gripper_width = robot_s.hnd.get_jawwidth()
onscreen = []
operation_count = 0
pre_pos = np.array([0, 0, math.radians(0)])
pre_angle = math.radians(0)

def move(task):
    global onscreen, onscreen_agv, operation_count, pre_pos, pre_angle, current_gripper_width

    arm_linear_speed = .01
    arm_angular_speed = .01
    gripper_speed = .001
    for item in onscreen:
        item.detach()
    onscreen.clear()

    pressed_keys = {'r': keyboard.is_pressed('r'),  # x+ global
                    't': keyboard.is_pressed('t'),  # x- global
                    'f': keyboard.is_pressed('f'),  # y+ global
                    'g': keyboard.is_pressed('g'),  # y- global
                    'v': keyboard.is_pressed('v'),  # z+ global
                    'b': keyboard.is_pressed('b'),  # z- gglobal
                    'y': keyboard.is_pressed('y'),  # r+ global
                    'u': keyboard.is_pressed('u'),  # r- global
                    'h': keyboard.is_pressed('h'),  # p+ global
                    'j': keyboard.is_pressed('j'),  # p- global
                    'n': keyboard.is_pressed('n'),  # yaw+ global
                    'm': keyboard.is_pressed('m'),  # yaw- global
                    'o': keyboard.is_pressed('o'),  # gripper open
                    'p': keyboard.is_pressed('p')}  # gripper close
    values_list = list(pressed_keys.values())

    if any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm', 'o', 'p']) and \
            sum(values_list) == 1:  # global
        tic = time.time()
        current_jnt_values = robot_s.get_jnt_values()
        current_arm_tcp_pos, current_arm_tcp_rotmat = robot_s.get_gl_tcp()
        rel_pos = np.zeros(3)
        rel_rotmat = np.eye(3)
        rel_gripper_distance = np.zeros(2)

        if pressed_keys['r']:
            rel_pos = np.array([arm_linear_speed * .5, 0, 0])
        elif pressed_keys['t']:
            rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
        elif pressed_keys['f']:
            rel_pos = np.array([0, arm_linear_speed * .5, 0])
        elif pressed_keys['g']:
            rel_pos = np.array([0, -arm_linear_speed * .5, 0])
        elif pressed_keys['v']:
            rel_pos = np.array([0, 0, arm_linear_speed * .5])
        elif pressed_keys['b']:
            rel_pos = np.array([0, 0, -arm_linear_speed * .5])
        elif pressed_keys['y']:
            rel_rotmat = rm.rotmat_from_euler(arm_angular_speed * .5, 0, 0)
        elif pressed_keys['u']:
            rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed * .5, 0, 0)
        elif pressed_keys['h']:
            rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed * .5, 0)
        elif pressed_keys['j']:
            rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
        elif pressed_keys['n']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed * .5)
        elif pressed_keys['m']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed * .5)
        elif pressed_keys['o']:
            rel_gripper_distance = np.array([0, gripper_speed * .5])
        elif pressed_keys['p']:
            rel_gripper_distance = np.array([0, -gripper_speed * .5])
        new_arm_tcp_pos = current_arm_tcp_pos + rel_pos
        new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
        new_jnt_values = robot_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat,
                                    seed_jnt_values=current_jnt_values)
        if new_jnt_values is not None:
            robot_s.fk(component_name=component_name, jnt_values=new_jnt_values)
            print("current_jnt_values =", new_jnt_values)
        else:
            raise NotImplementedError("IK is unsolved!")

        current_gripper_width = current_gripper_width + rel_gripper_distance[1]
        print("current_gripper_width =", current_gripper_width)

        if 0 <= current_gripper_width <= 0.085:
            robot_s.hnd.jaw_to(current_gripper_width)
        else:
            raise NotImplementedError("The jaw width is out of range!")
        toc = time.time()

    onscreen.append(robot_s.gen_meshmodel())
    onscreen[-1].attach_to(base)
    operation_count += 1
    return task.cont

if __name__ == '__main__':

    gm.gen_frame(length=3, thickness=0.01).attach_to(base)
    taskMgr.doMethodLater(1/60, move, "move",
                          extraArgs=None,
                          appendTask=True)
    base.run()
