import threading
from threading import Thread, Lock, current_thread
import time
import numpy as np
from visualization.panda.world import World
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.ur5e_ballpeg import ur5e_ballpeg as ur5e
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from robot_con.ur.ur5e import UR5ERtqHE as ur5e_real
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton

class MyWorld(World):

    def __init__(self, cam_pos, lookat_pos, 
                 up=np.array([0, 0, 1]), fov=40, w=1920, h=1080, 
                 lens_type="perspective", toggle_debug=False, 
                 auto_cam_rotate=False, backgroundcolor=None):
        super().__init__(cam_pos, lookat_pos, up, fov, w, h, 
                         lens_type, toggle_debug, 
                         auto_cam_rotate, backgroundcolor)
        self.create_button()

        self.path = []
        self.endplanningtask = 1
        self.pathLock = Lock()
        self.robot = None
        self.robot_connect = None
        self.component_name = None
        self.start_end_conf = []
        self.start_meshmodel = None
        self.goal_meshmodel = None
        self.robot_meshmodel = None
        self.target_conf = None


    def set_ip(self, robot_ip, pc_ip):
        self.robot_ip = robot_ip    # '192.168.58.2'
        self.pc_ip = pc_ip          # '192.168.58.70'

    
    def create_button(self):
        button_get_jnts = DirectButton(text="Get Joint Values", 
                                       command=self.get_robot_jnts,
                                       scale=(0.05, 0.05, 0.05),
                                       pos=(1, 0, 0.5))
        button_record_pose = DirectButton(text="Record Pose",
                                          command=self.record_robot_pose,
                                          scale=(0.05, 0.05, 0.05),
                                          pos=(1, 0, 0.4))
        button_planning = DirectButton(text="Plan Path",
                                       command=self.planning,
                                       scale=(0.05, 0.05, 0.05),
                                       pos=(1, 0, 0.3))
        button_execute = DirectButton(text="Execute Path",
                                      command=lambda: self.run_realrobot(self.robot_connect),
                                      scale=(0.05, 0.05, 0.05),
                                      pos=(1, 0, 0.2))

    
    def get_robot_jnts(self):
        print("当前机器人关节位置:", np.rad2deg(self.robot.get_jnt_values()))
        return self.robot.get_jnt_values()
    
    
    def record_robot_pose(self):
        record_conf = self.robot.get_jnt_values()
        
        if len(self.start_end_conf) == 2:
            print("[Warning] 清空之前示教的起始点和目标点")
            self.start_meshmodel.detach()
            self.goal_meshmodel.detach()
            self.start_end_conf = []
            self.start_end_conf.append(record_conf)
            self.robot.fk(self.component_name, record_conf)
            self.start_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.3])
            self.start_meshmodel.attach_to(self)

        elif len(self.start_end_conf) == 1:
            print("[Info] 已示教目标点")
            self.start_end_conf.append(record_conf)
            self.target_conf = self.start_end_conf[1]
            self.robot.fk(self.component_name, record_conf)
            self.goal_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True, rgba=[0,1,0,0.3])
            self.goal_meshmodel.attach_to(self)

        elif len(self.start_end_conf) == 0:
            print("[Info] 已示教起始点")
            self.start_end_conf.append(record_conf)
            self.robot.fk(self.component_name, record_conf)
            self.start_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.3])
            self.start_meshmodel.attach_to(self)

        else:
            raise ValueError("[Error] start_end_conf[]的长度不能超过2")


    def planning(self):
        """
            Motion planning
        """

        if self.robot_meshmodel is not None:
            self.robot_meshmodel.detach()

        self.endplanningtask = 0
        time_start = time.time()
        if len(self.start_end_conf) == 2:
            [start_conf, goal_conf] = self.start_end_conf
            rrtc_planner = rrtc.RRTConnect(base.robot)
            self.path = rrtc_planner.plan(component_name=base.component_name,
                                    start_conf=start_conf,
                                    goal_conf=goal_conf,
                                    obstacle_list=[],
                                    ext_dist=0.1,
                                    max_time=300)
            time_end = time.time()
            print("Planning time = ", time_end-time_start)
            print(len(self.path), self.path)
            self.start_end_conf = []    # 规划完毕,清空起始点和目标点
            self.start_meshmodel.detach()
            self.goal_meshmodel.detach()

            # Motion planning animation
            rbtmnp = [None]
            motioncounter = [0]
            taskMgr.doMethodLater(0.1, self.update, "update",
                                extraArgs=[rbtmnp, motioncounter, self.robot, self.path, self.component_name], 
                                appendTask=True)
            
        elif len(self.start_end_conf) == 1:
            print("[Warning] 请示教机器人运动目标点")

        elif len(self.start_end_conf) == 0:
            print("[Warning] 请示教机器人运动起始点")
        

    def run_realrobot(self, real_robot):
        """
            Control the real robot
        """

        if real_robot:
            print("[Info] Robot connected")
            # Real robot
            robot_r = ur5e_real(robot_ip=self.robot_ip, pc_ip=self.pc_ip)
            robot_r.move_jnts(np.rad2deg(self.path[0]))
            # Control real robot to reproduce the planned path
            self.pathLock.acquire()
            if self.path:
                robot_r.move_jntspace_path(self.path) # TODO:check
            self.path = []
            self.pathLock.release()
            self.endplanningtask = 1
            self.robot.fk(self.component_name, self.path[-1])
            self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
            self.robot_meshmodel.attach_to(base)
        else:
            print("[Info] Robot NOT connected")
            print("[Info] 模拟真实机器人运行时间...")
            time.sleep(5)
            self.robot.fk(self.component_name, self.path[-1])
            self.path = []
            self.target_conf = None
            self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
            self.robot_meshmodel.attach_to(base) 
            self.endplanningtask = 1
        
        print("[Info] 机器人运行结束")


    def update(self, rbtmnp, motioncounter, robot, path, armname, task):
        """
            Motion planning path animation
        """

        if motioncounter[0] < len(path):
            # update simulated robot
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0.5, 0, 0.7])
            rbtmnp[0].attach_to(base)
            tcp_ball = gm.gen_sphere(pos=robot.get_gl_tcp(armname)[0], 
                                    radius=0.01, rgba=[1, 1, 0, 1])
            tcp_ball.attach_to(base)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0

        if self.endplanningtask == 1:
            rbtmnp[0].detach()
            print("[Info] Animation 结束")
            return task.done
        else:
            return task.again


    def move_keyboard(self, rbtonscreen, robot, armname, mode, task):
        """
            Use keyboard to control robot
        """

        if rbtonscreen[0] is not None:
            rbtonscreen[0].detach()

        if mode == 'cart':
            arm_linear_speed = .01
            arm_angular_speed = .01
            if self.target_conf is not None:
                robot.fk(armname, self.target_conf)
            cur_jnt_values = robot.get_jnt_values()
            cur_tcp_pos, cur_tcp_rotmat = robot.get_gl_tcp()
            rel_pos = np.zeros(3)
            rel_rotmat = np.eye(3)

            if base.inputmgr.keymap['r']:
                rel_pos = np.array([arm_linear_speed * .5, 0, 0])
            elif base.inputmgr.keymap['t']:
                rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
            elif base.inputmgr.keymap['f']:
                rel_pos = np.array([0, arm_linear_speed * .5, 0])
            elif base.inputmgr.keymap['g']:
                rel_pos = np.array([0, -arm_linear_speed * .5, 0])
            elif base.inputmgr.keymap['v']:
                rel_pos = np.array([0, 0, arm_linear_speed * .5])
            elif base.inputmgr.keymap['b']:
                rel_pos = np.array([0, 0, -arm_linear_speed * .5])
            elif base.inputmgr.keymap['y']:
                rel_rotmat = rm.rotmat_from_euler(arm_angular_speed * .5, 0, 0)
            elif base.inputmgr.keymap['u']:
                rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed * .5, 0, 0)
            elif base.inputmgr.keymap['h']:
                rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed * .5, 0)
            elif base.inputmgr.keymap['j']:
                rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
            elif base.inputmgr.keymap['n']:
                rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed * .5)
            elif base.inputmgr.keymap['m']:
                rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed * .5)

            new_tcp_pos = cur_tcp_pos + rel_pos
            new_tcp_rotmat = rel_rotmat.dot(cur_tcp_rotmat)
            new_jnt_values = robot.ik(tgt_pos=new_tcp_pos, 
                                    tgt_rotmat=new_tcp_rotmat,
                                    seed_jnt_values=cur_jnt_values)
            if new_jnt_values is not None:
                robot.fk(armname, new_jnt_values)
                rbtonscreen[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 0, 1, 0.5])
                rbtonscreen[0].attach_to(base)
            else:
                raise NotImplementedError("IK is unsolved!")
            
        elif mode == 'jnt':
            jnt_angular_speed = .01
            if self.target_conf is not None:
                print("强制到达目标")
                robot.fk(armname, self.target_conf)
            cur_jnt_values = robot.get_jnt_values()
            rel_jnt = np.zeros(6)

            if base.inputmgr.keymap['r']:
                rel_jnt = np.array([jnt_angular_speed * .5, 0, 0, 0, 0, 0])
            elif base.inputmgr.keymap['t']:
                rel_jnt = np.array([-jnt_angular_speed * .5, 0, 0, 0, 0, 0])
            elif base.inputmgr.keymap['f']:
                rel_jnt = np.array([0, jnt_angular_speed * .5, 0, 0, 0, 0])
            elif base.inputmgr.keymap['g']:
                rel_jnt = np.array([0, -jnt_angular_speed * .5, 0, 0, 0, 0])
            elif base.inputmgr.keymap['v']:
                rel_jnt = np.array([0, 0, jnt_angular_speed * .5, 0, 0, 0])
            elif base.inputmgr.keymap['b']:
                rel_jnt = np.array([0, 0, -jnt_angular_speed * .5, 0, 0, 0])
            elif base.inputmgr.keymap['y']:
                rel_jnt = np.array([0, 0, 0, jnt_angular_speed * .5, 0, 0])
            elif base.inputmgr.keymap['u']:
                rel_jnt = np.array([0, 0, 0, -jnt_angular_speed * .5, 0, 0])
            elif base.inputmgr.keymap['h']:
                rel_jnt = np.array([0, 0, 0, 0, jnt_angular_speed * .5, 0])
            elif base.inputmgr.keymap['j']:
                rel_jnt = np.array([0, 0, 0, 0, -jnt_angular_speed * .5, 0])
            elif base.inputmgr.keymap['n']:
                rel_jnt = np.array([0, 0, 0, 0, 0, jnt_angular_speed * .5])
            elif base.inputmgr.keymap['m']:
                rel_jnt = np.array([0, 0, 0, 0, 0, -jnt_angular_speed * .5])

            new_jnt_values = cur_jnt_values + rel_jnt

            if robot.is_jnt_values_in_ranges(self.component_name, new_jnt_values):
                robot.fk(armname, new_jnt_values)
                rbtonscreen[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 0, 1, 0.5])
                rbtonscreen[0].attach_to(base)
            else:
                print("The given joint angles are out of joint limits.")

        return task.again


if __name__ == "__main__":
    
    # WRS planning simulation
    base = MyWorld(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
    gm.gen_frame().attach_to(base)

    ## Simulated robot
    base.component_name = 'arm'
    base.robot = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    base.robot_meshmodel = base.robot.gen_meshmodel()
    base.robot_meshmodel.attach_to(base)

    ## Set start and goal joint values
    # start_jnts = [-111.817, -87.609, -118.858, -55.275, 107.847, 20.778]
    # goal_jnts = [-127.146, -74.498, -85.835, -40.605, 71.584, 20.790]

    base.robot_connect = False
    teach_mode = 'jnt'

    if base.robot_connect:
        base.set_ip(robot_ip='192.168.58.2', pc_ip='192.168.58.70')

    # Keyboard movement control
    rbtonscreen = [None]
    taskMgr.doMethodLater(0.02, base.move_keyboard, "move_keyboard", 
                        extraArgs=[rbtonscreen, base.robot, base.component_name, teach_mode], 
                        appendTask=True)


    base.setFrameRateMeter(True)
    base.run()
    