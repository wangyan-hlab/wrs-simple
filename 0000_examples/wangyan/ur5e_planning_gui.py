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
from direct.gui.DirectGui import DirectButton, DirectOptionMenu, DirectSlider, DirectLabel

class MyWorld(World):

    def __init__(self, cam_pos, lookat_pos, 
                 up=np.array([0, 0, 1]), fov=40, w=1920, h=1080, 
                 lens_type="perspective", toggle_debug=False, 
                 auto_cam_rotate=False, backgroundcolor=None,
                 robot_connect=False, 
                 robot_ip='192.168.58.2', 
                 pc_ip='192.168.58.70'):
        super().__init__(cam_pos, lookat_pos, up, fov, w, h, 
                         lens_type, toggle_debug, 
                         auto_cam_rotate, backgroundcolor)

        self.path = []              # planning result
        self.endplanningtask = 1    # flag to stop animation
        self.robot = None           # sim robot
        self.robot_r = None         # real robot
        self.robot_plan = None      # visual robot for animation
        self.robot_teach = None     # visual robot for teaching
        self.robot_connect = None
        self.component_name = None
        self.start_end_conf = []    # saving teached start and goal confs
        self.start_meshmodel = None 
        self.goal_meshmodel = None
        self.robot_meshmodel = None
        self.teaching_mode = 'Joint'
        self.slider_values = []
        self.robot_connect = robot_connect
        self.robot_ip = robot_ip
        self.pc_ip = pc_ip

        if self.robot_connect:
            print("[Info] 机器人已连接")
            self.robot_r = ur5e_real(robot_ip=self.robot_ip, pc_ip=self.pc_ip)
            self.init_conf = self.robot_r.get_jnt_values()  # 实际机器人的初始关节角度
        else:
            print("[Info] 机器人未连接")
            self.init_conf = np.zeros(6)

        self.create_button()
        self.create_option_menu()
        self.create_sliders()

    
    def create_button(self):
        button_get_jnts = DirectButton(text="Get Joint Values",
                                       text_pos=(0, -0.4),
                                       command=self.get_robot_jnts,
                                       scale=(0.05, 0.05, 0.05),
                                       frameSize=(-5, 5, -1, 1),
                                       pos=(1, 0, 0.5))
        button_record_pose = DirectButton(text="Record Pose",
                                          text_pos=(0, -0.4),
                                          command=self.record_robot_pose,
                                          scale=(0.05, 0.05, 0.05),
                                          frameSize=(-5, 5, -1, 1),
                                          pos=(1, 0, 0.4))
        button_planning = DirectButton(text="Plan Path",
                                       text_pos=(0, -0.4), 
                                       command=self.planning,
                                       scale=(0.05, 0.05, 0.05),
                                       frameSize=(-5, 5, -1, 1),
                                       pos=(1, 0, 0.3))
        button_execute = DirectButton(text="Execute Path",
                                      text_pos=(0, -0.4), 
                                      command=lambda: self.run_realrobot(self.robot_connect),
                                      scale=(0.05, 0.05, 0.05),
                                      frameSize=(-5, 5, -1, 1),
                                      pos=(1, 0, 0.2))

    
    def create_option_menu(self):
        label = DirectLabel(text="Teaching Mode",
                            scale=0.05,
                            pos=(0.5, 0, 0.6))
        options = ["Joint", "Cartesian"]
        self.option_menu = DirectOptionMenu(text="Teaching Mode",
                                            text_pos=(-1, -0.4),
                                            scale=(0.05, 0.05, 0.05),
                                            frameSize=(-5, 5, -1, 1),
                                            pos=(1, 0, 0.6),
                                            items=options,
                                            initialitem=0)
    

    def create_sliders(self):
        slider_init = np.rad2deg(self.init_conf)
        slider_values = list(slider_init)  # 初始滑动条值
        for i in range(6):
            label = DirectLabel(text="Joint {}".format(i+1),
                                scale=0.05,
                                pos=(0.5, 0, 0.1 - i * 0.1))
            slider = DirectSlider(range=(-180, 180),
                                  value=slider_values[i],
                                  scale=(0.3, 0.5, 0.2),
                                  pos=(1, 0, 0.1 - i * 0.1),
                                  extraArgs=[i])  # 传递滑动条索引作为额外参数
            self.slider_values.append(slider)  # 存储滑动条实例


    def get_robot_jnts(self):
        print("当前机器人关节位置(deg):", np.rad2deg(self.robot.get_jnt_values()))
        return self.robot.get_jnt_values()
    
    
    def record_robot_pose(self):
        record_conf = self.robot_teach.get_jnt_values()
        
        if len(self.start_end_conf) == 2:
            print("[Warning] 清空之前示教的起始点和目标点")
            self.start_meshmodel.detach()
            self.goal_meshmodel.detach()
            self.start_end_conf = []
            self.start_end_conf.append(record_conf)
            self.robot_teach.fk(self.component_name, record_conf)
            self.start_meshmodel = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.3])
            self.start_meshmodel.attach_to(self)

        elif len(self.start_end_conf) == 1:
            print("[Info] 已示教目标点")
            self.start_end_conf.append(record_conf)
            self.robot_teach.fk(self.component_name, record_conf)
            self.goal_meshmodel = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[0,1,0,0.3])
            self.goal_meshmodel.attach_to(self)

        elif len(self.start_end_conf) == 0:
            print("[Info] 已示教起始点")
            self.start_end_conf.append(record_conf)
            self.robot_teach.fk(self.component_name, record_conf)
            self.start_meshmodel = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.3])
            self.start_meshmodel.attach_to(self)

        else:
            raise ValueError("[Error] start_end_conf[]的长度不能超过2")


    def planning(self):
        """
            Motion planning
        """

        self.endplanningtask = 0    # flag to start animation
        time_start = time.time()
        if len(self.start_end_conf) == 2:
            [start_conf, goal_conf] = self.start_end_conf
            rrtc_planner = rrtc.RRTConnect(base.robot_plan)
            self.path = rrtc_planner.plan(component_name=base.component_name,
                                          start_conf=start_conf,
                                          goal_conf=goal_conf,
                                          obstacle_list=[],
                                          ext_dist=0.05,
                                          max_time=300)
            time_end = time.time()
            print("Planning time = ", time_end-time_start)
            print(len(self.path), self.path)

            self.start_meshmodel.detach()
            self.goal_meshmodel.detach()

            # Motion planning animation
            rbtmnp = [None]
            motioncounter = [0]
            taskMgr.doMethodLater(0.1, self.update, "update",
                                extraArgs=[rbtmnp, motioncounter, self.robot_plan, self.path, self.component_name], 
                                appendTask=True)
            
        elif len(self.start_end_conf) == 1:
            print("[Warning] 请示教机器人运动目标点")

        elif len(self.start_end_conf) == 0:
            print("[Warning] 请示教机器人运动起始点")
        

    def run_realrobot(self, real_robot):
        """
            Control the real robot
        """

        if self.path:
            if self.robot_meshmodel is not None:
                self.robot_meshmodel.detach()
            if real_robot:
                print("[Info] Robot connected")
                self.endplanningtask = 1
                self.robot.fk(self.component_name, self.path[-1])
                self.robot_teach.fk(self.component_name, self.path[-1])
                self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
                self.robot_meshmodel.attach_to(base)
                # Real robot
                self.robot_r.move_jnts(np.rad2deg(self.path[0]))
                # Control real robot to reproduce the planned path
                if self.path:
                    self.robot_r.move_jntspace_path(self.path) # TODO:check
                self.path = []
                self.start_end_conf = []
                
            else:
                print("[Info] Robot NOT connected")
                self.endplanningtask = 1
                self.robot.fk(self.component_name, self.path[-1])
                self.robot_teach.fk(self.component_name, self.path[-1])
                self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
                self.robot_meshmodel.attach_to(base)
                print("[Info] 模拟真实机器人运行时间...")
                time.sleep(5)
                self.path = []
                self.start_end_conf = []
            
            print("[Info] 机器人运行结束")
        
        else:
            print("[Info] No path yet!")


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


    def move_keyboard(self, rbtonscreen, robot, armname, task):
        """
            Use keyboard to control robot
        """

        jnt_values = np.zeros(6)
        for i in range(6):
            jnt_values[i] = self.slider_values[i].getValue()
        robot.fk(armname, np.deg2rad(jnt_values))
        
        if rbtonscreen[0] is not None:
            rbtonscreen[0].detach()
        self.teaching_mode = self.option_menu.get()

        if self.teaching_mode == 'Cartesian':
            arm_linear_speed = 0.01
            arm_angular_speed = np.deg2rad(1)
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
                for i in range(6):
                    self.slider_values[i].setValue(np.rad2deg(new_jnt_values)[i])
            else:
                raise NotImplementedError("IK is unsolved!")
            
        elif self.teaching_mode == 'Joint':
            jnt_angular_speed = np.deg2rad(1)
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
                for i in range(6):
                    self.slider_values[i].setValue(np.rad2deg(new_jnt_values)[i])
            else:
                print("The given joint angles are out of joint limits.")

        return task.again


if __name__ == "__main__":
    
    # WRS planning simulation
    robot_connect = True
    robot_ip = '192.168.58.2'
    pc_ip = '192.168.58.70'
    base = MyWorld(cam_pos=[3, 3, 1], lookat_pos=[0, .5, 0], w=1960, h=1280,
                   robot_connect=robot_connect, robot_ip=robot_ip, pc_ip=pc_ip)
    gm.gen_frame().attach_to(base)

    ## Simulated robot
    base.component_name = 'arm'
    base.robot = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    base.robot_meshmodel = base.robot.gen_meshmodel()
    base.robot_meshmodel.attach_to(base)
    base.robot_teach = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    base.robot_plan = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)

    if base.robot_connect:
        print("[Info] 机器人初始角度(deg): ", np.rad2deg(base.init_conf))
        base.robot.fk(base.component_name, np.asarray(base.init_conf))

        if base.robot_meshmodel is not None:
            base.robot_meshmodel.detach()
        base.robot_meshmodel = base.robot.gen_meshmodel()
        base.robot_meshmodel.attach_to(base)

    # Keyboard movement control
    rbtonscreen = [None]
    taskMgr.doMethodLater(0.02, base.move_keyboard, "move_keyboard", 
                        extraArgs=[rbtonscreen, base.robot_teach, base.component_name], 
                        appendTask=True)

    base.setFrameRateMeter(True)
    base.run()
    