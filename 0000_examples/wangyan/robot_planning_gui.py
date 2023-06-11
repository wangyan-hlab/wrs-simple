import time
import numpy as np
from visualization.panda.world import World
from modeling import geometric_model as gm
from modeling import collision_model as cm
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectOptionMenu, \
        DirectSlider, DirectLabel, DirectEntry, DirectFrame


class FastSimWorld(World):

    def __init__(self, cam_pos=[3, 3, 1], lookat_pos=[0, .5, 0],
                 up=np.array([0, 0, 1]), fov=40, w=1920, h=1080, 
                 lens_type="perspective", toggle_debug=False, 
                 auto_cam_rotate=False, backgroundcolor=None,
                 robot_connect=False,
                 init_conf=np.zeros(6)):
        super().__init__(cam_pos, lookat_pos, up, fov, w, h, 
                         lens_type, toggle_debug, 
                         auto_cam_rotate, backgroundcolor)

        gm.gen_frame().attach_to(self)  # attach a world frame

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
        self.tcp_ball_meshmodel = []
        self.teaching_mode = 'Joint'
        self.slider_values = []
        self.robot_connect = robot_connect
        self.init_conf = init_conf
        self.real_robot_conf = np.zeros(6)
        self.joint_limits = None

        self.frame = DirectFrame(frameColor=(0, 0.5, 0.5, 0.2),
                                 pos=(0.5, 0, 0),
                                 frameSize=(0, 1.8,-1, 1))

        
    def start(self):
        self.create_button()
        self.create_option_menu()
        self.create_sliders()

    
    def create_button(self):
        button_get_jnts = DirectButton(text="Get Joint Values",
                                       text_pos=(0, -0.4),
                                       command=self.get_robot_jnts,
                                       scale=(0.05, 0.05, 0.05),
                                       frameSize=(-5, 5, -1, 1),
                                       pos=(0.7, 0, 0.5),
                                       parent=self.frame)
        button_record_pose = DirectButton(text="Record Pose",
                                          text_pos=(0, -0.4),
                                          command=self.record_robot_pose,
                                          scale=(0.05, 0.05, 0.05),
                                          frameSize=(-5, 5, -1, 1),
                                          pos=(0.7, 0, 0.4),
                                          parent=self.frame)
        button_planning = DirectButton(text="Plan Path",
                                       text_pos=(0, -0.4), 
                                       command=self.plan_path,
                                       scale=(0.05, 0.05, 0.05),
                                       frameSize=(-5, 5, -1, 1),
                                       pos=(0.7, 0, 0.3),
                                       parent=self.frame)
        button_execute = DirectButton(text="Execute Path",
                                      text_pos=(0, -0.4), 
                                      command=lambda: self.execute_path(self.robot_connect),
                                      scale=(0.05, 0.05, 0.05),
                                      frameSize=(-5, 5, -1, 1),
                                      pos=(0.7, 0, 0.2),
                                      parent=self.frame)

    
    def create_option_menu(self):
        label = DirectLabel(text="Teaching Mode",
                            scale=0.05,
                            pos=(0.2, 0, 0.6),
                            parent=self.frame,
                            frameColor=(1, 1, 1, 0.1))
        options = ["Joint", "Cartesian"]
        self.option_menu = DirectOptionMenu(text="Teaching Mode",
                                            text_pos=(-1, -0.4),
                                            scale=(0.05, 0.05, 0.05),
                                            frameSize=(-5, 5, -1, 1),
                                            pos=(0.7, 0, 0.6),
                                            items=options,
                                            initialitem=0,
                                            parent=self.frame)
    

    def create_sliders(self):
        if self.joint_limits is None:
            self.joint_limits = [[-180,180],[-180,180],[-180,180],
                                [-180,180],[-180,180],[-180,180]]
            
        slider_init = np.rad2deg(self.init_conf)
        slider_values = list(slider_init)  # 初始滑动条值
        for i in range(6):
            label = DirectLabel(text="Joint {}".format(i+1),
                                scale=0.05,
                                pos=(0.2, 0, -0.1 - i * 0.1),
                                parent=self.frame,
                                frameColor=(1, 1, 1, 0.1))
            slider = DirectSlider(range=(self.joint_limits[i][0], self.joint_limits[i][1]),
                                  value=slider_values[i],
                                  scale=(0.3, 0.5, 0.2),
                                  pos=(0.7, 0, -0.1 - i * 0.1),
                                  command=self.slider_changed,
                                  extraArgs=[i],
                                  parent=self.frame)  # 传递滑动条索引作为额外参数
            entry = DirectEntry(text='',
                                scale=0.05,
                                width=4,
                                pos=(1.1, 0, -0.1 - i * 0.1),
                                focus=0,
                                focusInCommand=self.entry_focused,
                                focusInExtraArgs=[i],
                                focusOutCommand=self.entry_blurred,
                                command=self.update_slider_value,
                                extraArgs=[i],
                                parent=self.frame,
                                frameColor=(1, 1, 1, 1))

            self.slider_values.append([slider, entry])  # 存储滑动条和文本框的实例


    def slider_changed(self, slider_index):
        [slider, entry] = self.slider_values[slider_index]
        value = round(slider.getValue(), 3)
        entry.enterText(str(value))  # 更新文本框的值


    def entry_focused(self, slider_index):
        self.accept('enter', self.update_slider_value, extraArgs=[slider_index])


    def entry_blurred(self):
        self.ignore('enter')


    def update_slider_value(self, slider_index):
        slider, entry = self.slider_values[slider_index]
        value_str = entry.get()
        if value_str.isdigit():
            value = float(value_str)
            slider.setValue(value)


    def get_robot_jnts(self):
        """
            获取真实机器人关节角度
        """
        pass
    
    
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


    def plan_path(self):
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
            taskMgr.doMethodLater(0.1, self.planning_animation, "planning_animation",
                                extraArgs=[rbtmnp, motioncounter, self.robot_plan, self.path, self.component_name], 
                                appendTask=True)
            
        elif len(self.start_end_conf) == 1:
            print("[Warning] 请示教机器人运动目标点")

        elif len(self.start_end_conf) == 0:
            print("[Warning] 请示教机器人运动起始点")
        
    
    def robot_move(self):
        pass


    def execute_path(self, real_robot):
        """
            Control the real robot
        """

        if self.path:
            if self.robot_meshmodel is not None:
                self.robot_meshmodel.detach()
            
            for tcp_ball in self.tcp_ball_meshmodel:
                tcp_ball.detach()

            if real_robot:
                print("[Info] Robot connected")
                self.endplanningtask = 1
                self.robot.fk(self.component_name, self.path[-1])
                self.robot_teach.fk(self.component_name, self.path[-1])
                self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
                self.robot_meshmodel.attach_to(base)
                self.robot_move()
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
            print("[Info] No path provided!")


    def planning_animation(self, rbtmnp, motioncounter, robot, path, armname, task):
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
            self.tcp_ball_meshmodel.append(tcp_ball)
            tcp_ball.attach_to(base)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0

        if self.endplanningtask == 1:
            rbtmnp[0].detach()
            tcp_ball.detach()
            print("[Info] Animation 结束")
            return task.done
        else:
            return task.again


    def movement_teaching(self, rbtonscreen, robot, armname, task):
        """
            Use keyboard to control robot
        """

        jnt_values = np.zeros(6)
        for i in range(6):
            jnt_values[i] = self.slider_values[i][0].getValue()
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
                    self.slider_values[i][0].setValue(np.rad2deg(new_jnt_values)[i])
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
                    self.slider_values[i][0].setValue(np.rad2deg(new_jnt_values)[i])
            else:
                print("The given joint angles are out of joint limits.")

        return task.again


    def set_robot(self, robot_s, component):
        """
            Set robot models in the window and enable movement teaching
        """

        self.component_name = component
        self.robot = robot_s
        self.robot_meshmodel = self.robot.gen_meshmodel()
        self.robot_meshmodel.attach_to(self)
        self.robot_teach = robot_s
        self.robot_plan = robot_s

        if self.robot_connect:
            print("[Info] 机器人初始角度(deg): ", np.rad2deg(self.init_conf))
            self.robot.fk(self.component_name, np.asarray(self.init_conf))

            if self.robot_meshmodel is not None:
                self.robot_meshmodel.detach()

            self.robot_meshmodel = self.robot.gen_meshmodel()
            self.robot_meshmodel.attach_to(self)
        
        # Always enable teaching when not planning
        rbtonscreen = [None]
        taskMgr.doMethodLater(0.02, self.movement_teaching, "movement_teaching", 
                        extraArgs=[rbtonscreen, self.robot_teach, self.component_name], 
                        appendTask=True)


if __name__ == "__main__":

    from robot_sim.robots.ur5e_ballpeg import ur5e_ballpeg as ur5e

    # WRS planning simulation
    robot_connect = False
    robot_ip = '192.168.58.2'
    pc_ip = '192.168.58.70'
    init_conf = np.zeros(6)
    base = FastSimWorld(robot_connect=robot_connect, init_conf=init_conf)
    base.start()
    
    robot_s = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    component = 'arm'
    base.set_robot(robot_s, component)
    
    base.setFrameRateMeter(True)
    base.run()
    