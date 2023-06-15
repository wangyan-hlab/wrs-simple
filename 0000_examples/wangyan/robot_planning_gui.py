import os
import time
import yaml
import tkinter as tk
from tkinter import filedialog
import numpy as np
from visualization.panda.world import World
from modeling import geometric_model as gm
from modeling import collision_model as cm
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectOptionMenu, \
        DirectSlider, DirectLabel, DirectEntry, DirectDialog, \
        DirectFrame, DirectCheckButton
from panda3d.core import VirtualFileSystem as vfs


class FastSimWorld(World):
    """
        A Fast Robot Simulator for teaching, planning, and executing

        Author: wangyan
        Date: 2023/06/14
    """

    def __init__(self, 
                 cam_pos=[3, 3, 1], 
                 lookat_pos=[0, 0.5, 0],
                 up=np.array([0, 0, 1]), 
                 fov=40, w=1920, h=1080, 
                 lens_type="perspective", 
                 toggle_debug=False, 
                 auto_cam_rotate=False, 
                 backgroundcolor=None,
                 robot_connect=False,
                 init_conf=np.zeros(6)):
        super().__init__(cam_pos, 
                         lookat_pos, 
                         up, 
                         fov, w, h, 
                         lens_type, 
                         toggle_debug, 
                         auto_cam_rotate, 
                         backgroundcolor)

        gm.gen_frame().attach_to(self)  # attach a world frame

        self.path = []              # planned path
        self.endplanningtask = 1    # flag to stop animation
        self.robot = None           # sim robot
        self.robot_r = None         # real robot
        self.robot_plan = None      # visual robot for animation
        self.component_name = None

        self.robot_teach = None     # visual robot for teaching
        self.teach_point_temp = {}
        self.start_end_conf = []    # saving teaching start and goal points
        self.start_meshmodel = None 
        self.goal_meshmodel = None
        self.robot_meshmodel = None
        self.tcp_ball_meshmodel = []
        self.slider_values = []
        self.tcp_values = []
        self.robot_connect = robot_connect
        self.init_conf = init_conf
        self.real_robot_conf = np.zeros(6)
        self.joint_limits = None
        
        self.vfs = vfs.get_global_ptr()

        
    def start(self):
        """
            Start the simulator
        """

        self.create_frame_gui()
        self.create_button_gui()
        self.create_option_menu_gui()
        self.create_joint_teaching_gui()
        self.create_cartesian_teaching_gui()
        self.create_point_mgr_menu_gui()
        self.create_path_mgr_menu_gui()
        self.create_model_mgr_menu_gui()

    
    """
        任务管理器模块
    """
    def save_task(self):
        """
            Exporting modeling, teaching, and moving data
        """
        
        self.save_modeling()
        self.save_teaching()
        self.save_moving()

    
    def load_task(self):
        """
            Importing modeling, teaching, and moving data
        """

        self.load_modeling()
        self.load_teaching()
        self.load_moving()


    """
        GUI模块
    """
    def create_frame_gui(self):
        """
            Creating frame widgets
        """
        
        self.frame_main = DirectFrame(frameColor=(0.5, 0.5, 0.5, 0.1),
                                 pos=(-1, 0, 0),
                                 frameSize=(-1, 0,-1, 1))

        self.frame_cartesian = DirectFrame(frameColor=(0, 1, 0, 0.1),
                                 pos=(0, 0, 0.),
                                 frameSize=(-1, 0, 0, 1),
                                 parent=self.frame_main)
        
        self.frame_middle = DirectFrame(frameColor=(1, 0, 0, 0.1),
                                 pos=(0, 0, -0.3),
                                 frameSize=(-1, 0, 0, 0.3),
                                 parent=self.frame_main)
        
        self.frame_joint = DirectFrame(frameColor=(1, 1, 0, 0.1),
                                 pos=(0, 0, -0.3),
                                 frameSize=(-1, 0,-1, 0),
                                 parent=self.frame_main)
        
        self.frame_manager = DirectFrame(frameColor=(0, 0, 1, 0.1),
                                 pos=(1, 0, 0.6),
                                 frameSize=(0, 1, 0, 0.4))
        

    def create_button_gui(self):
        """
            Creating button widgets
        """

        DirectButton(text="Get Joint Values",
                    text_pos=(0, -0.4),
                    command=self.get_robot_jnts,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-5, 5, -1, 1),
                    pos=(-0.7, 0, 0.2),
                    parent=self.frame_middle)
        
        DirectButton(text="Record",
                    text_pos=(0, -0.4),
                    command=self.record_teaching,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-5, 5, -1, 1),
                    pos=(-0.7, 0, 0.1),
                    parent=self.frame_middle)
        
        DirectButton(text="Plan",
                    text_pos=(0, -0.4), 
                    command=self.plan_moving,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-3, 3, -1, 1),
                    pos=(-0.2, 0, 0.2),
                    parent=self.frame_middle)
        
        DirectButton(text="Execute",
                    text_pos=(0, -0.4), 
                    command=lambda: self.execute_moving(self.robot_connect),
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-3, 3, -1, 1),
                    pos=(-0.2, 0, 0.1),
                    parent=self.frame_middle)

    
    def create_option_menu_gui(self):
        """
            Creating option menu widgets
        """

        DirectLabel(text="Frame",
                    scale=0.04,
                    pos=(-0.85, 0, 0.9),
                    parent=self.frame_cartesian,
                    frameColor=(1, 1, 1, 0.1))
        
        options = ["Base", "Tool"]
        self.option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                            scale=(0.04, 0.04, 0.04),
                                            frameSize=(0, 4, -1, 1),
                                            pos=(-0.7, 0, 0.9),
                                            items=options,
                                            initialitem=0,
                                            parent=self.frame_cartesian)
    

    def create_joint_teaching_gui(self):
        """
            Creating joint teaching widgets
        """

        if self.joint_limits is None:
            self.joint_limits = [[-180,180],[-180,180],[-180,180],
                                [-180,180],[-180,180],[-180,180]]
            
        slider_init = np.rad2deg(self.init_conf)
        slider_values = list(slider_init)  # 初始滑动条值

        for i in range(6):

            DirectLabel(text="Joint {}".format(i+1),
                        scale=0.035,
                        pos=(-0.9, 0, -0.1 - i * 0.1),
                        parent=self.frame_joint,
                        frameColor=(1, 1, 1, 0.1))
            
            DirectButton(text="-",
                        text_pos=(0, -0.4),
                        scale=(0.1, 0.04, 0.1),
                        pos=(-0.8, 0, -0.08 - i * 0.1),
                        frameSize=(-.3, .3, -.4, .0),
                        command=self.update_joint_slider_value_gui,
                        extraArgs=[i, -1],
                        parent=self.frame_joint)
            
            slider = DirectSlider(range=(self.joint_limits[i][0], self.joint_limits[i][1]),
                                  value=slider_values[i],
                                  scale=(0.25, 0.5, 0.2),
                                  pos=(-0.5, 0, -0.1 - i * 0.1),
                                  command=self.update_joint_entry_value_gui,
                                  extraArgs=[i],
                                  parent=self.frame_joint)
            
            DirectButton(text="+",
                        text_pos=(0, -0.4), 
                        scale=(0.1, 0.04, 0.1),
                        pos=(-0.2, 0, -0.08 - i * 0.1),
                        frameSize=(-.3, .3, -.4, .0),
                        command=self.update_joint_slider_value_gui,
                        extraArgs=[i, 1],
                        parent=self.frame_joint)
            
            entry = DirectEntry(text='',
                                scale=0.035,
                                width=3,
                                pos=(-0.15, 0, -0.1 - i * 0.1),
                                parent=self.frame_joint,
                                frameColor=(1, 1, 1, 1))

            self.slider_values.append([slider, entry])  # 存储滑动条和文本框的实例


    def update_joint_entry_value_gui(self, slider_index):
        """
            Updating joint entry values according to joint slider values
        """

        slider, entry = self.slider_values[slider_index]
        value = round(slider.getValue(), 2)
        entry.enterText(str(value))  # 更新文本框的值


    def update_joint_slider_value_gui(self, slider_index, direction):
        """
            Updating joint slider values according to joint +/- button clicks
        """

        jnt_angular_speed = 1
        slider, _ = self.slider_values[slider_index]
        slider_value = slider.getValue()
        slider_value += direction * jnt_angular_speed
        slider.setValue(slider_value)

    
    def create_cartesian_teaching_gui(self):
        """
            Creating Cartesian teaching widgets
        """

        tcp_dof = ["X", "Y", "Z", "RX", "RY", "RZ"]
        
        for i in range(6):

            DirectLabel(text=tcp_dof[i],
                    scale=0.035,
                    pos=(-0.8, 0, 0.8 - i * 0.1),
                    parent=self.frame_cartesian,
                    frameColor=(1, 1, 1, 0.1))

            DirectButton(text="-",
                        text_pos=(1, -0.2),
                        scale=(0.1, 0.04, 0.1),
                        pos=(-0.7, 0, 0.8 - i * 0.1),
                        frameSize=(0, 2, -.4, .4),
                        command=self.update_cartesian_entry_value_gui,
                        extraArgs=[i, -1],
                        parent=self.frame_cartesian)
            
            DirectButton(text="+",
                        text_pos=(1, -0.2), 
                        scale=(0.1, 0.04, 0.1),
                        pos=(-0.5, 0, 0.8 - i * 0.1),
                        frameSize=(0, 2, -.4, .4),
                        command=self.update_cartesian_entry_value_gui,
                        extraArgs=[i, 1],
                        parent=self.frame_cartesian)
            
            if i in [0, 1, 2]:
                DirectLabel(text=tcp_dof[i],
                            scale=0.035,
                            pos=(-0.85 + 0.3 * i, 0, 0.12),
                            parent=self.frame_cartesian,
                            frameColor=(1, 1, 1, 0.1))
                
                tcp_value_entry = DirectEntry(text='',
                                        scale=0.035,
                                        width=3.5,
                                        pos=(-0.8 + 0.3 * i, 0, 0.12),
                                        parent=self.frame_cartesian,
                                        frameColor=(1, 1, 1, 1))
            else:
                DirectLabel(text=tcp_dof[i],
                            scale=0.035,
                            pos=(-0.85 + 0.3 * (i-3), 0, 0.05),
                            parent=self.frame_cartesian,
                            frameColor=(1, 1, 1, 0.1))
                
                tcp_value_entry = DirectEntry(text='',
                                        scale=0.035,
                                        width=3.5,
                                        pos=(-0.8 + 0.3 * (i-3), 0, 0.05),
                                        parent=self.frame_cartesian,
                                        frameColor=(1, 1, 1, 1))
            
            DirectLabel(text="TCP Pose",
                            scale=0.035,
                            pos=(-0.85, 0, 0.19),
                            parent=self.frame_cartesian,
                            frameColor=(1, 1, 1, 0.1))

            self.tcp_values.append(tcp_value_entry)


    def update_cartesian_entry_value_gui(self, index, direction):
        """
            Updating joint slider values according to joint +/- button clicks
        """

        arm_linear_speed = 0.01
        arm_angular_speed = np.deg2rad(1)

        jnt_values = np.zeros(6)
        for i in range(6):
            jnt_values[i] = self.slider_values[i][0].getValue()
        self.robot_teach.fk(self.component_name, np.deg2rad(jnt_values))

        cur_jnt_values = self.robot_teach.get_jnt_values()
        cur_tcp_pos, cur_tcp_rotmat= self.robot_teach.get_gl_tcp()
        rel_pos = np.zeros(3)
        rel_rpy = np.zeros(3)
        rel_rotmat = np.eye(3)
        if index in [0, 1, 2]:
            rel_pos[index] = arm_linear_speed * direction * .5
        else:
            rel_rpy[index-3] = arm_angular_speed * direction * .5
            rel_rotmat = rm.rotmat_from_euler(rel_rpy[0], rel_rpy[1], rel_rpy[2])
        
        self.robot_frame = self.option_menu.get()

        if self.robot_frame == 'Base':
            new_tcp_pos = cur_tcp_pos + rel_pos
            new_tcp_rotmat = rel_rotmat.dot(cur_tcp_rotmat)
        elif self.robot_frame == 'Tool':
            rel_homomat = rm.homomat_from_posrot(rel_pos, rel_rotmat)
            cur_tcp_homomat = rm.homomat_from_posrot(cur_tcp_pos, cur_tcp_rotmat)
            new_tcp_homomat = cur_tcp_homomat.dot(rel_homomat)
            new_tcp_pos = new_tcp_homomat[:3, 3]
            new_tcp_rotmat = new_tcp_homomat[:3, :3]

        new_jnt_values = self.robot_teach.ik(tgt_pos=new_tcp_pos, 
                                        tgt_rotmat=new_tcp_rotmat,
                                        seed_jnt_values=cur_jnt_values)
        
        if new_jnt_values is not None:
            for i in range(6):
                self.slider_values[i][0].setValue(np.rad2deg(new_jnt_values)[i])
        else:
            print("[Warning] IK is unsolved!")

    
    def create_point_mgr_menu_gui(self):
        """
            Create the Point Manager menu
        """

        DirectLabel(text="Point Manager",
                    scale=0.04,
                    pos=(0.15, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.point_mgr_menu_frame = DirectFrame(pos=(0,0,0),
                                                frameSize=(0.02, 0.27, 0.02, 0.3),
                                                frameColor=(0.8, 0.8, 0.8, 1),
                                                sortOrder=1,
                                                parent=self.frame_manager)
        DirectButton(text="Edit",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.edit_teaching,
                    parent=self.point_mgr_menu_frame)
        
        DirectButton(text="Export",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.16),
                    frameSize=(0, 5, -1, 1),
                    command=self.save_teaching,
                    parent=self.point_mgr_menu_frame)
        
        DirectButton(text="Import",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.07),
                    frameSize=(0, 5, -1, 1),
                    command=self.load_teaching,
                    parent=self.point_mgr_menu_frame)


    def create_path_mgr_menu_gui(self):
        """
            Create the Path Manager menu
        """

        DirectLabel(text="Path Manager",
                    scale=0.04,
                    pos=(0.47, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.path_mgr_menu_frame = DirectFrame(pos=(0.32,0,0),
                                                frameSize=(0.02, 0.27, 0.02, 0.3),
                                                frameColor=(0.8, 0.8, 0.8, 1),
                                                sortOrder=1,
                                                parent=self.frame_manager)
        DirectButton(text="Delete",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.delete_moving,
                    parent=self.path_mgr_menu_frame)
        
        DirectButton(text="Export",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.16),
                    frameSize=(0, 5, -1, 1),
                    command=self.save_moving,
                    parent=self.path_mgr_menu_frame)
        
        DirectButton(text="Import",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.07),
                    frameSize=(0, 5, -1, 1),
                    command=self.load_moving,
                    parent=self.path_mgr_menu_frame)


    def create_model_mgr_menu_gui(self):
        """
            Create the Model Manager menu
        """

        DirectLabel(text="Model Manager",
                    scale=0.04,
                    pos=(0.79, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.model_mgr_menu_frame = DirectFrame(pos=(0.64,0,0),
                                                frameSize=(0.02, 0.27, 0.02, 0.3),
                                                frameColor=(0.8, 0.8, 0.8, 1),
                                                sortOrder=1,
                                                parent=self.frame_manager)
        DirectButton(text="Delete",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.delete_modeling,
                    parent=self.model_mgr_menu_frame)
        
        DirectButton(text="Export",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.16),
                    frameSize=(0, 5, -1, 1),
                    command=self.save_modeling,
                    parent=self.model_mgr_menu_frame)
        
        DirectButton(text="Import",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.07),
                    frameSize=(0, 5, -1, 1),
                    command=self.load_modeling,
                    parent=self.model_mgr_menu_frame)


    def get_robot_jnts(self):
        """
            Get real robot joint values
        """

        pass
    

    """
        MODELING - 模型构建模块
    """
    def static_modeling(self):
        """
            Setting static surrounding models
        """

        pass


    def wobj_modeling(self):
        """
            Setting work object models
        """

        pass

    
    def robot_modeling(self, robot_s, component):
        """
            Setting robot models
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
        
        # Teaching enabled right after robot modeling
        self.enable_teaching()


    def delete_modeling(self):
        """
            Deleting models
        """

        print("[TODO] deleting model")


    def save_modeling(self):
        """
            Exporting models
        """

        print("[TODO] exporting model")


    def load_modeling(self):
        """
            Importing models
        """

        print("[TODO] importing model")

    
    """
        TEACHING - 点位示教模块
    """

    def enable_teaching(self):
        """
            Enabling teaching
        """

        print("[Info] Teaching enabled")
        rbtonscreen = [None]
        taskMgr.doMethodLater(0.02, self.point_teaching, "point_teaching", 
                        extraArgs=[rbtonscreen, self.robot_teach, self.component_name], 
                        appendTask=True)
        
    
    def point_teaching(self, rbtonscreen, robot, armname, task):
        """
            Teaching points
        """

        # Joint-space teaching using sliders 
        jnt_values = np.zeros(6)
        for i in range(6):
            jnt_values[i] = self.slider_values[i][0].getValue()
        robot.fk(armname, np.deg2rad(jnt_values))
        tcp_pos, tcp_rotmat = robot.get_gl_tcp()

        if rbtonscreen[0] is not None:
            rbtonscreen[0].detach()

        cur_jnt_values = robot.get_jnt_values()

        if robot.is_jnt_values_in_ranges(self.component_name, cur_jnt_values):
            rbtonscreen[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 0, 1, 0.5])
            rbtonscreen[0].attach_to(base)
            for i in range(6):
                self.slider_values[i][0].setValue(np.rad2deg(cur_jnt_values)[i])
                if i in [0, 1, 2]:
                    value = round(tcp_pos[i]*1000, 2)
                    self.tcp_values[i].enterText(str(value))
                else:
                    tcp_rpy = rm.rotmat_to_euler(tcp_rotmat)
                    value = round(np.rad2deg(tcp_rpy[i-3]), 2)
                    self.tcp_values[i].enterText(str(value))
        else:
            print("The given joint angles are out of joint limits.")
 
        return task.again
    

    def record_teaching(self):
        """
            Recording teaching point
        """

        self.record_dialog = DirectDialog(dialogName='Record Point',
                              text='Enter the point name:',
                              scale=(0.7, 0.7, 0.7),
                              buttonTextList=['OK', 'Cancel'],
                              buttonValueList=[1, 0],
                              command=self.record_dialog_button_clicked_teaching)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.record_dialog)

        self.record_entry = entry


    def record_dialog_button_clicked_teaching(self, button_value):
        """
            Behaviors when 'Record Point' dialog buttons clicked
        """

        print("[Info] recording teaching")

        if button_value == 1:
            record_name = self.record_entry.get()
            print("Point name:", record_name)
            jnt_values = list(np.rad2deg(self.robot_teach.get_jnt_values()))
            for i in range(6):
                jnt_values[i] = round(float(jnt_values[i]), 3)
            self.teach_point_temp[record_name] = jnt_values
            print("teach_point_temp = ", self.teach_point_temp)
            # self.record_teaching()
            self.record_dialog.hide()

        else:
            self.record_dialog.hide()
            print("Record Point dialog closed")


    def record_teaching_old(self):
        """
            Recording teaching point
        """
        
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


    def edit_teaching(self):
        """
            Editing teaching point
        """
        
        print("[TODO] editing points")

        self.checkbox_values = []

        self.edit_point_dialog = DirectDialog(dialogName='Edit Points',
                              scale=(0.4, 0.4, 0.4),
                              buttonTextList=['Remove', 'Close'],
                              buttonValueList=[1, 0],
                              frameSize=(-1.5,1.5,-0.1-0.1*len(self.teach_point_temp),1),
                              frameColor=(0.7,0.7,0.7,0.5),
                              command=self.edit_point_dialog_button_clicked_teaching)
        
        self.edit_point_dialog.buttonList[0].setPos((1.0, 0, -0.05-0.1*len(self.teach_point_temp)))
        self.edit_point_dialog.buttonList[1].setPos((1.3, 0, -0.05-0.1*len(self.teach_point_temp)))

        DirectLabel(text="Point Name", 
                    pos=(-1.2, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        DirectLabel(text="Joint Values(deg)", 
                    pos=(0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        DirectLabel(text="Remove", 
                    pos=(1.2, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        row = 0
        for i, (point_name, joint_values) in enumerate(self.teach_point_temp.items()):
            DirectLabel(text=point_name, 
                        pos=(-1.2, 0, 0.6-row*0.1), 
                        scale=0.07, 
                        parent=self.edit_point_dialog)
            
            DirectLabel(text=str(joint_values), 
                        pos=(0, 0, 0.6-row*0.1), 
                        scale=0.07, 
                        parent=self.edit_point_dialog)
            
            DirectCheckButton(pos=(1.2, 0, 0.6-row*0.1),
                            scale=0.07, 
                            command=self.edit_point_checkbox_clicked_teaching,
                            extraArgs=[i],
                            frameColor=(1, 1, 1, 1),
                            parent=self.edit_point_dialog)
            row += 1

            self.checkbox_values.append([point_name, False])

    
    def edit_point_checkbox_clicked_teaching(self, isChecked, checkbox_index):
        """
            Change checkbox status
        """
        
        if isChecked:
            self.checkbox_values[checkbox_index][1] = True
        else:
            self.checkbox_values[checkbox_index][1] = False

    
    def edit_point_dialog_button_clicked_teaching(self, button_value):
        """
            Behaviors when 'Edit Point' dialog buttons clicked
        """

        if button_value == 1:
            for point_name, checkbox_state in self.checkbox_values:
                if checkbox_state:
                    removed_point = self.teach_point_temp.pop(point_name)
                    print("该示教点已被移除:", removed_point)
            self.edit_point_dialog.hide()
            print("Edit Point completed")

            self.edit_teaching()

        else:
            self.edit_point_dialog.hide()
            print("Edit Point dialog closed")


    def save_teaching(self):
        """
            Exporting teaching point
        """

        print("[Info] exporting points")

        self.export_point_dialog = DirectDialog(dialogName='Export Points',
                              text='Export points to:',
                              scale=(0.7, 0.7, 0.7),
                              buttonTextList=['OK', 'Cancel'],
                              buttonValueList=[1, 0],
                              command=self.export_point_dialog_button_clicked_gui)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.export_point_dialog)
        
        self.export_point_entry = entry

        
    def export_point_dialog_button_clicked_gui(self, button_value):
        """
            Behaviors when 'Export Point' dialog buttons clicked
        """

        if button_value == 1:
            filename = self.export_point_entry.get()
            
            this_dir = os.path.split(__file__)[0]
            dir = os.path.join(this_dir, 'config/points/')
            if not os.path.exists(dir):
                os.makedirs(dir)
            point_filepath = os.path.join(dir, f'{filename}.yaml')

            with open(point_filepath, 'w', encoding='utf-8') as outfile:
                yaml.dump(self.teach_point_temp, outfile, default_flow_style=False)

            self.export_point_dialog.hide()
            print("[Info] point yaml file saved")

        else:
            self.export_point_dialog.hide()
            print("Export Point dialog closed")


    def load_teaching(self):
        """
            Importing points
        """

        print("[Info] importing teaching")
        
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")],
                                              initialdir="./config/points")
        if filepath:
            print("导入的示教点文件:", filepath)

            self.teach_point_temp = {}
            with open(filepath, 'r', encoding='utf-8') as infile:
                self.teach_point_temp = yaml.load(infile, Loader=yaml.FullLoader)

            print("已导入示教点:", self.teach_point_temp)


    """
        MOVING - 运动规划执行模块
    """
    def plan_moving(self):
        """
            Planning the path
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
            taskMgr.doMethodLater(0.1, self.animation_moving, "animation_moving",
                                extraArgs=[rbtmnp, motioncounter, self.robot_plan, 
                                           self.path, self.component_name], 
                                appendTask=True)
            
        elif len(self.start_end_conf) == 1:
            print("[Warning] 请示教机器人运动目标点")

        elif len(self.start_end_conf) == 0:
            print("[Warning] 请示教机器人运动起始点")


    def animation_moving(self, rbtmnp, motioncounter, robot, path, armname, task):
        """
            Animation of the path
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


    def execute_moving(self, real_robot):
        """
            Executing the path 
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
                self.real_robot_moving()
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

    
    def real_robot_moving(self):
        """
            Moving the real robot
        """
        
        pass


    def delete_moving(self):
        """
            Deleting the path
        """

        print("[TODO] deleting path")

    def save_moving(self):
        """
            Exporting the path
        """
        
        print("[TODO] exporting path")


    def load_moving(self):
        """
            Importing the path
        """
        
        print("[TODO] importing path")

    
if __name__ == "__main__":

    from robot_sim.robots.ur5e import ur5e

    # WRS planning simulation
    robot_connect = False
    robot_ip = '192.168.58.2'
    pc_ip = '192.168.58.70'
    init_conf = np.zeros(6)
    base = FastSimWorld(robot_connect=robot_connect, init_conf=init_conf)
    base.start()
    
    robot_s = ur5e.ROBOT(enable_cc=True, peg_attached=False)
    component = 'arm'
    base.robot_modeling(robot_s, component)
    
    base.setFrameRateMeter(False)
    base.run()
    