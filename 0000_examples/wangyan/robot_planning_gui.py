import os, re, time, yaml, copy
import tkinter as tk
from tkinter import filedialog
import numpy as np
from visualization.panda.world import World
from modeling import geometric_model as gm
from modeling import collision_model as cm
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import VirtualFileSystem as vfs


class FastSimWorld(World):
    """
        A Fast Robot Simulator for teaching points, planning paths, and executing tasks

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

        self.robot_ip = '192.168.58.2'
        self.pc_ip = '192.168.58.70'
        self.robot_connect = robot_connect
        self.robot = None           # sim robot
        self.component_name = None
        self.robot_r = None         # real robot
        self.robot_teach = None     # robot object for teaching
        self.robot_plan = None      # robot object for animation

        self.teach_point_temp = {}  # saving points temporarily
        self.path_temp = {}         # saving paths temporarily
        self.task_temp = {}         # saving execution sequence of paths/points temporarily
        self.task_targets = []
        self.model_temp = {}        # saving models shown on the screen temporarily

        self.start_end_conf = []    # saving planning start and goal points
        self.path = []              # planned path
        self.endplanningtask = {}    # flag to stop animation

        self.conf_meshmodel = {}    # visual robot meshmodel for previewing point confs
        self.path_meshmodel = {}    # visual robot meshmodel for previewing paths
        self.start_meshmodel = None # visual robot meshmodel for previewing start conf
        self.goal_meshmodel = None  # visual robot meshmodel for previewing goal conf
        self.robot_meshmodel = None # visual robot meshmodel for the real robot
        self.rbtonscreen = [None]   # visual robot meshmodel for animation
        self.tcp_ball_meshmodel = {}    # visual tcp ball meshmodel for previewing paths
        self.static_models = []
        self.wobj_models = []
        
        self.slider_values = []
        self.tcp_values = []
        self.init_conf = init_conf
        self.real_robot_conf = np.zeros(6)
        self.joint_limits = None
        self.model_pose_values = []
        self.model_color_values = []
        self.model_init_pose_values = {}
        self.model_init_color_values = {}
        
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
        self.create_model_mgr_menu_gui()
        self.create_path_mgr_menu_gui()
        self.create_point_mgr_menu_gui()
        self.create_task_mgr_menu_gui()


    """
        GUI模块
    """
    def create_frame_gui(self):
        """
            Creating frame widgets
        """
        
        self.frame_main = DirectFrame(
                                frameColor=(0.5, 0.5, 0.5, 0.1),
                                pos=(-1, 0, 0),
                                frameSize=(-1, 0,-1, 1))

        self.frame_cartesian = DirectFrame(
                                frameColor=(0, 1, 0, 0.1),
                                pos=(0, 0, 0.),
                                frameSize=(-1, 0, 0, 1),
                                parent=self.frame_main)
        
        self.frame_middle = DirectFrame(
                                frameColor=(1, 0, 0, 0.1),
                                pos=(0, 0, -0.3),
                                frameSize=(-1, 0, 0, 0.3),
                                parent=self.frame_main)
        
        self.frame_joint = DirectFrame(
                                frameColor=(1, 1, 0, 0.1),
                                pos=(0, 0, -0.3),
                                frameSize=(-1, 0,-1, 0),
                                parent=self.frame_main)
        
        self.frame_manager = DirectFrame(
                                frameColor=(0, 0, 1, 0.1),
                                pos=(0.5, 0, 0.6),
                                frameSize=(0, 1.5, 0, 0.4))
        

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
        
        self.plan_button = DirectButton(
                    text="Plan",
                    text_pos=(0, -0.4), 
                    command=self.plan_moving,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-3, 3, -1, 1),
                    pos=(-0.2, 0, 0.2),
                    parent=self.frame_middle)
        
        self.execute_button = DirectButton(
                    text="Execute",
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
        self.option_menu = DirectOptionMenu(
                    text_pos=(1, -0.4),
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
            
            slider = DirectSlider(
                        range=(self.joint_limits[i][0], self.joint_limits[i][1]),
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
            
            entry = DirectEntry(
                        text='',
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
                
                tcp_value_entry = DirectEntry(
                        text='',
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
                
                tcp_value_entry = DirectEntry(
                        text='',
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


    def create_model_mgr_menu_gui(self):
        """
            Create the Model Manager menu
        """

        DirectLabel(text="Model Manager",
                    scale=0.04,
                    pos=(0.15, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.model_mgr_menu_frame = DirectFrame(
                    pos=(0, 0, 0),
                    frameSize=(0.02, 0.27, 0.02, 0.3),
                    frameColor=(0.8, 0.8, 0.8, 1),
                    sortOrder=1,
                    parent=self.frame_manager)
        
        DirectButton(text="Edit",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.edit_modeling,
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
        
    
    def create_path_mgr_menu_gui(self):
        """
            Create the Path Manager menu
        """

        DirectLabel(text="Path Manager",
                    scale=0.04,
                    pos=(0.47, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.path_mgr_menu_frame = DirectFrame(
                    pos=(0.32,0,0),
                    frameSize=(0.02, 0.27, 0.02, 0.3),
                    frameColor=(0.8, 0.8, 0.8, 1),
                    sortOrder=1,
                    parent=self.frame_manager)
        
        DirectButton(text="Edit",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.edit_moving,
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
        
    
    def create_point_mgr_menu_gui(self):
        """
            Create the Point Manager menu
        """

        DirectLabel(text="Point Manager",
                    scale=0.04,
                    pos=(0.79, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.point_mgr_menu_frame = DirectFrame(
                    pos=(0.64,0,0),
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
        

    def create_task_mgr_menu_gui(self):
        """
            Create the Task Manager menu
        """

        DirectLabel(text="Task Manager",
                    scale=0.04,
                    pos=(1.11, 0, 0.32),
                    parent=self.frame_manager,
                    frameColor=(1,1,1,0.5))
        
        self.task_mgr_menu_frame = DirectFrame(pos=(0.96,0,0),
                    frameSize=(0.02, 0.27, 0.02, 0.3),
                    frameColor=(0.8, 0.8, 0.8, 1),
                    sortOrder=1,
                    parent=self.frame_manager)
        
        DirectButton(text="Edit",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.25),
                    frameSize=(0, 5, -1, 1),
                    command=self.edit_task,
                    parent=self.task_mgr_menu_frame)
        
        DirectButton(text="Export",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.16),
                    frameSize=(0, 5, -1, 1),
                    command=self.save_task,
                    parent=self.task_mgr_menu_frame)
        
        DirectButton(text="Import",
                    text_pos=(2, -0.2),
                    scale=(0.04, 0.04, 0.04),
                    pos=(0.05, 0, 0.07),
                    frameSize=(0, 5, -1, 1),
                    command=self.load_task,
                    parent=self.task_mgr_menu_frame)


    def get_robot_jnts(self):
        """
            Get real robot joint values
        """

        pass
    

    """
        MODELING - 模型构建模块
    """
    def static_modeling(self, model_name, model_pose, model_color):
        """
            Setting static surrounding models
        """

        print("[Info] Setting static surrounding models")
        static_model = cm.CollisionModel(f"objects/other/static/{model_name}.stl")
        pos = model_pose[:3]
        rotmat = rm.rotmat_from_euler(model_pose[3],model_pose[4],model_pose[5])
        static_model.set_pose(pos, rotmat)
        static_model.set_rgba(model_color)
        static_model.attach_to(self)
        self.static_models.append([model_name, static_model])


    def wobj_modeling(self, model_name, model_pose, model_color):
        """
            Setting work object models
        """

        print("[Info] Setting work object models")
        wobj_model = cm.CollisionModel(f"objects/other/wobj/{model_name}.stl")
        pos = model_pose[:3]
        rotmat = rm.rotmat_from_euler(model_pose[3],model_pose[4],model_pose[5])
        wobj_model.set_pose(pos, rotmat)
        wobj_model.set_rgba(model_color)
        wobj_model.attach_to(self)
        self.wobj_models.append([model_name, wobj_model])

    
    def robot_modeling(self, robot_s, component, pos=np.zeros(3), rotmat=np.eye(3)):
        """
            Setting robot model
        """

        print("[Info] Setting robot model")
        self.component_name = component
        self.robot = robot_s
        self.robot.fix_to(pos, rotmat)
        self.robot_meshmodel = self.robot.gen_meshmodel()
        self.robot_meshmodel.attach_to(self)
        self.robot_teach = robot_s
        self.robot_teach.fix_to(pos, rotmat)
        self.robot_plan = robot_s
        self.robot_plan.fix_to(pos, rotmat)

        if self.robot_connect:
            print("[Info] 机器人初始角度(deg): ", np.rad2deg(self.init_conf))
            self.robot.fk(self.component_name, np.asarray(self.init_conf))

            if self.robot_meshmodel is not None:
                self.robot_meshmodel.detach()

            self.robot_meshmodel = self.robot.gen_meshmodel()
            self.robot_meshmodel.attach_to(self)
        
        # Teaching enabled right after robot modeling
        self.enable_teaching()


    def edit_modeling(self):
        """
            Editing models
        """

        print("[Info] editing model")

        self.edit_modeling_pose_color_values = []
        self.edit_modeling_checkbox_values = []

        self.edit_model_dialog = DirectDialog(
                    dialogName='Edit Models',
                    pos=(-0.5, 0, -0.2),
                    scale=(0.4, 0.4, 0.4),
                    buttonTextList=['Remove', 'Close'],
                    buttonValueList=[1, 0],
                    frameSize=(-1.5,1.5,-0.1-0.1*len(self.model_temp),1),
                    frameColor=(0.8,0.8,0.8,0.9),
                    command=self.edit_model_dialog_button_clicked_modeling,
                    parent=self.model_mgr_menu_frame)

        self.edit_model_dialog.buttonList[0].setPos((1.0, 0, -0.05-0.1*len(self.model_temp)))
        self.edit_model_dialog.buttonList[1].setPos((1.3, 0, -0.05-0.1*len(self.model_temp)))

        DirectLabel(text="Model Name", 
                    pos=(-1.0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_model_dialog)
        
        DirectLabel(text="Set Pose/Color", 
                    pos=(0.2, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_model_dialog)
        
        DirectLabel(text="Remove", 
                    pos=(1.0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_model_dialog)
        
        model_infos = []
        for i in range(len(self.model_temp.items())):
            model_type = list(self.model_temp)[i]

            if model_type == 'robot':
                if self.model_temp[model_type]:
                    model_infos.append([model_type, self.model_temp[model_type][0]])
            else:
                for model in self.model_temp[model_type]:
                    if model:
                        model_name = model[0]
                        model_infos.append([model_type, model_name])

        for i, [model_type, model_name] in enumerate(model_infos):
            if model_name:
                DirectLabel(text=model_name,
                                pos=(-1.0, 0, 0.6-i*0.15), 
                                scale=0.07, 
                                parent=self.edit_model_dialog)
                
                edit_pose_color_button = DirectButton(
                                text="Set",
                                text_pos=(0, -0.4),
                                pos=(0.2, 0, 0.62-i*0.15),
                                scale=0.07,
                                frameSize=(-2, 2, -0.8, 0.8),
                                command=self.edit_model_pose_color_modeling,
                                extraArgs=[i, model_type, model_name],
                                parent=self.edit_model_dialog)

                DirectCheckButton(pos=(1.0, 0, 0.6-i*0.15),
                                scale=0.07, 
                                command=self.edit_modeling_checkbox_status_change,
                                extraArgs=[i],
                                frameColor=(1, 1, 1, 1),
                                parent=self.edit_model_dialog)

            self.edit_modeling_pose_color_values.append([model_type, model_name, edit_pose_color_button])
            self.edit_modeling_checkbox_values.append([model_type, model_name, False])


    def edit_model_pose_color_modeling(self, index, model_type, model_name):
        """
            Behaviors when set pose/color buttons clicked
        """

        self.model_pose_values = []
        self.model_color_values = []

        self.pose_color_dialog = DirectDialog(pos=(-0.5, 0, -0.5-0.15*index),
                                    buttonTextList=['OK', 'Close'],
                                    buttonValueList=[1, 0],
                                    frameSize=(-1.0, 1.0, -0.5, 1.0),
                                    frameColor=(0.8,0.8,0.8,0.9),
                                    command=self.pose_color_dialog_button_clicked_modeling,
                                    extraArgs=[model_type, model_name],
                                    parent=self.edit_model_dialog)
        
        self.pose_color_dialog.buttonList[0].setPos((0.6, 0, -0.4))
        self.pose_color_dialog.buttonList[1].setPos((0.8, 0, -0.4))

        DirectLabel(text="Model Pose / Color",
                    pos=(-0.6, 0, 0.8),
                    scale=0.07,
                    parent=self.pose_color_dialog)
        # pose parameters
        pose_params = ['x','y','z','rx','ry','rz']
        pose_units = ['m', 'm', 'm', 'rad', 'rad', 'rad']
        for i in range(6):
            DirectLabel(text=pose_params[i],
                        pos=(-0.9, 0, 0.55-0.15*i),
                        scale=0.06,
                        parent=self.pose_color_dialog)
            model_init_pose_values = self.model_init_pose_values[f"{model_type}-{model_name}"]
            pose_entry = DirectEntry(scale=0.06,
                                    width=6,
                                    pos=(-0.8, 0, 0.55-0.15*i),
                                    initialText=str(model_init_pose_values[i]),
                                    focus=1,
                                    frameColor=(1, 1, 1, 1),
                                    parent=self.pose_color_dialog)
            DirectLabel(text=pose_units[i],
                        pos=(-0.3, 0, 0.55-0.15*i),
                        scale=0.06,
                        parent=self.pose_color_dialog)
            self.model_pose_values.append(pose_entry)
        # color parameters
        color_params = ['R','G','B','Alpha']
        if model_type != 'robot':
            for i in range(4):
                DirectLabel(text=color_params[i],
                            pos=(0.1, 0, 0.55-0.15*i),
                            scale=0.06,
                            parent=self.pose_color_dialog)
                model_init_color_values = self.model_init_color_values[f"{model_type}-{model_name}"]
                color_entry = DirectEntry(scale=0.06,
                                    width=6,
                                    pos=(0.2, 0, 0.55-0.15*i),
                                    initialText=str(model_init_color_values[i]),
                                    focus=1,
                                    frameColor=(1, 1, 1, 1),
                                    parent=self.pose_color_dialog)
                self.model_color_values.append(color_entry)
    

    def pose_color_dialog_button_clicked_modeling(self, button_value, model_type, model_name):
        """
            Behaviors of pose/color dialog buttons
        """

        if button_value == 1:   # implement pose/color parameters
            pose_value = []
            color_value = []

            for i in range(6):
                value = self.model_pose_values[i].get()
                pose_value.append(float(value))
            self.model_init_pose_values[f"{model_type}-{model_name}"] = pose_value

            if model_type != 'robot':
                for i in range(4):
                    value = self.model_color_values[i].get()
                    color_value.append(float(value))
            self.model_init_color_values[f"{model_type}-{model_name}"] = color_value
            
            if model_type == 'static':
                modelnames = [sublist[0] for sublist in self.static_models]
                model_index = modelnames.index(model_name)
                self.static_models[model_index][1].detach()
                self.static_models.pop(model_index)

                modelnames = [sublist[0] for sublist in self.model_temp['static']]
                self.model_temp['static'][modelnames.index(model_name)][1] = pose_value
                self.model_temp['static'][modelnames.index(model_name)][2] = color_value
                self.static_modeling(model_name, pose_value, color_value)
            
            elif model_type == 'wobj':
                modelnames = [sublist[0] for sublist in self.wobj_models]
                model_index = modelnames.index(model_name)
                self.wobj_models[model_index][1].detach()
                self.wobj_models.pop(model_index)

                modelnames = [sublist[0] for sublist in self.model_temp['wobj']]
                self.model_temp['wobj'][modelnames.index(model_name)][1] = pose_value
                self.model_temp['wobj'][modelnames.index(model_name)][2] = color_value
                self.wobj_modeling(model_name, pose_value, color_value)
            
            
            else:   # model_type == 'robot'
                for robot_model in [self.robot_meshmodel, self.start_meshmodel, 
                                        self.goal_meshmodel, self.rbtonscreen[0]]:
                    if robot_model is not None:
                        robot_model.detach()
                
                modelnames = self.model_temp['robot'][0]
                self.model_temp['robot'][1] = pose_value
                pos = pose_value[:3]
                rotmat = rm.rotmat_from_euler(pose_value[3],pose_value[4],pose_value[5])
                self.robot.fix_to(pos, rotmat)
                self.robot_teach.fix_to(pos, rotmat)
                self.robot_plan.fix_to(pos, rotmat)
                self.robot_meshmodel = self.robot.gen_meshmodel()
                self.robot_meshmodel.attach_to(self)
                
        else:   # close
            print("[Info] Pose/Color dialog closed")
            self.pose_color_dialog.hide()


    def edit_modeling_checkbox_status_change(self, isChecked, checkbox_index):
        """
            Change checkbox status
        """
        
        if isChecked:
            self.edit_modeling_checkbox_values[checkbox_index][2] = True
        else:
            self.edit_modeling_checkbox_values[checkbox_index][2] = False

    
    def edit_model_dialog_button_clicked_modeling(self, button_value):
        """
            Behaviors when 'Edit Model' dialog buttons clicked
        """

        if button_value == 1:   # remove
            for model_type, model_name, checkbox_state in self.edit_modeling_checkbox_values:
                if checkbox_state:
                    if model_type == 'robot':
                        self.model_temp[model_type] = []
                        removed_model = model_name
                    elif model_type == 'static':
                        # 获取'static'对应的列表
                        static_list = self.model_temp['static']
                        # 遍历列表，检查并删除满足条件的子列表
                        for sublist in static_list:
                            if sublist[0] == model_name:
                                static_list.remove(sublist)
                                removed_model = model_name
                        # 更新字典中'static'对应的值
                        self.model_temp['static'] = static_list
                    elif model_type == 'wobj':
                        # 获取'wobj'对应的列表
                        wobj_list = self.model_temp['wobj']
                        # 遍历列表，检查并删除满足条件的子列表
                        for sublist in wobj_list:
                            if sublist[0] == model_name:
                                wobj_list.remove(sublist)
                                removed_model = model_name
                        # 更新字典中'wobj'对应的值
                        self.model_temp['wobj'] = wobj_list

                    self.model_init_pose_values.pop(f"{model_type}-{model_name}")
                    if model_type != 'robot':
                        self.model_init_color_values.pop(f"{model_type}-{model_name}")

                    print("[Info] 该Model已被移除:", removed_model)
                    
            self.edit_model_dialog.hide()
            print("[Info] Edit Model completed")

            self.edit_modeling()

        else:   # close
            # remove robot meshmodel
            if 'robot' in self.model_temp:
                if not self.model_temp['robot']:
                    for robot_model in [self.robot_meshmodel, self.start_meshmodel, 
                                        self.goal_meshmodel, self.rbtonscreen[0]]:
                        if robot_model is not None:
                            robot_model.detach()
            # remove static meshmodel
            if 'static' in self.model_temp:
                new_modelnames = [sublist[0] for sublist in self.model_temp['static'] if sublist]
                old_modelnames = [sublist[0] for sublist in self.static_models if sublist]

                for modelname in old_modelnames:
                    if modelname not in new_modelnames:
                        model_index = old_modelnames.index(modelname)
                        self.static_models[model_index][1].detach()
                        removed_model = self.static_models.pop(model_index)
                        print("removed static model:", removed_model)
            # remove wobj meshmodel
            if 'wobj' in self.model_temp:
                new_modelnames = [sublist[0] for sublist in self.model_temp['wobj'] if sublist]
                old_modelnames = [sublist[0] for sublist in self.wobj_models if sublist]

                for modelname in old_modelnames:
                    if modelname not in new_modelnames:
                        model_index = old_modelnames.index(modelname)
                        self.wobj_models[model_index][1].detach()
                        removed_model = self.wobj_models.pop(model_index)
                        print("removed wobj model:", removed_model)

            self.edit_model_dialog.hide()
            print("[Info] Edit Model dialog closed")


    def save_modeling(self):
        """
            Exporting models
        """

        print("[Info] exporting models")

        self.export_model_dialog = DirectDialog(dialogName='Export Models',
                              text='Export models to:',
                              scale=(0.7, 0.7, 0.7),
                              buttonTextList=['OK', 'Cancel'],
                              buttonValueList=[1, 0],
                              command=self.export_model_dialog_button_clicked_modeling)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.export_model_dialog)
        
        self.export_model_entry = entry


    def export_model_dialog_button_clicked_modeling(self, button_value):
        """
            Behaviors when 'Export Model' dialog buttons clicked
        """

        if button_value == 1:   # ok
            filename = self.export_model_entry.get()
            
            this_dir = os.path.split(__file__)[0]
            dir = os.path.join(this_dir, 'config/models/')
            if not os.path.exists(dir):
                os.makedirs(dir)
            model_filepath = os.path.join(dir, f'{filename}.yaml')

            with open(model_filepath, 'w', encoding='utf-8') as outfile:
                yaml.dump(self.model_temp, outfile, default_flow_style=False)

            self.export_model_dialog.hide()
            print("[Info] 已保存Model的yaml文件")

        else:   # cancel
            self.export_model_dialog.hide()
            print("[Info] Export Model dialog closed")


    def load_modeling(self):
        """
            Importing models
        """

        print("[Info] importing models")

        self.import_model_dialog = DirectDialog(dialogName='Import Models',
                              text='Import models from:',
                              scale=(0.7, 0.7, 0.7),
                              buttonTextList=['YAML File', 'Robot Model', 'Other Models', 'Cancel'],
                              buttonValueList=[1, 2, 3, 0],
                              command=self.import_model_dialog_button_clicked_modeling)

        
    def import_model_dialog_button_clicked_modeling(self, button_value):
        """
            Behaviors when 'Import Model' dialog buttons clicked
        """
        
        if button_value == 1:   # import from yaml file
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")],
                                                initialdir="./config/models")
            if filepath:
                print("[Info] 导入的Model YAML文件:", filepath)

                self.model_temp = {}
                self.static_models = []
                self.wobj_models = []
                with open(filepath, 'r', encoding='utf-8') as infile:
                    self.model_temp = yaml.load(infile, Loader=yaml.FullLoader)

                print("[Info] 已从yaml文件导入Model:", self.model_temp.keys())
            
            # import models from yaml file
            if self.model_temp:
                robot_model = self.model_temp['robot'][0]
                robot_model_pose = self.model_temp['robot'][1]
                robot_pos = robot_model_pose[:3]
                robot_rotmat = rm.rotmat_from_euler(robot_model_pose[3],
                                                    robot_model_pose[4],
                                                    robot_model_pose[5])
                if robot_model == 'ur5e':
                    from robot_sim.robots.ur5e import ur5e
                    from robot_con.ur.ur5e import UR5ERtqHE as ur5e_real

                    robot_s = ur5e.ROBOT(enable_cc=True, peg_attached=False, 
                                         pos=robot_pos, rotmat=robot_rotmat)
                    component = 'arm'

                    if self.robot_connect:
                        print("[Info] 机器人已连接")
                        self.robot_r = ur5e_real(robot_ip=self.robot_ip, 
                                                pc_ip=self.pc_ip)
                        self.init_conf = self.robot_r.get_jnt_values()  # 实际机器人的初始关节角度
                    else:
                        print("[Info] 机器人未连接")
                        self.init_conf = np.zeros(6)

                elif robot_model == 'fr5':
                    from robot_sim.robots.fr5 import fr5
                    from fr_python_sdk.frmove import FRCobot as fr5_real

                    robot_s = fr5.ROBOT(enable_cc=True, peg_attached=False, 
                                        pos=robot_pos, rotmat=robot_rotmat,
                                        zrot_to_gndbase=0)
                    component = 'arm'
                    if self.robot_connect:
                        print("[Info] 机器人已连接")
                        self.robot_r = fr5_real(robot_ip=self.robot_ip)
                        self.init_conf = self.robot_r.GetJointPos(unit="rad")  # 实际机器人的初始关节角度
                    else:
                        print("[Info] 机器人未连接")
                        self.init_conf = np.zeros(6)

                self.model_init_pose_values[f"robot-{robot_model}"] = robot_model_pose
                self.robot_modeling(robot_s, component)
            
                # import static models
                static_models = self.model_temp['static']
                for static_model in static_models:
                    static_model_name = static_model[0]
                    static_model_pose = static_model[1]
                    static_model_color = static_model[2]
                    if static_model_name:
                        self.model_init_pose_values[f"static-{static_model_name}"] = static_model_pose
                        self.model_init_color_values[f"static-{static_model_name}"] = static_model_color
                    
                        self.static_modeling(static_model_name, static_model_pose, static_model_color)

                # import wobj models
                wobj_models = self.model_temp['wobj']
                for wobj_model in wobj_models:
                    wobj_model_name = wobj_model[0]
                    wobj_model_pose = wobj_model[1]
                    wobj_model_color = wobj_model[2]
                    if wobj_model_name:
                        self.model_init_pose_values[f"wobj-{wobj_model_name}"] = wobj_model_pose
                        self.model_init_color_values[f"wobj-{wobj_model_name}"] = wobj_model_color
                    
                        self.wobj_modeling(wobj_model_name, wobj_model_pose, wobj_model_color)
            
            self.import_model_dialog.hide()

        elif button_value == 2:
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")],
                                                initialdir="./objects/robot")
            if filepath:
                print("[Info] 导入的Robot Model文件:", filepath)

                with open(filepath, 'r', encoding='utf-8') as infile:
                    robot = yaml.load(infile, Loader=yaml.FullLoader)
                    self.model_temp['robot'] = robot['robot']
                     
                print("[Info] 已导入Robot Model:", self.model_temp['robot'])
            
            # import models from yaml file
            if 'robot' in self.model_temp:
                robot_model = self.model_temp['robot'][0]
                robot_model_pose = self.model_temp['robot'][1]
                robot_pos = robot_model_pose[:3]
                robot_rotmat = rm.rotmat_from_euler(robot_model_pose[3],
                                                    robot_model_pose[4],
                                                    robot_model_pose[5])
                
                if robot_model == 'ur5e':
                    from robot_sim.robots.ur5e import ur5e
                    from robot_con.ur.ur5e import UR5ERtqHE as ur5e_real

                    robot_s = ur5e.ROBOT(enable_cc=True, peg_attached=False,
                                         pos=robot_pos, rotmat=robot_rotmat)
                    component = 'arm'

                    if self.robot_connect:
                        print("[Info] 机器人已连接")
                        self.robot_r = ur5e_real(robot_ip=self.robot_ip, 
                                                pc_ip=self.pc_ip)
                        self.init_conf = self.robot_r.get_jnt_values()  # 实际机器人的初始关节角度
                    else:
                        print("[Info] 机器人未连接")
                        self.init_conf = np.zeros(6)

                elif robot_model == 'fr5':
                    from robot_sim.robots.fr5 import fr5
                    from fr_python_sdk.frmove import FRCobot as fr5_real

                    robot_s = fr5.ROBOT(enable_cc=True, peg_attached=False, 
                                        pos=robot_pos, rotmat=robot_rotmat,
                                        zrot_to_gndbase=0)
                    component = 'arm'
                    if self.robot_connect:
                        print("[Info] 机器人已连接")
                        self.robot_r = fr5_real(robot_ip=self.robot_ip)
                        self.init_conf = self.robot_r.GetJointPos(unit="rad")  # 实际机器人的初始关节角度
                    else:
                        print("[Info] 机器人未连接")
                        self.init_conf = np.zeros(6)

                self.model_init_pose_values[f"robot-{robot_model}"] = robot_model_pose
                self.robot_modeling(robot_s, component)
            self.import_model_dialog.hide()

        elif button_value == 3: # import from model file
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(filetypes=[("STL files", "*.stl")],
                                                  initialdir="./objects/other")
            
            if filepath:
                print("[Info] 导入的 Model 文件:", filepath)
                static_model_name_pattern = re.compile(r"/static/([\s\S]*?)\.stl")
                static_model = static_model_name_pattern.findall(filepath)
                wobj_model_name_pattern = re.compile(r"/wobj/([\s\S]*?)\.stl")
                wobj_model = wobj_model_name_pattern.findall(filepath)
            
                if static_model:
                    static_model_name = static_model[0]
                    static_model_pose = [0,0,0,0,0,0]
                    static_model_color = [1, 0, 0, 0.7]

                    if 'static' in self.model_temp:
                        self.model_temp['static'].append([static_model_name, static_model_pose, static_model_color])
                    else:
                        self.model_temp['static'] = [[static_model_name, static_model_pose, static_model_color]]
                    
                    self.model_init_pose_values[f"static-{static_model_name}"] = static_model_pose
                    self.model_init_color_values[f"static-{static_model_name}"] = static_model_color
                    self.static_modeling(static_model_name, static_model_pose, static_model_color)
                
                if wobj_model:
                    wobj_model_name = wobj_model[0]
                    wobj_model_pose = [0,0,0,0,0,0]
                    wobj_model_color = [1, 0, 0, 0.7]

                    if 'wobj' in self.model_temp:
                        self.model_temp['wobj'].append([wobj_model_name, wobj_model_pose, wobj_model_color])
                    else:
                        self.model_temp['wobj'] = [[wobj_model_name, wobj_model_pose, wobj_model_color]]

                    self.model_init_pose_values[f"wobj-{wobj_model_name}"] = wobj_model_pose
                    self.model_init_color_values[f"wobj-{wobj_model_name}"] = wobj_model_color
                    self.wobj_modeling(wobj_model_name, wobj_model_pose, wobj_model_color)
            
            self.import_model_dialog.hide()

        else:   # cancel
            self.import_model_dialog.hide()
            print("[Info] Import Model dialog closed")

    
    """
        TEACHING - 点位示教模块
    """

    def enable_teaching(self):
        """
            Enabling teaching
        """

        print("[Info] Teaching enabled")
        self.rbtonscreen = [None]
        taskMgr.doMethodLater(0.02, self.point_teaching, "point_teaching", 
                        extraArgs=[self.rbtonscreen, self.robot_teach, self.component_name], 
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
            rbtonscreen[0].attach_to(self)
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
            print("[Warning] The given joint angles are out of joint limits.")
 
        if 'robot' in self.model_temp:
            if not self.model_temp['robot']:
                return task.done
    
        return task.again
    

    def record_teaching(self):
        """
            Recording teaching point
        """

        print("[Info] recording point")

        self.record_point_dialog = DirectDialog(dialogName='Record Point',
                              text='Enter the point name:',
                              scale=(0.7, 0.7, 0.7),
                              buttonTextList=['OK', 'Cancel'],
                              buttonValueList=[1, 0],
                              command=self.record_point_dialog_button_clicked_teaching)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.record_point_dialog)

        self.record_point_entry = entry


    def record_point_dialog_button_clicked_teaching(self, button_value):
        """
            Behaviors when 'Record Point' dialog buttons clicked
        """

        if button_value == 1:   # ok
            record_name = self.record_point_entry.get()
            print("Point name:", record_name)
            jnt_values = list(np.rad2deg(self.robot_teach.get_jnt_values()))
            for i in range(6):
                jnt_values[i] = round(float(jnt_values[i]), 3)
            self.teach_point_temp[record_name] = jnt_values
            self.record_point_dialog.hide()

        else:   # close
            self.record_point_dialog.hide()
            print("[Info] Record Point dialog closed")


    def edit_teaching(self):
        """
            Editing teaching point
        """
        
        print("[Info] editing points")

        self.edit_point_checkbox_values = []

        self.edit_point_dialog = DirectDialog(dialogName='Edit Points',
                                            pos=(-0.5, 0, -0.2),
                                            scale=(0.4, 0.4, 0.4),
                                            buttonTextList=['Remove', 'Close'],
                                            buttonValueList=[1, 0],
                                            frameSize=(-1.5,1.5,-0.1-0.1*len(self.teach_point_temp),1),
                                            frameColor=(0.8,0.8,0.8,0.9),
                                            command=self.edit_point_dialog_button_clicked_teaching,
                                            parent=self.point_mgr_menu_frame)
        
        self.edit_point_dialog.buttonList[0].setPos((1.0, 0, -0.05-0.1*len(self.teach_point_temp)))
        self.edit_point_dialog.buttonList[1].setPos((1.3, 0, -0.05-0.1*len(self.teach_point_temp)))

        DirectLabel(text="Point Name", 
                    pos=(-1.3, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        DirectLabel(text="Joint Values(deg)", 
                    pos=(-0.2, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        DirectLabel(text="Preview", 
                    pos=(0.9, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        DirectLabel(text="Remove", 
                    pos=(1.3, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_point_dialog)
        
        for i, (point_name, joint_values) in enumerate(self.teach_point_temp.items()):
            DirectLabel(text=point_name, 
                        pos=(-1.3, 0, 0.6-i*0.15), 
                        scale=0.07, 
                        parent=self.edit_point_dialog)
            
            DirectLabel(text=str(joint_values), 
                        pos=(-0.2, 0, 0.6-i*0.15), 
                        scale=0.07, 
                        parent=self.edit_point_dialog)
            
            DirectButton(text="preview",
                         text_pos=(0, -0.4),
                         pos=(0.75, 0, 0.62-i*0.15),
                         scale=0.07,
                         frameSize=(-2, 2, -0.8, 0.8),
                         command=self.preview_point_button_clicked_moving,
                         extraArgs=[point_name],
                         parent=self.edit_point_dialog)
            
            DirectButton(text="delete",
                         text_pos=(0, -0.4),
                         pos=(1.05, 0, 0.62-i*0.15),
                         scale=0.07,
                         frameSize=(-2, 2, -0.8, 0.8),
                         command=self.del_preview_point_button_clicked_moving,
                         extraArgs=[point_name],
                         parent=self.edit_point_dialog)
            
            DirectCheckButton(pos=(1.3, 0, 0.6-i*0.15),
                            scale=0.07, 
                            command=self.edit_point_checkbox_status_change,
                            extraArgs=[i],
                            frameColor=(1, 1, 1, 1),
                            parent=self.edit_point_dialog)

            self.edit_point_checkbox_values.append([point_name, False])

    
    def preview_point_button_clicked_moving(self, point_name):
        """
            Behaviors when preview point button clicked
        """

        if point_name in self.conf_meshmodel:
            self.conf_meshmodel[point_name].detach()
        # Preview conf
        conf = np.deg2rad(self.teach_point_temp[point_name])
        self.robot_teach.fk(self.component_name, conf)
        self.conf_meshmodel[point_name] = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[1,1,0,0.3])
        self.conf_meshmodel[point_name].attach_to(self)


    def del_preview_point_button_clicked_moving(self, point_name):
        """
            Erase point preview
        """

        if point_name in self.conf_meshmodel:
            self.conf_meshmodel[point_name].detach()


    def edit_point_checkbox_status_change(self, isChecked, checkbox_index):
        """
            Change checkbox status
        """
        
        if isChecked:
            self.edit_point_checkbox_values[checkbox_index][1] = True
        else:
            self.edit_point_checkbox_values[checkbox_index][1] = False

    
    def edit_point_dialog_button_clicked_teaching(self, button_value):
        """
            Behaviors when 'Edit Point' dialog buttons clicked
        """

        if button_value == 1:   # remove
            for point_name, checkbox_state in self.edit_point_checkbox_values:
                if checkbox_state:
                    removed_point = self.teach_point_temp.pop(point_name)
                    print("[Info] 该示教点已被移除:", removed_point)
            self.edit_point_dialog.hide()
            print("[Info] Edit Point completed")

            self.edit_teaching()

        else:   # close
            for point_name, checkbox_state in self.edit_point_checkbox_values:
                self.del_preview_point_button_clicked_moving(point_name)

            self.edit_point_dialog.hide()
            print("[Info] Edit Point dialog closed")


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

        if button_value == 1:   # ok
            filename = self.export_point_entry.get()
            
            this_dir = os.path.split(__file__)[0]
            dir = os.path.join(this_dir, 'config/points/')
            if not os.path.exists(dir):
                os.makedirs(dir)
            point_filepath = os.path.join(dir, f'{filename}.yaml')

            with open(point_filepath, 'w', encoding='utf-8') as outfile:
                yaml.dump(self.teach_point_temp, outfile, default_flow_style=False)

            self.export_point_dialog.hide()
            print("[Info] 已保存Point的yaml文件")

        else:   # cancel
            self.export_point_dialog.hide()
            print("[Info] Export Point dialog closed")


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
            print("[Info] 导入的Point文件:", filepath)

            self.teach_point_temp = {}
            with open(filepath, 'r', encoding='utf-8') as infile:
                self.teach_point_temp = yaml.load(infile, Loader=yaml.FullLoader)

            print("[Info] 已导入Point:", self.teach_point_temp.keys())


    """
        MOVING - 运动规划执行模块
    """
    def plan_moving(self):
        """
            Planning the path
        """

        self.plan_dialog = DirectDialog(dialogName='Plan',
                                        pos=(0.4, 0, -0.2),
                                        scale=(0.4, 0.4, 0.4),
                                        buttonTextList=['Preview', 'Plan', 'Stop', 'Record', 'Cancel'],
                                        buttonValueList=[1, 2, 3, 4, 0],
                                        frameSize=(-1.0, 1.0, 0, 1.0),
                                        frameColor=(0.8, 0.8, 0.8, 0.9),
                                        command=self.plan_dialog_button_clicked_moving,
                                        parent=self.frame_middle)

        DirectLabel(text="Start",
                    pos=(-0.8, 0, 0.8),
                    scale=0.07,
                    parent=self.plan_dialog)
        
        DirectLabel(text="Goal", 
                    pos=(-0.8, 0, 0.5),
                    scale=0.07,
                    parent=self.plan_dialog)
        
        if self.teach_point_temp:
            options = list(self.teach_point_temp.keys())
        else:
            options = ['- no points -']

        self.start_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                            scale=(0.08, 0.08, 0.08),
                                            frameSize=(0, 15, -1, 1),
                                            frameColor=(1, 1, 1, 1),
                                            pos=(-0.4, 0, 0.8),
                                            items=options,
                                            initialitem=0,
                                            parent=self.plan_dialog)
        
        self.goal_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                            scale=(0.08, 0.08, 0.08),
                                            frameSize=(0, 15, -1, 1),
                                            frameColor=(1, 1, 1, 1),
                                            pos=(-0.4, 0, 0.5),
                                            items=options,
                                            initialitem=0,
                                            parent=self.plan_dialog)
        
        self.plan_dialog.buttonList[0].setPos((0.2, 0, 0.2))
        self.plan_dialog.buttonList[1].setPos((0.2, 0, 0.1))
        self.plan_dialog.buttonList[2].setPos((0.5, 0, 0.2))
        self.plan_dialog.buttonList[3].setPos((0.5, 0, 0.1))
        self.plan_dialog.buttonList[4].setPos((0.8, 0, 0.1))

        # Prevent from crash when planning without points
        if not self.teach_point_temp:
            for i in range(4):
                self.plan_dialog.buttonList[i]['state'] = DGG.DISABLED

        
    def plan_dialog_button_clicked_moving(self, button_value):
        """
            Behaviors when 'Plan' dialog buttons clicked
        """

        if self.teach_point_temp:
            start_conf = np.deg2rad(self.teach_point_temp[self.start_option_menu.get()])
            goal_conf = np.deg2rad(self.teach_point_temp[self.goal_option_menu.get()])

            self.start_end_conf = []
            self.start_end_conf.append(start_conf)
            self.start_end_conf.append(goal_conf)
        
        if button_value == 1:   # preview
            self.plan_show_startgoal_moving()

        elif button_value == 2: # plan
            self.endplanningtask['plan_preview'] = 0    # flag to start animation

            time_start = time.time()
            [start_conf, goal_conf] = self.start_end_conf
            rrtc_planner = rrtc.RRTConnect(self.robot_plan)
            static_obstacle_list = [sublist[1] for sublist in self.static_models]
            wobj_obstacle_list = [sublist[1] for sublist in self.wobj_models]
            obstacle_list = static_obstacle_list + wobj_obstacle_list
            self.path = rrtc_planner.plan(component_name=self.component_name,
                                        start_conf=start_conf,
                                        goal_conf=goal_conf,
                                        obstacle_list=obstacle_list,
                                        ext_dist=0.05,
                                        max_time=300)
            time_end = time.time()
            print("Planning time = ", time_end-time_start)
            print(f"Path length = {len(self.path)}\nPath = {self.path}")

            print("[Info] Plan animation started")
            # Motion planning animation
            rbtmnp = [None]
            motioncounter = [0]
            path_name = 'plan_preview'
            taskMgr.doMethodLater(0.1, self.animation_moving, "animation_moving",
                                extraArgs=[rbtmnp, motioncounter, self.robot_plan, 
                                        self.path, path_name, self.component_name], 
                                appendTask=True)
                
        elif button_value in [3, 4, 0]:
            self.endplanningtask['plan_preview'] = 1

            if self.start_meshmodel is not None:
                self.start_meshmodel.detach()
            if self.goal_meshmodel is not None:
                self.goal_meshmodel.detach()

            for i in range(len(self.path)):
                tcp_ball_name = f"plan_preview-{i}"
                if tcp_ball_name in self.tcp_ball_meshmodel:
                    self.tcp_ball_meshmodel[tcp_ball_name].detach()
                    self.tcp_ball_meshmodel.pop(f"plan_preview-{i}")

            if button_value == 3:   # stop
                print("[Info] Plan animation stopped")

            elif button_value == 4: # record

                self.plan_dialog.hide()

                self.record_path_dialog = DirectDialog(dialogName='Record Path',
                                    text='Enter the path name:',
                                    scale=(0.7, 0.7, 0.7),
                                    buttonTextList=['OK', 'Cancel'],
                                    buttonValueList=[1, 0],
                                    command=self.record_path_dialog_button_clicked_moving)

                entry = DirectEntry(scale=0.04,
                                    width=10,
                                    pos=(-0.2, 0, -0.1),
                                    initialText='',
                                    focus=1,
                                    frameColor=(1, 1, 1, 1),
                                    parent=self.record_path_dialog)
                
                self.record_path_entry = entry

            else:   # close
                self.start_end_conf = []
                self.plan_dialog.hide()
                print("[Info] Plan dialog closed")

    
    def record_path_dialog_button_clicked_moving(self, button_value):
        """
            Behaviors when 'Record Point' dialog buttons clicked
        """

        print("[Info] recording path")

        if button_value == 1:
            record_name = self.record_path_entry.get()
            path = copy.deepcopy(self.path)

            path_deg = []
            for pt in path:
                pt = list(np.rad2deg(pt))
                pt_deg = []
                for i in range(6):
                    pt[i] = round(float(pt[i]), 3)
                    pt_deg.append(pt[i])
                path_deg.append(pt_deg)

            self.path_temp[record_name] = path_deg
            self.record_path_dialog.hide()

        else:
            self.record_path_dialog.hide()
            print("[Info] Record Path dialog closed")


    def plan_show_startgoal_moving(self):
        """
            Showing start and goal point for planning
        """
        
        print("[Info] 显示Planning的start和goal")
        if self.start_meshmodel is not None:
            self.start_meshmodel.detach()
        if self.goal_meshmodel is not None:
            self.goal_meshmodel.detach()
        # Show start jnts
        self.robot_teach.fk(self.component_name, self.start_end_conf[0])
        self.start_meshmodel = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.3])
        self.start_meshmodel.attach_to(self)
        # Show goal jnts
        self.robot_teach.fk(self.component_name, self.start_end_conf[1])
        self.goal_meshmodel = self.robot_teach.gen_meshmodel(toggle_tcpcs=True, rgba=[0,1,0,0.3])
        self.goal_meshmodel.attach_to(self)


    def animation_moving(self, rbtmnp, motioncounter, robot, path, path_name, armname, task):
        """
            Animation of the path
        """

        if motioncounter[0] < len(path):
            # update simulated robot
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            tcp_ball_name = f"{path_name}-{motioncounter[0]}"
            if tcp_ball_name in self.tcp_ball_meshmodel:
                self.tcp_ball_meshmodel[f"{path_name}-{motioncounter[0]}"].detach()

            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0.5, 0, 0.7])
            rbtmnp[0].attach_to(self)
            tcp_ball = gm.gen_sphere(pos=robot.get_gl_tcp(armname)[0], 
                                    radius=0.01, rgba=[1, 1, 0, 1])
            self.tcp_ball_meshmodel[f"{path_name}-{motioncounter[0]}"] = tcp_ball
            self.tcp_ball_meshmodel[f"{path_name}-{motioncounter[0]}"].attach_to(self)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0

        if self.endplanningtask[path_name] == 1:
            rbtmnp[0].detach()
            
            for i in range(len(path)):
                tcp_ball_name = f"{path_name}-{i}"
                if tcp_ball_name in self.tcp_ball_meshmodel:
                    self.tcp_ball_meshmodel[tcp_ball_name].detach()
                    self.tcp_ball_meshmodel.pop(f"{path_name}-{i}")

            print("[Info] Animation 结束")
            return task.done
        else:
            return task.again


    def execute_moving(self, real_robot):
        """
            Executing the path 
        """

        if self.task_temp:
            execute_targets = []

            for target_num, [target_type, target_name] in self.task_temp.items():
                if target_type == 'path':
                    target = self.path_temp[target_name]
                    execute_targets.append(['path', np.deg2rad(target)])
                else:
                    target = self.teach_point_temp[target_name]
                    execute_targets.append(['point', np.deg2rad(target)])

            if execute_targets:
                if self.robot_meshmodel is not None:
                    self.robot_meshmodel.detach()
                if self.start_meshmodel is not None:
                    self.start_meshmodel.detach()
                if self.goal_meshmodel is not None:
                    self.goal_meshmodel.detach()
                
                for tcp_ball_name in self.tcp_ball_meshmodel:
                    self.tcp_ball_meshmodel[tcp_ball_name].detach()
                    self.tcp_ball_meshmodel.pop(tcp_ball_name)

                if real_robot:
                    print("[Info] Robot connected")
                    for path_name in self.endplanningtask:
                        self.endplanningtask[path_name] = 1
                    if execute_targets[-1][0] == 'path':
                        last_target = execute_targets[-1][1][-1]
                    else:
                        last_target = execute_targets[-1][1]
                    
                    self.robot.fk(self.component_name, last_target)
                    self.robot_teach.fk(self.component_name, last_target)
                    self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
                    self.robot_meshmodel.attach_to(self)
                    self.real_robot_moving(execute_targets)

                    cur_jnt_values = self.robot.get_jnt_values()
                    for i in range(6):
                        self.slider_values[i][0].setValue(np.rad2deg(cur_jnt_values)[i])
                    
                else:
                    print("[Info] Robot NOT connected")
                    for path_name in self.endplanningtask:
                        self.endplanningtask[path_name] = 1
                    if execute_targets[-1][0] == 'path':
                        last_target = execute_targets[-1][1][-1]
                    else:
                        last_target = execute_targets[-1][1]
                    self.robot.fk(self.component_name, last_target)
                    self.robot_teach.fk(self.component_name, last_target)
                    self.robot_meshmodel = self.robot.gen_meshmodel(toggle_tcpcs=True)
                    self.robot_meshmodel.attach_to(self)
                    print("[Info] 模拟真实机器人运行时间...")
                    time.sleep(5)

                    cur_jnt_values = self.robot.get_jnt_values()
                    for i in range(6):
                        self.slider_values[i][0].setValue(np.rad2deg(cur_jnt_values)[i])
                
                print("[Info] 机器人运行结束")
            
            else:
                print("[Info] No path provided!")

    
    def real_robot_moving(self):
        """
            Moving the real robot
        """
        
        pass


    def edit_moving(self):
        """
            Editing the path
        """

        print("[Info] editing paths")

        self.edit_path_checkbox_values = []

        self.edit_path_dialog = DirectDialog(dialogName='Edit Paths',
                                            pos=(-0.5, 0, -0.2),
                                            scale=(0.4, 0.4, 0.4),
                                            buttonTextList=['Remove', 'Close'],
                                            buttonValueList=[1, 0],
                                            frameSize=(-1.5,1.5,-0.1-0.1*len(self.path_temp),1),
                                            frameColor=(0.8,0.8,0.8,0.9),
                                            command=self.edit_path_dialog_button_clicked_moving,
                                            parent=self.path_mgr_menu_frame)
        
        self.edit_path_dialog.buttonList[0].setPos((1.0, 0, -0.05-0.1*len(self.path_temp)))
        self.edit_path_dialog.buttonList[1].setPos((1.3, 0, -0.05-0.1*len(self.path_temp)))

        DirectLabel(text="Path Name", 
                    pos=(-1.0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_path_dialog)
        
        DirectLabel(text="Preview", 
                    pos=(0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_path_dialog)
        
        DirectLabel(text="Remove", 
                    pos=(1.0, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_path_dialog)
        
        for i, (path_name, _) in enumerate(self.path_temp.items()):
            DirectLabel(text=path_name, 
                        pos=(-1.0, 0, 0.6-i*0.15), 
                        scale=0.07, 
                        parent=self.edit_path_dialog)

            DirectButton(text="preview",
                         text_pos=(0, -0.4),
                         pos=(-0.25, 0, 0.62-i*0.15),
                         scale=0.07,
                         frameSize=(-3, 3, -1, 1),
                         command=self.preview_path_button_clicked_moving,
                         extraArgs=[path_name],
                         parent=self.edit_path_dialog)
            
            DirectButton(text="delete",
                         text_pos=(0, -0.4),
                         pos=(0.25, 0, 0.62-i*0.15),
                         scale=0.07,
                         frameSize=(-3, 3, -1, 1),
                         command=self.del_preview_path_button_clicked_moving,
                         extraArgs=[path_name],
                         parent=self.edit_path_dialog)
            
            DirectCheckButton(pos=(1.0, 0, 0.6-i*0.15),
                            scale=0.07, 
                            command=self.edit_path_checkbox_status_change,
                            extraArgs=[i],
                            frameColor=(1, 1, 1, 1),
                            parent=self.edit_path_dialog)

            self.edit_path_checkbox_values.append([path_name, False])

    
    def preview_path_button_clicked_moving(self, path_name):
        """
            Behaviors when preview path button clicked
        """

        
        self.endplanningtask[path_name] = 0    # flag to start animation

        self.path = np.deg2rad(self.path_temp[path_name])
        print(f"Path name:{path_name}\nPath length = {len(self.path)}\nPath = {self.path}")
        print("[Info] Path preview animation started")
        # Path preview animation
        self.path_meshmodel[path_name] = [None]
        motioncounter = [0]
        taskMgr.doMethodLater(0.1, self.animation_moving, "animation_moving",
                            extraArgs=[self.path_meshmodel[path_name], motioncounter, 
                                       self.robot_plan, self.path, path_name, self.component_name], 
                            appendTask=True)


    def del_preview_path_button_clicked_moving(self, path_name):
        """
            Stop preview path button
        """    

        self.endplanningtask[path_name] = 1    # flag to stop animation

    
    def edit_path_checkbox_status_change(self, isChecked, checkbox_index):
        """
            Change checkbox status
        """
        
        if isChecked:
            self.edit_path_checkbox_values[checkbox_index][1] = True
        else:
            self.edit_path_checkbox_values[checkbox_index][1] = False


    def edit_path_dialog_button_clicked_moving(self, button_value):
        """
            Behaviors when 'Edit Path' dialog buttons clicked
        """

        if button_value == 1:   # remove
            for path_name, checkbox_state in self.edit_path_checkbox_values:
                if checkbox_state:
                    removed_path = self.path_temp.pop(path_name)
                    print("[Info] 该Path已被移除:", removed_path)
            self.edit_path_dialog.hide()
            print("[Info] Edit Path completed")

            self.edit_moving()

        else:   # close
            for path_name, checkbox_state in self.edit_path_checkbox_values:
                self.del_preview_path_button_clicked_moving(path_name)

            self.edit_path_dialog.hide()
            print("[Info] Edit Path dialog closed")


    def save_moving(self):
        """
            Exporting the path
        """
        
        print("[Info] exporting path")

        self.export_path_dialog = DirectDialog(dialogName='Export Path',
                                    text='Export path to:',
                                    scale=(0.7, 0.7, 0.7),
                                    buttonTextList=['OK', 'Cancel'],
                                    buttonValueList=[1, 0],
                                    command=self.export_path_dialog_button_clicked_moving)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.export_path_dialog)
        
        self.export_path_entry = entry


    def export_path_dialog_button_clicked_moving(self, button_value):
        """
            Behaviors when 'Export Path' dialog buttons clicked
        """

        if button_value == 1:   # ok
            filename = self.export_path_entry.get()
            
            this_dir = os.path.split(__file__)[0]
            dir = os.path.join(this_dir, 'config/paths/')
            if not os.path.exists(dir):
                os.makedirs(dir)
            path_filepath = os.path.join(dir, f'{filename}.yaml')

            with open(path_filepath, 'w', encoding='utf-8') as outfile:
                yaml.dump(self.path_temp, outfile, default_flow_style=False)

            self.export_path_dialog.hide()
            print("[Info] 已保存Path的yaml文件")

        else:   # cancel
            self.export_path_dialog.hide()
            print("[Info] Export Path dialog closed")


    def load_moving(self):
        """
            Importing the path
        """
        
        print("[Info] importing path")

        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")],
                                              initialdir="./config/paths")
        if filepath:
            print("[Info] 导入的Path文件:", filepath)

            self.path_temp = {}
            with open(filepath, 'r', encoding='utf-8') as infile:
                self.path_temp = yaml.load(infile, Loader=yaml.FullLoader)

            print("[Info] 已导入Path:", self.path_temp.keys())


    """
        TASK - 任务管理模块
    """
    def edit_task(self):
        """
            Editing tasks
        """

        print("[Info] editing tasks")

        self.edit_task_dialog = DirectDialog(dialogName='Edit Tasks',
                                        pos=(-0.5, 0, -0.2),
                                        scale=(0.4, 0.4, 0.4),
                                        buttonTextList=['Add', 'Apply', 'Close'],
                                        buttonValueList=[1, 2, 0],
                                        frameSize=(-1.5, 1.5, -0.1-0.2*len(self.task_temp), 1),
                                        frameColor=(0.8, 0.8, 0.8, 0.9),
                                        command=self.edit_task_dialog_button_clicked_task,
                                        parent=self.task_mgr_menu_frame)
        
        self.edit_task_dialog.buttonList[0].setPos((0.4, 0, 0.05-0.2*len(self.task_temp)))
        self.edit_task_dialog.buttonList[1].setPos((0.7, 0, 0.05-0.2*len(self.task_temp)))
        self.edit_task_dialog.buttonList[2].setPos((1.0, 0, 0.05-0.2*len(self.task_temp)))

        if not self.path_temp and not self.teach_point_temp:
            self.edit_task_dialog.buttonList[0]['state'] = DGG.DISABLED

        DirectLabel(text="Target Type", 
                    pos=(-1.1, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_task_dialog)
        
        DirectLabel(text="Target Name", 
                    pos=(-0.4, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_task_dialog)
        
        DirectLabel(text="Preview", 
                    pos=(0.5, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_task_dialog)
        
        DirectLabel(text="Remove", 
                    pos=(1.2, 0, 0.8),
                    scale=0.07,
                    parent=self.edit_task_dialog)
        
        self.task_targets = []
        for i, (target_num, [target_type, target_name]) in enumerate(self.task_temp.items()):
            DirectLabel(text=str(i+1),
                        pos=(-1.4, 0, 0.6-0.2*i),
                        scale=0.07,
                        parent=self.edit_task_dialog)
            
            initial_item = 0 if target_type == "point" else 1
            type_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                                scale=(0.07, 0.07, 0.07),
                                                frameSize=(0, 4, -1, 1),
                                                frameColor=(1, 1, 1, 1),
                                                pos=(-1.2, 0, 0.6-0.2*i),
                                                items=["point", "path"],
                                                initialitem=initial_item,
                                                command=self.update_target_name_task,
                                                extraArgs=[i],
                                                parent=self.edit_task_dialog)
            target_type = type_option_menu.get()

            if target_type == "point":
                name_options = list(self.teach_point_temp.keys())
            else:
                name_options = list(self.path_temp.keys())
            name_option_initialitem = name_options.index(target_name)

            name_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                                scale=(0.07, 0.07, 0.07),
                                                frameSize=(0, 10, -1, 1),
                                                frameColor=(1, 1, 1, 1),
                                                pos=(-0.7, 0, 0.6-0.2*i),
                                                items=name_options,
                                                initialitem=name_option_initialitem,
                                                command=self.update_task_temp_task,
                                                extraArgs=[i],
                                                parent=self.edit_task_dialog)

            DirectButton(text="preview",
                         text_pos=(0, -0.4),
                         pos=(0.3, 0, 0.6-i*0.2),
                         scale=0.07,
                         frameSize=(-2, 2, -0.8, 0.8),
                         command=self.preview_task_button_clicked_moving,
                         extraArgs=[target_type, target_name],
                         parent=self.edit_task_dialog)
            
            DirectButton(text="delete",
                         text_pos=(0, -0.4),
                         pos=(0.7, 0, 0.6-i*0.2),
                         scale=0.07,
                         frameSize=(-2, 2, -0.8, 0.8),
                         command=self.del_preview_task_button_clicked_moving,
                         extraArgs=[target_type, target_name],
                         parent=self.edit_task_dialog)
            
            DirectCheckButton(pos=(1.2, 0, 0.6-0.2*i),
                            scale=0.07, 
                            command=self.edit_task_checkbox_status_change,
                            extraArgs=[i],
                            frameColor=(1, 1, 1, 1),
                            parent=self.edit_task_dialog)
            
            self.task_targets.append([target_num, type_option_menu, name_option_menu, False])
        

    def preview_task_button_clicked_moving(self, target_type, target_name):
        """
            Behaviors when preview task button clicked
        """
    
        if target_type == 'point':
            print("[Info] Preview point")
            self.preview_point_button_clicked_moving(target_name)
        else:
            print("[Info] Preview path")
            self.preview_path_button_clicked_moving(target_name)

    
    def del_preview_task_button_clicked_moving(self, target_type, target_name):
        """
            Erase preview points / paths
        """

        if target_type == 'point':
            print("[Info] Delete preview point")
            self.del_preview_point_button_clicked_moving(target_name)
        else:
            print("[Info] Delete preview path")
            self.del_preview_path_button_clicked_moving(target_name)


    def edit_task_checkbox_status_change(self, isChecked, checkbox_index):
        """
            Change checkbox status
        """
        
        if isChecked:
            self.task_targets[checkbox_index][3] = True
        else:
            self.task_targets[checkbox_index][3] = False

    
    def update_target_name_task(self, value, index):
        """
            Change the target name list according to the target type
        """

        if value == "point":
            new_items = list(self.teach_point_temp.keys())
        else:
            new_items = list(self.path_temp.keys())
        
        self.task_targets[index][2].destroy()
        self.task_targets[index][2] = DirectOptionMenu(text_pos=(1, -0.4),
                                                scale=(0.07, 0.07, 0.07),
                                                frameSize=(0, 10, -1, 1),
                                                frameColor=(1, 1, 1, 1),
                                                pos=(-0.7, 0, 0.6-0.2*index),
                                                items=new_items,
                                                initialitem=0,
                                                command=self.update_task_temp_task,
                                                extraArgs=[index],
                                                parent=self.edit_task_dialog)
        
        target_name = self.task_targets[index][2].get()
        self.task_temp[str(index+1)] = [value, target_name]


    def update_task_temp_task(self, value, index):
        """
            Save target info to task_temp
        """

        target_type = self.task_targets[index][1].get() 
        self.task_temp[str(index+1)] = [target_type, value]

    
    def edit_task_dialog_button_clicked_task(self, button_value):
        """
            Behaviors when 'Task' dialog buttons clicked
        """
        
        if button_value == 1:   # add
            line_num = len(self.task_temp)
            DirectLabel(text=str(line_num+1),
                        pos=(-1.4, 0, 0.6-line_num*0.2),
                        scale=0.07,
                        parent=self.edit_task_dialog)
            
            type_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                                scale=(0.07, 0.07, 0.07),
                                                frameSize=(0, 4, -1, 1),
                                                frameColor=(1, 1, 1, 1),
                                                pos=(-1.2, 0, 0.6-line_num*0.2),
                                                items=["point", "path"],
                                                initialitem=0,
                                                command=self.update_target_name_task,
                                                extraArgs=[line_num],
                                                parent=self.edit_task_dialog)

            name_options = list(self.teach_point_temp.keys())
            name_option_menu = DirectOptionMenu(text_pos=(1, -0.4),
                                                scale=(0.07, 0.07, 0.07),
                                                frameSize=(0, 10, -1, 1),
                                                frameColor=(1, 1, 1, 1),
                                                pos=(-0.7, 0, 0.6-line_num*0.2),
                                                items=name_options,
                                                initialitem=0,
                                                command=self.update_task_temp_task,
                                                extraArgs=[line_num],
                                                parent=self.edit_task_dialog)

            DirectCheckButton(pos=(1.2, 0, 0.6-line_num*0.1),
                            scale=0.07, 
                            command=self.edit_task_checkbox_status_change,
                            extraArgs=[line_num],
                            frameColor=(1, 1, 1, 1),
                            parent=self.edit_task_dialog)
            
            self.task_targets.append([str(line_num+1), type_option_menu, name_option_menu, False])
            self.task_temp[str(line_num+1)] = ["target_type_placeholder", "target_name_placeholder"]

            for target_num, type_option_menu, name_option_menu, checkbox_state in self.task_targets:
                self.task_temp[target_num] = [type_option_menu.get(), name_option_menu.get()]

            self.edit_task_dialog.hide()
            self.edit_task()

        elif button_value == 2: # apply changes to targets
            # remove targets
            for target_num, _, _, checkbox_state in self.task_targets:
                if checkbox_state:
                    removed_target = self.task_temp.pop(target_num)
                    print("[Info] 该任务目标已被移除:", removed_target)
                    
            # reassign target_num
            task_temp_swap = {}
            for i, (_, [target_type, target_name]) in enumerate(self.task_temp.items()):
                task_temp_swap[str(i+1)] = [target_type, target_name]
            self.task_temp = task_temp_swap

            self.edit_task_dialog.hide()
            print("[Info] Edit Task completed")

            self.edit_task()

        else:   # close
            # refresh the task dict
            for target_num, type_option_menu, name_option_menu, checkbox_state in self.task_targets:
                self.task_temp[target_num] = [type_option_menu.get(), name_option_menu.get()]

                self.del_preview_task_button_clicked_moving(type_option_menu.get(), name_option_menu.get())

            self.edit_task_dialog.hide()
            print("[Info] Execute dialog closed")


    def save_task(self):
        """
            Exporting tasks
        """

        print("[Info] exporting task")

        self.export_task_dialog = DirectDialog(dialogName='Export Task',
                                    text='Export task to:',
                                    scale=(0.7, 0.7, 0.7),
                                    buttonTextList=['OK', 'Cancel'],
                                    buttonValueList=[1, 0],
                                    command=self.export_task_dialog_button_clicked_moving)

        entry = DirectEntry(scale=0.04,
                            width=10,
                            pos=(-0.2, 0, -0.1),
                            initialText='',
                            focus=1,
                            frameColor=(1, 1, 1, 1),
                            parent=self.export_task_dialog)
        
        self.export_task_entry = entry
            

    def export_task_dialog_button_clicked_moving(self, button_value):
        """
            Behaviors when 'Export Task' dialog buttons clicked
        """
        
        if button_value == 1:   # ok
            filename = self.export_task_entry.get()
            
            this_dir = os.path.split(__file__)[0]
            dir = os.path.join(this_dir, 'config/tasks/')
            if not os.path.exists(dir):
                os.makedirs(dir)
            task_filepath = os.path.join(dir, f'{filename}.yaml')

            with open(task_filepath, 'w', encoding='utf-8') as outfile:
                yaml.dump(self.task_temp, outfile, default_flow_style=False)

            self.export_task_dialog.hide()
            print("[Info] 已保存Task的yaml文件")

        else:   # close
            self.export_task_dialog.hide()
            print("[Info] Export Task dialog closed")

    
    def load_task(self):
        """
            Importing tasks
        """

        print("[Info] importing task")
        
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")],
                                              initialdir="./config/tasks")
        if filepath:
            print("[Info] 导入的Task文件:", filepath)

            self.task_temp = {}
            with open(filepath, 'r', encoding='utf-8') as infile:
                self.task_temp = yaml.load(infile, Loader=yaml.FullLoader)

            print("[Info] 已导入Task", filepath)
