import visualization.panda.world as wd
import modeling.dynamics.bullet.bdmodel as bdm
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import time
import pickle
from direct.gui.DirectGui import *
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.fr5.fr5_rtq85 as fr5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc


class GraspWorld(wd.World):
    def __init__(self, cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3]):
        super().__init__(cam_pos, lookat_pos)
        
        self.obgl_start_homomat = np.eye(4)
        milkcarton_gl_goal_pos = np.array([0.4, 0.4, 0.2])
        milkcarton_gl_goal_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.obgl_goal_homomat = rm.homomat_from_posrot(milkcarton_gl_goal_pos, milkcarton_gl_goal_rotmat)
        self.milkcarton_copy = None

    def get_obj_homomat(self, obj_bd_box, milkcarton, task):    
        # global obgl_start_homomat, milkcarton_copy
        self.obgl_start_homomat = obj_bd_box.get_homomat()
        self.milkcarton_copy = milkcarton.copy()
        print("obgl_start_homomat =", self.obgl_start_homomat)
        self.milkcarton_copy.set_rgba([1, 0, 0, .4])
        self.milkcarton_copy.set_homomat(self.obgl_start_homomat)
        return task.done

    def click_get_obj_homomat(self):
        taskMgr.doMethodLater(0.01, self.get_obj_homomat, "get_obj_homomat",
                                extraArgs=[milkcarton_bd_box, milkcarton],
                                appendTask=True)

    def create_frame_gui(self):
        """
            Creating frame widgets
        """
        self.frame_main = DirectFrame(
                            frameColor=(0.5, 0.5, 0.5, 0.1),
                            pos=(-1, 0, 0),
                            frameSize=(-1, 0,-1, 1))
        
        self.frame_middle = DirectFrame(
                                frameColor=(1, 0, 0, 0.1),
                                pos=(0, 0, -0.3),
                                frameSize=(-1, 0, 0, 0.3),
                                parent=self.frame_main)
        
    def create_button_gui(self):
        """
            Creating button widgets
        """
        DirectButton(text="Get Object Pose",
                    text_pos=(0, -0.4),
                    command=self.click_get_obj_homomat,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-5, 5, -1, 1),
                    pos=(-0.7, 0, 0.2),
                    parent=self.frame_middle)
    
        DirectButton(text="Plan Grasping",
                    text_pos=(0, -0.4),
                    command=self.grasp_planning,
                    scale=(0.04, 0.04, 0.04),
                    frameSize=(-5, 5, -1, 1),
                    pos=(-0.7, 0, 0.1),
                    parent=self.frame_middle)
        
    def grasp_planning(self):
        rrtc_s = rrtc.RRTConnect(robot_s)
        ppp_s = ppp.PickPlacePlanner(robot_s)

        original_grasp_info_list = gpa.load_pickle_file('milkcarton', './', 'rtq85_milkcarton.pickle')
        start_conf = robot_s.get_jnt_values()
        hnd_name = 'hnd'
        print(">>>>>>>", self.obgl_start_homomat)
        print("<<<<<<<", self.obgl_goal_homomat)
        conf_list, jawwidth_list, objpose_list = \
            ppp_s.gen_pick_and_place_motion(hnd_name=hnd_name,
                                            objcm=milkcarton,
                                            grasp_info_list=original_grasp_info_list,
                                            start_conf=start_conf,
                                            end_conf=start_conf,
                                            goal_homomat_list=[self.obgl_start_homomat, self.obgl_goal_homomat],
                                            approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
                                            approach_distance_list=[.3] * 2,
                                            depart_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 1])],
                                            depart_distance_list=[.3] * 2,
                                            obstacle_list=[obj])
    
        robot_attached_list = []
        object_attached_list = []
        counter = [0]
        
        def update(robot_s,
                hnd_name,
                milkcarton,
                robot_path,
                jawwidth_path,
                obj_path,
                robot_attached_list,
                object_attached_list,
                counter,
                task):
            if counter[0] >= len(robot_path):
                counter[0] = 0
            if len(robot_attached_list) != 0:
                for robot_attached in robot_attached_list:
                    robot_attached.detach()
                for object_attached in object_attached_list:
                    object_attached.detach()
                robot_attached_list.clear()
                object_attached_list.clear()
            pose = robot_path[counter[0]]
            robot_s.fk(hnd_name, pose)
            robot_s.jaw_to(hnd_name, jawwidth_path[counter[0]])
            robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
            robot_meshmodel.attach_to(base)
            robot_attached_list.append(robot_meshmodel)
            obj_pose = obj_path[counter[0]]
            objb_copy = milkcarton.copy()
            objb_copy.set_rgba([.9, .75, .35, 1])
            objb_copy.set_homomat(obj_pose)
            objb_copy.attach_to(base)
            object_attached_list.append(objb_copy)
            counter[0] += 1
            return task.again
        taskMgr.doMethodLater(0.01, update, "update",
                            extraArgs=[robot_s,
                                        hnd_name,
                                        milkcarton,
                                        conf_list,
                                        jawwidth_list,
                                        objpose_list,
                                        robot_attached_list,
                                        object_attached_list,
                                        counter],
                            appendTask=True)
        

if __name__ == '__main__':

    # base = wd.World(cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3])
    base = GraspWorld(cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3])
    base.create_frame_gui()
    base.create_button_gui()
    gm.gen_frame().attach_to(base)

    task_table = cm.CollisionModel("../objects/task_table.stl")
    task_table.set_rgba([.5, .5, .5, 1])
    task_table_bd = bdm.BDModel(task_table, mass=.0, type="convex", friction=0.9)
    task_table_bd.set_pos(np.array([0.8, -0.2, -0.025+0.3]))
    task_table_bd.start_physics()
    base.attach_internal_update_obj(task_table_bd)

    # milkcarton bodymodel
    # object to grasp
    milkcarton = cm.CollisionModel("../objects/milkcarton.stl")
    milkcarton.set_scale([.4, .4, .4])
    milkcarton.set_rgba([.9, .0, .0, 0.9])
    
    milkcarton_bd_box = bdm.BDModel(milkcarton, mass=.3, type="convex", friction=0.9)
    milkcarton_bd_box.set_pos(np.array([.4, .1, 0.8]))
    milkcarton_bd_box.start_physics()
    base.attach_internal_update_obj(milkcarton_bd_box)

    # object start
    gm.gen_frame().attach_to(milkcarton)
    # object goal
    milkcarton_goal_copy = milkcarton.copy()
    milkcarton_goal_copy.set_rgba([0, 1, 0, .4])
    milkcarton_goal_copy.set_homomat(base.obgl_goal_homomat)
    milkcarton_goal_copy.attach_to(base)

    robot_s = fr5.ROBOT(homeconf=np.radians([-40,-80,30,10,45,90]))
    robot_s.gen_meshmodel(rgba=[1, 0, 1, .3]).attach_to(base)

    # obstacle
    obj = cm.CollisionModel("../objects/bunnysim.stl")
    obj.set_pos(np.array([0.4, 0.22, 0.55]))
    obj.set_rpy(0, 0, 1.57)
    obj.set_scale([.5, .5, .5])
    obj.set_rgba([.1, .2, .8, 1])
    obj.attach_to(base)

    base.run()
