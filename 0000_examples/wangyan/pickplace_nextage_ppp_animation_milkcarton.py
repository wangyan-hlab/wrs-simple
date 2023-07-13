import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.nextage.nextage as nxt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

if __name__ == '__main__':

    base = wd.World(cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3])
    gm.gen_frame().attach_to(base)
    # object to grasp
    milkcarton = cm.CollisionModel("../objects/milkcarton.stl")
    milkcarton.set_rgba([.9, .75, .35, 1])
    # object start
    milkcarton_gl_pos = np.array([0.4, 0.05, 0.2])
    milkcarton_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
    obgl_start_homomat = rm.homomat_from_posrot(milkcarton_gl_pos, milkcarton_gl_rotmat)
    milkcarton.set_pos(milkcarton_gl_pos)
    milkcarton.set_rotmat(milkcarton_gl_rotmat)
    milkcarton.set_scale([.4, .4, .4])
    gm.gen_frame().attach_to(milkcarton)
    milkcarton_copy = milkcarton.copy()
    milkcarton_copy.set_rgba([1, 0, 0, .4])
    milkcarton_copy.attach_to(base)
    # object goal
    milkcarton_gl_goal_pos = np.array([0.4, 0.3, 0.2])
    milkcarton_gl_goal_rotmat = rm.rotmat_from_euler(0, 0, 0)
    obgl_goal_homomat = rm.homomat_from_posrot(milkcarton_gl_goal_pos, milkcarton_gl_goal_rotmat)
    milkcarton_goal_copy = milkcarton.copy()
    milkcarton_goal_copy.set_rgba([0, 1, 0, .4])
    milkcarton_goal_copy.set_homomat(obgl_goal_homomat)
    milkcarton_goal_copy.attach_to(base)

    robot_s = nxt.Nextage(mode='lft_arm_waist')
    robot_s.gen_meshmodel(rgba=[1, 0, 1, .3]).attach_to(base)

    obj = cm.CollisionModel("../objects/bunnysim.stl")
    obj.set_pos(np.array([0.4, 0.22, 0.55]))
    obj.set_rpy(0, 0, 1.57)
    obj.set_scale([.5, .5, .5])
    obj.set_rgba([.1, .2, .8, 1])
    obj.attach_to(base)
    # base.run()

    rrtc_s = rrtc.RRTConnect(robot_s)
    ppp_s = ppp.PickPlacePlanner(robot_s)

    original_grasp_info_list = gpa.load_pickle_file('milkcarton', './', 'nextage_milkcarton.pickle')
    hnd_name = robot_s.mode
    start_conf = robot_s.get_jnt_values(hnd_name[:4]+"hnd")
    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hnd_name,
                                        objcm=milkcarton,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
                                        approach_distance_list=[.15] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 1])],
                                        depart_distance_list=[.15] * 2,
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
    base.setFrameRateMeter(True)
    base.run()
