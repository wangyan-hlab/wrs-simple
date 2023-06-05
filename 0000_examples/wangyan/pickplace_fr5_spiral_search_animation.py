import pickle
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.fr5.fr5 as fr5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import motion.optimization_based.incremental_nik as inik
import matplotlib.pyplot as plt
import visualization.snsplot as snsplot
snsplot.set()

def spiral(start_angle, start_radius, linear_vel, radius_per_turn, max_radius, granularity=0.1):
    r, phi = start_radius, start_angle
    xpt, ypt = [], []
    while r <= max_radius:
        xpt.append(r * np.cos(phi))
        ypt.append(r * np.sin(phi))
        angle_vel = linear_vel/r
        phi += angle_vel*granularity
        r += radius_per_turn/(2*np.pi)*angle_vel*granularity
    return xpt, ypt

if __name__ == '__main__':

    base = wd.World(cam_pos=[-1, 1, 1], lookat_pos=[-0.4, 0.15, 0.5])
    gm.gen_frame().attach_to(base)

    robot_s = fr5.FR5_robot(zrot_to_gndbase=0, arm_jacobian_offset=np.array([0, 0, .145]), hnd_attached=True)
    tgt_pos = np.array([-0.4, 0, 0.6])
    tgt_rotmat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    jnt_values = robot_s.ik(component_name='arm', tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat,
                            seed_jnt_values=np.radians([-15,-60,20,0,-90,-90]))
    robot_s.fk(component_name="arm", jnt_values=jnt_values)
    robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 1, .3]).attach_to(base)
    # base.run()

    tcp_pos = robot_s.get_gl_tcp()[0]
    # object to grasp
    obj_name = 'peg'
    obj = cm.CollisionModel("../objects/" + obj_name + ".stl")
    obj_diam = 0.015
    obj_height = 0.05
    obj.set_rgba([.9, .75, .35, 1])
    # object start pose
    obj_gl_pos = tcp_pos + np.array([0, 0, -0.1])
    obj_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
    obgl_start_homomat = rm.homomat_from_posrot(obj_gl_pos, obj_gl_rotmat)
    obj.set_pos(obj_gl_pos)
    obj.set_rotmat(obj_gl_rotmat)
    gm.gen_frame().attach_to(obj)
    obj_copy = obj.copy()
    obj_copy.set_rgba([1, 0, 0, .4])
    obj_copy.attach_to(base)
    # object contact pose
    obj_gl_goal_pos = obj_gl_pos + np.array([0, 0.15, 0.025])
    obj_gl_goal_rotmat = rm.rotmat_from_euler(0, 0, 0)
    obgl_goal_homomat = rm.homomat_from_posrot(obj_gl_goal_pos, obj_gl_goal_rotmat)
    obj_goal_copy = obj.copy()
    obj_goal_copy.set_rgba([0, 1, 0, .4])
    obj_goal_copy.set_homomat(obgl_goal_homomat)
    obj_goal_copy.attach_to(base)
    # obstacle pose
    obstacle = cm.CollisionModel("../objects/hole.stl")
    obstacle_height = 0.035
    obstacle_error = np.array([0.001, 0.002, 0])     # x, y can be changed
    offset = np.array([0, 0, -(obj_height+obstacle_height)]) + obstacle_error
    obstacle_pos = obj_gl_goal_pos + offset
    obstacle.set_pos(obstacle_pos)
    obstacle.set_rpy(0, 0, 0)
    obstacle.set_rgba([.2, .2, .9, 1])
    obstacle.attach_to(base)
    # base.run()

    rrtc_s = rrtc.RRTConnect(robot_s)
    ppp_s = ppp.PickPlacePlanner(robot_s)
    original_grasp_info_list = gpa.load_pickle_file(obj_name, './', 'fr5_'+obj_name+'.pickle')
    hnd_name = "hnd"
    start_conf = robot_s.get_jnt_values(hnd_name)
    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hnd_name,
                                        objcm=obj,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[np.array([0,0,-1]), np.array([0,0,-1])],
                                        approach_distance_list=[.05] * 2,
                                        depart_direction_list=[np.array([0,0,1]), np.array([0,0,1])],
                                        depart_distance_list=[.05] * 2,
                                        obstacle_list=[])
    contact_id = 0
    for index, pose in enumerate(objpose_list):
        distance_to_contact = (pose[2, -1]-obj_height) - (obstacle_pos[2]+obstacle_height)
        if abs(distance_to_contact) <= 1e-3:
            if np.linalg.norm(pose[:2, -1] - obj_gl_goal_pos[:2]) == 0:
                contact_id = index
                # print("contact_id = ", contact_id)
                print(">> Contact!")
                print(">> Spiral search starts!")
                break

    conf_list = conf_list[:contact_id]
    objpose_list = objpose_list[:contact_id]
    jawwidth_list = jawwidth_list[:contact_id]
    obj_contact_pos = objpose_list[-1][:3, -1]
    obj_contact_orn = objpose_list[-1][:3, :3]
    robot_s.fk(component_name='arm', jnt_values=conf_list[-1])
    robot_pos, robot_orn = robot_s.get_gl_tcp()
    pos, orn = robot_pos, robot_orn
    delta_pos_obj_hand = obj_contact_pos - pos
    jnt_values = robot_s.get_jnt_values()
    spiral_x, spiral_y = spiral(start_angle=0, start_radius=1e-4, linear_vel=5e-3,
                                radius_per_turn=2e-4, max_radius=5e-3, granularity=0.05)

    error_list = []
    count = 0
    for x, y in zip(spiral_x, spiral_y):
        count += 1
        print("----------")
        print(">> spiral pt# {}/{}".format(count, len(spiral_x)))
        # print(">> x={}, y={}".format(x, y))
        pos = np.array([robot_pos[0]+x, robot_pos[1]+y, pos[2]])
        jnt_values = robot_s.ik(tgt_pos=pos, tgt_rotmat=orn, seed_jnt_values=jnt_values)
        conf_list.append(jnt_values)
        robot_s.fk(component_name='arm', jnt_values=jnt_values)
        pos = robot_s.get_gl_tcp()[0]
        obj_pos = np.array([pos[0], pos[1], obj_contact_pos[2]])
        objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_contact_orn))
        jawwidth_list.append(obj_diam)

        error = np.linalg.norm(obj_pos[:2] - obstacle_pos[:2])
        error_list.append(error)
        print(">> error = ", error)
        if error <= 2e-4:
            print(">> Hole found! \n>> Insert!")
            break
        elif count == len(spiral_x):
            raise NotImplementedError("Hole not found!")

    # adding a subsequent linear path
    linear_start_pos = pos
    linear_goal_pos = linear_start_pos + np.array([0, 0, -0.025])
    # print("obstacle_pos = ", obstacle_pos)
    # print("linear_start_pos = ", linear_start_pos)
    # print("linear_goal_pos = ", linear_goal_pos)
    robot_inik_solver = inik.IncrementalNIK(robot_s)
    linear_path = robot_inik_solver.gen_linear_motion(component_name='arm',
                                                      start_tcp_pos=linear_start_pos,
                                                      start_tcp_rotmat=orn,
                                                      goal_tcp_pos=linear_goal_pos,
                                                      goal_tcp_rotmat=orn,
                                                      obstacle_list=[], granularity=5e-5)
    for lp in linear_path:
        conf_list.append(lp)
        robot_s.fk(component_name='arm', jnt_values=lp)
        obj_pos = robot_s.get_gl_tcp()[0] + delta_pos_obj_hand
        objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_contact_orn))
        jawwidth_list.append(obj_diam)

    dict1 = {'conf_list': conf_list, 'objpose_list': objpose_list, 'jawwidth_list': jawwidth_list}
    pickle.dump(dict1, open('./spiral_search_motion_list.pickle', 'wb'))

    # plt.plot(np.linspace(0, len(error_list), len(error_list)), error_list)
    # plt.title('Error')
    # plt.show()

    robot_attached_list = []
    object_attached_list = []
    counter = [0]

    def update(robot_s, hnd_name, obj, robot_path, jawwidth_path, obj_path, robot_attached_list, object_attached_list, counter, task):

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
        objb_copy = obj.copy()
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1
        return task.again
    taskMgr.doMethodLater(0.02, update, "update",
                          extraArgs=[robot_s, hnd_name, obj,
                                     conf_list, jawwidth_list, objpose_list,
                                     robot_attached_list, object_attached_list, counter],
                          appendTask=True)
    base.setFrameRateMeter(True)
    base.run()
