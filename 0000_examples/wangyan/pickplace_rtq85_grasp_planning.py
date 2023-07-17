import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import grasping.annotation.utils as gau
# import robot_sim.end_effectors.gripper.schunkrh918.schunkrh918 as hnd
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as hnd

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
obj = cm.CollisionModel("../objects/milkcarton.stl")
obj.set_rgba([.9, .75, .35, 1])
obj.set_scale([.4, .4, .4])
obj.attach_to(base)
# hnd_s
# gripper_s = hnd.SchunkRH918()
gripper_s = hnd.Robotiq85()
# grasp_info_list = gpa.plan_grasps(gripper_s, obj,
#                                   openning_direction='loc_y',
#                                   max_samples=100,
#                                   min_dist_between_sampled_contact_points=.05)
grasp_info_list = gau.define_grasp_with_rotation(gripper_s,
                                                 obj,
                                                 gl_jaw_center_pos=np.array([0,0,0.06]),
                                                 gl_jaw_center_z=np.array([-1,0,0]),
                                                 gl_jaw_center_y=np.array([0,1,0]),
                                                 jaw_width=0.0324,
                                                 gl_rotation_ax=np.array([0,1,0]),
                                                 rotation_interval=np.radians(5))
gpa.write_pickle_file('milkcarton', grasp_info_list, './', 'rtq85_milkcarton.pickle')
for grasp_info in grasp_info_list:
    jaw_width, gl_jaw_center, gl_jaw_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.jaw_to(jaw_width)
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.gen_meshmodel(rgba=[.1,.1,.1,.2]).attach_to(base)
base.run()