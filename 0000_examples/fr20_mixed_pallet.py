import motion.optimization_based.incremental_nik as inik
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.fr20.fr20 as fr20
import numpy as np
import math
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[5, -5, 3], lookat_pos=[0, 0, -1])
    gm.gen_frame().attach_to(base)
    # object
    # object = cm.CollisionModel("./objects/bunnysim.stl")
    # object.set_pos(np.array([.55, -.3, 1.3]))
    # object.set_rgba([.5, .7, .3, 1])
    # object.attach_to(base)
    pallet_origin_pos = np.array([0.5, -1.5, -1])
    pallet_origin_rotmat = rm.rotmat_from_euler(0, 0, np.pi/2)
    gm.gen_frame(pallet_origin_pos, pallet_origin_rotmat).attach_to(base)

    bin_homomat = rm.homomat_from_posrot(pos=[0, -1, -0.5])
    gm.gen_box(homomat=bin_homomat, rgba=[0,0,0,0.1]).attach_to(base)

    box_sequence = np.array([[7, 12, 7, 0, 0, 0], [7, 13, 4, 0, 12, 0], [8, 6, 14, 7, 9, 0], [8, 10, 12, 7, 15, 0], [12, 9, 13, 7, 0, 0], [6, 9, 13, 19, 0, 0], [10, 13, 10, 15, 12, 0], [10, 3, 8, 15, 9, 0], [7, 13, 6, 0, 12, 4], [7, 12, 14, 0, 0, 7], [10, 3, 7, 15, 9, 8], [7, 13, 11, 0, 12, 10], [10, 13, 5, 15, 12, 10], [8, 10, 6, 17, 15, 15], [10, 9, 2, 7, 0, 13], [8, 9, 2, 17, 0, 13], [8, 6, 7, 7, 9, 14], [4, 9, 6, 7, 0, 15], [14, 9, 6, 11, 0, 15], [10, 3, 6, 15, 9, 15]])

    box_sequence = [np.concatenate((box[:3]*0.04, (box[3:]+box[:3]*0.5)*0.04)) for box in box_sequence]
    print(box_sequence)

    color_list = [[1,0,0,1], [1,1,0,1], [0,1,0,1], [0,0,1,1], [1,0,1,1]]
    for i, box in enumerate(box_sequence):
        box_homomat = np.dot(rm.homomat_from_posrot(pallet_origin_pos, pallet_origin_rotmat),
                             rm.homomat_from_posrot(pos=box[3:]))
        rgba = color_list[i % len(color_list)]
        gm.gen_box(extent=box[:3], homomat=box_homomat, rgba=rgba).attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_instance = fr20.ROBOT(hnd_attached=True)
    start_hnd_pos=np.array([0.4, -0.5, 1.3])
    start_hnd_rotmat=rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_hnd_pos=np.array([0.4, -0.3, 1.3])
    goal_hnd_rotmat=rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)

    pose_list = [np.radians([0, -120, 120, -90, -90, 0])]
    for jnt_values in pose_list:
        print(jnt_values)
        robot_instance.fk(component_name, jnt_values)
        robot_meshmodel = robot_instance.gen_meshmodel()
        robot_meshmodel.attach_to(base)
    base.run()