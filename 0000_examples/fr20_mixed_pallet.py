import motion.optimization_based.incremental_nik as inik
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import modeling.dynamics.bullet.bdmodel as bdm
import robot_sim.robots.fr20.fr20 as robot
import numpy as np
import math
import basis.robot_math as rm
import time
from test_inference import BppAgent
from unified_test import registration_envs
from example_sequences import SEQS
import argparse

if __name__ == '__main__':
    # inference box sequences
    # from acktr.arguments import get_args

    # registration_envs()
    # args = get_args()
    # tester = BppAgent(args=args)
    # bin_size = tester.bin_size

    # if not args.real:
    #     tester.test_sim()
    # else:
    #     tester.test_real()

    # sequences = tester.sequences

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seq_id', type=int, default=-1, help='sequence id to simulate'
    )
    args = parser.parse_args()

    sequences = SEQS
    selected_seq_id = args.seq_id

    color_list = ['red','tomato','pink','orange','yellow', 'green','cyan','blue','magenta','purple']

    # 3d simulation
    base = wd.World(cam_pos=[5, -5, 3], lookat_pos=[0, 0, -1])
    gm.gen_frame().attach_to(base)

    # generate the pallet frame
    pallet_l, pallet_w, pallet_h = 1, 1, 1
    pallet_size = np.array([pallet_l, pallet_w, pallet_h])
    pallet_origin_pos = np.array([1.5, -0.5, -0.5])
    pallet_origin_rotmat = rm.rotmat_from_euler(0, 0, np.pi/2)
    pallet_origin_homomat = rm.homomat_from_posrot(pallet_origin_pos, pallet_origin_rotmat)
    gm.gen_frame(pallet_origin_pos, pallet_origin_rotmat).attach_to(base)
    # generate the bin
    bin_pos = pallet_origin_pos + .5*np.array([-pallet_size[0], pallet_size[1], pallet_size[2]])
    bin_homomat = rm.homomat_from_posrot(pos=bin_pos)
    gm.gen_box(homomat=bin_homomat, rgba=[0,0,0,0.1]).attach_to(base)
    
    # generate the real box sequence
    grid_size = 20 * 0.001
    box_sequence = np.array(sequences[selected_seq_id])
    box_sequence = [np.concatenate((box[:3]*grid_size, (box[3:]+box[:3]*0.5)*grid_size)) for box in box_sequence]
    box_suction_sequence = [box[3:]+np.array([0,0,box[2]*.5]) for box in box_sequence]
    robot_goals = []
    for box_suction in box_suction_sequence:
        box_suction_homomat = rm.homomat_from_posrot(box_suction)
        box_suction_pos = np.dot(pallet_origin_homomat, box_suction_homomat)[:3, 3]
        box_suction_rotmat = np.dot(np.dot(pallet_origin_homomat, box_suction_homomat)[:3, :3],
                                    rm.rotmat_from_euler(0, np.pi, -np.pi/2))
        robot_goals.append([box_suction_pos, box_suction_rotmat])
        gm.gen_frame(box_suction_pos, box_suction_rotmat).attach_to(base)
    # print(len(box_sequence), box_sequence)

    alpha = 0.4
    color_list = [[1,0,0,alpha], [1,1,0,alpha], [0,1,0,alpha], [0,0,1,alpha], [1,0,1,alpha]]

    # robot_s
    component_name = 'arm'
    robot_instance = robot.ROBOT(hnd_attached=True)
    start_jnt = np.radians([0, -120, 120, -90, -90, 0])
    robot_instance.fk(component_name, start_jnt)
    robot_instance.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    goal_jnts = []
    for goal_id, robot_goal in enumerate(robot_goals):
        goal_jnt = robot_instance.ik(tgt_pos=robot_goal[0], 
                                     tgt_rotmat=robot_goal[1],
                                     seed_jnt_values=start_jnt)
        if goal_jnt is not None:
            goal_jnts.append(goal_jnt)
        else:
            print("Infeasible goal_id:", goal_id)
            goal_jnts.append(goal_jnts[-1])
    
    def update(box_sequence, goal_jnts, show_box_sequence, show_robot_sequence, robot, component_name, counter, color_list, task):
        if counter[0] < len(box_sequence):
            if base.inputmgr.keymap['space']:
                if show_robot_sequence[0] is not None:
                    show_robot_sequence[0].detach()
                print("box index:", counter[0])
                box = box_sequence[counter[0]]
                box_homomat = np.dot(rm.homomat_from_posrot(pallet_origin_pos, pallet_origin_rotmat),
                                rm.homomat_from_posrot(pos=box[3:]))
                rgba = color_list[counter[0] % len(color_list)]
                show_box_sequence.append(gm.gen_box(extent=box[:3], homomat=box_homomat, rgba=rgba))
                print("lenboxseq:", len(show_box_sequence))
                show_box_sequence[-1].attach_to(base)
                robot.fk(component_name, goal_jnts[counter[0]])
                show_robot_sequence[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1,1,0,0.3])
                show_robot_sequence[0].attach_to(base)

                counter[0] += 1
                base.inputmgr.keymap['space'] = False
        else:
            if base.inputmgr.keymap['space']:
                print("reset")
                while len(show_box_sequence) > 0:
                    show_box_sequence.pop(0).detach()
                if show_robot_sequence[0] is not None:
                    show_robot_sequence[0].detach()
                counter[0] = 0
                base.inputmgr.keymap['space'] = False

        return task.cont
    
    show_box_sequence = []
    show_robot_sequence = [None]
    counter = [0]
    taskMgr.add(update, "addbox", 
                extraArgs=[box_sequence, goal_jnts, show_box_sequence, show_robot_sequence, robot_instance, component_name, counter, color_list], 
                appendTask=True)
    base.run()