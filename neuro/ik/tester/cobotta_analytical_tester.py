import torch
import math
import time
import numpy as np
import pandas as pd
import neuro.ik.cobotta_fitting as cbf
import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as world
import robot_sim.robots.cobotta.cobotta as cbt_s

if __name__ == '__main__':
    base = world.World(cam_pos=np.array([1.5, 1, .7]))
    gm.gen_frame().attach_to(base)
    rbt_s = cbt_s.Cobotta()
    rbt_s.fk(jnt_values=np.zeros(6))
    rbt_s.gen_meshmodel(toggle_tcpcs=True, rgba=[.5,.5,.5,.3]).attach_to(base)
    rbt_s.gen_stickmodel(toggle_tcpcs=True).attach_to(base)
    tgt_pos = np.array([.25, .2, .2])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 3 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    contraint_pos = rbt_s.manipulator_dict['arm'].jnts[5]['gl_posq']
    contraint_rotmat = rbt_s.manipulator_dict['arm'].jnts[5]['gl_rotmatq']
    gm.gen_frame(pos=contraint_pos, rotmat=contraint_rotmat).attach_to(base)

    # numerical ik
    jnt_values = rbt_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    rbt_s.fk(jnt_values=jnt_values)
    rbt_s.gen_meshmodel(toggle_tcpcs=True, rgba=[.5,.5,.5,.3]).attach_to(base)
    rbt_s.gen_stickmodel(toggle_tcpcs=True).attach_to(base)
    contraint_pos = rbt_s.manipulator_dict['arm'].jnts[5]['gl_posq']
    contraint_rotmat = rbt_s.manipulator_dict['arm'].jnts[5]['gl_rotmatq']
    contraint_pos += rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1] * contraint_rotmat[:, 1]
    gm.gen_frame(pos=contraint_pos, rotmat=contraint_rotmat).attach_to(base)
    gm.gen_torus(contraint_rotmat[:, 2],
                 starting_vector=contraint_rotmat[:, 1],
                 portion=1,
                 center=contraint_pos,
                 radius=abs(rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1])).attach_to(base)

    gm.gen_torus(contraint_rotmat[:, 2],
                 starting_vector=contraint_rotmat[:, 1],
                 portion=1,
                 center=contraint_pos,
                 radius=abs(rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1])+abs(rbt_s.manipulator_dict['arm'].jnts[5]['loc_pos'][1])).attach_to(base)

    gm.gen_torus(contraint_rotmat[:, 2],
                 starting_vector=contraint_rotmat[:, 1],
                 portion=1,
                 center=contraint_pos-rbt_s.manipulator_dict['arm'].jnts[1]['gl_rotmatq'].dot(np.array([0,rbt_s.manipulator_dict['arm'].jnts[4]['loc_pos'][1],0])),
                 radius=abs(rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1])+abs(rbt_s.manipulator_dict['arm'].jnts[5]['loc_pos'][1])).attach_to(base)
    # jnt_values[3:]=0
    # rbt_s.fk(jnt_values=jnt_values)
    # rbt_s.gen_meshmodel(toggle_tcpcs=True, rgba=[.5,.5,.5,.3]).attach_to(base)
    # rbt_s.gen_stickmodel(toggle_tcpcs=True).attach_to(base)
    #
    # jnt_values[2:]=0
    # rbt_s.fk(jnt_values=jnt_values)
    # rbt_s.gen_meshmodel(toggle_tcpcs=True, rgba=[.5,.5,.5,.3]).attach_to(base)
    # rbt_s.gen_stickmodel(toggle_tcpcs=True).attach_to(base)
    #
    # gm.gen_sphere(pos=rbt_s.manipulator_dict['arm'].jnts[2]['gl_posq'],
    #               radius=np.linalg.norm(rbt_s.manipulator_dict['arm'].jnts[3]['loc_pos']),
    #               rgba=[.5,.5,.5,.2], subdivisions=5).attach_to(base)
    # contraint_pos = rbt_s.manipulator_dict['arm'].jnts[2]['gl_posq']
    # contraint_rotmat = rbt_s.manipulator_dict['arm'].jnts[2]['gl_rotmatq']
    # gm.gen_frame(pos=contraint_pos, rotmat=contraint_rotmat).attach_to(base)
    # gm.gen_torus(rbt_s.manipulator_dict['arm'].jnts[1]['gl_rotmatq'][:,2],
    #              starting_vector=rbt_s.manipulator_dict['arm'].jnts[1]['gl_rotmatq'][:,0],
    #              portion=1,
    #              center=contraint_pos,
    #              radius=abs(rbt_s.manipulator_dict['arm'].jnts[3]['loc_pos'][1]) + abs(
    #                  rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1])).attach_to(base)
    # # neural ik
    # model = cbf.Net(n_hidden=100, n_jnts=6)
    # model.load_state_dict(torch.load("cobotta_model.pth"))
    # tgt_rpy = rm.rotmat_to_euler(tgt_rotmat)
    # xyzrpy = torch.from_numpy(np.hstack((tgt_pos,tgt_rpy)))
    # jnt_values = model(xyzrpy.float()).to('cpu').detach().numpy()
    # rbt_s.fk(jnt_values=jnt_values)
    # rbt_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    base.run()
