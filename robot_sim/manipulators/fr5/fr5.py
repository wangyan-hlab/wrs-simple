import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi

class FR5(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6),
                 name='fr5', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name, cdprimitive_type="cylinder")
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.155])
        self.jlc.jnts[1]['motion_rng'] = np.deg2rad([-175, 175])
        self.jlc.jnts[2]['loc_pos'] = np.array([0, 0.138, 0])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(.0, np.pi/2, .0)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['motion_rng'] = np.deg2rad([-265, 85])
        self.jlc.jnts[3]['loc_pos'] = np.array([0, -0.138, 0.425])
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[3]['motion_rng'] = np.deg2rad([-160, 160])
        self.jlc.jnts[4]['loc_pos'] = np.array([.0, .0, 0.395])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(.0, np.pi/2, 0)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[4]['motion_rng'] = np.deg2rad([-265, 85])
        self.jlc.jnts[5]['loc_pos'] = np.array([0, 0.130, 0])
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[5]['motion_rng'] = np.deg2rad([-175, 175])
        self.jlc.jnts[6]['loc_pos'] = np.array([0, 0, .102])
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[6]['motion_rng'] = np.deg2rad([-175, 175])
        self.jlc.jnts[7]['loc_pos'] = np.array([0, .102, 0])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-np.pi/2, 0, 0)
        # links
        arm_color1 = [.65, .65, .65, 1.0]
        arm_color2 = [0.6, 0.5, 0.3, 1.0]
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 2.0
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.jlc.lnks[0]['rgba'] = arm_color1
        self.jlc.lnks[1]['name'] = "shoulder"
        self.jlc.lnks[1]['loc_pos'] = np.array([.0, .0, -0.155])
        self.jlc.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(.0, .0, math.pi)
        self.jlc.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.jlc.lnks[1]['mass'] = 1.95
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "shoulder.stl")
        self.jlc.lnks[1]['rgba'] = arm_color2
        self.jlc.lnks[2]['name'] = "upperarm"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, -.138, .0])
        self.jlc.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(math.pi/2, 0, math.pi)
        self.jlc.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.jlc.lnks[2]['mass'] = 3.42
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "upperarm.stl")
        self.jlc.lnks[2]['rgba'] = arm_color1
        self.jlc.lnks[3]['name'] = "forearm"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi / 2, math.pi/2)
        self.jlc.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.jlc.lnks[3]['mass'] = 1.437
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "forearm.stl")
        self.jlc.lnks[3]['rgba'] = arm_color2
        self.jlc.lnks[4]['name'] = "wrist1"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi/2, math.pi/2, 0)
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[4]['mass'] = 0.871
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist1.stl")
        self.jlc.lnks[4]['rgba'] = arm_color1
        self.jlc.lnks[5]['name'] = "wrist2"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[5]['mass'] = 0.8
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist2.stl")
        self.jlc.lnks[5]['rgba'] = arm_color2
        self.jlc.lnks[6]['name'] = "wrist3"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, 0, 0)
        self.jlc.lnks[6]['com'] = np.array([.0, .0, -0.02])
        self.jlc.lnks[6]['mass'] = 0.8
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist3.stl")
        self.jlc.lnks[6]['rgba'] = arm_color1
        self.jlc.reinitialize()
        # collision checker
        if enable_cc:
            super().enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = FR5(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel(rgba=[1,1,1,0.3])
    manipulator_meshmodel.attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)

    manipulator_instance = FR5(enable_cc=True)
    manipulator_instance.fix_to(pos=[0, .5, 0], rotmat=rm.rotmat_from_euler(0, 0, 0))
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)

    base.run()