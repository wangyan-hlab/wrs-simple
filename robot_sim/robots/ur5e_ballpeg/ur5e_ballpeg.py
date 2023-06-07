import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur5e.ur5e as ur5e
import robot_sim.end_effectors.external_link.peg.peg as peg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import robot_sim.robots.robot_interface as ri

class UR5EBallPeg(ri.RobotInterface):

    """
        author: wangyan
        date: 2022/08/12, Suzhou
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur5e',
                 enable_cc=True, peg_attached=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        arm_homeconf = np.zeros(6)
        self.arm = ur5e.UR5E(pos=np.zeros(3), rotmat=np.eye(3), homeconf=arm_homeconf, enable_cc=False)
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.peg_attached = peg_attached
        if peg_attached:
            self.peg_rotmat = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            self.hnd = peg.PegLink(pos=self.arm.jnts[-1]['gl_posq'],
                                   rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'], self.peg_rotmat),
                                   enable_cc=False)
            self.arm.tcp_jnt_id = -1
            self.arm.tcp_loc_pos = self.hnd.center_pos
            self.arm.tcp_loc_rotmat = self.hnd.center_rotmat
            self.hnd_dict['arm'] = self.hnd
            self.hnd_dict['hnd'] = self.hnd
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        if self.peg_attached:
            self.cc.add_cdlnks(self.hnd.jlc, [0])
        activelist_arm = [self.arm.lnks[0],
                          self.arm.lnks[1],
                          self.arm.lnks[2],
                          self.arm.lnks[3],
                          self.arm.lnks[4],
                          self.arm.lnks[5],
                          self.arm.lnks[6]]
        if self.peg_attached:
            activelist_peg = [self.hnd.jlc.lnks[0]]
            activelist = activelist_arm + activelist_peg
        else:
            activelist = activelist_arm
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist_arm = [self.arm.lnks[3],
                    self.arm.lnks[5],
                    self.arm.lnks[6]]
        if self.peg_attached:
            intolist_peg = [self.hnd.jlc.lnks[0]]
            intolist = intolist_arm + intolist_peg
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[2]]
        intolist_arm = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6]]
        if self.peg_attached:
            intolist_peg = [self.hnd.jlc.lnks[0]]
            intolist = intolist_arm + intolist_peg
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3]]
        intolist_arm = [self.arm.lnks[6]]
        if self.peg_attached:
            intolist_peg = [self.hnd.jlc.lnks[0]]
            intolist = intolist_arm + intolist_peg
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)

    def get_hnd_on_manipulator(self, manipulator_name):
        if manipulator_name == 'arm':
            return self.hnd
        else:
            raise ValueError("The given jlc does not have a hand!")

    def get_gl_tcp(self, manipulator_name="arm"):
        return super().get_gl_tcp(manipulator_name=manipulator_name)

    def get_jnt_values(self, component_name="arm"):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.arm.fix_to(self.pos, self.rotmat)
        if self.peg_attached:
            self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'],
                            rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'], self.peg_rotmat))

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: nparray 1x6
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka, 20210403osaka
        """

        def update_component(component_name='arm', jnt_values=np.zeros(6)):
            self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            if self.peg_attached:
                self.hnd_dict[component_name].fix_to(
                        pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                        rotmat=np.dot(self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'], self.peg_rotmat))
        super().fk(component_name, jnt_values)
        # examine length
        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not available!")

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='ur5e_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=True,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if self.peg_attached:
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        return meshmodel

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ur5e_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if self.peg_attached:
            self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[-2, -2, 1], lookat_pos=[0, 0, 0], w=960, h=720)
    gm.gen_frame().attach_to(base)
    ur5e = UR5EBallPeg(enable_cc=True, peg_attached=True)
    pos1 = np.array([-0.03,  0.31,  0.42])
    rotmat1 = np.array([[0,  0.866, -0.5],
                        [0,  0.5,   0.866],
                        [1,  0,     0]])
    conf1 = ur5e.ik(component_name="arm", tgt_pos=pos1, tgt_rotmat=rotmat1)
    ur5e.fk(component_name="arm", jnt_values=conf1)
    ur5e.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
    pos2 = np.array([-0.03, 0.51, 0.42])
    rotmat2 = np.array([[0, 0.866, -0.5],
                        [0, 0.5, 0.866],
                        [1, 0, 0]])
    conf2 = np.radians([90.91, -97.36, 113.41, -106.04, -90.0, 0.89])
    ur5e.fk(component_name="arm", jnt_values=conf2)
    ur5e.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    base.run()
