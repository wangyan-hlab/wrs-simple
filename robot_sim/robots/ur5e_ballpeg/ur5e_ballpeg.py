import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur5e.ur5e as ur5e
import robot_sim.end_effectors.external_link.peg.peg as peg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3

class UR5EBallPeg(ri.RobotInterface):

    """
        author: wangyan
        date: 2022/08/12, Suzhou
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), 
                 name='ur5e',
                 homeconf = np.zeros(6),
                 enable_cc=True, 
                 peg_attached=True, 
                 zrot_to_gndbase=np.radians(135)):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.ground_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name="robot_to_ground_base")
        self.ground_base.jnts[0]['loc_pos'] = np.array([0, 0, 0])
        self.ground_base.lnks[0]['name'] = "ground_base"
        self.ground_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.ground_base.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes/ground_base.stl"),
            cdprimit_type="user_defined", expand_radius=.002,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.ground_base.lnks[0]['rgba'] = [.5, .5, .5, 1.0]
        self.ground_base.reinitialize()

        
        self.arm = ur5e.UR5E(pos=np.zeros(3), rotmat=np.eye(3), homeconf=homeconf, enable_cc=False)
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

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.0, 0.0, -.01),
                                              x=.075 + radius, y=.075 + radius, z=.01 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3325),
                                              x=.05 + radius, y=.05 + radius, z=.3125 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.0, 0.0, -.655),
                                              x=.225 + radius, y=.225 + radius, z=.01 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(.1534, .1675, -.505),
                                              Point3(.2445, .2304, -.645))
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(-.1534, .1675, -.505),
                                              Point3(-.2445, .2304, -.645))
        collision_node.addSolid(collision_primitive_r0)
        collision_primitive_l1 = CollisionBox(Point3(-.1534, -.1675, -.505),
                                              Point3(-.2445, -.2304, -.645))
        collision_node.addSolid(collision_primitive_l1)
        collision_primitive_r1 = CollisionBox(Point3(.1534, -.1675, -.505),
                                              Point3(.2445, -.2304, -.645))
        collision_node.addSolid(collision_primitive_r1)
        return collision_node

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
                        self.arm.lnks[4],
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
        self.ground_base.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.ground_base.jnts[0]['gl_posq'],
                        rotmat=np.dot(self.ground_base.jnts[0]['gl_rotmatq'],
                                      rm.rotmat_from_euler(0,0,0)))
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
        elif component_name == "robot_to_ground_base":
            self.ground_base.fk(jnt_values)
            self.arm.fix_to(pos=self.ground_base.jnts[0]['gl_posq'],
                            rotmat=np.dot(self.ground_base.jnts[0]['gl_rotmatq'],
                                          rm.rotmat_from_euler(0,0,0)))
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
        self.ground_base.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 rgba=rgba).attach_to(meshmodel)
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
        self.ground_base.gen_stickmodel(tcp_loc_pos=None,
                                  tcp_loc_rotmat=None,
                                  toggle_tcpcs=False,
                                  toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
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
    conf = np.radians([90.91, -97.36, 113.41, -106.04, -90.0, 0.89])
    ur5e.fk(component_name="arm", jnt_values=conf)
    ur5e.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    base.run()
