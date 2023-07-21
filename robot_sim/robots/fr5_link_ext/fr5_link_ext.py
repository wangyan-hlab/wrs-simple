import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri

class FR5_robot(ri.RobotInterface):

    """
        author: wangyan
        date: 2022/03/14, Suzhou
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='fr5', homeconf=np.zeros(7), enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.ground_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name="fr5_to_ground_base")
        self.ground_base.jnts[0]['loc_pos'] = np.array([0, 0, 0])
        self.ground_base.lnks[0]['name'] = "ground_base"
        self.ground_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.ground_base.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes/ground_base.stl"),
            cdprimit_type="user_defined", expand_radius=.002,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.ground_base.lnks[0]['rgba'] = [.5, .5, .5, 1.0]
        self.ground_base.reinitialize()

        self.arm = jl.JLChain(pos=self.ground_base.jnts[0]['gl_posq'],
                              rotmat=np.dot(self.ground_base.jnts[0]['gl_rotmatq'],
                                        rm.rotmat_from_euler(0, 0, np.pi*135/180)),
                              homeconf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.arm.jnts[1]['loc_pos'] = np.array([0, 0, 0.152])
        self.arm.jnts[1]['motion_rng'] = np.deg2rad([-175, 175])
        self.arm.jnts[2]['loc_pos'] = np.array([0, 0.138, 0])
        self.arm.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(.0, np.pi/2, .0)
        self.arm.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.arm.jnts[2]['motion_rng'] = np.deg2rad([-265, 85])
        self.arm.jnts[3]['loc_pos'] = np.array([0, -0.138, 0.425])
        self.arm.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.arm.jnts[3]['motion_rng'] = np.deg2rad([-160, 160])
        self.arm.jnts[4]['loc_pos'] = np.array([.0, .0, 0.395])
        self.arm.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(.0, np.pi/2, 0)
        self.arm.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.arm.jnts[4]['motion_rng'] = np.deg2rad([-265, 85])
        self.arm.jnts[5]['loc_pos'] = np.array([0, 0.102, 0])
        self.arm.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.arm.jnts[5]['motion_rng'] = np.deg2rad([-175, 175])
        self.arm.jnts[6]['loc_pos'] = np.array([0, 0, .102])
        self.arm.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.arm.jnts[6]['motion_rng'] = np.deg2rad([-175, 175])
        self.arm.jnts[7]['loc_pos'] = np.array([0, .100, 0])
        self.arm.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-np.pi/2, 0, 0)
        # links
        arm_color1 = [.65, .65, .65, 1.0]
        arm_color2 = [0.8, 0.2, 0.0, 1.0]
        self.arm.lnks[0]['name'] = "base"
        self.arm.lnks[0]['loc_pos'] = np.zeros(3)
        self.arm.lnks[0]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[0]['mass'] = 2.0
        self.arm.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.arm.lnks[0]['rgba'] = arm_color1
        self.arm.lnks[1]['name'] = "shoulder"
        self.arm.lnks[1]['loc_pos'] = np.zeros(3)
        self.arm.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.arm.lnks[1]['mass'] = 1.95
        self.arm.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "shoulder.stl")
        self.arm.lnks[1]['rgba'] = arm_color2
        self.arm.lnks[2]['name'] = "upperarm"
        self.arm.lnks[2]['loc_pos'] = np.zeros(3)
        self.arm.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.arm.lnks[2]['mass'] = 3.42
        self.arm.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "upperarm.stl")
        self.arm.lnks[2]['rgba'] = arm_color1
        self.arm.lnks[3]['name'] = "forearm"
        self.arm.lnks[3]['loc_pos'] = np.zeros(3)
        self.arm.lnks[3]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.arm.lnks[3]['mass'] = 1.437
        self.arm.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "forearm.stl")
        self.arm.lnks[3]['rgba'] = arm_color2
        self.arm.lnks[4]['name'] = "wrist1"
        self.arm.lnks[4]['loc_pos'] = np.zeros(3)
        self.arm.lnks[4]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.arm.lnks[4]['mass'] = 0.871
        self.arm.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist1.stl")
        self.arm.lnks[4]['rgba'] = arm_color1
        self.arm.lnks[5]['name'] = "wrist2"
        self.arm.lnks[5]['loc_pos'] = np.zeros(3)
        self.arm.lnks[5]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.arm.lnks[5]['mass'] = 0.8
        self.arm.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist2.stl")
        self.arm.lnks[5]['rgba'] = arm_color2
        self.arm.lnks[6]['name'] = "wrist3"
        self.arm.lnks[6]['loc_pos'] = np.zeros(3)
        self.arm.lnks[6]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[6]['com'] = np.array([.0, .0, -0.02])
        self.arm.lnks[6]['mass'] = 0.8
        self.arm.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist3.stl")
        self.arm.lnks[6]['rgba'] = arm_color1
        self.arm.lnks[7]['loc_pos'] = np.zeros(3)
        self.arm.lnks[7]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)
        self.arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "ext_link2.stl")
        self.arm.lnks[7]['rgba'] = [.8, .8, 0, 1.0]

        self.arm.reinitialize()
        self.manipulator_dict['arm'] = self.arm
        # collision detection
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
        self.cc.add_cdlnks(self.ground_base, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6, 7])
        # lnks used for cd with external stationary objects
        activelist = [self.arm.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.arm.lnks[7]
                      ]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        fromlist = [self.ground_base.lnks[0]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.arm.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.arm.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[2]]
        intolist = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.arm.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3]]
        intolist = [self.arm.lnks[6],
                    self.arm.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)

    def get_gl_tcp(self, manipulator_name="arm"):
        return super().get_gl_tcp(manipulator_name=manipulator_name)

    def get_jnt_values(self, component_name="arm"):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()

    def fix_to(self, pos, rotmat):
        super().fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.ground_base.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.ground_base.jnts[0]['gl_posq'],
                        rotmat=self.ground_base.jnts[0]['gl_rotmatq'])

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: nparray 1x6
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka, 20210403osaka
        """

        def update_component(component_name='arm', jnt_values=np.zeros(7)):
            self.manipulator_dict[component_name].fk(jnt_values=jnt_values)

        super().fk(component_name, jnt_values)
        # examine length
        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            update_component(component_name, jnt_values)
        elif component_name == "fr5_to_ground_base":
            self.ground_base.fk(jnt_values)
            self.arm.fix_to(pos=self.ground_base.jnts[0]['gl_posq'],
                            rotmat=self.ground_base.jnts[0]['gl_rotmatq'])
        else:
            raise ValueError("The given component name is not available!")

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='fr5_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.ground_base.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=toggle_tcpcs,
                                 toggle_jntscs=toggle_jntscs,
                                 rgba=rgba).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
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
                       name='fr5_stickmodel'):
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
        return stickmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0], w=960, h=720)
    gm.gen_frame().attach_to(base)
    fr5 = FR5_robot()
    conf1 = np.radians([0, 0, 0, 0, 0, 0, 0])
    fr5.fk(component_name="arm", jnt_values=conf1)
    print("collision=", fr5.is_collided())
    fr5.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    conf2 = np.radians([0, -90, 90, 0, -90, 0, 0])
    fr5.fk(component_name="arm", jnt_values=conf2)
    print("collision=", fr5.is_collided())
    # print("global_tcp=", fr5.get_gl_tcp())
    # print("jacobian=", fr5.jacobian())
    # print("manipulability=", fr5.manipulability())
    fr5.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, 1]).attach_to(base)
    fr5.show_cdprimit()  # show the collision model

    # ns = rm.null_space(fr5.jacobian())
    # print("null space = ", ns)
    # print("check = ", np.dot(fr5.jacobian(), ns))

    base.run()