import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.fr20.fr20 as fr
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri

class ROBOT(ri.RobotInterface):

    """
        author: wangyan
        date: 2023/07/14, Suzhou
    """
    
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='fr20', homeconf=np.zeros(6),
                 enable_cc=True, hnd_attached=True, zrot_to_gndbase=np.radians(0)):
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

        self.arm = fr.FR20(pos=self.ground_base.jnts[0]['gl_posq'],
                          rotmat=np.dot(self.ground_base.jnts[0]['gl_rotmatq'],
                                        rm.rotmat_from_euler(0, 0, zrot_to_gndbase)),
                          homeconf=homeconf,
                          enable_cc=False)
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_attached = hnd_attached
        if hnd_attached:
            self.hnd_pos = np.array([0, 0, 0])
            self.hnd_rotmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            self.hnd = rtq.Robotiq85(pos=np.dot(self.arm.jnts[-1]['gl_rotmatq'], self.hnd_pos)+self.arm.jnts[-1]['gl_posq'],
                                     rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'], self.hnd_rotmat),
                                     enable_cc=False)
            # tool center point
            self.arm.tcp_jnt_id = -1
            self.arm.tcp_loc_pos = self.hnd_rotmat.dot(self.hnd.jaw_center_pos) + self.hnd_pos
            self.arm.tcp_loc_rotmat = self.hnd_rotmat.dot(self.hnd.jaw_center_rotmat)
            # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
            self.oih_infos = []
            self.hnd_dict['arm'] = self.hnd
            self.hnd_dict['hnd'] = self.hnd
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
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        if self.hnd_attached:
            self.cc.add_cdlnks(self.hnd.lft_outer, [0,1,2,3,4])
            self.cc.add_cdlnks(self.hnd.lft_inner, [1])
            self.cc.add_cdlnks(self.hnd.rgt_outer, [1,2,3,4])
            self.cc.add_cdlnks(self.hnd.rgt_inner, [1])
        # lnks used for cd with external stationary objects
        activelist_arm = [self.arm.lnks[0],
                          self.arm.lnks[1],
                          self.arm.lnks[2],
                          self.arm.lnks[3],
                          self.arm.lnks[4],
                          self.arm.lnks[5],
                          self.arm.lnks[6]]
        if self.hnd_attached:
            activelist_hnd = [self.hnd.lft_outer.lnks[0],
                              self.hnd.lft_outer.lnks[1],
                              self.hnd.lft_outer.lnks[2],
                              self.hnd.lft_outer.lnks[3],
                              self.hnd.lft_outer.lnks[4],
                              self.hnd.rgt_outer.lnks[1],
                              self.hnd.rgt_outer.lnks[2],
                              self.hnd.rgt_outer.lnks[3],
                              self.hnd.rgt_outer.lnks[4]]
            activelist = activelist_arm + activelist_hnd
        else:
            activelist = activelist_arm
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        fromlist = [self.ground_base.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist_arm = [self.arm.lnks[3],
                        self.arm.lnks[4],
                        self.arm.lnks[5],
                        self.arm.lnks[6]]
        if self.hnd_attached:
            intolist_hnd = [self.hnd.lft_outer.lnks[0],
                            self.hnd.lft_outer.lnks[1],
                            self.hnd.lft_outer.lnks[2],
                            self.hnd.lft_outer.lnks[3],
                            self.hnd.lft_outer.lnks[4],
                            self.hnd.rgt_outer.lnks[1],
                            self.hnd.rgt_outer.lnks[2],
                            self.hnd.rgt_outer.lnks[3],
                            self.hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[2]]
        intolist_arm = [self.arm.lnks[4],
                        self.arm.lnks[5],
                        self.arm.lnks[6]]
        if self.hnd_attached:
            intolist_hnd = [self.hnd.lft_outer.lnks[0],
                            self.hnd.lft_outer.lnks[1],
                            self.hnd.lft_outer.lnks[2],
                            self.hnd.lft_outer.lnks[3],
                            self.hnd.lft_outer.lnks[4],
                            self.hnd.rgt_outer.lnks[1],
                            self.hnd.rgt_outer.lnks[2],
                            self.hnd.rgt_outer.lnks[3],
                            self.hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3]]
        intolist_arm = [self.arm.lnks[6]]
        if self.hnd_attached:
            intolist_hnd = [self.hnd.lft_outer.lnks[0],
                            self.hnd.lft_outer.lnks[1],
                            self.hnd.lft_outer.lnks[2],
                            self.hnd.lft_outer.lnks[3],
                            self.hnd.lft_outer.lnks[4],
                            self.hnd.rgt_outer.lnks[1],
                            self.hnd.rgt_outer.lnks[2],
                            self.hnd.rgt_outer.lnks[3],
                            self.hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
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

    def align_axis_down(self):
        seed_jnt_values = self.arm.get_jnt_values()
        position = self.arm.get_gl_tcp()[0]
        orientation_new = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        jnt_values = self.arm.ik(tgt_pos=position, tgt_rotmat=orientation_new, seed_jnt_values=seed_jnt_values)
        return jnt_values

    def fix_to(self, pos, rotmat):
        super().fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.ground_base.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.ground_base.jnts[0]['gl_posq'],
                        rotmat=np.dot(self.ground_base.jnts[0]['gl_rotmatq'],
                                      rm.rotmat_from_euler(0,0,0)))
        if self.hnd_attached:
            self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'],
                            rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'], self.hnd_rotmat))

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
            if self.hnd_attached:
                self.hnd_dict[component_name].fix_to(
                    pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                    rotmat=np.dot(self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'], self.hnd_rotmat))

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

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        # TODO “ValueError: The link needs to be added to collider using the addjlcobj function first!”
        
        rel_pos, rel_rotmat = self.arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))

        return rel_pos, rel_rotmat
    
    def get_loc_pose_from_hio(self, hio_pos, hio_rotmat, component_name='arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        arm = self.arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return self.cvt_gl_to_loc_tcp(component_name, gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3])

    def get_gl_pose_from_hio(self, hio_pos, hio_rotmat, component_name='arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        arm = self.arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]

    def get_oih_cm_list(self, hnd_name='hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        """
        oih_infos = self.oih_infos
        return_list = []
        for obj_info in oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def get_oih_glhomomat_list(self, hnd_name='hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        oih_infos = self.oih_infos
        return_list = []
        for obj_info in oih_infos:
            return_list.append(rm.homomat_from_posrot(obj_info['gl_pos']), obj_info['gl_rotmat'])
        return return_list

    def get_oih_relhomomat(self, objcm, hnd_name='hnd'):
        """
        TODO: useless? 20210320
        oih = object in hand list
        :param objcm
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        oih_info_list = self.oih_infos
        for obj_info in oih_info_list:
            if obj_info['collisionmodel'] is objcm:
                return rm.homomat_from_posrot(obj_info['rel_pos']), obj_info['rel_rotmat']

    def release(self, hnd_name, objcm, jaw_width=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :param hnd_name:
        :return:
        """
        oih_infos = self.oih_infos
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                oih_infos.remove(obj_info)
                break

    def release_all(self, jaw_width=None, hnd_name='hnd'):
        """
        release all objects from the specified hand
        :param jaw_width:
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210125
        """
        oih_infos = self.oih_infos
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            self.cc.delete_cdobj(obj_info)
        oih_infos.clear()

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='fr20_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.ground_base.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 rgba=rgba).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if self.hnd_attached:
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
                       name='fr20_stickmodel'):
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
        if self.hnd_attached:
            self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, -2, 1], lookat_pos=[0, 0, 0], w=960, h=720)
    gm.gen_frame().attach_to(base)
    fr20 = ROBOT()
    conf1 = np.radians([0, 0, 0, 0, 0, 0])
    fr20.fk(component_name="arm", jnt_values=conf1)
    print("collision=", fr20.is_collided())
    fr20.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    arm_jacobian_offset = np.array([0, 0, .145])
    fr20 = ROBOT(hnd_attached=True)
    conf2 = np.radians([-93, -98, -73, -97, 90, 0])
    fr20.fk(component_name="arm", jnt_values=conf2)
    print("global_tcp=", fr20.get_gl_tcp())
    print("collision=", fr20.is_collided())
    print("jacobian=", fr20.jacobian())
    # print("manipulability=", fr20.manipulability())
    fr20.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # fr20.show_cdprimit()  # show the collision model

    # ns = rm.null_space(fr20.jacobian())
    # print("null space = ", ns)
    # print("check = ", np.dot(fr20.jacobian(), ns))

    base.run()