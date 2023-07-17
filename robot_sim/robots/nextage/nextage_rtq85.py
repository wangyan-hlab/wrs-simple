import os
import copy
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri


class Nextage(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='nextage',
                 enable_cc=True, hnd_attached='bothhnd', mode='lft_arm_waist'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        central_homeconf = np.radians(np.array([.0, .0, .0]))
        lft_arm_homeconf = np.radians(np.array([central_homeconf[0], 15, 0, -143, 0, 0, 0]))
        rgt_arm_homeconf = np.radians(np.array([central_homeconf[0], -15, 0, -143, 0, 0, 0]))
        # central
        self.central_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=central_homeconf, name='centeral_body')
        self.central_body.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.central_body.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.central_body.jnts[1]['motion_rng'] = [-2.84489, 2.84489]
        self.central_body.jnts[2]['loc_pos'] = np.array([0, 0, 0.5695])
        self.central_body.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.central_body.jnts[2]['motion_rng'] = [-1.22173, 1.22173]
        self.central_body.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.central_body.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.central_body.jnts[3]['motion_rng'] = [-0.349066, 1.22173]
        self.central_body.lnks[0]['name'] = "nextage_base"
        self.central_body.lnks[0]['loc_pos'] = np.array([0, 0, 0.97])
        self.central_body.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "waist_link_mesh.dae"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._waist_combined_cdnp)
        self.central_body.lnks[0]['rgba'] = [.77, .77, .77, 1.0]
        self.central_body.lnks[1]['name'] = "nextage_chest"
        self.central_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.central_body.lnks[1]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "chest_joint0_link_mesh.dae"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._chest_combined_cdnp)
        self.central_body.lnks[1]['rgba'] = [.8, .8, .8, 1]
        self.central_body.lnks[2]['name'] = "head_joint0_link_mesh"
        self.central_body.lnks[2]['loc_pos'] = np.array([0, 0, 0.5695])
        self.central_body.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "head_joint0_link_mesh.dae")
        self.central_body.lnks[2]['rgba'] = [.35, .35, .35, 1]
        self.central_body.lnks[3]['name'] = "nextage_chest"
        self.central_body.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.central_body.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "head_joint1_link_mesh.dae")
        self.central_body.lnks[3]['rgba'] = [.63, .63, .63, 1]
        self.central_body.reinitialize()
        # lft
        self.lft_arm = jl.JLChain(pos=self.central_body.jnts[1]['gl_posq'],
                                  rotmat=self.central_body.jnts[1]['gl_rotmatq'],
                                  homeconf=lft_arm_homeconf, name='lft_arm')
        self.lft_arm.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_arm.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.lft_arm.jnts[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
        self.lft_arm.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
        self.lft_arm.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.lft_arm.jnts[2]['motion_rng'] = [-1.53589, 1.53589]
        self.lft_arm.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.lft_arm.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.lft_arm.jnts[3]['motion_rng'] = [-2.44346, 1.0472]
        self.lft_arm.jnts[4]['loc_pos'] = np.array([0, 0.095, -0.25])
        self.lft_arm.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.lft_arm.jnts[4]['motion_rng'] = [-2.75762, 0]
        self.lft_arm.jnts[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.lft_arm.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.lft_arm.jnts[5]['motion_rng'] = [-1.8326, 2.87979]
        self.lft_arm.jnts[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.lft_arm.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.lft_arm.jnts[6]['motion_rng'] = [-1.74533, 1.74533]
        self.lft_arm.jnts[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        self.lft_arm.jnts[7]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_arm.jnts[7]['motion_rng'] = [-2.84489, 2.84489]
        self.lft_arm.lnks[2]['name'] = "lft_arm_joint0"
        self.lft_arm.lnks[2]['loc_pos'] = np.array([0, 0.145, 0.370296])
        self.lft_arm.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(-0.261799, 0, 0)
        self.lft_arm.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint0_link_mesh.dae")
        self.lft_arm.lnks[2]['rgba'] = [.35, .35, .35, 1]
        self.lft_arm.lnks[3]['name'] = "lft_arm_joint1"
        self.lft_arm.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.lft_arm.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint1_link_mesh.dae")
        self.lft_arm.lnks[3]['rgba'] = [.57, .57, .57, 1]
        self.lft_arm.lnks[4]['name'] = "lft_arm_joint2"
        self.lft_arm.lnks[4]['loc_pos'] = np.array([0, 0.095, -0.25])
        self.lft_arm.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint2_link_mesh.dae")
        self.lft_arm.lnks[4]['rgba'] = [.35, .35, .35, 1]
        self.lft_arm.lnks[5]['name'] = "lft_arm_joint3"
        self.lft_arm.lnks[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.lft_arm.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint3_link_mesh.dae")
        self.lft_arm.lnks[5]['rgba'] = [.35, .35, .35, 1]
        self.lft_arm.lnks[6]['name'] = "lft_arm_joint4"
        self.lft_arm.lnks[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.lft_arm.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint4_link_mesh.dae")
        self.lft_arm.lnks[6]['rgba'] = [.7, .7, .7, 1]
        self.lft_arm.lnks[7]['name'] = "lft_arm_joint5"
        self.lft_arm.lnks[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        # self.lft_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint5_link_mesh.dae")
        self.lft_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint5_link_mesh_new.stl")
        self.lft_arm.lnks[7]['rgba'] = [.57, .57, .57, 1]
        self.lft_arm.reinitialize()
        # rgt
        self.rgt_arm = jl.JLChain(pos=self.central_body.jnts[1]['gl_posq'],
                                  rotmat=self.central_body.jnts[1]['gl_rotmatq'],
                                  homeconf=rgt_arm_homeconf, name='rgt_arm')
        self.rgt_arm.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_arm.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.rgt_arm.jnts[2]['loc_pos'] = np.array([0, -0.145, 0.370296])
        self.rgt_arm.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(0.261799, 0, 0)
        self.rgt_arm.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.rgt_arm.jnts[2]['motion_rng'] = [-1.53589, 1.53589]
        self.rgt_arm.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_arm.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.rgt_arm.jnts[3]['motion_rng'] = [-2.44346, 1.0472]
        self.rgt_arm.jnts[4]['loc_pos'] = np.array([0, -0.095, -0.25])
        self.rgt_arm.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.rgt_arm.jnts[4]['motion_rng'] = [-2.75762, 0]
        self.rgt_arm.jnts[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.rgt_arm.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.rgt_arm.jnts[5]['motion_rng'] = [-1.8326, 2.87979]
        self.rgt_arm.jnts[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.rgt_arm.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.rgt_arm.jnts[6]['motion_rng'] = [-1.74533, 1.74533]
        self.rgt_arm.jnts[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        self.rgt_arm.jnts[7]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_arm.jnts[7]['motion_rng'] = [-2.84489, 2.84489]
        self.rgt_arm.lnks[2]['name'] = "rgt_arm_joint0"
        self.rgt_arm.lnks[2]['loc_pos'] = np.array([0, -0.145, 0.370296])
        self.rgt_arm.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0.261799, 0, 0)
        self.rgt_arm.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint0_link_mesh.dae")
        self.rgt_arm.lnks[2]['rgba'] = [.35, .35, .35, 1]
        self.rgt_arm.lnks[3]['name'] = "rgt_arm_joint1"
        self.rgt_arm.lnks[3]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_arm.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint1_link_mesh.dae")
        self.rgt_arm.lnks[3]['rgba'] = [.57, .57, .57, 1]
        self.rgt_arm.lnks[4]['name'] = "rgt_arm_joint2"
        self.rgt_arm.lnks[4]['loc_pos'] = np.array([0, -0.095, -0.25])
        self.rgt_arm.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint2_link_mesh.dae")
        self.rgt_arm.lnks[4]['rgba'] = [.35, .35, .35, 1]
        self.rgt_arm.lnks[5]['name'] = "rgt_arm_joint3"
        self.rgt_arm.lnks[5]['loc_pos'] = np.array([-0.03, 0, 0])
        self.rgt_arm.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint3_link_mesh.dae")
        self.rgt_arm.lnks[5]['rgba'] = [.35, .35, .35, 1]
        self.rgt_arm.lnks[6]['name'] = "rgt_arm_joint4"
        self.rgt_arm.lnks[6]['loc_pos'] = np.array([0, 0, -0.235])
        self.rgt_arm.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint4_link_mesh.dae")
        self.rgt_arm.lnks[6]['rgba'] = [.7, .7, .7, 1]
        self.rgt_arm.lnks[7]['name'] = "rgt_arm_joint5"
        self.rgt_arm.lnks[7]['loc_pos'] = np.array([-0.047, 0, -0.09])
        # self.rgt_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "rarm_joint5_link_mesh.dae")
        self.rgt_arm.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "larm_joint5_link_mesh_new.stl")
        self.rgt_arm.lnks[7]['rgba'] = [.57, .57, .57, 1]
        self.rgt_arm.reinitialize()

        self.hnd_attached = hnd_attached
        self.hnd_origin_pos = np.array([-.079, 0, 0])
        self.hnd_origin_rotmat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        if hnd_attached == "bothhnd":
            self.lft_hnd = rtq85.Robotiq85(
                pos=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.lft_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat),
                enable_cc=False)
            self.rgt_hnd = rtq85.Robotiq85(
                pos=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.rgt_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat),
                enable_cc=False)
            # tool center point
            self.lft_arm.tcp_jntid = -1
            self.lft_arm.tcp_loc_pos = self.hnd_origin_rotmat.dot(self.lft_hnd.jaw_center_pos) + self.hnd_origin_pos
            self.lft_arm.tcp_loc_rotmat = self.hnd_origin_rotmat.dot(self.lft_hnd.jaw_center_rotmat)
            self.rgt_arm.tcp_jntid = -1
            self.rgt_arm.tcp_loc_pos = self.hnd_origin_rotmat.dot(self.rgt_hnd.jaw_center_pos) + self.hnd_origin_pos
            self.rgt_arm.tcp_loc_rotmat = self.hnd_origin_rotmat.dot(self.rgt_hnd.jaw_center_rotmat)
            # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
            self.lft_oih_infos = []
            self.rgt_oih_infos = []
            self.hnd_dict['lft_arm'] = self.lft_hnd
            self.hnd_dict['lft_hnd'] = self.lft_hnd
            self.hnd_dict['lft_arm_waist'] = self.lft_hnd
            self.hnd_dict['rgt_arm'] = self.rgt_hnd
            self.hnd_dict['rgt_hnd'] = self.rgt_hnd
            self.hnd_dict['rgt_arm_waist'] = self.rgt_hnd
        elif hnd_attached == "lft_hnd":
            self.lft_hnd = rtq85.Robotiq85(
                pos=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.lft_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat),
                enable_cc=False)
            # tool center point
            self.lft_arm.tcp_jntid = -1
            self.lft_arm.tcp_loc_pos = self.hnd_origin_rotmat.dot(self.lft_hnd.jaw_center_pos) + self.hnd_origin_pos
            self.lft_arm.tcp_loc_rotmat = self.hnd_origin_rotmat.dot(self.lft_hnd.jaw_center_rotmat)
            # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
            self.lft_oih_infos = []
            self.hnd_dict['lft_arm'] = self.lft_hnd
            self.hnd_dict['lft_hnd'] = self.lft_hnd
        elif hnd_attached == "rgt_hnd":
            self.rgt_hnd = rtq85.Robotiq85(
                pos=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.rgt_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat),
                enable_cc=False)
            # tool center point
            self.rgt_arm.tcp_jntid = -1
            self.rgt_arm.tcp_loc_pos = self.hnd_origin_rotmat.dot(self.rgt_hnd.jaw_center_pos) + self.hnd_origin_pos
            self.rgt_arm.tcp_loc_rotmat = self.hnd_origin_rotmat.dot(self.rgt_hnd.jaw_center_rotmat)
            # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
            self.rgt_oih_infos = []
            self.hnd_dict['rgt_arm'] = self.rgt_hnd
            self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['rgt_arm_waist'] = self.rgt_arm
        self.manipulator_dict['lft_arm_waist'] = self.lft_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm
        self.manipulator_dict['lft_hnd'] = self.lft_arm
        # considering waist or not when getting joint values
        self.mode = mode
        if self.mode == 'lft_arm_waist' or self.mode == 'rgt_arm_waist':
            self.read_waist = True
        elif self.mode == 'lft_arm' or self.mode == 'rgt_arm':
            self.read_waist = False
        else:
            raise ValueError("The given mode name is not available!")

    @staticmethod
    def _waist_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.183, 0, -1.68),
                                              x=.3 + radius, y=.3 + radius, z=.26 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.183, 0, -1.28),
                                              x=.3 + radius, y=.135 + radius, z=.15 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0, 0, -.93),
                                              x=.06 + radius, y=.06 + radius, z=.2 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    @staticmethod
    def _chest_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.0505, 0, .45),
                                              x=.136 + radius, y=.12 + radius, z=.09 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.028, 0, .3),
                                              x=.1 + radius, y=.07 + radius, z=.05 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_l0 = CollisionBox(Point3(0.005, 0.16, .515),
                                              x=.037 + radius, y=.055 + radius, z=.02 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.005, -0.16, .515),
                                              x=.037 + radius, y=.055 + radius, z=.02 + radius)
        collision_node.addSolid(collision_primitive_r0)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.central_body, [0, 1, 2, 3])
        self.cc.add_cdlnks(self.lft_arm, [2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.rgt_arm, [2, 3, 4, 5, 6, 7])
        if self.hnd_attached == "bothhnd":
            self.cc.add_cdlnks(self.lft_hnd.lft_outer, [0, 1, 2, 3, 4])
            self.cc.add_cdlnks(self.lft_hnd.rgt_outer, [1, 2, 3, 4])
            self.cc.add_cdlnks(self.rgt_hnd.lft_outer, [0, 1, 2, 3, 4])
            self.cc.add_cdlnks(self.rgt_hnd.rgt_outer, [1, 2, 3, 4])
        elif self.hnd_attached == "lft_hnd":
            self.cc.add_cdlnks(self.lft_hnd.lft_outer, [0, 1, 2, 3, 4])
            self.cc.add_cdlnks(self.lft_hnd.rgt_outer, [1, 2, 3, 4])
        elif self.hnd_attached == "rgt_hnd":
            self.cc.add_cdlnks(self.rgt_hnd.lft_outer, [0, 1, 2, 3, 4])
            self.cc.add_cdlnks(self.rgt_hnd.rgt_outer, [1, 2, 3, 4])
        activelist_arm = [self.lft_arm.lnks[2],
                          self.lft_arm.lnks[3],
                          self.lft_arm.lnks[4],
                          self.lft_arm.lnks[5],
                          self.lft_arm.lnks[6],
                          self.lft_arm.lnks[7],
                          self.rgt_arm.lnks[2],
                          self.rgt_arm.lnks[3],
                          self.rgt_arm.lnks[4],
                          self.rgt_arm.lnks[5],
                          self.rgt_arm.lnks[6],
                          self.rgt_arm.lnks[7]]
        if self.hnd_attached == "bothhnd":
            activelist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                              self.lft_hnd.lft_outer.lnks[1],
                              self.lft_hnd.lft_outer.lnks[2],
                              self.lft_hnd.lft_outer.lnks[3],
                              self.lft_hnd.lft_outer.lnks[4],
                              self.lft_hnd.rgt_outer.lnks[1],
                              self.lft_hnd.rgt_outer.lnks[2],
                              self.lft_hnd.rgt_outer.lnks[3],
                              self.lft_hnd.rgt_outer.lnks[4],
                              self.rgt_hnd.lft_outer.lnks[0],
                              self.rgt_hnd.lft_outer.lnks[1],
                              self.rgt_hnd.lft_outer.lnks[2],
                              self.rgt_hnd.lft_outer.lnks[3],
                              self.rgt_hnd.lft_outer.lnks[4],
                              self.rgt_hnd.rgt_outer.lnks[1],
                              self.rgt_hnd.rgt_outer.lnks[2],
                              self.rgt_hnd.rgt_outer.lnks[3],
                              self.rgt_hnd.rgt_outer.lnks[4],
                              ]
            activelist = activelist_arm + activelist_hnd
        elif self.hnd_attached == "lft_hnd":
            activelist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                              self.lft_hnd.lft_outer.lnks[1],
                              self.lft_hnd.lft_outer.lnks[2],
                              self.lft_hnd.lft_outer.lnks[3],
                              self.lft_hnd.lft_outer.lnks[4],
                              self.lft_hnd.rgt_outer.lnks[1],
                              self.lft_hnd.rgt_outer.lnks[2],
                              self.lft_hnd.rgt_outer.lnks[3],
                              self.lft_hnd.rgt_outer.lnks[4]]
            activelist = activelist_arm + activelist_hnd
        elif self.hnd_attached == "rgt_hnd":
            activelist_hnd = [self.rgt_hnd.lft_outer.lnks[0],
                              self.rgt_hnd.lft_outer.lnks[1],
                              self.rgt_hnd.lft_outer.lnks[2],
                              self.rgt_hnd.lft_outer.lnks[3],
                              self.rgt_hnd.lft_outer.lnks[4],
                              self.rgt_hnd.rgt_outer.lnks[1],
                              self.rgt_hnd.rgt_outer.lnks[2],
                              self.rgt_hnd.rgt_outer.lnks[3],
                              self.rgt_hnd.rgt_outer.lnks[4]]
            activelist = activelist_arm + activelist_hnd
        else:
            activelist = activelist_arm
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.central_body.lnks[0],
                    self.central_body.lnks[1],
                    self.central_body.lnks[3],
                    self.lft_arm.lnks[2],
                    self.rgt_arm.lnks[2]]
        intolist_arm = [self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_arm.lnks[7],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_arm.lnks[7]]
        if self.hnd_attached == "bothhnd":
            intolist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                            self.lft_hnd.lft_outer.lnks[1],
                            self.lft_hnd.lft_outer.lnks[2],
                            self.lft_hnd.lft_outer.lnks[3],
                            self.lft_hnd.lft_outer.lnks[4],
                            self.lft_hnd.rgt_outer.lnks[1],
                            self.lft_hnd.rgt_outer.lnks[2],
                            self.lft_hnd.rgt_outer.lnks[3],
                            self.lft_hnd.rgt_outer.lnks[4],
                            self.rgt_hnd.lft_outer.lnks[0],
                            self.rgt_hnd.lft_outer.lnks[1],
                            self.rgt_hnd.lft_outer.lnks[2],
                            self.rgt_hnd.lft_outer.lnks[3],
                            self.rgt_hnd.lft_outer.lnks[4],
                            self.rgt_hnd.rgt_outer.lnks[1],
                            self.rgt_hnd.rgt_outer.lnks[2],
                            self.rgt_hnd.rgt_outer.lnks[3],
                            self.rgt_hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        elif self.hnd_attached == "lft_hnd":
            intolist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                            self.lft_hnd.lft_outer.lnks[1],
                            self.lft_hnd.lft_outer.lnks[2],
                            self.lft_hnd.lft_outer.lnks[3],
                            self.lft_hnd.lft_outer.lnks[4],
                            self.lft_hnd.rgt_outer.lnks[1],
                            self.lft_hnd.rgt_outer.lnks[2],
                            self.lft_hnd.rgt_outer.lnks[3],
                            self.lft_hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        elif self.hnd_attached == "rgt_hnd":
            intolist_hnd = [self.rgt_hnd.lft_outer.lnks[0],
                            self.rgt_hnd.lft_outer.lnks[1],
                            self.rgt_hnd.lft_outer.lnks[2],
                            self.rgt_hnd.lft_outer.lnks[3],
                            self.rgt_hnd.lft_outer.lnks[4],
                            self.rgt_hnd.rgt_outer.lnks[1],
                            self.rgt_hnd.rgt_outer.lnks[2],
                            self.rgt_hnd.rgt_outer.lnks[3],
                            self.rgt_hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        else:
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)
        fromlist_arm = [self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_arm.lnks[7]]
        intolist_arm = [self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_arm.lnks[7]]
        if self.hnd_attached == "bothhnd":
            fromlist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                            self.lft_hnd.lft_outer.lnks[1],
                            self.lft_hnd.lft_outer.lnks[2],
                            self.lft_hnd.lft_outer.lnks[3],
                            self.lft_hnd.lft_outer.lnks[4],
                            self.lft_hnd.rgt_outer.lnks[1],
                            self.lft_hnd.rgt_outer.lnks[2],
                            self.lft_hnd.rgt_outer.lnks[3],
                            self.lft_hnd.rgt_outer.lnks[4]]
            intolist_hnd = [self.rgt_hnd.lft_outer.lnks[0],
                            self.rgt_hnd.lft_outer.lnks[1],
                            self.rgt_hnd.lft_outer.lnks[2],
                            self.rgt_hnd.lft_outer.lnks[3],
                            self.rgt_hnd.lft_outer.lnks[4],
                            self.rgt_hnd.rgt_outer.lnks[1],
                            self.rgt_hnd.rgt_outer.lnks[2],
                            self.rgt_hnd.rgt_outer.lnks[3],
                            self.rgt_hnd.rgt_outer.lnks[4]]
            fromlist = fromlist_arm + fromlist_hnd
            intolist = intolist_arm + intolist_hnd
        elif self.hnd_attached == "lft_hnd":
            fromlist_hnd = [self.lft_hnd.lft_outer.lnks[0],
                            self.lft_hnd.lft_outer.lnks[1],
                            self.lft_hnd.lft_outer.lnks[2],
                            self.lft_hnd.lft_outer.lnks[3],
                            self.lft_hnd.lft_outer.lnks[4],
                            self.lft_hnd.rgt_outer.lnks[1],
                            self.lft_hnd.rgt_outer.lnks[2],
                            self.lft_hnd.rgt_outer.lnks[3],
                            self.lft_hnd.rgt_outer.lnks[4]]
            fromlist = fromlist_arm + fromlist_hnd
        elif self.hnd_attached == "rgt_hnd":
            intolist_hnd = [self.rgt_hnd.lft_outer.lnks[0],
                            self.rgt_hnd.lft_outer.lnks[1],
                            self.rgt_hnd.lft_outer.lnks[2],
                            self.rgt_hnd.lft_outer.lnks[3],
                            self.rgt_hnd.lft_outer.lnks[4],
                            self.rgt_hnd.rgt_outer.lnks[1],
                            self.rgt_hnd.rgt_outer.lnks[2],
                            self.rgt_hnd.rgt_outer.lnks[3],
                            self.rgt_hnd.rgt_outer.lnks[4]]
            intolist = intolist_arm + intolist_hnd
        else:
            fromlist = fromlist_arm
            intolist = intolist_arm
        self.cc.set_cdpair(fromlist, intolist)

    def get_hnd_on_manipulator(self, manipulator_name):
        if manipulator_name == 'rgt_arm':
            return self.rgt_hnd
        elif manipulator_name == 'lft_arm':
            return self.lft_hnd
        else:
            raise ValueError("The given jlc does not have a hand!")

    def fix_to(self, pos, rotmat):
        super().fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.central_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(self.pos, self.rotmat)
        if self.hnd_attached == "bothhnd":
            self.lft_hnd.fix_to(
                pos=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.lft_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat))
            self.rgt_hnd.fix_to(
                pos=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.rgt_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat))
        elif self.hnd_attached == "lft_hnd":
            self.lft_hnd.fix_to(
                pos=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.lft_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat))
        elif self.hnd_attached == "rgt_hnd":
            self.rgt_hnd.fix_to(
                pos=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) + self.rgt_arm.jnts[-1]['gl_posq'],
                rotmat=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat))

    def fk(self, component_name, jnt_values):
        """
        waist angle is transmitted to arms
        :param jnt_values: nparray 1x6 or 1x14 depending on component_names
        :hnd_name 'lft_arm', 'rgt_arm', 'lft_arm_waist', 'rgt_arm_wasit', 'both_arm'
        :param component_name:
        :return:
        author: wangyan
        date: 20220308, Suzhou
        """

        def update_oih(component_name='rgt_arm_waist'):
            # inline function for update objects in hand
            if component_name[:7] == 'rgt_arm':
                oih_info_list = self.rgt_oih_infos
            elif component_name[:7] == 'lft_arm':
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            hnd_on_manipulator = self.get_hnd_on_manipulator(component_name[:7])
            if hnd_on_manipulator is not None:
                hnd_on_manipulator.fix_to(
                    pos=np.dot(self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'], self.hnd_origin_pos) +
                        self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                    rotmat=np.dot(self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'], self.hnd_origin_rotmat))
            update_oih(component_name=component_name)
            return status

        # examine length
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
            waist_value = self.central_body.jnts[1]['motion_val']
            return update_component(component_name, np.append(waist_value, jnt_values))
        elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move a single arm plus the waist!")
            status = update_component(component_name, jnt_values)
            self.central_body.jnts[1]['motion_val'] = jnt_values[0]
            self.central_body.fk()
            the_other_manipulator_name = 'lft_arm' if component_name[:7] == 'rgt_arm' else 'rgt_arm'
            self.manipulator_dict[the_other_manipulator_name].jnts[1]['motion_val'] = jnt_values[0]
            self.manipulator_dict[the_other_manipulator_name].fk()
            self.get_hnd_on_manipulator(the_other_manipulator_name).fix_to(
                pos=np.dot(self.manipulator_dict[the_other_manipulator_name].jnts[-1]['gl_rotmatq'],self.hnd_origin_pos) +
                    self.manipulator_dict[the_other_manipulator_name].jnts[-1]['gl_posq'],
                rotmat=np.dot(self.manipulator_dict[the_other_manipulator_name].jnts[-1]['gl_rotmatq'],self.hnd_origin_rotmat))
            return status # if waist is out of range, the first status will always be out of rng
        elif component_name == 'botharm':
            raise NotImplementedError
        elif component_name == 'all':
            raise NotImplementedError
        else:
            raise ValueError("The given component name is not available!")

    def ik(self,
           component_name,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           max_niter=100,
           local_minima="accept",
           toggle_debug=False):
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            old_tgt_jnts = self.manipulator_dict[component_name].tgtjnts
            self.manipulator_dict[component_name].tgtjnts = range(2, self.manipulator_dict[component_name].ndof + 1)
            ik_results = self.manipulator_dict[component_name].ik(tgt_pos,
                                                                  tgt_rotmat,
                                                                  seed_jnt_values=seed_jnt_values,
                                                                  tcp_jnt_id=tcp_jnt_id,
                                                                  tcp_loc_pos=tcp_loc_pos,
                                                                  tcp_loc_rotmat=tcp_loc_rotmat,
                                                                  max_niter=max_niter,
                                                                  local_minima=local_minima,
                                                                  toggle_debug=toggle_debug)
            self.manipulator_dict[component_name].tgtjnts = old_tgt_jnts
            return ik_results
        elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
            return self.manipulator_dict[component_name].ik(tgt_pos,
                                                            tgt_rotmat,
                                                            seed_jnt_values=seed_jnt_values,
                                                            tcp_jnt_id=tcp_jnt_id,
                                                            tcp_loc_pos=tcp_loc_pos,
                                                            tcp_loc_rotmat=tcp_loc_rotmat,
                                                            max_niter=max_niter,
                                                            local_minima=local_minima,
                                                            toggle_debug=toggle_debug)
        elif component_name == 'both_arm':
            raise NotImplementedError
        elif component_name == 'all':
            raise NotImplementedError
        else:
            raise ValueError("The given component name is not available!")

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            return super().rand_conf(component_name)[1:]
        elif component_name == 'lft_arm_waist' or component_name == 'rgt_arm_waist':
            return super().rand_conf(component_name)
        elif component_name == 'both_arm':
            return np.hstack((super().rand_conf('lft_arm')[1:], super().rand_conf('rgt_arm')[1:]))
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='lft_hnd', jawwidth=0.05):
        self.hnd_dict[hnd_name].jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :return:
        """
        hnd_name = hnd_name[:4]+"hnd"
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        # TODO “ValueError: The link needs to be added to collider using the addjlcobj function first!”
        if hnd_name == 'lft_hnd':
            rel_pos, rel_rotmat = self.lft_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.central_body.lnks[0],
                        self.central_body.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6]]
            self.lft_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        elif hnd_name == 'rgt_hnd':
            rel_pos, rel_rotmat = self.rgt_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.central_body.lnks[0],
                        self.central_body.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6]]
            self.rgt_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        elif hnd_name == 'bothhnd':
            rel_pos, rel_rotmat = self.lft_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.central_body.lnks[0],
                        self.central_body.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_hnd.lft_outer.lnks[0],
                        self.rgt_hnd.lft_outer.lnks[2],
                        self.rgt_hnd.rgt_outer.lnks[2]]
            self.lft_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))

            rel_pos, rel_rotmat = self.rgt_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.central_body.lnks[0],
                        self.central_body.lnks[1],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_hnd.lft.lnks[0],
                        self.lft_hnd.lft.lnks[2],
                        self.lft_hnd.rgt.lnks[2]]
            self.rgt_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")

        return rel_pos, rel_rotmat

    def get_gl_tcp(self, manipulator_name="lft_arm"):
        return super().get_gl_tcp(manipulator_name=manipulator_name[:7])

    def get_jnt_values(self, component_name="lft_arm", waist=None):
        if component_name in self.manipulator_dict:
            if waist is None:
                waist = self.read_waist
            if waist:
                return self.manipulator_dict[component_name].get_jnt_values()
            else:
                return self.manipulator_dict[component_name].get_jnt_values()[1:]

    def get_loc_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return self.cvt_gl_to_loc_tcp(component_name, gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3])

    def get_gl_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        else:
            raise ValueError("Component name for Nextage Robot must be \'lft_arm\' or \'rgt_arm\'!")
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]

    def get_oih_cm_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def get_oih_glhomomat_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            return_list.append(rm.homomat_from_posrot(obj_info['gl_pos']), obj_info['gl_rotmat'])
        return return_list

    def get_oih_relhomomat(self, objcm, hnd_name='lft_hnd'):
        """
        TODO: useless? 20210320
        oih = object in hand list
        :param objcm
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_info_list = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_info_list = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
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
        hnd_name = hnd_name[:4]+"hnd"
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                oih_infos.remove(obj_info)
                break

    def release_all(self, jaw_width=None, hnd_name='lft_hnd'):
        """
        release all objects from the specified hand
        :param jaw_width:
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210125
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            self.cc.delete_cdobj(obj_info)
        oih_infos.clear()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='nextage_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.central_body.gen_stickmodel(tcp_loc_pos=None,
                                         tcp_loc_rotmat=None,
                                         toggle_tcpcs=False,
                                         toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if self.hnd_attached == "bothhnd":
            self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
            self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        elif self.hnd_attached == "lft_hnd":
            self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        elif self.hnd_attached == "rgt_hnd":
            self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='nextage_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.central_body.gen_meshmodel(tcp_loc_pos=None,
                                        tcp_loc_rotmat=None,
                                        toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        if self.hnd_attached == "bothhnd":
            self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
            self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
        elif self.hnd_attached == "lft_hnd":
            self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)
            for obj_info in self.lft_oih_infos:
                objcm = obj_info['collisionmodel']
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                objcm.copy().attach_to(meshmodel)
        elif self.hnd_attached == "rgt_hnd":
            self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       rgba=rgba).attach_to(meshmodel)

            for obj_info in self.rgt_oih_infos:
                objcm = obj_info['collisionmodel']
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                objcm.copy().attach_to(meshmodel)

        return meshmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis
    #
    base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    nxt_instance = Nextage(enable_cc=True, hnd_attached='bothhnd', mode='rgt_arm_waist')
    jnt_values = np.radians([45, 0, -60, -120, 0, 0, 90])
    component_name = nxt_instance.mode
    nxt_instance.fk(component_name, jnt_values)
    nxt_instance.jaw_to(hnd_name='lft_hnd', jawwidth=0.05)
    nxt_instance.jaw_to(hnd_name='rgt_hnd', jawwidth=0.01)
    nxt_instance.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    print(nxt_instance.get_jnt_values("lft_arm"))
    print(nxt_instance.get_jnt_values("rgt_arm"))
    # nxt_instance.show_cdprimit()
    base.run()

    # tgt_pos = np.array([.4, 0, .2])
    # tgt_rotmat = rm.rotmat_from_euler(0, math.pi * 2 / 3, -math.pi / 4)
    # ik test
    component_name = 'lft_arm_waist'
    tgt_pos = np.array([-.3, .45, .55])
    tgt_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # tgt_rotmat = np.eye(3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat, toggle_debug=True)
    toc = time.time()
    print(toc - tic)
    nxt_instance.fk(component_name, jnt_values)
    nxt_meshmodel = nxt_instance.gen_meshmodel()
    nxt_meshmodel.attach_to(base)
    nxt_instance.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = nxt_instance.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()

    # hold test
    component_name = 'lft_arm'
    obj_pos = np.array([.35, .5, .4])
    obj_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    objfile = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm = cm.CollisionModel(objfile, cdprimit_type='cylinder')
    objcm.set_pos(obj_pos)
    objcm.set_rotmat(obj_rotmat)
    objcm.attach_to(base)
    objcm_copy = objcm.copy()
    nxt_instance.hold(objcm=objcm_copy, jawwidth=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = nxt_instance.ik(component_name, tgt_pos, tgt_rotmat)
    nxt_instance.fk(component_name, jnt_values)
    # nxt_instance.show_cdprimit()
    nxt_meshmodel = nxt_instance.gen_meshmodel()
    nxt_meshmodel.attach_to(base)

    base.run()
