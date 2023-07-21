# import robot_sim.robots.fr5.fr5 as fr5
import robot_sim.robots.fr5.fr5_rtq85 as fr5
import modeling.dynamics.bullet.bdmodel as bdm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm
import numpy as np
import math


def get_lnk_bdmodel(lnk):
    bd_lnk = bdm.BDModel(lnk["collision_model"], mass=0, type="box", dynamic=False)
    bd_lnk.set_homomat(rm.homomat_from_posrot(lnk["gl_pos"], lnk["gl_rotmat"]))
    bd_lnk.set_rgba([.8, .7, 0, 1])
    return bd_lnk


def get_hnd_lnk_bdmodel(lnk_eef):
    bd_lnk_eef = bdm.BDModel(lnk_eef["collision_model"], mass=0, type="box", dynamic=False)
    bd_lnk_eef.set_homomat(rm.homomat_from_posrot(lnk_eef["gl_pos"], lnk_eef["gl_rotmat"]))
    bd_lnk_eef.set_rgba([.8, .8, 0.8, 1])
    return bd_lnk_eef


def get_robot_bdmodel(robot_s):
    bd_lnk_list = []
    bd_hnd_lnk_list = []

    for arm_name in ["arm"]:
        for lnk_id in [0,1,2,3,4,5,6]:
            lnk = robot_s.manipulator_dict[arm_name].lnks[lnk_id]
            bd_lnk_list.append(get_lnk_bdmodel(lnk))

    for hnd_name in ["hnd"]:
        for hnd_lnk_id in [0,1,2,3,4]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].lft_outer.lnks[hnd_lnk_id]
            bd_hnd_lnk_list.append(get_hnd_lnk_bdmodel(hnd_lnk))
        for hnd_lnk_id in [1]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].lft_inner.lnks[hnd_lnk_id]
            bd_hnd_lnk_list.append(get_hnd_lnk_bdmodel(hnd_lnk))
        for hnd_lnk_id in [1,2,3,4]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].rgt_outer.lnks[hnd_lnk_id]
            bd_hnd_lnk_list.append(get_hnd_lnk_bdmodel(hnd_lnk))
        for hnd_lnk_id in [1]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].rgt_inner.lnks[hnd_lnk_id]
            bd_hnd_lnk_list.append(get_hnd_lnk_bdmodel(hnd_lnk))

    return bd_lnk_list, bd_hnd_lnk_list


def update_robot_bdmodel(robot_s, bd_lnk_list, bd_hnd_lnk_list):
    cnter = 0
    cnter_hnd = 0

    for arm_name in ["arm"]:
        for lnk_id in [0, 1, 2, 3, 4, 5, 6]:
            lnk = robot_s.manipulator_dict[arm_name].lnks[lnk_id]
            bd_lnk_list[cnter].set_homomat(rm.homomat_from_posrot(lnk["gl_pos"], lnk["gl_rotmat"]))
            cnter += 1

    for hnd_name in ["hnd"]:
        for lnk_id in [0,1,2,3,4]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].lft_outer.lnks[lnk_id]
            bd_hnd_lnk_list[cnter_hnd].set_homomat(rm.homomat_from_posrot(hnd_lnk["gl_pos"], hnd_lnk["gl_rotmat"]))
            cnter_hnd += 1
        for lnk_id in [1]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].lft_inner.lnks[lnk_id]
            bd_hnd_lnk_list[cnter_hnd].set_homomat(rm.homomat_from_posrot(hnd_lnk["gl_pos"], hnd_lnk["gl_rotmat"]))
            cnter_hnd += 1
        for lnk_id in [1,2,3,4]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].rgt_outer.lnks[lnk_id]
            bd_hnd_lnk_list[cnter_hnd].set_homomat(rm.homomat_from_posrot(hnd_lnk["gl_pos"], hnd_lnk["gl_rotmat"]))
            cnter_hnd += 1
        for lnk_id in [1]:
            hnd_lnk = robot_s.hnd_dict[hnd_name].rgt_inner.lnks[lnk_id]
            bd_hnd_lnk_list[cnter_hnd].set_homomat(rm.homomat_from_posrot(hnd_lnk["gl_pos"], hnd_lnk["gl_rotmat"]))
            cnter_hnd += 1


if __name__ == '__main__':
    import os

    base = wd.World(cam_pos=[10, 0, 5], lookat_pos=[0, 0, 1])
    base.setFrameRateMeter(True)
    gm.gen_frame().attach_to(base)

    task_table = cm.CollisionModel("../objects/task_table.stl")
    task_table.set_rgba([.5, .5, .5, 1])
    task_table_bd = bdm.BDModel(task_table, mass=.0, type="convex", friction=0.9)
    task_table_bd.set_pos(np.array([0.35, 0, -0.025]))
    task_table_bd.start_physics()
    base.attach_internal_update_obj(task_table_bd)

    robot_s = fr5.ROBOT(zrot_to_gndbase=0)
    robot_s.fk(component_name="arm", jnt_values=np.radians([0,-20,30,0,0,0]))
    # robot_s.show_cdprimit()
    bd_lnk_list, bd_hnd_lnk_list = get_robot_bdmodel(robot_s)
    for bdl in bd_lnk_list:
        bdl.start_physics()
        base.attach_internal_update_obj(bdl)
    for bdl_hnd in bd_hnd_lnk_list:
        bdl_hnd.start_physics()
        base.attach_internal_update_obj(bdl_hnd)

     # obj_box = cm.gen_box(extent=[.05, .05, .2], rgba=[.3, 0, 0, 1])
    # obj_box = cm.gen_sphere(radius=.2, rgba=[.0, 0, 1.0, 1])
    obj_box = cm.CollisionModel("../objects/milkcarton.stl")
    obj_box.set_scale([.5, .5, .5])
    obj_box.set_rgba([.1, .2, .8, 1])
    obj_bd_box = bdm.BDModel(obj_box, mass=.3, type="convex", friction=0.9)
    obj_bd_box.set_pos(np.array([.5, -.2, 1.0]))
    obj_bd_box.start_physics()
    base.attach_internal_update_obj(obj_bd_box)

    def update(robot_s, bd_lnk_list, bd_hnd_lnk_list, task):
        box_pos = obj_bd_box.get_pos()
        box_homomat = obj_bd_box.get_homomat()
        print(">>>>>>>", box_pos, box_homomat)
        if base.inputmgr.keymap['space'] is True:
            jnt_values = robot_s.get_jnt_values("arm")
            rand = np.random.rand(6)*.01
            # jnt_values = jnt_values - rand
            jnt_values = jnt_values - np.array([0.01,0,0,0,0,0])
            robot_s.fk(component_name="arm", jnt_values=jnt_values)
            update_robot_bdmodel(robot_s, bd_lnk_list, bd_hnd_lnk_list)
            base.inputmgr.keymap['space'] = False
        return task.cont
    
    taskMgr.add(update, "update", extraArgs=[robot_s, bd_lnk_list, bd_hnd_lnk_list], appendTask=True)

    base.run()
