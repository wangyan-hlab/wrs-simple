import math
import numpy as np
import basis.robot_math as rm
import modeling.mesh_tools as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    # mt.convert_to_stl("robotiq_arg2f_85_base_link.stl", "robotiq_arg2f_85_base_link_cvt.stl", rotmat=rm.rotmat_from_axangle([0,1,0], -math.pi/2))
    # mt.convert_to_stl("robotiq_arg2f_85_inner_knuckle.stl", "robotiq_arg2f_85_inner_knuckle_cvt.stl", scale_ratio=.001)
    # mt.convert_to_stl("robotiq_arg2f_85_outer_finger.stl", "robotiq_arg2f_85_outer_finger_cvt.stl", scale_ratio=.001)
    mt.convert_to_stl("robotiq_arg2f_85_inner_finger.stl", "robotiq_arg2f_85_inner_finger_cvt2.stl", scale_ratio=.001)