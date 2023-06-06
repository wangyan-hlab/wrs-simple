
#%%
# Set start and goal joint values
start_jnts = [-111.817, -87.609, -118.858, -55.275, 107.847, 20.778]
goal_jnts = [-126.658, -90.459, -127.047, -17.006, 73.267, 20.767]

#%%
# WRS planning simulation
import time
import numpy as np
from visualization.panda import world as wd
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.fr5 import fr5 as fr5
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from fr_python_sdk.frmove import FRCobot

def genSphere(pos, radius=0.005, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
gm.gen_frame().attach_to(base)
component_name = 'arm'
# simulated robot
robot_s = fr5.FR5_robot(enable_cc=True, peg_attached=False)
# robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
# robot_meshmodel.attach_to(base)

start_conf = np.deg2rad(start_jnts)
goal_conf = np.deg2rad(goal_jnts)
robot_s.fk(component_name, start_conf)
robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1,0,0,0.5]).attach_to(base)
robot_s.fk(component_name, goal_conf)
robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0,1,0,0.5]).attach_to(base)
# planner
time_start = time.time()
rrtc_planner = rrtc.RRTConnect(robot_s)
path = rrtc_planner.plan(component_name=component_name,
                            start_conf=start_conf,
                            goal_conf=goal_conf,
                            obstacle_list=[],
                            ext_dist=0.1,
                            max_time=300)
time_end = time.time()
print("Planning time = ", time_end-time_start)

print(len(path), path)

def update(rbtmnp, motioncounter, robot, path, armname, task):
    if motioncounter[0] < len(path):
        # update simulated robot
        if rbtmnp[0] is not None:
            rbtmnp[0].detach()
        pose = path[motioncounter[0]]
        robot.fk(armname, pose)
        rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
        rbtmnp[0].attach_to(base)
        genSphere(robot.get_gl_tcp(component_name)[0], radius=0.01, rgba=[1, 1, 0, 1])
        
        motioncounter[0] += 1
    else:
        motioncounter[0] = 0
    return task.again
rbtmnp = [None]
motioncounter = [0]
taskMgr.doMethodLater(0.07, update, "update",
                        extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], appendTask=True)

base.setFrameRateMeter(True)
base.run()

#%%
# Real robot
from fr_python_sdk.frmove import FRCobot
robot_r = FRCobot()
real_robot = True

#%%
# Move to start_jnts
if real_robot:
    robot_r.MoveJ(start_jnts)

#%%
# Control real robot to reproduce the planned path
if real_robot:
    t = 0.008
    acc = 0.0
    vel = 0.0
    lookahead_time = 0.0
    P = 0.0
    path_deg = []
    for pose in path:
        pose = list(pose)
        for i in range(6):
            pose[i] = float(np.rad2deg(pose[i]))
        print("目标关节位置:", pose)
        path_deg.append(pose)
    
    robot_r.MoveJSeq(path_deg, granularity=0.02)

    print("关节差值: ", np.asarray(robot_r.GetJointPos()) - np.asarray(path_deg[-1]))
    if np.allclose(np.asarray(robot_r.GetJointPos()), 
                   np.asarray(path_deg[-1]), atol=1.0):
        print("[INFO] ServoJ 运动到关节位置序列终点")

#%%
# Move to start_jnts
if real_robot:
    robot_r.MoveJ(start_jnts)

#%%
# Move to goal_jnts
if real_robot:
    robot_r.MoveJ(goal_jnts)

# %%
