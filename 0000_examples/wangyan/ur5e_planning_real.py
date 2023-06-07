import threading
from threading import Thread, Lock, current_thread
import time
import numpy as np
from visualization.panda import world as wd
from modeling import geometric_model as gm
from modeling import collision_model as cm
from robot_sim.robots.ur5e_ballpeg import ur5e_ballpeg as ur5e
from motion.probabilistic import rrt_connect as rrtc
from basis import robot_math as rm
from robot_con.ur.ur5e import UR5ERtqHE as ur5e_real

global path, endtask 
path = []
pathLock = Lock()
endtask = 0

def run_realrobot(real_robot):

    global path, endtask
    start_jnts = [-111.817, -87.609, -118.858, -55.275, 107.847, 20.778]
    
    # Move to start_jnts
    if real_robot:
        print("I am a robot.")
        # Real robot
        robot_r = ur5e_real(robot_ip='192.168.58.2', pc_ip='192.168.58.70')
        robot_r.move_jnts(start_jnts)
    else:
        print("I am not a robot.")
        time.sleep(10)
        endtask = 1
    
    # Control real robot to reproduce the planned path
    pathLock.acquire()

    if real_robot and path:
        robot_r.move_jntspace_path(path) # TODO:check

    pathLock.release()
    print(f"线程{current_thread().name}已退出")


if __name__ == "__main__":
    
    # Set start and goal joint values
    start_jnts = [-111.817, -87.609, -118.858, -55.275, 107.847, 20.778]
    goal_jnts = [-127.146, -74.498, -85.835, -40.605, 71.584, 20.790]

    # WRS planning simulation
    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0.5], w=960, h=720)
    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    # simulated robot
    robot_s = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    # robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    # robot_meshmodel.attach_to(base)

    start_conf = np.deg2rad(start_jnts)
    goal_conf = np.deg2rad(goal_jnts)
    robot_s.fk(component_name, start_conf)
    robot_start = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_start.attach_to(base)
    robot_s.fk(component_name, goal_conf)
    robot_goal = robot_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 1, 0, 0.3])
    robot_goal.attach_to(base)
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
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0.5, 0, 0.7])
            rbtmnp[0].attach_to(base)
            tcp_ball = gm.gen_sphere(pos=robot.get_gl_tcp(component_name)[0], 
                                    radius=0.01, rgba=[1, 1, 0, 1])
            tcp_ball.attach_to(base)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0

        if endtask == 1:
            robot_start.detach()
            robot_goal.detach()
            rbtmnp[0].detach()
            robot.fk(armname, path[-1])
            robot.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
            print("Task ends")
            return task.done
        else:
            return task.again

    rbtmnp = [None]
    motioncounter = [0]
    
    thread_rr = Thread(target=lambda: run_realrobot(False), daemon=True, name="real_robot")
    thread_rr.start()
    # print("程序因线程real_robot陷入阻塞")
    # thread_rr.join(timeout=1)

    taskMgr.doMethodLater(0.1, update, "update",
                        extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], 
                        appendTask=True)

    base.setFrameRateMeter(True)
    base.run()
    