import time
import numpy as np
from robot_con.ur.ur5e import UR5ERtqHE as ur5e_real
from robot_planning_gui import FastSimWorld

class URFastSimWorld(FastSimWorld):

    def __init__(self, cam_pos=[3, 3, 1], lookat_pos=[0, .5, 0],
                 up=np.array([0, 0, 1]), fov=40, w=1920, h=1080, 
                 lens_type="perspective", toggle_debug=False, 
                 auto_cam_rotate=False, backgroundcolor=None,
                 robot_connect=False, 
                 robot_ip='192.168.58.2', 
                 pc_ip='192.168.58.70'):
        super().__init__(cam_pos, lookat_pos, up, fov, w, h, 
                         lens_type, toggle_debug, 
                         auto_cam_rotate, backgroundcolor,
                         robot_connect,robot_ip, pc_ip)

        if self.robot_connect:
            print("[Info] 机器人已连接")
            self.robot_r = ur5e_real(robot_ip=self.robot_ip, pc_ip=self.pc_ip)
            self.init_conf = self.robot_r.get_jnt_values()  # 实际机器人的初始关节角度
        else:
            print("[Info] 机器人未连接")
            self.init_conf = np.zeros(6)
        
    def robot_move(self):
        """
            UR5e robot move
        """
        self.robot_r.move_jnts(np.rad2deg(self.path[0]))
        if self.path:
            self.robot_r.move_jntspace_path(self.path)


if __name__ == "__main__":

    from robot_sim.robots.ur5e_ballpeg import ur5e_ballpeg as ur5e

    # WRS planning simulation
    robot_connect = False
    robot_ip = '192.168.58.2'
    pc_ip = '192.168.58.70'

    base = URFastSimWorld(robot_connect=robot_connect, robot_ip=robot_ip, pc_ip=pc_ip)
    
    robot_s = ur5e.UR5EBallPeg(enable_cc=True, peg_attached=False)
    component = 'arm'
    base.set_robot(robot_s, component)
    
    base.setFrameRateMeter(True)
    base.run()
    