import time
import numpy as np
from robot_planning_gui import FastSimWorld

class URFastSimWorld(FastSimWorld):

    def __init__(self, cam_pos=[3, 3, 1], lookat_pos=[0, -.5, 0],
                 up=np.array([0, 0, 1]), fov=40, w=1920, h=1080, 
                 lens_type="perspective", toggle_debug=False, 
                 auto_cam_rotate=False, backgroundcolor=None,
                 robot_connect=False,
                 init_conf=np.zeros(6)):
        super().__init__(cam_pos, lookat_pos, up, fov, w, h, 
                         lens_type, toggle_debug, 
                         auto_cam_rotate, backgroundcolor, 
                         robot_connect, init_conf)


    def get_robot_jnts(self):
        """
            获取真实机器人关节角度
        """

        if self.robot_connect:
            print("[Info] 机器人已连接")
            self.real_robot_conf = self.robot_r.get_jnt_values()  # 实际机器人的关节角度
        else:
            print("[Info] 机器人未连接")
            self.real_robot_conf = np.zeros(6)

        for i in range(6):
            self.slider_values[i][0].setValue(np.rad2deg(self.real_robot_conf)[i])

        self.robot.fk(self.component_name, np.asarray(self.real_robot_conf))

        if self.robot_meshmodel is not None:
            self.robot_meshmodel.detach()

        self.robot_meshmodel = self.robot.gen_meshmodel()
        self.robot_meshmodel.attach_to(self)
    
    
    def real_robot_moving(self, targets):
        """
            UR5e robot move
        """

        for target in targets:
            if target[0] == 'point':
                self.robot_r.move_jnts(np.rad2deg(target[1]))
            else:
                self.robot_r.move_jnts(np.rad2deg(target[1][0]))
                self.robot_r.move_jntspace_path(target[1])


if __name__ == "__main__":

    # WRS planning simulation
    robot_connect = False
    base = URFastSimWorld(robot_connect=robot_connect)
    base.start()

    base.run()
    