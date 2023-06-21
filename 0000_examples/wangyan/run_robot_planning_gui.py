import time
import numpy as np
from robot_planning_gui import FastSimWorld


if __name__ == "__main__":

    robot_connect = True
    base = FastSimWorld(robot_connect=robot_connect)
    # whether check interpolated_confs, speeds, and accs 
    # before execute planning trajectory
    base.set_toggle_debug = False
    base.start()
    
    base.run()
    