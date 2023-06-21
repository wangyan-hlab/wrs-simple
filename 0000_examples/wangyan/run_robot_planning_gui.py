import time
import numpy as np
from robot_planning_gui import FastSimWorld


if __name__ == "__main__":

    robot_connect = True
    base = FastSimWorld(robot_connect=robot_connect)
    base.start()
    
    base.run()
    