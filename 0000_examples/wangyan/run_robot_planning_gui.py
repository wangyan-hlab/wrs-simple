import time
import numpy as np
from fastsim_gui import FastSimWorld


if __name__ == "__main__":

    robot_connect = False
    base = FastSimWorld(robot_connect=robot_connect)
    base.start()
    
    base.run()
    