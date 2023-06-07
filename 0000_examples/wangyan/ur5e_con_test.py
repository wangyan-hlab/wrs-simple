#%%
from robot_con.ur.ur5e import UR5ERtqHE
import visualization.panda.world as wd
import numpy as np

# base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])

#%%
ur5e = UR5ERtqHE(robot_ip='192.168.58.2', pc_ip='192.168.58.70')
jnt_values_rad = ur5e.get_jnt_values()
jnt_values_deg = np.rad2deg(jnt_values_rad)
print(jnt_values_deg)

#%%
start_jnt_deg = [-111.817, -87.609, -118.858, -55.275, 107.847, 20.778]
ur5e.move_jnts(start_jnt_deg)

#%%
goal_jnt_deg = [-127.146, -74.498, -85.835, -40.605, 71.584, 20.790]
ur5e.move_jnts(goal_jnt_deg)

# base.run()
# %%
