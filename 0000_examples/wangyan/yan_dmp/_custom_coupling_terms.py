import math
import numpy as np
from scipy.interpolate import interp1d
import pytransform3d.rotations as pr
import pytransform3d.batch_rotations as pbr
import pytransform3d.transformations as pt


EPSILON = 1e-10


def obstacle_avoidance_acceleration_3d(
        y, yd, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
    """Compute acceleration for obstacle avoidance in 3D.

    Parameters
    ----------
    y : array, shape (..., 3)
        Current position(s).

    yd : array, shape (..., 3)
        Current velocity / velocities.

    obstacle_position : array, shape (3,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 3)
        Accelerations.
    """
    if len(y) == 3:
        obstacle_diff = obstacle_position - y
        r = 0.5 * np.pi * pr.norm_vector(np.cross(obstacle_diff, yd))
        R = pr.matrix_from_compact_axis_angle(r)
        theta = np.arccos(
            np.dot(obstacle_diff, yd)
            / (np.linalg.norm(obstacle_diff) * np.linalg.norm(yd) + EPSILON))
        cdd = gamma * np.dot(R, yd) * theta * np.exp(-beta * theta)

        return cdd

    elif len(y) == 4:
        return np.zeros(3)

    else:
        raise ValueError("Current position/quaternion must be of shape (3, ) or (4, )!")


class CouplingTermObstacleAvoidance3D:  # for DMP
    """Coupling term for obstacle avoidance in 3D."""
    def __init__(self, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
        self.obstacle_position = obstacle_position
        self.gamma = gamma
        self.beta = beta

    def coupling(self, y, yd):
        cdd = obstacle_avoidance_acceleration_3d(
            y, yd, self.obstacle_position, self.gamma, self.beta)
        return np.zeros_like(cdd), cdd
