"""A calibration board that exists only in memory."""

import numpy as np
from scipy.spatial.transform import Rotation

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import CalibrationBoard


class SimulatedBoard(CalibrationBoard):
    """
    A board lying at `board_in_base` (4 x 4, the robot's base frame, metres), seen by the camera that
    `robot` carries at `camera_in_sensor`.  It reads no image: it answers with where the board is
    from the camera, off by `shake` [m] and `tremble` [rad] as a detector would be, and with None
    when the middle of the board is outside the picture.
    """

    def __init__(self, robot, camera_in_sensor, board_in_base, shake=0.0003, tremble=np.radians(0.2), rng=None):
        self.robot, self.camera_in_sensor = robot, Pose.from_matrix(camera_in_sensor)
        self.board_in_base, self.shake, self.tremble = Pose.from_matrix(board_in_base), shake, tremble
        self.rng = rng or np.random.default_rng(2)

    def pose_in(self, colour, intrinsics):
        seen = (self.robot.pose() @ self.camera_in_sensor).inverse() @ self.board_in_base
        pixel = np.asarray(intrinsics) @ seen.t / seen.t[2]
        if seen.t[2] <= 0 or not (0 <= pixel[0] < np.shape(colour)[1] and 0 <= pixel[1] < np.shape(colour)[0]):
            return None
        error = Pose(Rotation.from_rotvec(self.rng.normal(0, self.tremble, 3)).as_matrix(), self.rng.normal(0, self.shake, 3))
        return seen @ error
