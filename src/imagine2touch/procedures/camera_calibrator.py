"""
Finding where the wrist camera sits, and where the table is.

A board of printed tags lies on the table.  The robot holds the camera over it from many directions;
in every view the camera reports where it sees the board and the robot where it holds the sensor.
`geometry/CameraCalibration` turns those pairs into the two calibrations every other procedure needs.
"""

import time

import numpy as np

from imagine2touch.geometry import CameraCalibration, Pose, Viewpoints
from imagine2touch.interfaces import CalibrationBoard, DepthCamera, Robot


class CameraCalibrator:
    """
        calibration = CameraCalibrator(robot, camera, board, expected_camera_in_sensor).calibrate(board_centre)

    `board` is a CalibrationBoard (interfaces.py).  `expected_camera_in_sensor` (4 x 4) is a rough
    guess of the camera's place on the sensor, from a ruler or a drawing: it only serves to point the
    camera at the board.
    `board_centre` is roughly where the middle of the board is, in the frame of the robot's base [m].
    """

    def __init__(self, robot: Robot, camera: DepthCamera, board: CalibrationBoard, expected_camera_in_sensor, distance=0.15,
                 tilts=(0.0, np.radians(20), np.radians(40)), per_tilt=6, settle=0.5, sleep=time.sleep):
        self.robot, self.camera, self.board, self.sleep, self.settle = robot, camera, board, sleep, settle
        self.expected = Pose.from_matrix(expected_camera_in_sensor)
        self.distance, self.tilts, self.per_tilt = distance, tilts, per_tilt

    def views(self, board_centre):
        """The camera poses to look from, in the robot's base frame."""
        return Viewpoints.around(board_centre, self.distance, self.tilts, self.per_tilt)

    def observe(self, camera_pose):
        """
        Takes a camera pose in the base frame.  Returns where the robot then holds the sensor and where
        the camera sees the board, or None if the robot did not get there or the board was not seen.
        """
        if not self.robot.move_to(camera_pose @ self.expected.inverse()):
            return None
        self.sleep(self.settle)
        seen = self.board.pose_in(self.camera.capture()[0], self.camera.intrinsics)
        return None if seen is None else (self.robot.pose(), seen)

    def calibrate(self, board_centre):
        """Looks from every view and returns the CameraCalibration of the views in which the board was seen."""
        pairs = [pair for pair in map(self.observe, self.views(board_centre)) if pair is not None]
        if len(pairs) < 3:
            raise RuntimeError(f"the board was seen in {len(pairs)} views, a calibration needs three: check the lighting and the board's place")
        return CameraCalibration.fit([sensor for sensor, _ in pairs], [seen for _, seen in pairs])
