"""Where the wrist camera sits relative to the touch sensor."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from imagine2touch.geometry import Pose


class CameraCalibration:
    """
    The robot holds the camera over a board of printed tags that lies still, from several directions.
    For every view we know where the robot says the sensor is, and where the camera sees the board.
    The board did not move, so there is one place of the camera on the sensor for which all views
    put the board at the same place: `fit` finds it, and that place of the board with it.

        calibration = CameraCalibration.fit(sensor_poses, board_poses)
        calibration.camera_in_sensor        # the camera in the frame of the sensor
        calibration.board_in_base           # the board in the frame of the robot's base
        calibration.spread                  # how far the views disagree on where the board is [m]

    The board is where the world frame of a setup is: its plane is the table.  Lengths are metres.
    """

    POSITION_WEIGHT = 50        # a metre of disagreement counts as much as this many radians, as in the paper's code

    def __init__(self, camera_in_sensor, board_in_base, spread, turn):
        self.camera_in_sensor, self.board_in_base, self.spread, self.turn = camera_in_sensor, board_in_base, spread, turn

    @classmethod
    def fit(cls, sensor_poses, board_poses):
        """
        Takes, for at least three views that differ in direction, the pose of the sensor in the robot's
        base frame and the pose of the board in the camera's frame.  Returns the calibration.
        """
        if len(sensor_poses) != len(board_poses) or len(sensor_poses) < 3:
            raise ValueError("a calibration needs the same number of sensor and board poses, at least three")
        camera = cls._from_pairs(sensor_poses, board_poses)
        board = cls._mean([s @ camera @ b for s, b in zip(sensor_poses, board_poses)])

        def disagreement(x):
            camera, board = cls._unpack(x[:6]), cls._unpack(x[6:])
            seen = [board.inverse() @ s @ camera @ b for s, b in zip(sensor_poses, board_poses)]
            return np.concatenate([np.r_[cls.POSITION_WEIGHT * p.t, Rotation.from_matrix(p.R).as_rotvec()] for p in seen])

        x = least_squares(disagreement, np.r_[cls._pack(camera), cls._pack(board)]).x
        camera, board = cls._unpack(x[:6]), cls._unpack(x[6:])
        seen = [board.inverse() @ s @ camera @ b for s, b in zip(sensor_poses, board_poses)]
        spread = np.array([np.linalg.norm(p.t) for p in seen])
        turn = np.array([np.linalg.norm(Rotation.from_matrix(p.R).as_rotvec()) for p in seen])
        return cls(camera, board, spread, turn)

    @staticmethod
    def _from_pairs(sensor_poses, board_poses):
        """
        A first answer in closed form (Park and Martin, 1994).  Between two views the sensor moves by A
        and the board, as the camera sees it, by B; the camera X on the sensor satisfies A X = X B.
        """
        A, B = [], []
        for i in range(len(sensor_poses)):
            for j in range(i + 1, len(sensor_poses)):
                A.append(sensor_poses[j].inverse() @ sensor_poses[i])
                B.append(board_poses[j] @ board_poses[i].inverse())
        a = np.array([Rotation.from_matrix(p.R).as_rotvec() for p in A])
        b = np.array([Rotation.from_matrix(p.R).as_rotvec() for p in B])
        U, _, Vt = np.linalg.svd(a.T @ b)                       # the rotation that turns every b into its a
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        lhs = np.concatenate([p.R - np.eye(3) for p in A])
        rhs = np.concatenate([R @ q.t - p.t for p, q in zip(A, B)])
        return Pose(R, np.linalg.lstsq(lhs, rhs, rcond=None)[0])

    @staticmethod
    def _mean(poses):
        return Pose(Rotation.from_matrix(np.array([p.R for p in poses])).mean().as_matrix(), np.mean([p.t for p in poses], axis=0))

    @staticmethod
    def _pack(pose):
        return np.r_[Rotation.from_matrix(pose.R).as_rotvec(), pose.t]

    @staticmethod
    def _unpack(x):
        return Pose(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])

    def save(self, camera_path, world_path):
        """Writes the two 4 x 4 files the scripts take as --camera and --world."""
        np.save(camera_path, self.camera_in_sensor.matrix)
        np.save(world_path, self.board_in_base.matrix)
