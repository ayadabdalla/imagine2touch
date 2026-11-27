"""What the procedures need from the hardware.  `devices/` and `simulation/` implement these classes."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from imagine2touch.geometry import Pose


class Robot(ABC):
    @abstractmethod
    def pose(self) -> Pose:
        """The pose of the touch sensor in the frame of the robot's base [m]."""

    @abstractmethod
    def joints(self) -> np.ndarray:
        """The seven joint angles [rad]."""

    @abstractmethod
    def wrench(self) -> np.ndarray:
        """The force [N] and torque [Nm] the robot estimates at the sensor, (6,)."""

    @abstractmethod
    def move_to(self, pose: Pose) -> bool:
        """Move the sensor to `pose` and wait until the arm is still.  False if the pose was not reached."""

    @abstractmethod
    def step_to(self, position: np.ndarray, pose: Pose) -> None:
        """Start a straight move of the sensor to `position`, keeping the orientation of `pose`; returns at once."""


class TouchSensor(ABC):
    @abstractmethod
    def read(self, samples: int = 1) -> np.ndarray:
        """
        The mean of the last `samples` readings: 15 field values, three for each of the five
        magnetometers, in the order centre, top, right, bottom, left.
        """


class DepthCamera(ABC):
    intrinsics: np.ndarray          # 3 x 3 camera matrix

    @abstractmethod
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """A colour image (h, w, 3) and a depth image (h, w) in metres, aligned with each other."""


class CalibrationBoard(ABC):
    """
    A flat board lying on the table, which the wrist camera is calibrated against.  Its frame becomes
    the world frame: the board's third axis points out of its printed side, up from the table.
    devices/AprilTagBoard is the printed one; simulation/SimulatedBoard stands in for it where the
    simulated camera draws no picture.
    """

    @abstractmethod
    def pose_in(self, colour: np.ndarray, intrinsics: np.ndarray) -> Optional[Pose]:
        """
        Takes a colour image (h, w, 3) and the 3 x 3 matrix of the camera that took it.  Returns the pose
        of the board in the frame of that camera [m], or None if the board is not seen well enough.
        """
