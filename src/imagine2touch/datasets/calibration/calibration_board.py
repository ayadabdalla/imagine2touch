"""A printed board that the wrist camera can find in a picture, to be calibrated against."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from imagine2touch.geometry import Pose


class CalibrationBoard(ABC):
    """
    A flat board lying on the table.  Its frame becomes the world frame: the board's third axis
    points out of its printed side, up from the table.  AprilTagBoard is the printed one;
    simulation/SimulatedBoard stands in for it where the simulated camera draws no picture.
    """

    @abstractmethod
    def pose_in(self, colour: np.ndarray, intrinsics: np.ndarray) -> Optional[Pose]:
        """
        Takes a colour image (h, w, 3) and the 3 x 3 matrix of the camera that took it.  Returns the pose
        of the board in the frame of that camera [m], or None if the board is not seen well enough.
        """
