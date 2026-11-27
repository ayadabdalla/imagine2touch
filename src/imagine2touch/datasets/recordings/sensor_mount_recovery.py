"""Where the touch sensor sat on the robot's flange, recovered for recordings that did not save where the sensor stopped."""

import numpy as np
from scipy.spatial.transform import Rotation

from imagine2touch.geometry import FrankaKinematics, Pose


class SensorMountRecovery:
    """
    The recordings hold the robot's joint angles at every touch.  From those we can compute where the
    flange (the end of the arm) was, but the sensor sits further out, on a hand and a printed finger.
    The step from flange to sensor was set in the robot's controller and was never saved to a file.

    We recover it from the data.  At every touch the robot was told where to put the sensor.  So, on
    average, the sensor ended up at that place, and the step from the flange to that place is the
    step from flange to sensor.  `fit` averages it over all touches made with the same mount.

    Where a recording says how each approach ended, only touches that the sensor itself stopped are
    used, since those are the ones that reached their target.
    """

    @staticmethod
    def fit(recordings):
        """Takes recordings made with the same mount.  Returns the pose of the sensor in the frame of the flange [mm]."""
        rotations, positions = [], []
        for recording in recordings:
            for i in recording.felt():
                flange = FrankaKinematics.flange_pose(recording.joints[i])
                commanded_rotation = Rotation.from_quat(recording.rotation[i]).as_matrix()
                commanded_position = 1000 * recording.target[i]                      # the recordings are in metres
                rotations.append(flange.R.T @ commanded_rotation)                    # the commanded sensor pose,
                positions.append(flange.to_local(commanded_position))                # seen from the flange
        if not rotations:
            return Pose()
        return Pose(Rotation.from_matrix(np.array(rotations)).mean().as_matrix(), np.median(positions, axis=0))
