"""Where an object's scan stood under the robot, for sessions that did not record it."""

import numpy as np
from scipy.spatial import cKDTree

from imagine2touch.datasets.recordings.point_cloud_file import PointCloudFile


class ScanPlacement:
    """
    A scan is stored in the frame of the table.  The later sessions saved, with every recording, the
    calibration that maps it into the robot's frame.  The sessions of March 2023 did not: their
    recording code read it from a file of its own, `reskin/utils/socket_robot_transform.npy`, kept
    here as TABLE_IN_ROBOT_MARCH_2023 (from commit 1a93322b of the recording code, 29 March 2023).

    `fit` is the check on that calibration, and the way to place scans when none is known.  Every
    target the robot was sent to is a point of the scan, so there is one shift that puts all targets
    of a recording onto its scan; `fit` finds it for every recording and returns the median.  For the
    sessions of March 2023 it agrees with the stored calibration within 2 mm.
    """

    TABLE_IN_ROBOT_MARCH_2023 = np.array([[0.99815636, -0.03295978, -0.05096604, 0.50102551],
                                          [0.03387397, 0.99927847, 0.01717841, -0.00378739],
                                          [0.05036307, -0.01887316, 0.99855263, -0.02844102],
                                          [0.0, 0.0, 0.0, 1.0]])

    @classmethod
    def stored(cls):
        """The 4 x 4 transform from the robot's base frame to the frame of the scans, for the sessions of March 2023."""
        return np.linalg.inv(cls.TABLE_IN_ROBOT_MARCH_2023)

    PROBES, SHORTLIST, CAP = 6, 40, 0.005

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def shift_of(self, recording):
        """The shift [m] that puts the targets of one recording on its scan, and the distances left [m]."""
        targets, cloud = recording.target, PointCloudFile.read(recording.scan_path)
        tree = cKDTree(cloud)
        probe = self.rng.choice(len(targets), min(self.PROBES, len(targets)), replace=False)
        candidates = targets[probe[0]] - cloud                  # every shift that puts one target on a scan point
        score = sum(np.minimum(tree.query(targets[i] - candidates)[0], self.CAP) for i in probe[1:])
        shortlist = candidates[np.argsort(score)[:self.SHORTLIST]]
        shift = min(shortlist, key=lambda t: np.median(tree.query(targets - t)[0]))
        for _ in range(20):                                     # settle on the median offset of the well-fitting targets
            distance, nearest = tree.query(targets - shift)
            keep = distance <= np.percentile(distance, 80)
            shift = shift + np.median((targets - shift)[keep] - cloud[nearest[keep]], axis=0)
        return shift, tree.query(targets - shift)[0]

    def fit(self, recordings, least=8):
        """
        The 4 x 4 transform from the robot's base frame to the frame of the scans, shared by the
        recordings, and the per-recording shifts it was taken from.  Recordings with fewer than `least`
        touches say too little and are left out.
        """
        shifts = np.array([self.shift_of(r)[0] for r in recordings if r.n >= least])
        world_from_base = np.eye(4)
        world_from_base[:3, 3] = -np.median(shifts, axis=0)
        return world_from_base, shifts
