"""
The shape classification dataset (paper: Table I and Fig. 3; thesis: Sec. 5.1).

The sensor is mounted on a fixture.  The robot holds one of five 3D-printed stamps, each smaller
than the sensing surface (a letter T, a circle, an angle, a triangle and a cross), and presses it into
the skin at eight positions of a grid, to four depths each, again and again.  Every press is one
touch signal; the question is whether the signal tells which stamp it was.  A stamp fits the finger
in two ways, turned by 180 degrees to each other; which one was used was not recorded.

The 1,280 presses of the paper, 256 of every stamp, are in `data/shape_classification.npz` with the
paper's split into training, validation and test presses, made from the recordings by
datasets/recordings/stamp_recording_converter.py.
"""

from pathlib import Path

import numpy as np

from imagine2touch.paths import DATA_DIR


class ShapeClassificationDataset:
    """
      signal     (N, 15) how much each value changed: |reading of the press - reading out of contact before it|
      role       (N,) the paper's split: "train", "validation" or "test"
      stamp      (N,) index into STAMPS
      session    (N,) index into `sessions`, the recorded folders
      position   (N, 2) where on the sensing surface the stamp was pressed [m]
      force      (N,) force along the press that the robot estimated [N]
    """

    FILE = "shape_classification.npz"
    STAMPS = ("T", "circle", "angle", "triangle", "cross")          # the order of the paper's Table I

    def __init__(self, path=None):
        self.path = Path(path or DATA_DIR / self.FILE)
        if self.available:
            z = np.load(self.path)
            self.signal, self.stamp, self.session = np.abs(z["signal"].astype(float)), z["stamp"], z["session"]
            self.position, self.force, self.sessions = z["position"], z["force"], [str(s) for s in z["sessions"]]
            roles = [str(r) for r in z["roles"]] if "roles" in z else ["train"]
            index = z["role"] if "role" in z else np.zeros(len(self.stamp), int)
            self.role = np.array([roles[i] if i >= 0 else "unused" for i in index])

    @property
    def available(self):
        return self.path.exists()

    def __len__(self):
        return len(self.stamp)

    @property
    def position_index(self):
        """(N,) which of the grid positions each press was made at."""
        return np.unique(self.position.round(5), axis=0, return_inverse=True)[1].ravel()

    def balanced(self, per_stamp, rng):
        """Rows of `per_stamp` presses of every stamp, drawn at random."""
        return np.concatenate([rng.choice(np.flatnonzero(self.stamp == k), per_stamp, replace=False)
                               for k in range(len(self.STAMPS))])
