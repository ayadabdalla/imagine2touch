"""
The Imagine2touch dataset: every touch of the robot with what it saw and what it felt.

A Franka arm carries a ReSkin touch sensor (five magnetometers under a magnetic skin) and a depth
camera at its wrist.  It scans a 3D-printed object into a point cloud, then repeats: pick a point of
the scan, look at it with the wrist camera from 4 cm along the surface normal, and move the sensor
along that normal until the touch reading jumps.

The dataset is one file, `data/imagine2touch.npz`, made from the robot's recordings by
datasets/recordings/recording_reducer.py: by default the six sessions of the paper, 1,630 touches.  All lengths
are millimetres and all poses are in the frame of the robot's base.
"""

from pathlib import Path

import numpy as np

from imagine2touch.geometry import ObjectScan, Pose, SensorGrid
from imagine2touch.paths import DATA_DIR
from imagine2touch.datasets.recordings.approach_outcome import ApproachOutcome


class VisuoTactileDataset:
    """
    One row per approach of the robot.

    Touch
      signal            (N, 15) how much the reading changed at contact, |reading - ambient|: three field
                        components for each magnetometer, in the order centre, top, right, bottom, left
      reading, ambient  (N, 15) the two readings themselves
      outcome           (N,) what ended the approach: "sensor" the reading jumped, "wrench" the robot
                        felt its force limit first, "air" it passed the point without either, "aborted",
                        or "unknown" in the sessions of March 2023, which did not record it
      wrench            (N, 6) force [N] and torque [Nm] the robot estimated when it stopped; NaN if not recorded
    Vision (the patch the sensor is about to cover, seen by the wrist camera from 4 cm away)
      depth             (N, 48, 48) depth [mm]; 0 = no reading
      depth_processed   (N, 48, 48) the same with missing pixels filled
      mask              (N, 48, 48) the pixels within 2 mm of the nearest one: the part the sensor meets
      rgb               (N, 48, 48, 3) colour image
      camera_to_sensor  (N,) the depth the camera shows for a surface lying on the sensing surface
                        where the sensor stopped; NaN in the sessions of March 2023, whose camera was tilted
    Robot
      commanded, pose   the N poses the sensor was sent to and the N poses it stopped at
      joints            (N, 7) joint angles when it stopped [rad]
      tool, early_tool  the fitted pose of the sensor in the frame of the robot's flange, in the
                        sessions of September and of March 2023
    Objects
      object, shape     (N,) indices into `objects`, the recorded sessions, and `shapes`, the printed shapes
      experiment        (N,) number of the experiment within the session
      scan(k)           the scan of session k, an ObjectScan; empty for the sessions of March 2023, whose
                        scans cannot be placed in the robot's frame (`has_scan(k)`)
    """

    FILE = "imagine2touch.npz"
    PAPER_SESSIONS = ("flower", "hexagon", "parallelogram", "star", "square", "cutter")     # the paper's Fig. 1e

    def __init__(self, path=None):
        self.path = Path(path or DATA_DIR / self.FILE)
        self.grid = SensorGrid(size=48)
        if not self.available:
            return
        z = np.load(self.path)
        self.objects, self.shapes = [str(o) for o in z["objects"]], [str(s) for s in z["shapes"]]
        self.object, self.shape, self.experiment = z["object"], z["shape"], z["experiment"]
        self.outcome = np.array([ApproachOutcome.label_of(int(code)) for code in z["outcome"]])
        self.signal, self.reading, self.ambient = np.abs(z["signal"]), z["reading"], z["ambient"]
        self.wrench, self.joints, self.camera_to_sensor = z["wrench"], z["joints"], z["camera_to_sensor"]
        self.depth, self.depth_processed, self.mask, self.rgb = z["depth"], z["depth_processed"], z["mask"], z["rgb"]
        self.pose = [Pose(R, t) for R, t in zip(z["pose_R"], z["pose_t"])]
        self.commanded = [Pose(R, t) for R, t in zip(z["commanded_R"], z["commanded_t"])]
        self.tool, self.early_tool = Pose(z["tool_R"], z["tool_t"]), Pose(z["early_tool_R"], z["early_tool_t"])
        self._scan_points, self._scan_start, self._scans = z["scan"], z["scan_start"], {}

    @property
    def available(self):
        return self.path.exists()

    def __len__(self):
        return len(self.object)

    # ------------------------------------------------------------------ selections
    def sessions(self, which):
        """
        Sessions as indices into `objects`.  `which` is a list of session names, in which case their
        order is kept, or the ending of the names, e.g. '_v_6'.
        """
        if isinstance(which, str):
            return [k for k, name in enumerate(self.objects) if name.endswith(which)]
        return [self.objects.index(name) for name in which]

    def in_sessions(self, which):
        """Mask of the touches recorded in the sessions `which` (see `sessions`)."""
        return np.isin(self.object, self.sessions(which))

    def shape_of_session(self, k):
        return self.shapes[self.shape[np.flatnonzero(self.object == k)[0]]]

    def pose_error(self, i):
        """How far the sensor stopped from the point it was sent to [mm]."""
        return float(np.linalg.norm(self.pose[i].t - self.commanded[i].t))

    # ------------------------------------------------------------------ the scans
    def scan(self, k):
        if k not in self._scans:
            self._scans[k] = ObjectScan(self._scan_points[self._scan_start[k]:self._scan_start[k + 1]])
        return self._scans[k]

    def has_scan(self, k):
        return len(self.scan(k)) > 0

    def scan_depth(self, i, k=None, pose=None):
        """
        The depth image (48, 48) that the scan of session k shows to the sensor at the pose where touch
        i stopped (or at `pose`): what the sensor would meet there if that object stood in the robot's
        workspace.  By default k is the session of touch i, the object that was really touched.
        """
        return self.grid.depth_of(self.scan(int(self.object[i]) if k is None else k), pose or self.pose[i])

    def scan_depths(self, rows, k=None):
        return np.stack([self.scan_depth(i, k) for i in rows]).astype(np.float32)

    def distance_to_scan(self, i, k):
        """Distance from the sensor, where touch i stopped, to the nearest point of the scan of session k [mm]."""
        return self.scan(k).distance_to(self.pose[i].t)
