"""What the simulated robot stands in front of: a table and one object on it, taken from a scan."""

import numpy as np
from scipy.ndimage import grey_closing

from imagine2touch.geometry import ObjectScan


class SimulatedScene:
    """
    The scan of an object, standing on a table.  `points` (n, 3) is the scan in the frame of the
    robot's base [m], whose z axis points up.  The table is at `table_height` in that frame (by
    default the lowest point of the scan).

    The world frame has its origin on the table under the object (`origin`, by default under the
    middle of the scan) and is not rotated against the base frame.

      world_from_base, base_from_world    the two changes of frame, 4 x 4
      to_world(points), to_base(points)   the same for points (..., 3)

    For the robot and the camera the object is a height above the table at every place: the highest
    scan point in each square of side `cell`.
    """

    def __init__(self, points, table_height=None, origin=None, cell=0.0005):
        points = np.asarray(points, float)
        table = points[:, 2].min() if table_height is None else table_height
        middle = points[:, :2].mean(0) if origin is None else np.asarray(origin, float)[:2]
        self.base_from_world = np.eye(4)
        self.base_from_world[:3, 3] = [middle[0], middle[1], table]
        self.world_from_base = np.linalg.inv(self.base_from_world)
        self.points, self.cell = self.to_world(points), cell
        self._corner = self.points[:, :2].min(0) - cell
        index = np.floor((self.points[:, :2] - self._corner) / cell).astype(int)
        heights = np.zeros(index.max(0) + 2)
        np.maximum.at(heights, (index[:, 0], index[:, 1]), self.points[:, 2])
        self.heights = grey_closing(heights, size=3)            # a scan has small holes; the object does not

    @classmethod
    def of_session(cls, dataset, session):
        """Takes a VisuoTactileDataset and the name of one of its sessions.  Returns the scene with that object's scan."""
        return cls(dataset.scan(dataset.objects.index(session)).points / 1000)

    # ------------------------------------------------------------------ the two frames
    def to_world(self, points):
        return np.asarray(points, float) @ self.world_from_base[:3, :3].T + self.world_from_base[:3, 3]

    def to_base(self, points):
        return np.asarray(points, float) @ self.base_from_world[:3, :3].T + self.base_from_world[:3, 3]

    @property
    def up(self):
        """The direction up from the table, in the frame of the robot's base."""
        return self.world_from_base[2, :3]

    # ------------------------------------------------------------------ the surface
    @property
    def object_height(self):
        """The height of the object's highest point above the table."""
        return float(self.heights.max())

    def surface_height(self, x, y):
        """Height of the surface above the table at the world coordinates x, y: the object, or 0 beside it."""
        i = np.floor((np.asarray(x, float) - self._corner[0]) / self.cell).astype(int)
        j = np.floor((np.asarray(y, float) - self._corner[1]) / self.cell).astype(int)
        inside = (i >= 0) & (i < self.heights.shape[0]) & (j >= 0) & (j < self.heights.shape[1])
        return np.where(inside, self.heights[np.clip(i, 0, self.heights.shape[0] - 1), np.clip(j, 0, self.heights.shape[1] - 1)], 0.0)

    def depth_below_surface(self, points):
        """How far points (n, 3) of the base frame are under the surface; 0 for points above it."""
        x, y, z = self.to_world(points).T
        return np.maximum(self.surface_height(x, y) - z, 0)

    def scan(self):
        """Returns the scan in the world frame, (n, 3), and the surface normal at every point, (n, 3)."""
        return self.points, ObjectScan.normals_of(self.points)
