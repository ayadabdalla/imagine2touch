"""The scan of an object: the point cloud the robot built before touching it."""

import numpy as np
from scipy.spatial import cKDTree



class ObjectScan:
    """
    A scan in the frame of the robot's base [mm], thinned to one point per cube of side VOXEL.

    A raw scan is a layer of points several millimetres thick whose front is the object's surface,
    with a few stray points in front of it.  Thinning keeps a cube only if it holds at least DENSE
    points, which removes the stray ones.  A sensor looking at the scan (SensorGrid) keeps only the
    points within SHELL of the nearest one along its direction of touch: the front of the layer.
    """

    VOXEL, DENSE, SHELL = 0.4, 3, 1.0

    def __init__(self, points):
        self.points = np.asarray(points, float)
        self._tree = None

    @property
    def point_spacing(self):
        """The typical distance between neighbouring points [mm]."""
        if len(self.points) < 2:
            return self.VOXEL
        sample = self.points[::max(len(self.points) // 2000, 1)]
        return float(np.median(self._neighbours.query(sample, k=2)[0][:, 1]))

    @classmethod
    def from_raw_points(cls, points):
        """Thin a raw scan (n, 3).  Scans that were already thinned when they were recorded are kept as they are."""
        points = np.asarray(points, float)
        if len(points) == 0 or cls(points).point_spacing > cls.VOXEL / 2:
            return cls(points)
        cubes = np.floor(points / cls.VOXEL).astype(np.int64)
        _, inverse, count = np.unique(cubes, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(count), 3))
        np.add.at(sums, inverse.ravel(), points)
        return cls((sums / count[:, None])[count >= cls.DENSE])

    def __len__(self):
        return len(self.points)

    @property
    def _neighbours(self):
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def distance_to(self, point):
        """Distance from `point` to the nearest point of the scan."""
        return float(self._neighbours.query(point)[0])

    def points_near(self, point, radius):
        """The points of the scan within `radius` of `point`."""
        return self.points[self._neighbours.query_ball_point(point, radius)]

    @staticmethod
    def normals_of(points, neighbours=30):
        """
        Takes points (n, 3) of a surface whose outside is up (+z).  Returns the surface normal at every
        point, (n, 3): the direction across which its nearest points spread least, pointing up.
        """
        points = np.asarray(points, float)
        _, nearest = cKDTree(points).query(points, k=min(neighbours, len(points)))
        spread = points[nearest] - points[nearest].mean(1, keepdims=True)
        normals = np.linalg.eigh(np.einsum("nki,nkj->nij", spread, spread))[1][:, :, 0]
        return normals * np.where(normals[:, 2:] < 0, -1, 1)
