"""The pixel grid of the patch that the touch sensor covers."""

import numpy as np

from .depth_renderer import DepthRenderer


class SensorGrid:
    """
    The square patch in front of the touch sensor as a grid of `size` x `size` pixels of side `side`,
    centred on the sensor.  The sensor's pose has its origin at the centre of the sensing surface
    and faces the direction of touch.  Depth is measured from a plane `behind` the sensing surface, so that a
    surface that presses into the sensor still has a positive depth, and up to `reach` in front
    of the sensing surface; beyond that the patch shows nothing.
    """

    def __init__(self, size=32, side=17.0, behind=3.0, reach=7.0):
        self.size, self.side, self.behind, self.reach = size, side, behind, reach
        self.pitch = side / size
        self.centres = (np.arange(size) - (size - 1) / 2) * self.pitch

    @property
    def empty(self):
        """The depth of a pixel that shows nothing."""
        return self.behind + self.reach

    def surface_in_front(self, scan, pose):
        """The points of an ObjectScan in front of the patch that a sensor at `pose` meets first; points hidden behind them are dropped."""
        if len(scan) == 0:
            return scan.points
        local = pose.to_local(scan.points_near(pose.t, self.side + self.empty))
        local = local[(local[:, 2] >= -self.behind) & (local[:, 2] <= self.reach)]
        if len(local) == 0:
            return local
        cell = max(scan.VOXEL, scan.point_spacing)
        _, column = np.unique(np.floor(local[:, :2] / cell).astype(np.int64), axis=0, return_inverse=True)
        column = column.ravel()
        nearest = np.full(column.max() + 1, np.inf)
        np.minimum.at(nearest, column, local[:, 2])
        return pose.apply(local[local[:, 2] <= nearest[column] + scan.SHELL])

    def depth_of(self, scan, pose):
        """The depth image (size, size) an ObjectScan shows to a sensor at `pose`: what the sensor would meet there."""
        radius = 1.5 * max(scan.VOXEL, scan.point_spacing)
        return DepthRenderer(self, radius).render(self.surface_in_front(scan, pose), pose)
