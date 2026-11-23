"""The depth image that a set of points shows to the touch sensor."""

import numpy as np


class DepthRenderer:
    """
    Draws every point as a disc of `radius` facing the sensor; a pixel takes the nearest disc over
    its centre.  Depth counts from `grid.behind` behind the sensing surface, and pixels over which
    there is nothing hold `grid.empty`.
    """

    def __init__(self, grid, radius):
        self.grid, self.radius = grid, radius
        self._reach = int(np.floor(radius / grid.pitch + 0.5)) + 1      # how many pixels a disc can spread over

    def render(self, points, pose):
        """The depth image (size, size) of `points` (n, 3) for a sensor at `pose`."""
        grid = self.grid
        local = pose.to_local(points) + [0, 0, grid.behind]
        half = grid.centres[-1] + self.radius
        u, v, z = local[(local[:, 2] >= 0) & (local[:, 2] <= grid.empty) & (np.abs(local[:, :2]) <= half).all(1)].T
        image = np.full((grid.size, grid.size), grid.empty)
        row = np.rint((u - grid.centres[0]) / grid.pitch).astype(int)
        col = np.rint((v - grid.centres[0]) / grid.pitch).astype(int)
        for di in range(-self._reach, self._reach + 1):
            for dj in range(-self._reach, self._reach + 1):
                i, j = row + di, col + dj
                inside = (i >= 0) & (i < grid.size) & (j >= 0) & (j < grid.size)
                inside[inside] = ((grid.centres[i[inside]] - u[inside]) ** 2
                                  + (grid.centres[j[inside]] - v[inside]) ** 2 <= self.radius ** 2)
                np.minimum.at(image, (i[inside], j[inside]), z[inside])
        return image
