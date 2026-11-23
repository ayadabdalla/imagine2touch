"""A depth camera that exists only in memory."""

import numpy as np

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import DepthCamera


class SimulatedCamera(DepthCamera):
    """A pinhole depth camera fixed to the sensor of `robot` at `camera_in_sensor`, looking along the direction of touch."""

    # the wrist camera of the paper's robot, rounded: beside and behind the sensor, turned a quarter turn about its axis
    MOUNT = np.array([[0.0, 1, 0, -0.059], [-1, 0, 0, 0.014], [0, 0, 1, -0.063], [0, 0, 0, 1]])

    def __init__(self, robot, camera_in_sensor=None, width=160, height=120, focal=300.0):
        self.robot, self.width, self.height = robot, width, height
        self.intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        self.camera_in_sensor = self.MOUNT.copy() if camera_in_sensor is None else np.asarray(camera_in_sensor, float)

    def capture(self):
        """Colour (h, w, 3) and depth (h, w) [m]: every pixel's ray is followed until it goes below the surface."""
        scene = self.robot.scene
        camera = self.robot.pose() @ Pose.from_matrix(self.camera_in_sensor)
        j, i = np.meshgrid(np.arange(self.width), np.arange(self.height))
        rays = np.stack([(j - self.intrinsics[0, 2]) / self.intrinsics[0, 0],
                         (i - self.intrinsics[1, 2]) / self.intrinsics[1, 1], np.ones(j.shape)], -1)
        depth = np.zeros(j.shape)
        for z in np.arange(0.02, 0.3, 0.0005):
            p = scene.to_world(camera.apply(rays * z))
            hit = (depth == 0) & (p[..., 2] <= scene.surface_height(p[..., 0], p[..., 1]))
            depth[hit] = z
        grey = np.clip(255 * (1 - depth / 0.3), 0, 255).astype(np.uint8)
        return np.repeat(grey[..., None], 3, -1), depth
