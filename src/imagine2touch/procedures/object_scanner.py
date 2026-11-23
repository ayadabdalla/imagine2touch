"""
Scanning an object into a point cloud (paper: Fig. 1a, the 3D model the robot touches and recognises).

The robot holds the wrist camera over the object from straight above and from four sides.  Every
depth image becomes points in the frame of the table; the points of all views inside a box around
the object, thinned, are the scan.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from imagine2touch.geometry import ObjectScan, Pose, Viewpoints


class ObjectScanner:
    """
        points, normals = ObjectScanner(robot, camera, world_from_base, camera_in_sensor).scan()

    `world_from_base` and `camera_in_sensor` are the two calibrations (4 x 4).  The world frame lies on
    the table with its z axis pointing up; `box` is the part of it that holds the object, as its
    lowest and highest corner [m].  Its floor is above the table, which keeps the table out of the scan.
    The camera looks at the middle of the box from `distance` away, tilted by `tilt` in the side views.
    The sensor sits beside the camera, so in every view the camera is turned about its line of sight
    until the sensor is as high above the table as it gets.  A view is left out if the sensor would
    still come within `margin` of the box, or if the robot does not reach it.
    """

    def __init__(self, robot, camera, world_from_base, camera_in_sensor, box=((-0.05, -0.05, 0.01), (0.05, 0.05, 0.1)),
                 distance=0.12, tilt=np.pi / 4, margin=0.005, half_side=0.0085, settle=0.5, sleep=time.sleep):
        self.robot, self.camera, self.sleep, self.settle = robot, camera, sleep, settle
        self.world_from_base = Pose.from_matrix(world_from_base)
        self.camera_in_sensor = Pose.from_matrix(camera_in_sensor)
        self.low, self.high = np.asarray(box[0], float), np.asarray(box[1], float)
        self.distance, self.tilt, self.margin, self.half_side = distance, tilt, margin, half_side

    TURNS = 12      # how many turns about the line of sight are tried

    def sensor_corners(self, camera_pose):
        """The four corners of the sensing surface with the camera at `camera_pose`, in the frame of that pose's outside, (4, 3)."""
        sensor = camera_pose @ self.camera_in_sensor.inverse()
        return sensor.apply(np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]) * self.half_side)

    def sensor_is_clear(self, camera_pose):
        """Whether the sensing surface stays out of the box, and above the table, with the camera at `camera_pose` (world frame)."""
        corners = self.sensor_corners(camera_pose)
        inside = ((corners > self.low - self.margin) & (corners < self.high + self.margin)).all(1)
        return not inside.any() and corners[:, 2].min() > self.margin

    def views(self):
        """The camera poses to scan from, in the world frame: one above the box and four around it, each turned to keep the sensor high."""
        turns = [Pose(Rotation.from_euler("z", angle).as_matrix()) for angle in 2 * np.pi * np.arange(self.TURNS) / self.TURNS]
        views = Viewpoints.around((self.low + self.high) / 2, self.distance, tilts=(0.0, self.tilt), per_tilt=4)
        return [max((view @ turn for turn in turns), key=lambda pose: self.sensor_corners(pose)[:, 2].min()) for view in views]

    def points_seen(self, depth):
        """Takes a depth image (h, w) [m] taken where the robot is now.  Returns the points it shows inside the box, world frame, (n, 3)."""
        K = self.camera.intrinsics
        row, column = np.nonzero(np.isfinite(depth) & (depth > 0))
        z = depth[row, column]
        in_camera = np.column_stack([(column - K[0, 2]) / K[0, 0] * z, (row - K[1, 2]) / K[1, 1] * z, z])
        points = (self.world_from_base @ self.robot.pose() @ self.camera_in_sensor).apply(in_camera)
        return points[((points > self.low) & (points < self.high)).all(1)]

    def look(self, camera_pose):
        """Takes a camera pose in the world frame.  Returns the points seen from there, or None if the view was left out."""
        if not self.sensor_is_clear(camera_pose):
            return None
        sensor = self.world_from_base.inverse() @ camera_pose @ self.camera_in_sensor.inverse()
        if not self.robot.move_to(sensor):
            return None
        self.sleep(self.settle)                        # let the arm and the camera's exposure come to rest
        return self.points_seen(self.camera.capture()[1])

    def scan(self, views=None):
        """
        Looks from every view (by default `views()`).  Returns the scan as points (n, 3) in the world
        frame [m] and the surface normal at each, (n, 3), pointing up.
        """
        seen = [self.look(view) for view in (self.views() if views is None else views)]
        seen = [points for points in seen if points is not None and len(points)]
        if not seen:
            raise RuntimeError("no view showed the object: check the box, the calibrations and the reach of the robot")
        points = ObjectScan.from_raw_points(1000 * np.concatenate(seen)).points / 1000
        return points, ObjectScan.normals_of(points)
