"""The checks a touch has to pass before and while the robot moves."""

from dataclasses import dataclass

import numpy as np

from imagine2touch.geometry import Pose


@dataclass
class SafetySettings:
    steepest: float = 0.25              # least cosine between the direction of touch and straight down
    clearance: float = 0.003            # nothing may stand higher than this in the sensor's way [m]
    camera_floor: float = 0.005         # the camera stays this far above the table when it looks [m]
    table_clearance: float = 0.005      # the sensor stays this far above the table [m]
    flange_floor: float = -0.06         # the flange may dip this far below it, beside the object's stand [m]
    sensor_box: tuple = (0.0085, 0.0085, 0.0)       # half sides of the sensor and of the flange across the
    flange_box: tuple = (0.015, 0.09, 0.058)        # direction of touch, and their length behind the sensor [m]


class SafetyChecker:
    """
    All checks are made in the world frame, whose z axis points up from the table.
    `world_from_base` (4 x 4) maps the robot's base frame to it; poses are given in the base frame.
    """

    def __init__(self, world_from_base, settings=None):
        self.world_from_base = Pose.from_matrix(world_from_base)
        self.settings = settings or SafetySettings()

    # ------------------------------------------------------------------ before the robot moves (world frame)
    def faces_up(self, direction):
        """Whether a direction of touch points down steeply enough to be reached from above."""
        return float(-np.asarray(direction)[2]) > self.settings.steepest

    def path_is_clear(self, cloud, point, direction, half_side):
        """
        Whether the sensor can reach `point` along `direction`: within the square of `half_side` around
        the point, no point of the cloud may stand more than the clearance in front of it.
        """
        local = Pose.from_normal(direction, point).to_local(cloud)
        inside = (np.abs(local[:, :2]) < half_side).all(1)
        return not bool((local[inside, 2] < -self.settings.clearance).any())

    def can_touch(self, cloud, point, normal, half_side):
        """Both of the above for a scan point and its surface normal."""
        return self.faces_up(-normal) and self.path_is_clear(cloud, point, -normal, half_side)

    # ------------------------------------------------------------------ while it moves (poses in the base frame)
    def lowest_point(self, pose, box):
        """The height above the table of the lowest corner of a box around the sensor at `pose`."""
        x, y, z = box
        corners = pose.apply(np.array([[x, y, -z], [-x, y, -z], [x, -y, -z], [-x, -y, -z]]))
        return float(self.world_from_base.apply(corners)[:, 2].min())

    def sensor_clears_table(self, pose):
        return self.lowest_point(pose, self.settings.sensor_box) >= self.settings.table_clearance

    def flange_clears_stand(self, pose):
        return self.lowest_point(pose, self.settings.flange_box) >= self.settings.flange_floor

    def camera_clears_table(self, camera_pose):
        return float(self.world_from_base.apply(camera_pose.t)[2]) >= self.settings.camera_floor
