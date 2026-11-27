"""
Collecting touches (paper: Fig. 1a, 1b).

For one touch the robot
  1. picks a point of the object's scan at random among those it can reach safely,
  2. holds the wrist camera 4 cm in front of the point, along the surface normal, and keeps the part
     of the depth image that the sensor is about to cover,
  3. puts the sensor where the camera was and moves it onto the surface until it feels it,
  4. stores the image, the reading and where it stopped, and backs away.
"""

import time

import numpy as np

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import DepthCamera, Robot, TouchSensor
from .contact_approach import ContactApproach
from .safety_checker import SafetyChecker


class TouchCollector:
    """
    `cloud` (n, 3) and `normals` (n, 3) are the scan of the object in the world frame, whose z axis
    points up from the table; `world_from_base` and `camera_in_sensor` are the two calibrations.
    Points to touch are drawn from at most `candidates` scan points that pass the safety checks.
    The sensor is turned about its direction of touch so that its first axis points as near to
    `sensor_x` (base frame) as possible: backwards, as the paper's robot held it.
    """

    def __init__(self, robot: Robot, sensor: TouchSensor, camera: DepthCamera, cloud, normals, world_from_base, camera_in_sensor,
                 look=0.04, retreat=0.02, half_side=0.0085, candidates=2000, sensor_x=(-1.0, 0.0, 0.0),
                 approach_settings=None, safety_settings=None, rng=None, sleep=time.sleep):
        self.robot, self.sensor, self.camera = robot, sensor, camera
        self.cloud, self.normals = np.asarray(cloud, float), np.asarray(normals, float)
        self.world_from_base = np.asarray(world_from_base, float)
        self.base_from_world = Pose.from_matrix(np.linalg.inv(self.world_from_base))
        self.camera_in_sensor = Pose.from_matrix(camera_in_sensor)
        self.look, self.retreat, self.half_side, self.sensor_x = look, retreat, half_side, sensor_x
        self.rng = rng or np.random.default_rng()
        self.safety = SafetyChecker(self.world_from_base, safety_settings)
        self.approach = ContactApproach(robot, sensor, approach_settings, self.safety, sleep)
        self.candidates = np.arange(len(self.cloud))[::max(len(self.cloud) // candidates, 1)]
        self.safe = np.array([self.safety.can_touch(self.cloud, self.cloud[i], self.normals[i], half_side)
                              for i in self.candidates])

    def plan(self, index):
        """For scan point `index`: the target and the poses to look from, start from and back away to, in the base frame."""
        target = self.base_from_world.apply(self.cloud[index])
        direction = self.base_from_world.R @ (-self.normals[index] / np.linalg.norm(self.normals[index]))
        start = Pose.from_normal(direction, target - self.look * direction, self.sensor_x)
        # shift the sensor sideways so that the camera, not the sensor, is in front of the point
        sideways = np.array([-self.camera_in_sensor.t[0], -self.camera_in_sensor.t[1], 0])
        looking = Pose(start.R, start.apply(sideways))
        away = Pose(start.R, target - (self.look + self.retreat) * direction)
        return target, start, looking, away

    def crop(self, image, target):
        """The part of a camera image that the sensing surface will cover around `target` (base frame)."""
        camera = self.robot.pose() @ self.camera_in_sensor
        corners = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]) * self.half_side
        corners = camera.to_local(target) + corners @ self.camera_in_sensor.R.T
        pixels = (self.camera.intrinsics @ corners.T / corners[:, 2]).T
        c0, c1 = sorted((int(pixels[:, 0].min()), int(pixels[:, 0].max())))
        r0, r1 = sorted((int(pixels[:, 1].min()), int(pixels[:, 1].max())))
        return image[max(r0, 0):r1, max(c0, 0):c1]

    def touch(self, index=None):
        """One touch, at scan point `index` or at a safe point drawn at random.  A dictionary, or None if it was skipped."""
        index = int(self.rng.choice(self.candidates[self.safe])) if index is None else index
        target, start, looking, away = self.plan(index)
        if not self.safety.camera_clears_table(looking @ self.camera_in_sensor) or not self.robot.move_to(looking):
            return None
        rgb, depth = self.camera.capture()
        rgb, depth = self.crop(rgb, target), self.crop(depth, target)
        if not self.robot.move_to(start):
            return None
        result = self.approach.run(target, start)
        record = None
        if result.ambient is not None:                 # without the reading before contact a touch is of no use
            record = dict(index=index, target=target, rotation=start.quaternion, prepare=start.t, pose=result.pose.matrix,
                          joints=self.robot.joints(), wrench=result.wrench, outcome=int(result.outcome),
                          reading=self.sensor.read(1), ambient=result.ambient, jump=result.jump or 0.0,
                          rgb=rgb, depth=np.round(1000 * depth).astype(np.uint16))
            if not np.isfinite(record["reading"]).all():       # a simulated sensor without an answer here
                record = None
        self.robot.move_to(away)
        return record

    def collect(self, touches, writer=None, attempts=None):
        """Touch until `touches` are recorded (or `attempts` are used up); returns the records."""
        records, attempts = [], attempts or 10 * touches
        while len(records) < touches and attempts > 0:
            attempts -= 1
            record = self.touch()
            if record is not None:
                records.append(record)
                if writer is not None:
                    writer.add(record)
        if writer is not None:
            writer.save(self.cloud, self.world_from_base, self.camera_in_sensor.matrix)
        return records
