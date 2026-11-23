"""
Moving the sensor onto a surface until it feels it.

The robot walks along the direction of touch in small steps.  After every step it compares the
touch reading with the one before; a jump means contact.  If the robot's own force estimate passes
its limit first, the robot stops on that.  If neither happens at the target, the target is pushed a
little further a few times, with the reading before the first push kept as the reference, and then
the approach gives up: the sensor met nothing.

Shortly before the target, the sensor takes the reading that is later subtracted from the reading
at contact (the ambient reading).
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from imagine2touch.datasets.recordings import ApproachOutcome
from imagine2touch.geometry import Pose


@dataclass
class ApproachSettings:
    step: float = 0.004                # length of one step [m]
    pause: float = 0.1                 # time between steps [s]
    jump: float = 25.0                 # change of the reading between two steps that counts as contact
    max_force: float = 8.0             # [N]
    max_torque: float = 8.0            # [Nm]
    tolerance: float = 0.002           # the target counts as reached within this distance [m]
    extensions: int = 3                # how often the target is pushed further
    extension: float = 0.002           # by this much [m]
    ambient_distance: float = 0.01     # the ambient reading is taken this close to the target [m]
    max_distance: float = 0.15         # targets further away than this are refused [m]
    max_turn: float = np.radians(5)    # the sensor must already face the direction of touch


@dataclass
class ApproachResult:
    outcome: ApproachOutcome
    pose: Pose                                     # where the sensor stopped
    wrench: np.ndarray
    ambient: Optional[np.ndarray] = None           # reading taken just before the target
    reading: Optional[np.ndarray] = None           # reading when it stopped
    jump: Optional[float] = None                   # the change of the reading that stopped it
    steps: int = 0


class ContactApproach:
    """
        result = ContactApproach(robot, sensor).run(target, pose)

    `safety` (a SafetyChecker) switches on the checks that keep the tool above the table.
    `sleep` is the function that waits between steps; the tests pass one that does not wait.
    """

    def __init__(self, robot, sensor, settings=None, safety=None, sleep=time.sleep):
        self.robot, self.sensor, self.safety, self.sleep = robot, sensor, safety, sleep
        self.settings = settings or ApproachSettings()

    def _stop(self, outcome, **more):
        return ApproachResult(outcome, self.robot.pose(), self.robot.wrench(), **more)

    def _over_the_limit(self, wrench):
        s = self.settings
        return np.linalg.norm(wrench[:3]) > s.max_force or np.linalg.norm(wrench[3:]) > s.max_torque

    def run(self, target, pose):
        """Move the sensor to the point `target` along the direction of touch of `pose`, the orientation it holds meanwhile."""
        s, robot, sensor = self.settings, self.robot, self.sensor
        target, direction, start = np.asarray(target, float), pose.normal, robot.pose()
        if np.linalg.norm(target - start.t) > s.max_distance or start.angle_to(pose) > s.max_turn:
            return self._stop(ApproachOutcome.ABORTED)
        waypoint, previous, ambient, extended, steps = start.t.copy(), None, None, 0, 0
        while True:
            if np.linalg.norm(target - robot.pose().t) <= s.tolerance:
                if extended == s.extensions:
                    return self._stop(ApproachOutcome.AIR, ambient=ambient, steps=steps)
                target, extended = target + s.extension * direction, extended + 1
            else:
                waypoint = waypoint + s.step * direction
                robot.step_to(waypoint, pose)
                self.sleep(s.pause)
                steps += 1
            reading = sensor.read(2)
            if previous is not None:
                if np.array_equal(reading, previous):
                    raise RuntimeError("the touch sensor repeats its reading: check the cables")
                jump = float(np.linalg.norm(reading - previous))
                if jump > s.jump:
                    return self._stop(ApproachOutcome.SENSOR, ambient=ambient, reading=reading, jump=jump, steps=steps)
            if extended == 0 or previous is None:
                previous = reading                     # once the target is pushed further, the reference stays
            if ambient is None and np.linalg.norm(target - robot.pose().t) < s.ambient_distance:
                ambient = sensor.read(10)
            if self.safety is not None:
                here = Pose(pose.R, robot.pose().t)
                if not self.safety.sensor_clears_table(here):
                    return self._stop(ApproachOutcome.AIR, ambient=ambient, steps=steps)
                if not self.safety.flange_clears_stand(here):
                    return self._stop(ApproachOutcome.ABORTED, ambient=ambient, steps=steps)
            if self._over_the_limit(robot.wrench()):
                return self._stop(ApproachOutcome.WRENCH, ambient=ambient, reading=reading, steps=steps)
