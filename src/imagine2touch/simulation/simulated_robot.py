"""A robot arm that exists only in memory."""

import numpy as np

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import Robot


class SimulatedRobot(Robot):
    """
    Moves the sensor where it is told, except into the object of `scene` or the table: the skin of the sensor
    gives way by `skin`, and further motion only raises the force, by `stiffness` per metre it was
    told to go on.  `half_side` is half the side of the square sensing surface.
    """

    def __init__(self, scene, skin=0.0015, stiffness=4000.0, half_side=0.0085, error=0.0, rng=None):
        self.scene, self.skin, self.stiffness, self.half_side, self.error = scene, skin, stiffness, half_side, error
        self.rng = rng or np.random.default_rng(0)
        self._pose = Pose.from_normal(-scene.up, scene.to_base([0, 0, 0.2]))      # 20 cm above the table, facing it
        self._blocked = 0.0
        self.moves = []

    def footprint(self, pose, n=35):
        """Points of the sensing surface at `pose`, in the base frame, (n*n, 3): one every half millimetre, as fine as the scene."""
        a = np.linspace(-self.half_side, self.half_side, n)
        u, v = [g.ravel() for g in np.meshgrid(a, a)]
        return pose.apply(np.column_stack([u, v, np.zeros(len(u))]))

    DEPTH = 0.03            # scan points further behind the sensing surface than this are not in its way [m]

    def press(self, pose):
        """
        How deep the object stands inside the sensing surface at `pose` [m], as an array of one value.
        Measured along the sensor's own direction against the points of the scan in front of its patch,
        so that a tilted sensor meets an edge or a slope where a real one would.
        """
        local = pose.to_local(self.scene.to_base(self.scene.points))
        inside = (np.abs(local[:, :2]) <= self.half_side).all(1) & (local[:, 2] > -self.DEPTH)
        return np.array([max(-local[inside, 2].min(), 0.0) if inside.any() else 0.0])

    def gap(self, pose):
        """
        How far in front of the sensing surface at `pose` the object's nearest point lies [m];
        infinite if the object has no point in front of the patch within DEPTH.
        """
        local = pose.to_local(self.scene.to_base(self.scene.points))
        ahead = (np.abs(local[:, :2]) <= self.half_side).all(1) & (local[:, 2] > 0) & (local[:, 2] < self.DEPTH)
        return float(local[ahead, 2].min()) if ahead.any() else np.inf

    def _settle(self, pose):
        """The pose the sensor ends in when told to go to `pose`: backed out along its direction until the skin holds."""
        under_table = max(-self.scene.to_world(self.footprint(pose, n=3))[:, 2].min(), 0.0)
        descent = max(-self.scene.up @ pose.normal, 1e-6)       # how much of a step along the direction of touch goes down
        self._blocked = max(self.press(pose).max() - self.skin, under_table / descent, 0.0)
        return Pose(pose.R, pose.t - self._blocked * pose.normal)

    # ---------------------------------------------------------------- the Robot interface
    def pose(self):
        return self._pose

    def joints(self):
        return np.zeros(7)

    def wrench(self):
        return np.concatenate([-self.stiffness * self._blocked * self._pose.normal, np.zeros(3)])

    def move_to(self, pose):
        noise = self.rng.normal(0, self.error, 3) if self.error else 0
        self._pose = self._settle(Pose(pose.R, pose.t + noise))
        self.moves.append(("move", pose.t.copy()))
        return True

    def step_to(self, position, pose):
        self._pose = self._settle(Pose(pose.R, np.asarray(position, float)))
        self.moves.append(("step", np.asarray(position, float).copy()))
