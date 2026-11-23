"""A touch sensor that exists only in memory and answers with recorded touches."""

import numpy as np

from imagine2touch.interfaces import TouchSensor


class SimulatedSensor(TouchSensor):
    """
    The sensor that `robot` carries.  Out of contact it reads its rest field and noise.  In contact
    it adds the signal that the real sensor gave at the recorded touch nearest to where it is now,
    growing from nothing to the full signal while the skin is pressed.

    The nearest recorded touch is typically 2 to 3 mm away and may have come from another direction:
    the signal is a real one, but of a slightly different touch.

    With `reach` [m] and `turn` [rad] it answers only if a recorded touch lies that close and was made
    from within that angle of its own direction; otherwise it reads NaN, and the touch does not count.

    `positions` (n, 3) are where the real sensor stopped at n recorded touches, in the frame of the
    robot's base [m], and `directions` (n, 3) the directions it touched along; `signals` (n, 15) is
    what it felt there, and `rest` (15,) its reading in the air.
    """

    def __init__(self, robot, positions, signals, rest, directions=None, reach=None, turn=None, noise=0.5, rng=None):
        self.robot, self.reach, self.turn, self.noise = robot, reach, turn, noise
        self.positions, self.signals, self.rest = np.asarray(positions, float), np.asarray(signals, float), np.asarray(rest, float)
        self.directions = None if directions is None else np.asarray(directions, float)
        self.rng = rng or np.random.default_rng(1)
        self.reads = 0

    @classmethod
    def of_session(cls, robot, dataset, session, max_signal=500, **settings):
        """
        Takes a VisuoTactileDataset and the name of one of its sessions.  Returns a sensor that answers
        with that session's touches; readings above `max_signal` are glitches and are left out.
        """
        rows = np.flatnonzero((dataset.object == dataset.objects.index(session)) & (np.abs(dataset.signal).max(1) <= max_signal))
        positions = np.array([dataset.pose[i].t for i in rows]) / 1000
        directions = np.array([dataset.pose[i].normal for i in rows])
        return cls(robot, positions, dataset.signal[rows], dataset.ambient[rows].mean(0), directions, **settings)

    def response(self, pose):
        """
        The change of the 15 field values at `pose`, without noise.  Out of contact it is zero.  In
        contact where no recorded touch lies within the limits, it is NaN: the sensor has no answer.
        """
        pressed = min(self.robot.press(pose).max() / self.robot.skin, 1.0)
        if pressed == 0:
            return np.zeros(15)
        distance = np.linalg.norm(self.positions - pose.t, axis=1)
        if self.turn is not None:                      # only recorded touches made from about this direction
            distance = np.where(self.directions @ pose.normal >= np.cos(self.turn), distance, np.inf)
        if not np.isfinite(distance.min()) or (self.reach is not None and distance.min() > self.reach):
            return np.full(15, np.nan)                 # no recorded touch to answer with
        return pressed * self.signals[distance.argmin()]

    def read(self, samples=1):
        self.reads += samples
        noise = self.rng.normal(0, self.noise / np.sqrt(samples), 15)
        return self.rest + self.response(self.robot.pose()) + noise
