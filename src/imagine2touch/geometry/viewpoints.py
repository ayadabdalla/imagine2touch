"""Places from which a camera looks at one point."""

import numpy as np

from .pose import Pose


class Viewpoints:
    """
    The paper's robot looked at an object, and at the board it was calibrated with, from straight
    above and from places tilted away from that: all at one distance, all facing the same point.
    """

    @staticmethod
    def around(centre, distance, tilts=(0.0, np.pi / 4), per_tilt=4, up=(0, 0, 1)):
        """
        Takes the point to look at, the distance to keep from it, and the angles to tilt away from
        `up` [rad].  Returns the camera poses, in the frame and units of `centre`: one for a tilt of
        zero, `per_tilt` evenly around `up` for every other tilt.  Each faces the centre (third axis)
        with its first axis kept as near to the frame's x axis as that allows.
        """
        centre, up = np.asarray(centre, float), np.asarray(up, float) / np.linalg.norm(up)
        side = Pose.from_normal(up).R[:, :2]                    # two directions across `up`
        poses = []
        for tilt in tilts:
            turns = [0.0] if np.isclose(tilt, 0) else 2 * np.pi * np.arange(per_tilt) / per_tilt
            for turn in turns:
                away = np.cos(tilt) * up + np.sin(tilt) * (np.cos(turn) * side[:, 0] + np.sin(turn) * side[:, 1])
                poses.append(Pose.from_normal(-away, centre + distance * away, first_axis=[1, 0, 0]))
        return poses
