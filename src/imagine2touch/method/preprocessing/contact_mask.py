"""The part of a depth image that the sensor will meet."""

import numpy as np


class ContactMask:
    """
    True where the surface is within `tolerance` [mm] of the pixel nearest the camera.  The sensor
    approaches along the camera's axis, so these are the pixels it meets first (thesis, Sec. 5.2.2).
    Pixels without a reading (0) are never in the mask.
    """

    def __init__(self, tolerance=2):
        self.tolerance = tolerance

    def __call__(self, depth):
        """Masks (..., h, w) of depth images (..., h, w)."""
        depth = np.asarray(depth, float)
        valid = depth > 0
        nearest = np.where(valid, depth, np.inf).min(axis=(-2, -1), keepdims=True)
        return valid & (depth - nearest <= self.tolerance)
