"""A touch signal as a picture."""

import numpy as np


class TactileImage:
    """
    The five magnetometers of the sensor sit in a cross.  A touch signal becomes a 3 x 3 image in
    which each pixel of the cross holds the three field components of one magnetometer, and the four
    corners are empty (zero).

    The cross is laid out as in Imagine2touch's own code (`visualize_reskin_image`) and in the sensor's
    visualiser: the sensor sends centre, top, right, bottom, left, and they are drawn there.  This is
    the sensor's own view of itself; the wrist camera's images of the same touch are turned against it.
    """

    PIXELS = ((1, 1), (0, 1), (1, 2), (2, 1), (1, 0))      # centre, top, right, bottom, left
    CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))

    @classmethod
    def of(cls, signal):
        """Images (n, 3, 3, 3) of signals (n, 15)."""
        signal = np.asarray(signal, float).reshape(-1, 5, 3)
        image = np.zeros((len(signal), 3, 3, 3))
        for magnetometer, (row, col) in enumerate(cls.PIXELS):
            image[:, row, col] = signal[:, magnetometer]
        return image

    @classmethod
    def along_touch(cls, signal):
        """Images (n, 3, 3) of the field along the direction of touch, the component the paper's Fig. 2 shows."""
        return cls.of(signal)[..., 2]
