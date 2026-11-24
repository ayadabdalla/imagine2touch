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
    GREY = (0.2989, 0.5870, 0.1140)         # the weights of red, green and blue in a grey picture

    @classmethod
    def of(cls, signal):
        """Images (n, 3, 3, 3) of signals (n, 15)."""
        signal = np.asarray(signal, float).reshape(-1, 5, 3)
        image = np.zeros((len(signal), 3, 3, 3))
        for magnetometer, (row, col) in enumerate(cls.PIXELS):
            image[:, row, col] = signal[:, magnetometer]
        return image

    @staticmethod
    def aligned(images, colour=False):
        """
        Takes camera images (..., rows, columns), or with `colour` (..., rows, columns, 3).  Returns
        them turned as the tactile image is laid out, so that the part of the surface over a
        magnetometer is drawn where its pixel is.  The turn is the one the data supports: the right of
        the camera image is the top of the sensor, and its bottom the sensor's left.
        """
        images = np.asarray(images)
        rows, columns = (-3, -2) if colour else (-2, -1)
        return np.swapaxes(np.flip(images, (rows, columns)), rows, columns)

    @classmethod
    def grey(cls, signal):
        """
        Images (n, 3, 3) as the paper draws a touch signal (its Fig. 2): at each magnetometer the
        sizes of the three components of the change of the field, weighted by GREY and summed.  The
        corners, where no magnetometer sits, are NaN.
        """
        image = np.abs(cls.of(signal)) @ np.array(cls.GREY)
        for corner in cls.CORNERS:
            image[(slice(None), *corner)] = np.nan
        return image
