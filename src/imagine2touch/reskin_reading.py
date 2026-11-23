"""One reading of the ReSkin sensor as its driver delivers it."""

import numpy as np


class ReSkinReading:
    """
    A reading is (time, delay, values, sensor id) with 20 values: for each of the five magnetometers
    its temperature and three field components.  The touch signal is made of the 15 field values,
    in the order the sensor sends its magnetometers: centre, top, right, bottom, left (the sensor's own
    visualiser and Imagine2touch's `visualize_reskin_image` both lay them out so; see TactileImage).
    """

    TEMPERATURES = (0, 4, 8, 12, 16)
    MAGNETOMETERS = ("centre", "top", "right", "bottom", "left")

    @classmethod
    def field(cls, reading):
        """The 15 field values of a reading."""
        values = np.asarray(np.asarray(reading, object).reshape(-1)[2], float)
        return np.delete(values, cls.TEMPERATURES)

    @classmethod
    def mean_field(cls, readings):
        """The mean field of several readings."""
        return np.mean([cls.field(reading) for reading in readings], axis=0)

    @classmethod
    def from_field(cls, field, time=0.0, sensor=1):
        """A reading in the driver's layout that holds `field`, with temperatures of zero."""
        values = np.insert(np.asarray(field, float), [0, 3, 6, 9, 12], 0.0)
        return [time, 0.0, values.tolist(), sensor]
