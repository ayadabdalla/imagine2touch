"""The layout of a ReSkin reading."""

import numpy as np

from imagine2touch.reskin_reading import ReSkinReading


def test_a_reading_gives_its_field_and_is_rebuilt_from_it():
    reading = [0.0, 0.0, [30, 1, 2, 3, 30, 4, 5, 6, 30, 7, 8, 9, 30, 10, 11, 12, 30, 13, 14, 15], 1]
    assert ReSkinReading.field(reading).tolist() == list(range(1, 16))       # the five temperatures are dropped
    assert ReSkinReading.field(ReSkinReading.from_field(np.arange(15.0))).tolist() == list(range(15))
    assert ReSkinReading.mean_field([ReSkinReading.from_field(np.zeros(15)), ReSkinReading.from_field(np.full(15, 2.0))]).tolist() == [1.0] * 15
    assert len(ReSkinReading.MAGNETOMETERS) == 5
