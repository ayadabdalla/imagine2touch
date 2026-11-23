"""Masks, the filling of depth images and the scaling of signals."""

import numpy as np
import pytest

from imagine2touch.method.preprocessing import ContactMask, DepthFill, SIGNAL_SCALERS


def test_contact_mask_keeps_the_pixels_nearest_the_camera():
    depth = np.full((6, 6), 110)
    depth[:3] = 104                  # a step: the upper half is 6 mm nearer
    depth[0, 0] = 0                  # one pixel without a reading
    mask = ContactMask(tolerance=2)(depth)
    assert mask[:3].sum() == 17 and not mask[3:].any() and not mask[0, 0]


def test_fill_missing_takes_the_nearest_reading():
    depth = np.array([[5, 0, 0, 9]], np.uint16)
    assert DepthFill()(depth).tolist() == [[5, 5, 9, 9]]
    assert DepthFill()(np.zeros((2, 2), np.uint16)).sum() == 0


@pytest.mark.parametrize("mode", sorted(SIGNAL_SCALERS))
def test_signal_scaling_is_undone_exactly(mode):
    signal = np.random.default_rng(0).normal(3, 20, (50, 15))
    scaler = SIGNAL_SCALERS[mode]().fit(signal)
    scaled = scaler.transform(signal)
    assert scaler.inverse(scaled) == pytest.approx(signal, abs=1e-3)
    if mode == "minmax":
        assert scaled.min() == pytest.approx(0, abs=1e-6) and scaled.max() == pytest.approx(1, abs=1e-6)
    elif mode == "quantile":                # every value's rank among the training signals
        assert scaled.min(0) == pytest.approx(0, abs=1e-6) and scaled.max(0) == pytest.approx(1, abs=1e-6)
        assert np.sort(scaled[:, 0]) == pytest.approx(np.linspace(0, 1, 50), abs=1e-6)
    else:
        assert scaled.mean(0) == pytest.approx(0, abs=1e-5) and scaled.std(0) == pytest.approx(1, abs=1e-5)
