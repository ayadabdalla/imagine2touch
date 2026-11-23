"""The tactile image and the kinds of image a model can be given."""

import numpy as np

from imagine2touch.datasets import IMAGE_INPUTS, ScanMask, TactileImage
from imagine2touch.geometry import SensorGrid


def test_tactile_image_is_a_cross_with_empty_corners():
    signal = np.arange(1, 16, dtype=float)[None]
    image = TactileImage.of(signal)[0]
    assert image.shape == (3, 3, 3)
    assert all((image[corner] == 0).all() for corner in TactileImage.CORNERS)
    assert image[1, 1].tolist() == [1, 2, 3] and image[0, 1].tolist() == [4, 5, 6]      # centre, top
    assert image[1, 2].tolist() == [7, 8, 9] and image[2, 1].tolist() == [10, 11, 12]   # right, bottom
    assert image[1, 0].tolist() == [13, 14, 15]                                         # left


def test_scan_mask_marks_the_nearest_surface_and_is_laid_out_like_the_camera():
    grid = SensorGrid(size=8)
    depth = np.full((1, 8, 8), grid.empty)          # nothing in view,
    depth[0, :, :2] = 3.0                           # except a surface along the first two columns
    depth[0, :, 2] = 7.0                            # and a lower one beside it
    mask = ScanMask(tolerance=2).from_depth(depth, grid)
    assert mask[0, :, -2:].all() and mask.sum() == 16       # the near surface only, on the other side of the image


def test_image_inputs_know_what_they_are():
    assert set(IMAGE_INPUTS) == {"mask", "depth", "scan", "scan_mask"}
    assert IMAGE_INPUTS["mask"].binary and IMAGE_INPUTS["scan_mask"].binary and not IMAGE_INPUTS["depth"].binary
    assert IMAGE_INPUTS["scan"].rendered and not IMAGE_INPUTS["mask"].rendered
