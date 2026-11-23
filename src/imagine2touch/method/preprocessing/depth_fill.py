"""Filling the pixels of a depth image that have no reading."""

import numpy as np
from scipy.ndimage import distance_transform_edt


class DepthFill:
    """
    Sets every pixel without a reading (0) to the nearest pixel that has one.  The recordings of the
    paper were filled by a longer rule that also looks at the colour image
    (src/imagine2touch/models/depth_correction_utils.py at commit c2ea2c3); its output is stored in the dataset.
    """

    def __call__(self, depth):
        depth = np.asarray(depth)
        if (depth > 0).all() or not (depth > 0).any():
            return depth.copy()
        nearest = distance_transform_edt(depth == 0, return_distances=False, return_indices=True)
        return depth[tuple(nearest)]
