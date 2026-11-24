"""One touch with all its images brought to a common layout and scale."""

import numpy as np

from imagine2touch.datasets import TactileImage


class TouchView:
    """
    The wrist camera measures depth from the camera, 10 cm away; the scan is rendered from the
    sensor.  To compare them, both are shown here as the distance of the surface in front of the
    sensing surface at the pose where the sensor stopped [mm], in the layout of the camera's image.
    In sessions that cannot tell where the sensing surface was in the camera's image, the camera's
    depth is counted from its nearest pixel instead, and there is no image from the scan
    (`has_scan`).

      rgb           (48, 48, 3) colour image of the patch
      camera_gap    (48, 48) from the wrist camera; NaN where it has no reading
      scan_gap      (48, 48) from the object's scan; `reach` where the scan shows nothing
      mask          (48, 48) the contact mask of the camera's image
      field         (3, 3, 3) the touch signal as three crosses: the field's two components across
                    the direction of touch and the one along it; NaN in the corners
      grey          (3, 3) the touch signal as the paper draws it (TactileImage.grey)
    """

    COMPONENTS = ("across", "across", "along the touch")

    def __init__(self, dataset, i):
        self.dataset, self.index = dataset, int(i)
        grid = dataset.grid
        self.reach = grid.reach
        self.rgb, self.mask = TactileImage.aligned(dataset.rgb[i], colour=True), TactileImage.aligned(dataset.mask[i])
        depth = dataset.depth[i].astype(float)
        zero = dataset.camera_to_sensor[i]
        if not np.isfinite(zero):
            zero = depth[depth > 0].min() if (depth > 0).any() else 0.0
        self.camera_gap = TactileImage.aligned(np.where(depth > 0, depth - zero, np.nan))
        self.has_scan = dataset.has_scan(int(dataset.object[i]))
        self.scan_gap = None
        if self.has_scan:
            self.scan_gap = TactileImage.aligned((dataset.scan_depth(i) - grid.behind)[:, ::-1])    # the camera's columns run the other way
        image = TactileImage.of(dataset.signal[i])[0]
        for corner in TactileImage.CORNERS:
            image[corner] = np.nan
        self.field = np.moveaxis(image, -1, 0)
        self.grey = TactileImage.grey(dataset.signal[i])[0]
        self.signal = dataset.signal[i].astype(float)
        self.pose, self.outcome = dataset.pose[i], str(dataset.outcome[i])
        self.session = int(dataset.object[i])
        self.shape = dataset.shapes[dataset.shape[i]]
        self.force = float(np.linalg.norm(dataset.wrench[i][:3]))

    @property
    def footprint(self):
        """The corners of the sensing surface in the robot's base frame, (5, 3), closed."""
        half = self.dataset.grid.side / 2
        corners = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0], [-half, -half, 0]])
        return self.pose.apply(corners)
