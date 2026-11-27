"""The kinds of image a model can be given for a touch."""

from abc import ABC, abstractmethod

import numpy as np

from imagine2touch.method.preprocessing import ContactMask


class ImageInput(ABC):
    """
    Turns touches of a VisuoTactileDataset into images (n, 48, 48) scaled to about [0, 1].

    `name` is the key in IMAGE_INPUTS and in the configuration.  `binary` says whether the images
    are masks, which decides how a model that rebuilds its input scores the rebuilt image.
    `rendered` says whether the image comes from an object's scan, and so exists for objects
    that were not touched (`session`), or from the wrist camera, which saw only the touched one.
    """

    name: str
    binary = False
    rendered = False

    @abstractmethod
    def build(self, dataset, rows, session=None):
        """Images of the touches `rows`; for rendered inputs, of the scan of `session` at those touches."""


class CameraMask(ImageInput):
    """The contact mask of the wrist camera's depth image: what the paper's code trains the model on (the paper: a processed depth image)."""

    name, binary = "mask", True

    def build(self, dataset, rows, session=None):
        return dataset.mask[np.asarray(rows)].astype(np.float32)


class CameraDepth(ImageInput):
    """The wrist camera's depth image with missing pixels filled, relative to its nearest pixel."""

    name = "depth"

    def build(self, dataset, rows, session=None):
        depth = dataset.depth_processed[np.asarray(rows)].astype(np.float32)
        nearest = depth.reshape(len(depth), -1).min(1)[:, None, None]
        return np.clip((depth - nearest) / dataset.grid.empty, 0, 1)


class ScanDepth(ImageInput):
    """The depth image that an object's scan shows at the pose where the sensor stopped."""

    name, rendered = "scan", True

    def build(self, dataset, rows, session=None):
        return dataset.scan_depths(rows, session) / dataset.grid.empty


class ScanMask(ImageInput):
    """
    The contact mask of that depth image, laid out like the camera's mask: what the paper's robot
    renders from the 3D model of an object it has in mind.
    """

    name, binary, rendered = "scan_mask", True, True

    def __init__(self, tolerance=2):
        self.contact_mask = ContactMask(tolerance)

    def build(self, dataset, rows, session=None):
        return self.from_depth(dataset.scan_depths(rows, session), dataset.grid)

    def from_depth(self, depth, grid):
        """Masks of scan depth images [mm].  The camera's image columns run against the sensor's second axis, hence the flip."""
        seen = np.where(np.asarray(depth) < grid.empty, depth, 0)
        return self.contact_mask(seen)[..., ::-1].astype(np.float32)


IMAGE_INPUTS = {cls.name: cls() for cls in (CameraMask, CameraDepth, ScanDepth, ScanMask)}
