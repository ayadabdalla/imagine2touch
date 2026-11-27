"""
One recording of the robot: one experiment on one object, as a set of files.

    <object>/
      <object>_tactile/experiment_<e>_reskin            touch reading at contact, one per touch
      <object>_tactile/experiment_<e>_reskin_ambient    reading taken just before the contact
      <object>_poses/experiment_<e>_final_contact       the point on the object the robot was sent to [m]
      <object>_poses/experiment_<e>_rotations           the orientation it was sent in, quaternion (x, y, z, w)
      <object>_poses/experiment_<e>_prepare_contact     the point 4 cm in front, from which the camera looked [m]
      <object>_poses/experiment_<e>_joints_state        the seven joint angles when the robot stopped [rad]
      <object>_poses/experiment_<e>_reached_pose        the 4x4 pose it stopped at (new recordings only)
      <object>_forces/experiment_<e>_forces_torques     the wrench the robot estimated when it stopped
      <object>_forces/experiment_<e>_contact_returns    what stopped the approach (ApproachOutcome)
      <object>_images/depth/experiment_<e>_depth_<i>.tif   wrist camera depth of the patch to touch [mm]
      <object>_images/rgb/experiment_<e>_rgb_<i>.png       and its colour image
      <object>_images/depth_processed/depth_processed_<k>.tif   depth with missing pixels filled, 48 x 48
      <object>_images/masks/masks_<k>.tif                  the pixels nearest the camera, 48 x 48
      pcds/experiment_<e>_combined.pcd                  the scan of the object, in the world frame [m]
      pcds/experiment_<e>_robot_in_world.npy            maps the robot's base frame to the world frame
      pcds/experiment_<e>_wcamera_in_tcp.npy            the wrist camera in the frame of the sensor

Touches count from i = 1 within an experiment; processed images count from k = 1 through all
experiments of an object in turn.  All files without an extension are numpy arrays.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image

from imagine2touch.datasets.tactile_image import ReSkinReading
from imagine2touch.geometry.object_scan import ObjectScan
from .approach_outcome import ApproachOutcome
from .point_cloud_file import PointCloudFile


class Recording:
    """
    Experiment `number` of the object folder `folder`.  After `load()`, `signal`, `reading`,
    `ambient`, `target`, `rotation`, `prepare`, `joints`, `wrench` and `outcome` hold one row per
    touch in the units of the files, and `reached` the stopped poses if they were recorded.
    """

    CROP = 48        # side of the images the model sees [pixels]
    ARRAYS = dict(target="poses/final_contact", rotation="poses/rotations", prepare="poses/prepare_contact",
                  joints="poses/joints_state", reached="poses/reached_pose", wrench="forces/forces_torques",
                  outcome="forces/contact_returns", jump="forces/reskin_contact_magnitudes",
                  reading="tactile/reskin", ambient="tactile/reskin_ambient")
    REQUIRED = ("target", "rotation", "prepare", "joints", "reading", "ambient")
    OPTIONAL = ("wrench", "outcome")        # the sessions of March 2023 did not record them
    CALIBRATIONS = dict(world_from_base="robot_in_world", camera_in_sensor="wcamera_in_tcp")

    def __init__(self, folder, number, first_image=1):
        self.folder, self.number, self.first_image = Path(folder), number, first_image
        self.object = self.folder.name

    # ------------------------------------------------------------------ where things are
    @property
    def shape(self):
        """'star_v_1' -> 'star': the same printed shape was recorded in several sessions."""
        return re.sub(r"_v_\d+$", "", self.object)

    def array_path(self, key):
        group, name = self.ARRAYS[key].split("/")
        return self.folder / f"{self.object}_{group}" / f"experiment_{self.number}_{name}"

    def calibration_path(self, key):
        return self.folder / "pcds" / f"experiment_{self.number}_{self.CALIBRATIONS[key]}.npy"

    @property
    def scan_path(self):
        return self.folder / "pcds" / f"experiment_{self.number}_combined.pcd"

    def image_folder(self, kind):
        return self.folder / f"{self.object}_images" / kind

    @property
    def image_count(self):
        return len(list(self.image_folder("depth").glob(f"experiment_{self.number}_depth_*.tif")))

    # ------------------------------------------------------------------ reading
    @property
    def complete(self):
        """Whether the recording holds what a touch needs: the readings and where the robot was."""
        return all(self.array_path(key).exists() for key in self.REQUIRED)

    @property
    def has_placement(self):
        """Whether the recording stored where its scan stood under the robot; the sessions of March 2023 did not."""
        return self.calibration_path("world_from_base").exists()

    placement = None        # set by the reducer for recordings without a stored one (TablePoseRecovery)

    def raw_scan(self):
        """The points (n, 3) of the scan as it was stored, in the frame of the table [m]."""
        return PointCloudFile.read(self.scan_path)

    def felt(self):
        """The touches at which the sensor felt the surface, as indices; every touch if the session did not record how its approaches ended."""
        known = (self.outcome != ApproachOutcome.UNKNOWN).any()
        return np.flatnonzero(self.outcome == ApproachOutcome.SENSOR) if known else np.arange(self.n)

    @property
    def has_scan(self):
        """Whether the object's scan can be placed in the robot's frame."""
        return self.scan_path.exists() and (self.has_placement or self.placement is not None)

    @property
    def camera_is_aligned(self):
        """Whether the camera is known to look along the direction of touch; in March 2023 it was tilted."""
        camera = self.camera_in_sensor
        return camera is not None and abs(camera[2, 2] - 1) < 1e-6

    @staticmethod
    def _ambient(raw, n):
        """
        The reading out of contact for each of n touches, (n, 15), and whether one was taken per touch.
        Some sessions took one before every touch.  Those of March 2023 took ten readings before
        every tenth touch; their mean serves the ten touches that follow.
        """
        if raw.ndim == 2 and raw.shape[1] == 15:
            groups = np.asarray(raw, float)
        else:
            groups = np.array([ReSkinReading.mean_field(group) for group in raw])
        if len(groups) >= n:
            return groups[:n], True
        return groups[np.minimum(np.arange(n) // 10, len(groups) - 1)], False

    def load(self):
        raw = {key: np.load(self.array_path(key), allow_pickle=True) for key in self.REQUIRED}
        self.n = n = min(len(array) for key, array in raw.items() if key != "ambient")
        self.reading = np.array([ReSkinReading.field(reading) for reading in raw["reading"][:n]])
        self.ambient, self.ambient_per_touch = self._ambient(raw["ambient"], n)
        self.signal = self.reading - self.ambient
        for key in ("target", "rotation", "prepare", "joints"):
            setattr(self, key, np.asarray(raw[key][:n], float))
        recorded = all(self.array_path(key).exists() for key in self.OPTIONAL)
        if recorded:
            recorded = min(len(np.load(self.array_path(key), allow_pickle=True)) for key in self.OPTIONAL) >= n
        self.wrench = np.asarray(np.load(self.array_path("wrench"), allow_pickle=True)[:n], float) if recorded else np.full((n, 6), np.nan)
        self.outcome = np.asarray(np.load(self.array_path("outcome"), allow_pickle=True)[:n], int) if recorded else np.zeros(n, int)
        self.reached = np.load(self.array_path("reached"))[:n] if self.array_path("reached").exists() else None
        return self

    @property
    def world_from_base(self):
        return np.load(self.calibration_path("world_from_base")) if self.has_placement else self.placement

    @property
    def camera_in_sensor(self):
        """The calibration of the wrist camera; some sessions stored one for all their experiments, some none."""
        for path in (self.calibration_path("camera_in_sensor"), self.folder / "pcds" / "wcamera_in_tcp.npy"):
            if path.exists():
                return np.load(path)
        return None

    def scan(self):
        """The scan of the object, thinned, in the frame of the robot's base [mm]; empty if it cannot be placed."""
        if not self.has_scan:
            return ObjectScan(np.zeros((0, 3)))
        base_from_world = np.linalg.inv(self.world_from_base)
        world = self.raw_scan()
        return ObjectScan.from_raw_points(1000 * (world @ base_from_world[:3, :3].T + base_from_world[:3, 3]))

    def images(self, i):
        """Depth [mm], processed depth [mm], mask and colour image of touch i (from 0), each 48 x 48."""
        k, c = self.first_image + i, self.CROP
        return (self._image(self.image_folder("depth") / f"experiment_{self.number}_depth_{i + 1}.tif", (c, c), np.uint16),
                self._image(self.image_folder("depth_processed") / f"depth_processed_{k}.tif", (c, c), np.uint16),
                self._image(self.image_folder("masks") / f"masks_{k}.tif", (c, c), bool),
                self._image(self.image_folder("rgb") / f"experiment_{self.number}_rgb_{i + 1}.png", (c, c, 3), np.uint8))

    @classmethod
    def _image(cls, path, shape, dtype):
        if not path.exists():
            return np.zeros(shape, dtype)
        return cls.centre_crop(np.asarray(Image.open(path))).astype(dtype).reshape(shape)

    @classmethod
    def centre_crop(cls, image, size=None):
        """The middle `size` x `size` pixels of an image, padded with zeros where the image is smaller."""
        size, image = size or cls.CROP, np.asarray(image)
        out = np.zeros((size, size) + image.shape[2:], image.dtype)
        h, w = min(size, image.shape[0]), min(size, image.shape[1])
        a, b = (image.shape[0] - h) // 2, (image.shape[1] - w) // 2
        c, d = (size - h) // 2, (size - w) // 2
        out[c:c + h, d:d + w] = image[a:a + h, b:b + w]
        return out
