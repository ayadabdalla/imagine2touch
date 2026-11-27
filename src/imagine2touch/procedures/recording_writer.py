"""Stores collected touches as a recording."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from imagine2touch.datasets import ReSkinReading
from imagine2touch.datasets.recordings import PointCloudFile, Recording
from imagine2touch.method.preprocessing import ContactMask, DepthFill


class RecordingWriter:
    """
    Writes the records of a TouchCollector as experiment `number` of the folder `<root>/<name>`, in
    the layout of datasets/recordings/recording.py, so that new recordings are reduced like those of the paper.
    Unlike those, it also stores the pose at which the sensor stopped.
    """

    def __init__(self, root, name, number=1):
        self.recording, self.records = Recording(Path(root) / name, number), []
        self.fill, self.mask = DepthFill(), ContactMask()

    def add(self, record):
        self.records.append(record)

    def _save(self, path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:                    # the recordings' arrays have no file extension
            np.save(f, array, allow_pickle=True)

    def save(self, cloud, world_from_base, camera_in_sensor):
        recording, column = self.recording, lambda key: np.array([record[key] for record in self.records])
        readings = np.empty((len(self.records), 1, 4), object)
        for i, record in enumerate(self.records):
            readings[i, 0] = ReSkinReading.from_field(record["reading"], time.time())
        arrays = dict(reading=readings, ambient=column("ambient"), target=column("target"), rotation=column("rotation"),
                      prepare=column("prepare"), joints=column("joints"), reached=column("pose"), wrench=column("wrench"),
                      outcome=column("outcome"), jump=column("jump"))
        for key, array in arrays.items():
            self._save(recording.array_path(key), array)
        self._save(recording.calibration_path("world_from_base"), world_from_base)
        self._save(recording.calibration_path("camera_in_sensor"), camera_in_sensor)
        PointCloudFile.write(recording.scan_path, cloud)
        for kind in ("depth", "rgb", "depth_processed", "masks"):
            recording.image_folder(kind).mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(self.records, start=1):
            stem = f"experiment_{recording.number}"
            Image.fromarray(record["depth"]).save(recording.image_folder("depth") / f"{stem}_depth_{i}.tif")
            Image.fromarray(record["rgb"]).save(recording.image_folder("rgb") / f"{stem}_rgb_{i}.png")
            filled = self.fill(Recording.centre_crop(record["depth"]))
            Image.fromarray(filled).save(recording.image_folder("depth_processed") / f"depth_processed_{i}.tif")
            Image.fromarray(self.mask(filled).astype(np.uint16)).save(recording.image_folder("masks") / f"masks_{i}.tif")
