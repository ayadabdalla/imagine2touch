"""
Converts the recordings of the stamp presses to the file behind ShapeClassificationDataset, as the
paper's experiment selected and split them.

    python -m imagine2touch.datasets.recordings.stamp_recording_converter --source PATH/TO/Classification_experiment

Every sub-folder `<stamp>_experiment_<n>` holds the arrays `reskin_readings`, `ambient_readings` (ten
readings out of contact per group of about eight presses), `pose` and `force`.  The folders carry
the names the stamps had in the lab: Cylinder, Mothalath (triangle), Salib (cross), T and V (angle).

The paper's experiment (its notebook `investigate_data.ipynb` and `indenter_experiment/data.py`):
  - the reading out of contact of a press is the first of the ten readings of its group;
  - of the 2,628 recorded presses, 256 of every stamp are drawn at random, as many as the cross has:
    1,280 in all, of which 181 of every stamp train the classifier, 25 validate and 50 test it.
The draw repeats the notebook's: the folders in the notebook's order, the random generator seeded
with 0, every stamp's presses shuffled once for a first split and once more for this one.
"""

import argparse
import collections
import sys
import types
from pathlib import Path

import numpy as np

from imagine2touch.paths import DATA_DIR
from imagine2touch.reskin_reading import ReSkinReading
from imagine2touch.datasets.shape_classification_dataset import ShapeClassificationDataset

FOLDERS = {"T": "T", "Cylinder": "circle", "V": "angle", "Mothalath": "triangle", "Salib": "cross"}


class StampRecordingConverter:
    # the order in which the paper's notebook listed the folders, and so the order of its presses
    ORDER = ("T_experiment_1", "Mothalath_experiment_2", "Salib_experiment_2", "V_experiment_1", "T_experiment_2",
             "Mothalath_experiment_3", "Mothalath_experiment_1", "Cylinder_experiment_1", "V_experiment_2", "Salib_experiment_1")
    DRAW_ORDER = ("T", "cross", "angle", "triangle", "circle")      # the order of the classes in the paper's code
    TEST, VALIDATION, SEED = 50, 25, 0
    ROLES = ("train", "validation", "test")

    def __init__(self, source):
        folders = {path.name: path for path in Path(source).iterdir() if path.is_dir()}
        missing = [name for name in self.ORDER if name not in folders]
        if missing:
            raise FileNotFoundError(f"{source}: missing the folders {missing}")
        self.folders = [folders[name] for name in self.ORDER]
        # some arrays hold records of the sensor's driver; a stand-in with the same fields reads them
        record = collections.namedtuple("ReSkinData", "time, acq_delay, data, dev_id")
        for name in ("reskin_sensor", "reskin_sensor.sensor"):
            sys.modules.setdefault(name, types.ModuleType(name)).ReSkinData = record

    @staticmethod
    def session(folder):
        """Signal, position and force of every press in one folder."""
        load = lambda name: np.load(folder / name, allow_pickle=True)
        reading = np.array([ReSkinReading.field(r) for r in load("reskin_readings")])
        ambient = np.array([ReSkinReading.field(group[0]) for group in load("ambient_readings")])     # the first of the ten
        repeats = int(np.ceil(len(reading) / len(ambient)))
        return reading - np.repeat(ambient, repeats, axis=0)[:len(reading)], load("pose"), load("force")

    def split(self, stamp):
        """
        The paper's draw: for every press (n,) its role, an index into ROLES, or -1 if it was not
        drawn.  Every stamp gets as many presses as the rarest one has.
        """
        stamps = ShapeClassificationDataset.STAMPS
        order = [stamps.index(name) for name in self.DRAW_ORDER]
        per_stamp = min(int((stamp == k).sum()) for k in order)
        generator = np.random.RandomState(self.SEED)
        for k in order:                                     # the notebook's first split, which only moves the generator on
            generator.shuffle(np.flatnonzero(stamp == k))
        role = np.full(len(stamp), -1)
        for k in order:
            presses = np.flatnonzero(stamp == k)
            generator.shuffle(presses)
            role[presses[:self.TEST]] = self.ROLES.index("test")
            role[presses[self.TEST:self.TEST + self.VALIDATION]] = self.ROLES.index("validation")
            role[presses[self.TEST + self.VALIDATION:per_stamp]] = self.ROLES.index("train")
        return role

    def convert(self, path=None, report=print, everything=False):
        """Write the dataset file; `everything` keeps the presses the paper's draw left out, with role -1."""
        stamps = ShapeClassificationDataset.STAMPS
        rows = {key: [] for key in ("signal", "stamp", "session", "position", "force")}
        for k, folder in enumerate(self.folders):
            signal, position, force = self.session(folder)
            rows["signal"].append(signal), rows["position"].append(position), rows["force"].append(force)
            rows["stamp"].append(np.full(len(signal), stamps.index(FOLDERS[folder.name.split("_")[0]])))
            rows["session"].append(np.full(len(signal), k))
        arrays = {key: np.concatenate(values) for key, values in rows.items()}
        arrays["role"] = self.split(arrays["stamp"])
        if not everything:
            keep = arrays["role"] >= 0
            arrays = {key: values[keep] for key, values in arrays.items()}
        for key in ("signal", "force"):
            arrays[key] = arrays[key].astype(np.float32)
        path = Path(path or DATA_DIR / ShapeClassificationDataset.FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sessions=[folder.name for folder in self.folders], roles=list(self.ROLES), **arrays)
        dataset = ShapeClassificationDataset(path)
        report(f"{path}: {len(dataset):,} presses, " + ", ".join(
            f"{int((dataset.role == name).sum())} to {name}" for name in self.ROLES))
        return dataset


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true", help="also keep the presses the paper's draw left out")
    args = ap.parse_args()
    StampRecordingConverter(args.source).convert(args.out, everything=args.all)
