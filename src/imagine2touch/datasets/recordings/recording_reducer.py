"""
Reduces the recordings of the robot to the file behind VisuoTactileDataset.

    python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data
    python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data --all
    python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data \\
        --sessions cylinder_v_6 flower_v_6 hexagon_v_6 parallelogram_v_6 square_v_6 star_v_6 --out data/imagine2touch_v6.npz

The first keeps the six sessions of the paper; --all keeps every session that is complete enough,
including those recorded after the paper's; --sessions keeps the ones it names.

The recordings store where the sensor was sent, not where it stopped.  The pose it stopped at is
computed from the joint angles, which were recorded in contact, with Franka's kinematics and the
fitted position of the sensor on the flange (ToolCalibration), fitted once for the sessions of
March 2023 and once for those of September 2023.  Where a session did not store where its scan
stood under the robot, the calibration of its recording code is used (ScanPlacement).  Recordings that do store the reached pose, those
of procedures/RecordingWriter, are taken as they are.
"""

import argparse
from pathlib import Path

import numpy as np

from imagine2touch.geometry import FrankaKinematics, Pose
from imagine2touch.paths import DATA_DIR
from .recording_reader import RecordingReader
from imagine2touch.datasets.calibration import ScanPlacement, ToolCalibration
from imagine2touch.datasets.visuo_tactile_dataset import VisuoTactileDataset

COLUMNS = ("object", "shape", "experiment", "outcome", "signal", "reading", "ambient", "wrench", "joints", "depth",
           "depth_processed", "mask", "rgb", "camera_to_sensor", "pose_R", "pose_t", "commanded_R", "commanded_t")
SINGLE_PRECISION = ("signal", "reading", "ambient", "wrench")


class RecordingReducer:
    def __init__(self, source, sessions=None):
        self.recordings = RecordingReader(source, sessions).recordings()
        if not self.recordings:
            raise FileNotFoundError(f"{source}: no complete recordings")
        unplaced = [r for r in self.recordings if not r.has_placement and r.scan_path.exists()]
        for recording in unplaced:                              # the sessions of March 2023: the calibration of their recording code
            recording.placement = ScanPlacement.stored()
        fitted = [r for r in self.recordings if r.reached is None]
        self.tool = ToolCalibration.fit([r for r in fitted if r.ambient_per_touch])            # September 2023
        self.early_tool = ToolCalibration.fit([r for r in fitted if not r.ambient_per_touch])   # March 2023
        self.objects = sorted({r.object for r in self.recordings})
        self.shapes = sorted({r.shape for r in self.recordings})

    def stopped_pose(self, recording, i):
        """The pose at which the sensor stopped at touch i [mm]."""
        if recording.reached is not None:
            return Pose(recording.reached[i][:3, :3], 1000 * recording.reached[i][:3, 3])
        return FrankaKinematics.flange_pose(recording.joints[i]) @ (self.tool if recording.ambient_per_touch else self.early_tool)

    def touch(self, recording, i):
        """The row of touch i of a recording."""
        pose = self.stopped_pose(recording, i)
        commanded = Pose.from_quaternion(recording.rotation[i], 1000 * recording.target[i])
        travel = (pose.t - 1000 * recording.prepare[i]) @ commanded.normal        # how far the sensor went after the camera looked
        depth, processed, mask, rgb = recording.images(i)
        return dict(object=self.objects.index(recording.object), shape=self.shapes.index(recording.shape),
                    experiment=recording.number, outcome=recording.outcome[i], signal=recording.signal[i],
                    reading=recording.reading[i], ambient=recording.ambient[i], wrench=recording.wrench[i],
                    joints=recording.joints[i], depth=depth, depth_processed=processed, mask=mask, rgb=rgb,
                    camera_to_sensor=(travel - 1000 * recording.camera_in_sensor[2, 3]) if recording.camera_is_aligned else np.nan,
                    pose_R=pose.R, pose_t=pose.t,
                    commanded_R=commanded.R, commanded_t=commanded.t)

    def reduce(self, path=None, report=print):
        """Write the dataset file and return it as a VisuoTactileDataset."""
        rows, scans = {key: [] for key in COLUMNS}, {}
        for recording in self.recordings:
            scans.setdefault(recording.object, recording.scan().points.astype(np.float32))
            for i in range(recording.n):
                for key, value in self.touch(recording, i).items():
                    rows[key].append(value)
            report(f"{recording.object} experiment {recording.number}: {recording.n} touches")
        clouds = [scans[name] for name in self.objects]
        path = Path(path or DATA_DIR / VisuoTactileDataset.FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, objects=self.objects, shapes=self.shapes, tool_R=self.tool.R, tool_t=self.tool.t,
                            early_tool_R=self.early_tool.R, early_tool_t=self.early_tool.t,
                            scan=np.concatenate(clouds), scan_start=np.cumsum([0] + [len(c) for c in clouds]),
                            **{key: np.asarray(values, np.float32 if key in SINGLE_PRECISION else None)
                               for key, values in rows.items()})
        dataset = VisuoTactileDataset(path)
        report(f"{path}: {len(dataset):,} touches of {len(self.shapes)} shapes in {len(self.objects)} sessions, "
               f"{path.stat().st_size / 1e6:.0f} MB")
        return dataset


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, nargs="+", help="the folder of the recordings, one sub-folder per object; "
                    "of several folders, a later one replaces sessions of the same name in an earlier one")
    ap.add_argument("--out", default=None, help=f"the file to write (default: {DATA_DIR / VisuoTactileDataset.FILE})")
    ap.add_argument("--all", action="store_true", help="keep every session, not only those of the paper")
    ap.add_argument("--sessions", nargs="+", metavar="NAME", help="keep these sessions, by the names of their folders")
    args = ap.parse_args()
    sessions = args.sessions or (None if args.all else VisuoTactileDataset.PAPER_SESSIONS)
    RecordingReducer(args.source, sessions).reduce(args.out)
