"""
Brings the scan of an object that was never touched into the form the recognition experiments read.

    python -m imagine2touch.datasets.recordings.scan_converter --source PATH/TO/hanger_combined.pcd --out data/scans/hanger.npz

The source is a point cloud in the frame of the scans [m], as the paper's recognition script loaded
its candidates (`reskin/task/data/<object>/<object>_combined.pcd` of the original repository).  It is
moved into the frame of the robot's base with the calibration of the March 2023 sessions
(TablePoseRecovery), turned to millimetres and thinned like every scan of the dataset (ObjectScan).
The file holds `points` (n, 3).
"""

import argparse
from pathlib import Path

import numpy as np

from imagine2touch.geometry import ObjectScan
from .point_cloud_file import PointCloudFile
from .table_pose_recovery import TablePoseRecovery

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    points = np.asarray(PointCloudFile.read(args.source), float)
    in_base = points @ TablePoseRecovery.TABLE_IN_ROBOT_MARCH_2023[:3, :3].T + TablePoseRecovery.TABLE_IN_ROBOT_MARCH_2023[:3, 3]
    scan = ObjectScan.from_raw_points(1000 * in_base)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, points=scan.points.astype(np.float32))
    print(f"{args.out}: {len(scan):,} points, {' x '.join(f'{e:.0f}' for e in np.ptp(scan.points, 0))} mm")
