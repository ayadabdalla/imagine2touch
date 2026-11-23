"""
Scan an object into a point cloud (needs the robot, or --simulate).

    python -m imagine2touch.procedures.scripts.scan_object --object star --world calibration/world.npy --camera calibration/camera.npy --out scans
    python -m imagine2touch.procedures.scripts.scan_object --simulate --object star --out scans

Stand the object at the origin of the world frame (`python -m imagine2touch.procedures.scripts.calibrate_camera`).  The robot looks at
it from above and from four sides and writes `<out>/<object>.pcd`, the scan in the world frame [m],
and `<out>/<object>.npy`, its surface normals: the --scan of collect_data.py and the --candidates of
recognize_object.py.  --box is the part of the world frame that holds the object: lowest corner,
highest corner [m].  Its floor is one centimetre above the table, as in the paper, so that the table
stays out of the scan.

With --simulate, --object is one of the paper's objects and the simulated camera scans its scan.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from imagine2touch.datasets import VisuoTactileDataset
from imagine2touch.datasets.recordings import PointCloudFile
from imagine2touch.procedures import ObjectScanner
from imagine2touch.simulation import SimulatedCamera, SimulatedRobot, SimulatedScene

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True, help="the name the scan is saved under")
    ap.add_argument("--out", default="scans")
    ap.add_argument("--world"), ap.add_argument("--camera")
    ap.add_argument("--box", type=float, nargs=6, default=[-0.05, -0.05, 0.01, 0.05, 0.05, 0.1], metavar="M")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    if args.simulate:
        scene = SimulatedScene.of_session(VisuoTactileDataset(), args.object)
        robot = SimulatedRobot(scene)
        camera = SimulatedCamera(robot, width=320, height=240, focal=600.0)
        world_from_base, camera_in_sensor, sleep = scene.world_from_base, camera.camera_in_sensor, lambda seconds: None
    else:
        from imagine2touch.devices.franka_robot import FrankaRobot
        from imagine2touch.devices.realsense_camera import RealSenseCamera
        if not (args.world and args.camera):
            ap.error("--world and --camera are needed with the robot")
        robot, camera, sleep = FrankaRobot(), RealSenseCamera(), time.sleep
        world_from_base, camera_in_sensor = np.linalg.inv(np.load(args.world)), np.load(args.camera)
    scanner = ObjectScanner(robot, camera, world_from_base, camera_in_sensor, box=(args.box[:3], args.box[3:]), sleep=sleep)
    points, normals = scanner.scan()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    PointCloudFile.write(out / f"{args.object}.pcd", points)
    np.save(out / f"{args.object}.npy", normals)
    size = 1000 * (points.max(0) - points.min(0))
    print(f"{len(points)} points of {args.object}, {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm, in {out / (args.object + '.pcd')}")
