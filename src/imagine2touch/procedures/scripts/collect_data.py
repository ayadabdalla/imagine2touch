"""
Collect touches of one object (needs the robot, or --simulate).

    python -m imagine2touch.procedures.scripts.collect_data --simulate --object star --touches 50 --out data/simulated
    python -m imagine2touch.procedures.scripts.collect_data --object star_v_7 --scan scans/star.pcd --touches 300 --out data/recordings \\
        --world world.npy --camera camera.npy

With --simulate, --object is one of the paper's objects: the simulated robot touches its scan and
feels the signals recorded on it.  A touch is kept only where the real sensor was, coming from about
the same direction; elsewhere the simulated sensor has no answer and the touch is left out, so no
image is paired with a signal from another place.

With the robot, --scan is the object's scan in the world frame [m] (a .pcd, with its normals in a
.npy of the same name), and --world and --camera are the two calibrations as 4 x 4 .npy files: the
world frame as the robot sees it, and the wrist camera in the frame of the sensor.  The recordings
are then reduced like the paper's:

    python -m imagine2touch.datasets.recordings.recording_reducer --source data/recordings --all
"""

import argparse
import time

import numpy as np

from imagine2touch.datasets import VisuoTactileDataset
from imagine2touch.datasets.recordings import ApproachOutcome, PointCloudFile
from imagine2touch.procedures import RecordingWriter, TouchCollector
from imagine2touch.simulation import SimulatedCamera, SimulatedRobot, SimulatedScene, SimulatedSensor

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True, help="name of the session, e.g. star_v_7")
    ap.add_argument("--touches", type=int, default=50)
    ap.add_argument("--experiment", type=int, default=1)
    ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--simulate", action="store_true", help="use the simulated robot on the object's scan from the dataset")
    ap.add_argument("--scan"), ap.add_argument("--world"), ap.add_argument("--camera")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.simulate:
        dataset = VisuoTactileDataset()
        scene = SimulatedScene.of_session(dataset, args.object)
        robot = SimulatedRobot(scene, rng=rng)
        sensor = SimulatedSensor.of_session(robot, dataset, args.object, rng=rng)
        camera = SimulatedCamera(robot)
        cloud, normals = scene.scan()
        world_from_base, camera_in_sensor, sleep = scene.world_from_base, camera.camera_in_sensor, lambda seconds: None
    else:
        from imagine2touch.devices.franka_robot import FrankaRobot
        from imagine2touch.devices.realsense_camera import RealSenseCamera
        from imagine2touch.devices.reskin_sensor import ReSkinSensor
        robot, sensor, camera, sleep = FrankaRobot(), ReSkinSensor(), RealSenseCamera(), time.sleep
        cloud, normals = PointCloudFile.read(args.scan), np.load(args.scan.replace(".pcd", ".npy"))
        world_from_base, camera_in_sensor = np.linalg.inv(np.load(args.world)), np.load(args.camera)
    collector = TouchCollector(robot, sensor, camera, cloud, normals, world_from_base, camera_in_sensor, rng=rng, sleep=sleep)
    records = collector.collect(args.touches, RecordingWriter(args.out, args.object, args.experiment))
    count = lambda outcome: sum(record["outcome"] == outcome for record in records)
    print(f"{len(records)} touches of {args.object} in {args.out}: {count(ApproachOutcome.SENSOR)} stopped by the sensor, "
          f"{count(ApproachOutcome.WRENCH)} by the force limit, {count(ApproachOutcome.AIR)} met nothing")
