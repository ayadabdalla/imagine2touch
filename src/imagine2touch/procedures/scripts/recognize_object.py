"""
Recognise an object by touching it (paper: Table II; needs the robot, or --simulate).

    python -m imagine2touch.procedures.scripts.recognize_object --simulate --object star --touches 10
    python -m imagine2touch.procedures.scripts.recognize_object --model results/models/imagine2touch.pt --candidates scans/star.pcd scans/square.pcd \\
        --world world.npy --camera camera.npy

With --simulate, the scan of --object, one of the paper's recognition objects, stands in front of the
simulated robot.  As in the paper, the robot aims every touch at the object it currently believes to be
there, and names the real one among the paper's three recognition objects.  Where it meets the object, the
touch is scored from the real touch recorded nearest to it; where it meets nothing it feels nothing.
The model is a saved one, made by `python -m imagine2touch.experiments.scripts.train_model`.  --input is
the kind of image it was trained on.
With the robot, --candidates are the scans of the objects to choose among, in the world frame [m].
"""

import argparse
import time
from pathlib import Path

import numpy as np

from imagine2touch.datasets import VisuoTactileDataset
from imagine2touch.datasets.recordings import PointCloudFile
from imagine2touch.geometry import ObjectScan
from imagine2touch.method.models import TouchModel
from imagine2touch.paths import ROOT
from imagine2touch.procedures import TouchCollector, TouchingRecogniser
from imagine2touch.simulation import SimulatedCamera, SimulatedRobot, SimulatedScene, SimulatedSensor

MODEL = ROOT / "results" / "models" / "imagine2touch.pt"
OBJECTS = ("parallelogram", "square", "star")       # the recognition objects of the paper's Table II


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--touches", type=int, default=10)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--object", default="star", choices=OBJECTS, help="with --simulate: the object to touch")
    ap.add_argument("--model", default=MODEL), ap.add_argument("--candidates", nargs="+"), ap.add_argument("--world"), ap.add_argument("--camera")
    ap.add_argument("--evidence", default="touch", choices=["touch", "proprioception"],
                    help="what a touch says about a candidate: the touch model, or the paper's baseline, where the sensor stopped")
    ap.add_argument("--input", default="scan", choices=["scan", "scan_mask"],
                    help="how a candidate is shown to the model: the depth image of its scan, or the contact mask made from it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not Path(args.model).exists():
        ap.error(f"no model at {args.model}; train one with: python -m imagine2touch.experiments.scripts.train_model")
    rng, replay, model = np.random.default_rng(args.seed), None, TouchModel.load(args.model)
    if args.simulate:
        dataset = VisuoTactileDataset()
        scene = SimulatedScene.of_session(dataset, args.object)
        robot = SimulatedRobot(scene, rng=rng)
        sensor, camera = SimulatedSensor.of_session(robot, dataset, args.object), SimulatedCamera(robot)
        # every candidate's scan where it stood under the real robot, in this scene's world frame [m]
        scans = {name: scene.to_world(dataset.scan(dataset.objects.index(name)).points / 1000) for name in OBJECTS}
        normals = {name: ObjectScan.normals_of(points) for name, points in scans.items()}
        collector = TouchCollector(robot, sensor, camera, scans[args.object], normals[args.object], scene.world_from_base,
                                   camera.camera_in_sensor, candidates=1, rng=rng, sleep=lambda seconds: None)
        replay = sensor.recorded_touch
    else:
        from imagine2touch.devices.franka_robot import FrankaRobot
        from imagine2touch.devices.realsense_camera import RealSenseCamera
        from imagine2touch.devices.reskin_sensor import ReSkinSensor
        scans = {path: PointCloudFile.read(path) for path in args.candidates}
        normals = {path: np.load(path.replace(".pcd", ".npy")) for path in args.candidates}
        first = args.candidates[0]
        collector = TouchCollector(FrankaRobot(), ReSkinSensor(), RealSenseCamera(), scans[first], normals[first],
                                   np.linalg.inv(np.load(args.world)), np.load(args.camera), candidates=1, rng=rng, sleep=time.sleep)
    recogniser = TouchingRecogniser(collector, scans, normals, model, args.input, args.evidence, replay=replay, rng=rng)
    name, probabilities = recogniser.recognise(args.touches)
    print("aimed at: " + ", ".join(recogniser.guesses))
    for candidate, p in probabilities.items():
        print(f"  {candidate:14s} {100 * p:5.1f}% of the votes")
    print(f"recognised: {name}")
