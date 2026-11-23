"""
Which object is the robot touching?  (Paper: Table II, repeated by a simulated robot.)

In the paper the robot touched an object ten times and chose among candidates for which it had 3D
scans.  Those runs were not recorded.  Here a simulated robot repeats them on the real data: the
scan of one of the paper's recognition objects stands in front of it, it follows the paper's
procedure (procedures/touching_recogniser.py).  Every touch counts, as on the paper's robot.  Where
the sensor meets the object it feels the signal recorded at the nearest real touch; where it meets
nothing, nothing.  (`reach` and `turn` restrict the touches to those near a real one.)
The model is trained on the configuration's training objects, never on a candidate.

One record per method and touched object, over `runs` runs of `touches` touches:
  per_touch       share of single touches that vote for the touched object, the number of Table II
  after_touches   share of runs that name it
The methods are "touch model" and "proprioception", the paper's baseline: the scan nearest to where
the sensor stopped.
"""

import numpy as np

from imagine2touch.geometry import ObjectScan
from imagine2touch.procedures import TouchCollector, TouchingRecogniser
from imagine2touch.simulation import SimulatedCamera, SimulatedRobot, SimulatedScene, SimulatedSensor
from .experiment import Experiment


class ObjectRecognition(Experiment):
    name = "object_recognition"
    METHODS = {"touch model": "touch", "proprioception": "proprioception"}

    def run(self, seed):
        dataset, shapes = self.bench.touches, self.bench.config.data.recognition
        sessions = {shape: dataset.objects[k] for shape in shapes for k in self.bench.selection().sessions
                    if dataset.shape_of_session(k) == shape}
        model, records = self.bench.trained_model(seed, "mask"), []
        for touched, shape in enumerate(shapes):
            scene = SimulatedScene.of_session(dataset, sessions[shape])
            # every candidate's scan where it stood under the real robot, in this scene's world frame [m]
            scans = {name: scene.to_world(dataset.scan(dataset.objects.index(sessions[name])).points / 1000) for name in shapes}
            normals = {name: ObjectScan.normals_of(points) for name, points in scans.items()}
            for method, evidence in self.METHODS.items():
                votes, named = [], 0
                for run in range(self.settings.runs):
                    rng = np.random.default_rng([seed, touched, run])
                    recogniser = self.recogniser(scene, sessions[shape], scans, normals, model, evidence, rng)
                    recogniser.recognise(self.settings.touches, attempts=40 * self.settings.touches)
                    if recogniser.recogniser.touches == 0:          # no touch of this run could be counted
                        continue
                    named += recogniser.recogniser.name(rng) == shape
                    cost = np.array(recogniser.recogniser.costs)
                    best = cost == cost.min(1, keepdims=True)
                    votes.append((best[:, touched] / best.sum(1)).mean())
                records.append(self.record(seed, "simulated", rule=method, shape=shape, per_touch=float(np.mean(votes)) if votes else np.nan,
                                           after_touches=named / self.settings.runs, chance=1 / len(shapes),
                                           runs=self.settings.runs))
        return records

    def recogniser(self, scene, session, scans, normals, model, evidence, rng):
        """A simulated robot in front of `scene`, ready to recognise the object among `scans`."""
        dataset = self.bench.touches
        robot = SimulatedRobot(scene, rng=rng)
        sensor = SimulatedSensor.of_session(robot, dataset, session, reach=self.settings.reach,
                                           turn=None if self.settings.turn is None else np.radians(self.settings.turn), rng=rng)
        camera = SimulatedCamera(robot)
        first = next(iter(scans))
        collector = TouchCollector(robot, sensor, camera, scans[first], normals[first], scene.world_from_base,
                                   camera.camera_in_sensor, candidates=1, rng=rng, sleep=lambda seconds: None)
        return TouchingRecogniser(collector, scans, normals, model, "scan_mask", evidence, per_candidate=self.settings.targets, rng=rng)
