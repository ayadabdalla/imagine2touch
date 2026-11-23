"""
Data shared by the tests.  The simulated robot touches the scans of three of the paper's objects and
feels the recorded signals; the presses of the stamps are the paper's.  The tests
therefore need the two files in data/ and are skipped without them.
"""

import numpy as np
import pytest

from imagine2touch.datasets import ShapeClassificationDataset, VisuoTactileDataset
from imagine2touch.datasets.recordings.recording_reducer import RecordingReducer
from imagine2touch.procedures import RecordingWriter, TouchCollector
from imagine2touch.simulation import SimulatedCamera, SimulatedRobot, SimulatedScene, SimulatedSensor

PAPER = VisuoTactileDataset()
SHAPES = ("square", "star", "hexagon")       # sessions of the paper's dataset


def paper_data():
    if not PAPER.available:
        pytest.skip("needs data/imagine2touch.npz (see the README)")
    return PAPER


def simulated_scene(shape="square"):
    return SimulatedScene.of_session(paper_data(), shape)


def simulated_collector(shape, seed=0, **robot):
    scene, rng = simulated_scene(shape), np.random.default_rng(seed)
    arm = SimulatedRobot(scene, rng=rng, **robot)
    sensor, camera = SimulatedSensor.of_session(arm, PAPER, shape), SimulatedCamera(arm, width=120, height=90, focal=225.0)
    cloud, normals = scene.scan()
    return TouchCollector(arm, sensor, camera, cloud, normals, scene.world_from_base, camera.camera_in_sensor,
                          candidates=300, rng=rng, sleep=lambda seconds: None)


@pytest.fixture(scope="session")
def recordings(tmp_path_factory):
    """A folder of recordings in the layout of the real ones: 14 touches of each of three shapes."""
    root = tmp_path_factory.mktemp("recordings")
    for k, shape in enumerate(SHAPES):
        simulated_collector(shape, seed=k).collect(14, RecordingWriter(root, f"{shape}_v_1"))
    return root


@pytest.fixture(scope="session")
def touches(recordings, tmp_path_factory):
    return RecordingReducer(recordings).reduce(tmp_path_factory.mktemp("data") / "imagine2touch.npz", report=lambda line: None)


@pytest.fixture(scope="session")
def stamps():
    """The presses of the five stamps, from the paper's dataset."""
    dataset = ShapeClassificationDataset()
    if not dataset.available:
        pytest.skip("needs data/shape_classification.npz (see the README)")
    return dataset
