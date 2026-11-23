"""The simulated robot, sensor and camera."""

import numpy as np
import pytest

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import DepthCamera, Robot, TouchSensor
from imagine2touch.simulation import SimulatedRobot, SimulatedSensor

from conftest import PAPER, simulated_collector, simulated_scene


def test_simulated_devices_are_devices():
    collector = simulated_collector("square")
    assert isinstance(collector.robot, Robot) and isinstance(collector.sensor, TouchSensor)
    assert isinstance(collector.camera, DepthCamera)


def test_the_simulated_sensor_answers_with_the_recorded_touch():
    scene = simulated_scene("star")
    cloud, normals = scene.scan()
    assert len(cloud) == len(normals) and np.linalg.norm(normals, axis=1) == pytest.approx(1) and (normals[:, 2] >= 0).all()
    robot = SimulatedRobot(scene)
    sensor = SimulatedSensor.of_session(robot, PAPER, "star")
    k = int(np.abs(sensor.signals).max(1).argmax())                             # the strongest recorded touch
    above = scene.to_base([0, 0, scene.object_height + 0.02])
    robot.move_to(Pose.from_normal([0, 0, -1], above))                          # in the air: nothing felt
    assert np.abs(sensor.read() - sensor.rest).max() < 5
    sensor.positions[k] = above - [0, 0, 0.02]                                  # say that touch was made here, and press in
    robot.move_to(Pose.from_normal([0, 0, -1], above - [0, 0, 0.03]))
    assert sensor.read() - sensor.rest == pytest.approx(sensor.signals[k], abs=5)
