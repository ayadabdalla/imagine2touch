"""The procedures of the robot, in the simulated scene: no hardware is touched by these tests."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from imagine2touch.datasets.recordings import ApproachOutcome
from imagine2touch.geometry import ObjectScan, Pose
from imagine2touch.procedures import ApproachSettings, CameraCalibrator, ContactApproach, ObjectScanner, SafetyChecker, TouchingRecogniser
from imagine2touch.simulation import SimulatedBoard, SimulatedCamera, SimulatedRobot, SimulatedSensor

from conftest import PAPER, SHAPES, simulated_collector, simulated_scene


QUIET = dict(sleep=lambda seconds: None)


def approach(robot, sensor, target, pose, safety=None):
    return ContactApproach(robot, sensor, safety=safety, **QUIET).run(target, pose)


def above_the_object(scene, height=0.04, xy=(0.0, 0.0)):
    """A pose looking straight down at the point (x, y) of the scene from `height` above its surface, and that point."""
    target = scene.to_base([xy[0], xy[1], float(scene.surface_height(*xy))])
    return Pose.from_normal([0, 0, -1], target + [0, 0, height]), target


def loud_sensor(robot):
    """A sensor whose every recorded signal is strong enough to stop the approach."""
    sensor = SimulatedSensor.of_session(robot, PAPER, "square")
    sensor.signals = sensor.signals[np.abs(sensor.signals).max(1) > 2 * ApproachSettings().jump]
    sensor.positions = sensor.positions[:len(sensor.signals)] * 0 + robot.scene.to_base([0, 0, robot.scene.object_height])
    return sensor


def silent_sensor(robot):
    sensor = SimulatedSensor.of_session(robot, PAPER, "square")
    sensor.signals = 0 * sensor.signals
    return sensor


def angle_between(a, b):
    """The angle [rad] by which the pose a is turned against the pose b."""
    return float(np.arccos(np.clip((np.trace(a.R.T @ b.R) - 1) / 2, -1, 1)))


def test_approach_stops_when_the_sensor_feels_the_surface():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    start, target = above_the_object(scene)
    robot.move_to(start)
    result = approach(robot, loud_sensor(robot), target, start)
    assert result.outcome == ApproachOutcome.SENSOR and result.jump > ApproachSettings().jump
    assert result.ambient is not None and result.ambient.shape == (15,)
    assert robot.press(result.pose).max() <= robot.skin + 1e-9                 # it stopped on the surface, pressed by at most the skin


def test_approach_stops_on_the_force_limit_when_the_sensor_stays_silent():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    start, target = above_the_object(scene)
    robot.move_to(start)
    result = approach(robot, silent_sensor(robot), target, start)
    assert result.outcome == ApproachOutcome.WRENCH and np.linalg.norm(result.wrench[:3]) > ApproachSettings().max_force


def test_approach_gives_up_in_the_air():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    start, target = above_the_object(scene, height=0.04)
    robot.move_to(Pose(start.R, start.t + [0, 0, 0.03]))
    result = approach(robot, loud_sensor(robot), target + [0, 0, 0.03], start)      # a point 3 cm above the surface
    s = ApproachSettings()
    assert result.outcome == ApproachOutcome.AIR
    assert result.pose.t[2] > target[2] + 0.03 - s.extensions * s.extension - s.step


def test_approach_refuses_far_targets_and_wrong_orientations():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    start, target = above_the_object(scene, height=0.3)
    robot.move_to(start)
    assert approach(robot, loud_sensor(robot), target, start).outcome == ApproachOutcome.ABORTED
    near, target = above_the_object(scene)
    robot.move_to(near)
    tilted = Pose.from_normal([0.3, 0, -1], near.t)
    assert approach(robot, loud_sensor(robot), target, tilted).outcome == ApproachOutcome.ABORTED
    assert len([m for m in robot.moves if m[0] == "step"]) == 0               # the robot never moved towards the object


def test_approach_keeps_the_sensor_above_the_table():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    start, target = above_the_object(scene, xy=(0.08, 0.0))                     # beside the object: only the table is below
    target = target - [0, 0, 0.01]                                              # a point 1 cm under the table
    robot.move_to(Pose(start.R, target + [0, 0, 0.03]))
    safety = SafetyChecker(scene.world_from_base)
    result = approach(robot, silent_sensor(robot), target, start, safety)
    assert result.outcome == ApproachOutcome.AIR
    assert scene.to_world(result.pose.t)[2] >= safety.settings.table_clearance - ApproachSettings().step


def test_safety_checks():
    safety = SafetyChecker(np.eye(4))
    assert safety.faces_up([0, 0, -1]) and not safety.faces_up([1, 0, -0.1])
    flat = np.array([[x, y, 0.0] for x in np.linspace(-0.02, 0.02, 21) for y in np.linspace(-0.02, 0.02, 21)])
    assert safety.path_is_clear(flat, [0, 0, 0], [0, 0, -1], 0.0085)
    bump = np.vstack([flat, [[0.004, 0.0, 0.01]]])                             # something stands 1 cm high beside the point
    assert not safety.path_is_clear(bump, [0, 0, 0], [0, 0, -1], 0.0085)
    assert safety.path_is_clear(bump, [-0.015, 0, 0], [0, 0, -1], 0.0085)      # but not in the way further left
    down = Pose.from_normal([0, 0, -1], [0, 0, 0.05])
    assert safety.lowest_point(down, (0.01, 0.01, 0.0)) == pytest.approx(0.05)
    assert safety.sensor_clears_table(down) and not safety.sensor_clears_table(Pose(down.R, [0, 0, 0.001]))
    assert safety.can_touch(flat, flat[0], np.array([0, 0, 1.0]), 0.0085) and not safety.can_touch(flat, flat[0], np.array([1.0, 0, 0]), 0.0085)


def test_collector_looks_first_then_touches_and_backs_away():
    collector = simulated_collector("star")
    record = collector.touch()
    assert record is not None and record["outcome"] == ApproachOutcome.SENSOR
    kinds = [kind for kind, _ in collector.robot.moves]
    assert kinds[0] == "move" and kinds[1] == "move" and kinds[-1] == "move" and "step" in kinds
    seen = record["depth"][record["depth"] > 0].min()                          # nearest surface from 4 cm + the camera's offset
    assert seen == pytest.approx(1000 * (collector.look - collector.camera_in_sensor.t[2]), abs=4)        # a real surface is not flat
    footprint = 2 * collector.half_side * collector.camera.intrinsics[0, 0] / (seen / 1000)     # the sensor's side in pixels
    assert record["depth"].shape[0] == pytest.approx(footprint, abs=3) and record["depth"].shape[1] == pytest.approx(footprint, abs=3)
    assert np.linalg.norm(collector.robot.pose().t - record["target"]) > collector.look     # it backed away


def test_recordings_of_the_simulated_robot_reduce_like_the_real_ones(touches):
    assert len(touches) == 14 * len(SHAPES) and sorted(touches.shapes) == sorted(SHAPES)
    stopped = np.flatnonzero(touches.outcome == "sensor")
    assert len(stopped) > 0.6 * len(touches)                   # some recorded signals are too weak to stop the approach
    for i in stopped[:10]:
        # the scan, seen from where the sensor stopped, lies on the sensing surface, pressed by at most the skin.
        # Looked at away from the rim of the patch, where scan points just outside it are drawn into the image.
        nearest = touches.scan_depth(i)[6:-6, 6:-6].min()
        assert touches.grid.behind - 1.7 <= nearest <= touches.grid.behind + 0.5
        # and so does the wrist camera's image
        seen = touches.depth[i][touches.depth[i] > 0].min()
        assert seen - touches.camera_to_sensor[i] == pytest.approx(nearest - touches.grid.behind, abs=3)
    assert touches.mask[stopped].any() and np.abs(touches.signal[stopped]).max() > 25


def test_another_object_looks_different_from_where_the_sensor_stopped(touches):
    k = touches.shapes.index("hexagon"), touches.objects.index("star_v_1")
    on_hexagon = np.flatnonzero((touches.shape == k[0]) & (touches.outcome == "sensor"))
    differences = [np.abs(touches.scan_depth(i) - touches.scan_depth(i, k[1])).max() for i in on_hexagon]
    assert max(differences) > 1.0                                              # somewhere the star is not where the hexagon is


def test_touching_recogniser_scores_every_candidate_at_every_touch():
    class Flat:                                              # a model that expects the same signal everywhere
        def predict(self, images):
            return np.zeros((len(images), 15))

        def standardise(self, signals):
            return np.asarray(signals, float)

    collector = simulated_collector("star")
    world = collector.base_from_world.inverse()                 # every scan in the world frame of the scene that is touched
    scans = {shape: world.apply(PAPER.scan(PAPER.objects.index(shape)).points / 1000) for shape in SHAPES}
    normals = {shape: ObjectScan.normals_of(points) for shape, points in scans.items()}
    for evidence in ("touch", "proprioception"):
        recogniser = TouchingRecogniser(collector, scans, normals, Flat(), evidence=evidence, per_candidate=60,
                                        rng=np.random.default_rng(0))
        assert recogniser.images(collector.robot.pose()).shape == (3, 48, 48)
        name, probabilities = recogniser.recognise(touches=3)
        assert name in SHAPES and recogniser.recogniser.touches == 3 and set(recogniser.guesses) <= set(SHAPES)
        assert sum(probabilities.values()) == pytest.approx(1) and set(probabilities) == set(SHAPES)


def test_camera_calibration_finds_the_camera_and_the_table():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    camera = SimulatedCamera(robot, width=64, height=48, focal=120.0)           # the simulated board reads no image
    board = SimulatedBoard(robot, camera.camera_in_sensor, scene.base_from_world)
    guess = np.round(camera.camera_in_sensor, 2)                                # the mount as a ruler would give it
    calibration = CameraCalibrator(robot, camera, board, guess, **QUIET).calibrate(scene.to_base([0, 0, 0]))
    mount, table = Pose.from_matrix(camera.camera_in_sensor), Pose.from_matrix(scene.base_from_world)
    assert len(calibration.spread) == 13 and calibration.spread.mean() < 0.001
    assert np.linalg.norm(calibration.camera_in_sensor.t - mount.t) < 0.003 and angle_between(calibration.camera_in_sensor, mount) < np.radians(1)
    assert np.linalg.norm(calibration.board_in_base.t - table.t) < 0.003 and angle_between(calibration.board_in_base, table) < np.radians(1)
    assert calibration.board_in_base.normal @ scene.up > 0.99                   # the world's z axis points up from the table


def test_camera_calibration_is_exact_without_noise_and_refuses_too_few_views():
    scene = simulated_scene("square")
    robot = SimulatedRobot(scene)
    camera = SimulatedCamera(robot, width=64, height=48, focal=120.0)
    board = SimulatedBoard(robot, camera.camera_in_sensor, scene.base_from_world, shake=0, tremble=0)
    calibrator = CameraCalibrator(robot, camera, board, np.round(camera.camera_in_sensor, 2), **QUIET)
    calibration = calibrator.calibrate(scene.to_base([0, 0, 0]))
    assert calibration.camera_in_sensor.matrix == pytest.approx(camera.camera_in_sensor, abs=1e-6)
    assert calibration.spread.max() < 1e-6
    calibrator.tilts, calibrator.per_tilt = (0.0,), 1
    with pytest.raises(RuntimeError):
        calibrator.calibrate(scene.to_base([0, 0, 0]))


def test_scanner_rebuilds_the_surface_it_looks_at():
    scene = simulated_scene("star")
    robot = SimulatedRobot(scene)
    camera = SimulatedCamera(robot, width=160, height=120, focal=300.0)
    scanner = ObjectScanner(robot, camera, scene.world_from_base, camera.camera_in_sensor, **QUIET)
    views = scanner.views()
    assert len(views) == 5 and all(scanner.sensor_is_clear(view) for view in views)
    points, normals = scanner.scan()
    assert len(points) == len(normals) and (normals[:, 2] >= 0).all()
    assert points[:, 2].min() > 0.01                                            # the table and what is near it stay out
    gap = points[:, 2] - scene.surface_height(points[:, 0], points[:, 1])
    assert abs(np.median(gap)) < 0.0005 and (np.abs(gap) < 0.001).mean() > 0.9  # the scan lies on the real scan's surface
    high = np.column_stack(np.nonzero(scene.heights > 0.012))                   # every part of the object above the box's floor
    cells = scene._corner + (high + 0.5) * scene.cell
    assert (cKDTree(points[:, :2]).query(cells)[0] < 0.001).mean() > 0.95       # was seen


def test_scanner_leaves_out_a_view_that_brings_the_sensor_into_the_box():
    scene = simulated_scene("star")
    robot = SimulatedRobot(scene)
    camera = SimulatedCamera(robot, width=64, height=48, focal=120.0)
    scanner = ObjectScanner(robot, camera, scene.world_from_base, camera.camera_in_sensor, **QUIET)
    sensor_in_the_box = Pose.from_normal([0, 0, -1], [0, 0, 0.05]) @ scanner.camera_in_sensor
    assert not scanner.sensor_is_clear(sensor_in_the_box) and scanner.look(sensor_in_the_box) is None
    assert robot.moves == []                                                    # the robot was not sent there
