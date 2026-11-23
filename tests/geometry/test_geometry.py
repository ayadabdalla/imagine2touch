"""Kinematics, poses, the depth image a sensor sees, the scan of an object, and the places a camera looks from."""

import numpy as np
import pytest

from imagine2touch.geometry import DepthRenderer, FrankaKinematics, ObjectScan, Pose, SensorGrid, Viewpoints


def test_flange_of_the_stretched_arm():
    pose = FrankaKinematics.flange_pose(np.zeros(7))
    assert pose.t == pytest.approx([88.0, 0.0, 333 + 316 + 384 - 107])      # link lengths from Franka's data sheet
    assert pose.normal == pytest.approx([0, 0, -1], abs=1e-12)              # the flange points down


def test_first_joint_turns_the_flange_about_the_vertical():
    assert FrankaKinematics.flange_pose([np.pi / 2, 0, 0, 0, 0, 0, 0]).t == pytest.approx([0.0, 88.0, 926.0], abs=1e-9)


def test_pose_inverse_and_composition():
    rng = np.random.default_rng(0)
    a = Pose.from_normal(rng.normal(size=3), rng.normal(size=3))
    b = Pose.from_quaternion(a.quaternion, [1.0, 2.0, 3.0])
    x = rng.normal(size=(5, 3))
    assert (a @ a.inverse()).matrix == pytest.approx(np.eye(4), abs=1e-12)
    assert (a @ b).apply(x) == pytest.approx(a.apply(b.apply(x)))
    assert a.to_local(a.apply(x)) == pytest.approx(x)
    assert b.R == pytest.approx(a.R)


def test_depth_image_of_a_plane_and_of_nothing():
    grid = SensorGrid(size=16)
    renderer = DepthRenderer(grid, 0.6)
    u, v = np.meshgrid(np.arange(-12, 12, 0.2), np.arange(-12, 12, 0.2))
    plane = np.column_stack([u.ravel(), v.ravel(), np.full(u.size, 1.5)])     # 1.5 mm in front of the sensor
    assert renderer.render(plane, Pose()) == pytest.approx(grid.behind + 1.5)
    assert renderer.render(plane, Pose(t=[0, 0, 0.5])) == pytest.approx(grid.behind + 1.0)
    assert (renderer.render(plane + [0, 0, 50], Pose()) == grid.empty).all()


def test_depth_image_shows_only_what_is_under_the_sensor():
    grid = SensorGrid(size=16)
    half = np.array([[u, v, 1.0] for u in np.arange(0.5, 12, 0.2) for v in np.arange(-12, 12, 0.2)])   # one side only
    image = DepthRenderer(grid, 0.6).render(half, Pose())
    assert (image[:6] == grid.empty).all() and (image[10:] < grid.empty).all()


def test_a_scan_is_thinned_and_shows_its_front():
    rng = np.random.default_rng(0)
    layer = np.column_stack([rng.uniform(-12, 12, (60000, 2)), rng.uniform(1, 5, 60000)])     # a thick layer of points
    stray = np.array([[0.1, 0.1, -1.0], [1.3, -0.7, 0.0]])                                    # and two in front of it
    scan = ObjectScan.from_raw_points(np.vstack([layer, stray]))
    assert len(scan) < len(layer) and scan.points[:, 2].min() > 0.9                           # the stray points are gone
    grid = SensorGrid(size=16)
    image = grid.depth_of(scan, Pose())
    assert np.median(image) == pytest.approx(grid.behind + 1.3, abs=0.4)                      # the front of the layer
    assert scan.distance_to([0, 0, -9]) == pytest.approx(10.2, abs=0.5)


def test_viewpoints_face_the_centre_from_one_distance():
    centre = np.array([0.4, 0.1, 0.02])
    views = Viewpoints.around(centre, 0.15, tilts=(0.0, np.radians(20), np.radians(40)), per_tilt=6)
    assert len(views) == 13                                                     # one from above, six for each tilt
    for view in views:
        assert np.linalg.norm(view.t - centre) == pytest.approx(0.15)
        assert view.apply([0, 0, 0.15]) == pytest.approx(centre)                # the centre is straight ahead
    assert views[0].normal == pytest.approx([0, 0, -1])
    heights = sorted({round(float(view.t[2] - centre[2]), 6) for view in views})
    assert heights == pytest.approx(sorted(0.15 * np.cos([np.radians(40), np.radians(20), 0.0])), abs=1e-5)
