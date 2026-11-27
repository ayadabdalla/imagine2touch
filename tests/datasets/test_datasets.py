"""The recordings and their parts, and the two datasets if their files are there."""

import numpy as np
import pytest

from imagine2touch.datasets import ShapeClassificationDataset, VisuoTactileDataset
from imagine2touch.datasets.recordings import ApproachOutcome, PointCloudFile, Recording
from imagine2touch.method.preprocessing import ContactMask


def test_recording_helpers(tmp_path):
    points = np.random.default_rng(0).normal(size=(40, 3))
    PointCloudFile.write(tmp_path / "scan.pcd", points)
    assert PointCloudFile.read(tmp_path / "scan.pcd") == pytest.approx(points, abs=1e-6)
    assert Recording(tmp_path / "star_v_1", 1).shape == "star" and Recording(tmp_path / "star", 1).shape == "star"
    assert Recording.centre_crop(np.ones((52, 53))).shape == (48, 48) and Recording.centre_crop(np.ones((40, 40))).sum() == 1600
    assert ApproachOutcome.label_of(2) == "sensor" and ApproachOutcome.label_of(0) == "unknown" and ApproachOutcome.label_of(1) == "aborted" and ApproachOutcome.SENSOR == 2


@pytest.mark.skipif(not VisuoTactileDataset().available, reason="data/imagine2touch.npz is not there")
def test_the_visuo_tactile_dataset():
    t = VisuoTactileDataset()
    paper = t.in_sessions(list(t.PAPER_SESSIONS))
    assert paper.sum() == 1630                                                  # the number the paper states
    assert sorted({t.shapes[k] for k in t.shape[paper]}) == ["cutter", "flower", "hexagon", "parallelogram", "square", "star"]
    assert (t.outcome[t.in_sessions(["flower"])] == "unknown").all()
    assert all(t.has_scan(k) for k in range(len(t.objects)))                    # every scan is placed under the robot
    # with the calibration of the recording code, the points the robot was sent to are points of the scan
    flower = np.flatnonzero(t.in_sessions(["flower"]))
    assert np.median([t.scan(int(t.object[i])).distance_to(t.commanded[i].t) for i in flower]) < 0.05
    assert np.median([t.distance_to_scan(i, int(t.object[i])) for i in flower]) < 1.5         # and the sensor stops on it
    assert t.signal.shape[1] == 15 and t.mask.shape[1:] == (48, 48)
    assert t.signal == pytest.approx(np.abs(t.reading - t.ambient), abs=1e-3)
    # the stored masks follow the rule of preprocessing.contact_mask
    assert (ContactMask()(t.depth_processed[paper]) == t.mask[paper]).mean() > 0.97
    # the fitted sensor sits 125 mm in front of the flange, and the robot stops within millimetres of its target
    assert t.early_tool.t[2] == pytest.approx(125, abs=3)
    stopped = np.flatnonzero(paper & (t.outcome == "sensor"))
    assert np.median([t.pose_error(i) for i in stopped]) < 6


@pytest.mark.skipif(not ShapeClassificationDataset().available, reason="data/shape_classification.npz is not there")
def test_the_presses_of_the_stamps():
    s = ShapeClassificationDataset()
    assert len(s) == 1280                                                       # the number the paper states
    assert [int((s.stamp == k).sum()) for k in range(len(s.STAMPS))] == [256] * 5
    assert s.position_index.max() == 7
    assert [int((s.role == r).sum()) for r in ("train", "validation", "test")] == [905, 125, 250]       # the paper's split
    assert all(int(((s.role == "test") & (s.stamp == k)).sum()) == 50 for k in range(len(s.STAMPS)))
    assert np.bincount(s.stamp[s.balanced(100, np.random.default_rng(0))]).tolist() == [100] * 5
    if s.depth is not None:                                                     # the depth column is optional
        deep, shallow = s.depth > 1.0e-3, s.depth < 0.4e-3
        assert np.abs(s.force[deep]).mean() > np.abs(s.force[shallow]).mean()   # a deeper press takes more force
