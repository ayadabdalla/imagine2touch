"""The pictures of the data, drawn from the simulated datasets."""

import numpy as np
import pytest
from PIL import Image

from imagine2touch.visualization import DatasetVisualizer, StampVisualizer, TouchView


def test_a_touch_view_brings_camera_and_scan_to_one_scale(touches):
    i = int(np.flatnonzero(touches.outcome == "sensor")[0])
    view = TouchView(touches, i)
    assert view.rgb.shape == (48, 48, 3) and view.camera_gap.shape == view.scan_gap.shape == (48, 48)
    assert view.field.shape == (3, 3, 3) and np.isnan(view.field[:, 0, 0]).all()        # three crosses, empty corners
    # where the sensor stopped, both sources put the nearest surface at the sensing surface, pressed by at most the skin
    assert -1.7 <= np.nanmin(view.camera_gap) <= 1.6 and -1.7 <= view.scan_gap.min() <= 0.2
    assert view.footprint.shape == (5, 3) and np.linalg.norm(view.footprint[0] - view.footprint[1]) == pytest.approx(touches.grid.side)


def test_the_pictures_are_drawn(touches, stamps, tmp_path):
    visualizer = DatasetVisualizer(touches, sessions="_v_1")
    assert visualizer.placed == visualizer.sessions                                     # the simulated scans can be placed
    assert len(visualizer.clear_touches()) == len(touches.objects)                      # one touch of every object
    visualizer.overview(tmp_path / "overview.png", examples=2)
    visualizer.sessions_overview(touches.objects, tmp_path / "sessions.png", per_object=1)
    visualizer.touch(0, tmp_path / "touch.png")
    visualizer.animation(tmp_path / "touches.gif", per_shape=1)
    StampVisualizer(stamps).overview(tmp_path / "stamps.png")
    assert Image.open(tmp_path / "touches.gif").n_frames == len(touches.objects)
    assert all((tmp_path / name).stat().st_size > 5000 for name in ("overview.png", "touch.png", "stamps.png"))
