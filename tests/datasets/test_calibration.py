"""The board the wrist camera is calibrated against."""

import numpy as np
import pytest

from imagine2touch.datasets.calibration import AprilTagBoard
from imagine2touch.geometry import Viewpoints


def test_april_tag_board_is_found_in_a_picture_of_itself():
    cv2 = pytest.importorskip("cv2")
    board, pixels, K = AprilTagBoard(), 120, np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    sheet = board.image(pixels)
    margin, metre = int(round(pixels * board.gap / board.tag)), board.tag / pixels
    height = board.rows * board.tag + (board.rows - 1) * board.gap
    on_board = np.array([[metre, 0, -margin * metre], [0, -metre, height + margin * metre], [0, 0, 1]])   # sheet pixel -> place on the board
    for view in Viewpoints.around([height / 2, height / 2, 0], 0.2, tilts=(0.0, np.radians(30))):
        seen = view.inverse()                                                   # the board from the camera
        picture = cv2.warpPerspective(sheet, K @ np.column_stack([seen.R[:, 0], seen.R[:, 1], seen.t]) @ on_board, (640, 480), borderValue=200)
        found = board.pose_in(picture, K)
        assert np.linalg.norm(found.t - seen.t) < 0.002 and np.arccos(np.clip((np.trace(found.R.T @ seen.R) - 1) / 2, -1, 1)) < np.radians(1)
    assert board.pose_in(np.full((480, 640), 200, np.uint8), K) is None         # an empty picture
