"""
A printed board of AprilTags, found in a picture with OpenCV (`pip install -e ".[robot]"`).

Written from the calibration scripts of the paper's code (commit c2ea2c3 of this repository), which
used the board detector of robot_io.  Checked on rendered pictures of the board, not on a camera's.
"""

import numpy as np

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import CalibrationBoard


class AprilTagBoard(CalibrationBoard):
    """
    `columns` x `rows` tags of the family 36h11, each `tag` metres wide with `gap` metres between
    them, numbered from `first` along the rows, starting at the top left.  The defaults are the
    paper's board.  `save()` writes the board for printing at its real size.

    The board's frame has its origin at the lower left corner of the lower left tag, x to the right,
    y up the board and z out of the printed side: lying on a table, its z axis points up.
    """

    def __init__(self, columns=3, rows=3, tag=0.03, gap=0.009, first=9, least_tags=2):
        import cv2
        self.cv2, self.columns, self.rows, self.tag, self.gap, self.first, self.least_tags = cv2, columns, rows, tag, gap, first, least_tags
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())

    def corners_of(self, tag_id):
        """The four corners of a tag on the board [m], (4, 3), in the order OpenCV reports them: from the top left, clockwise."""
        column, row = (tag_id - self.first) % self.columns, (tag_id - self.first) // self.columns
        left = column * (self.tag + self.gap)
        top = (self.rows - row) * self.tag + (self.rows - 1 - row) * self.gap
        return np.array([[left, top, 0], [left + self.tag, top, 0], [left + self.tag, top - self.tag, 0], [left, top - self.tag, 0]])

    def image(self, pixels_per_tag=240):
        """The board as a grey image to print, with a white margin of one gap around it."""
        step = int(round(pixels_per_tag * (1 + self.gap / self.tag)))
        margin = step - pixels_per_tag
        sheet = np.full((self.rows * step + margin, self.columns * step + margin), 255, np.uint8)
        for k in range(self.columns * self.rows):
            top, left = margin + (k // self.columns) * step, margin + (k % self.columns) * step
            sheet[top:top + pixels_per_tag, left:left + pixels_per_tag] = \
                self.cv2.aruco.generateImageMarker(self.dictionary, self.first + k, pixels_per_tag)
        return sheet

    def save(self, path, pixels_per_tag=240):
        """
        Writes the board to print, as a .pdf or an image file.  The file carries its real size:
        printed without scaling, a tag comes out `tag` metres wide.
        """
        from PIL import Image
        dots_per_inch = pixels_per_tag / (self.tag / 0.0254)
        Image.fromarray(self.image(pixels_per_tag)).save(path, dpi=(dots_per_inch, dots_per_inch), resolution=dots_per_inch)

    def pose_in(self, colour, intrinsics):
        grey = self.cv2.cvtColor(np.ascontiguousarray(colour), self.cv2.COLOR_RGB2GRAY) if np.ndim(colour) == 3 else colour
        corners, ids, _ = self.detector.detectMarkers(grey)
        known = [] if ids is None else [k for k, i in enumerate(ids.ravel()) if 0 <= i - self.first < self.columns * self.rows]
        if len(known) < self.least_tags:
            return None
        on_board = np.concatenate([self.corners_of(int(ids.ravel()[k])) for k in known])
        in_image = np.concatenate([corners[k].reshape(4, 2) for k in known])
        found, rotation, position = self.cv2.solvePnP(on_board, in_image, np.asarray(intrinsics, float), None)
        if not found:
            return None
        return Pose(self.cv2.Rodrigues(rotation)[0], position.ravel())
