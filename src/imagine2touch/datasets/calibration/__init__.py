"""
The calibrations needed to place a touch on its object.

  ToolCalibration      where the touch sensor sits relative to the robot's flange
  ScanPlacement        where an object's scan stood relative to the robot
  CameraCalibration    where the wrist camera sits relative to the sensor, and where the table is
  CalibrationBoard     the printed board the camera is calibrated against: AprilTagBoard

The first two were not saved with the paper's touches; `recordings/RecordingReducer` recovers them to
build the dataset.  The camera's calibration is made before touching, by `procedures/CameraCalibrator`,
which looks at the board.
"""

from .april_tag_board import AprilTagBoard
from .calibration_board import CalibrationBoard
from .camera_calibration import CameraCalibration
from .scan_placement import ScanPlacement
from .tool_calibration import ToolCalibration

__all__ = ["AprilTagBoard", "CalibrationBoard", "CameraCalibration", "ScanPlacement", "ToolCalibration"]
