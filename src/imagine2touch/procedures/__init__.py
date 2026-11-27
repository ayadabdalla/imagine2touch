"""
What the robot does to prepare for touching objects, and when it touches them.  The hardware is reached
through the interfaces of `interfaces.py` only, so the same code runs on the real devices
(`devices/`) and on the simulated ones (`simulation/`).

  CameraCalibrator   finds where the wrist camera sits and where the table is, from views of a board of tags
  ObjectScanner      looks at an object from several sides and builds its scan
  ContactApproach    moves the sensor onto a surface until it feels it
  SafetyChecker      the checks a touch has to pass before and while the robot moves
  TouchCollector     one touch after another: look, approach, record, back away
  RecordingWriter    stores the touches in the layout the datasets are built from
  TouchingRecogniser   touches an object and names it among candidates
"""

from .camera_calibrator import CameraCalibrator
from .contact_approach import ApproachResult, ApproachSettings, ContactApproach
from .touching_recogniser import TouchingRecogniser
from .object_scanner import ObjectScanner
from .recording_writer import RecordingWriter
from .safety_checker import SafetyChecker, SafetySettings
from .touch_collector import TouchCollector

__all__ = ["ApproachResult", "ApproachSettings", "CameraCalibrator", "ContactApproach", "ObjectScanner", "TouchingRecogniser", "RecordingWriter",
           "SafetyChecker", "SafetySettings", "TouchCollector"]
