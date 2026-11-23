"""
The recordings as the robot wrote them, and their conversion into the two dataset files.

  Recording                  one session of one object, read from its folder
  RecordingReader            finds the recordings in a folder
  PointCloudFile             reads and writes the files of the scans
  ApproachOutcome            how an approach of the sensor ended, as the robot's code is written to a recording
  RecordingReducer           recordings of touches -> imagine2touch.npz, with the help of `calibration/`
  StampRecordingConverter    recordings of stamp presses -> shape_classification.npz

    python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data
"""

from .approach_outcome import ApproachOutcome
from .point_cloud_file import PointCloudFile
from .recording import Recording
from .recording_reader import RecordingReader

__all__ = ["ApproachOutcome", "PointCloudFile", "Recording", "RecordingReader", "RecordingReducer", "StampRecordingConverter"]


def __getattr__(name):
    # the two converters are also run as programs (python -m ...): loading them only when asked keeps that clean
    if name == "RecordingReducer":
        from .recording_reducer import RecordingReducer
        return RecordingReducer
    if name == "StampRecordingConverter":
        from .stamp_recording_converter import StampRecordingConverter
        return StampRecordingConverter
    raise AttributeError(name)
