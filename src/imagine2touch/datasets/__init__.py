"""
The data: the two datasets of the paper, and how they are made from the robot's recordings.

  VisuoTactileDataset           every touch of an object: what the camera saw, what the sensor felt,
                                where the sensor was, and the scan of the object
  ShapeClassificationDataset    presses of five small stamps into the mounted sensor
  ReSkinReading                 a touch reading as the recordings store it: the layout of the ReSkin's driver
  TactileImage                  a touch signal as the 3 x 3 image of the sensor's magnetometers
  ImageInput                    the kinds of image a model can be given for a touch (IMAGE_INPUTS):
                                CameraMask, CameraDepth, ScanDepth, ScanMask

  recordings/                   the raw folders the robot wrote, and their conversion into the two files

With the two files downloaded, only the classes above are needed.
"""

from .image_inputs import IMAGE_INPUTS, CameraDepth, CameraMask, ImageInput, ScanDepth, ScanMask
from .shape_classification_dataset import ShapeClassificationDataset
from .tactile_image import ReSkinReading, TactileImage
from .visuo_tactile_dataset import VisuoTactileDataset

__all__ = ["IMAGE_INPUTS", "ImageInput", "CameraDepth", "CameraMask", "ScanDepth", "ScanMask",
           "ReSkinReading", "ShapeClassificationDataset", "TactileImage", "VisuoTactileDataset"]
