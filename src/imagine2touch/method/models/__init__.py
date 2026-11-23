"""
The models of the paper.

  TouchNetwork        what a network of the touch model has to be; ARCHITECTURES holds the two:
    Imagine2TouchMLP    the network of the paper and the thesis
    Imagine2TouchCNN    the network of the paper's last code
  TouchModel          a network with its scaling and training: images in, touch signals out
  ShapeClassifier     names the stamp pressed into the sensor (paper: Table I)
"""

from .imagine2touch_cnn import Imagine2TouchCNN
from .imagine2touch_mlp import Imagine2TouchMLP
from .shape_classifier import ShapeClassifier
from .imagine2touch_model import TouchModel
from .imagine2touch_network import TouchNetwork

ARCHITECTURES = {cls.name: cls for cls in (Imagine2TouchMLP, Imagine2TouchCNN)}

__all__ = ["ARCHITECTURES", "Imagine2TouchCNN", "Imagine2TouchMLP", "ShapeClassifier", "TouchModel", "TouchNetwork"]
