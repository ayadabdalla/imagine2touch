"""
Pictures of the data, for the README and for looking around.

  DatasetVisualizer   the visuo-tactile dataset: the objects and where they were touched, what one
                      touch holds, example touches, and an animation that walks through touches
  StampVisualizer     the shape classification dataset: what each stamp does to the sensor
  CorrespondenceFigure   the paper's Fig. 2: the touches laid out by their masks, with their signals
  TouchView           one touch with all its images brought to a common layout and scale
  Style               the colours and type shared by the pictures
"""

from .correspondence_figure import CorrespondenceFigure
from .dataset_visualizer import DatasetVisualizer
from .stamp_visualizer import StampVisualizer
from .style import Style
from .touch_view import TouchView

__all__ = ["CorrespondenceFigure", "DatasetVisualizer", "StampVisualizer", "Style", "TouchView"]
