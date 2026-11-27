"""
The experiments of the paper, on the recorded data.

  Benchmark         runs experiments over seeds and keeps their records; holds the data and model they share
  Experiment        what an experiment has to be; EXPERIMENTS holds the four:
    ShapeClassification   paper, Table I
    Correspondence        paper, Fig. 2
    TouchPrediction       the touch model on the paper's split of the objects
    ObjectRecognition     paper, Table II, repeated by a simulated robot (procedures/, simulation/)
    ToolRecognition       the same on the paper's second set of objects, the tools
  TouchSelection    the touches the experiments share, with their images
  Config            the settings of a run (configs/benchmark.yaml)
  ResultsAnalyzer   the records of a run as tables;  metrics.py: the measures reported
"""

from .benchmark import Benchmark
from .config import Config
from .correspondence import Correspondence
from .experiment import Experiment
from .metrics import certainty_accuracy, explained_share, relative_error
from .object_recognition import ObjectRecognition, ToolRecognition
from .results_analyzer import ResultsAnalyzer
from .shape_classification import ShapeClassification
from .touch_prediction import TouchPrediction

EXPERIMENTS = {cls.name: cls for cls in (ShapeClassification, Correspondence, TouchPrediction, ObjectRecognition, ToolRecognition)}

__all__ = ["Benchmark", "Config", "Experiment", "EXPERIMENTS", "Correspondence", "ObjectRecognition", "ResultsAnalyzer",
           "ShapeClassification", "ToolRecognition", "TouchPrediction", "certainty_accuracy", "explained_share", "relative_error"]
