"""
Recognising an object by touch (paper: Sec. II, Object Recognition; thesis: Sec. 4.2).

The robot touches an unknown object N times.  For every touch, a method scores each candidate
object; a rule turns the scores of all touches into one decision.

  ObjectRecogniser      collects the touches and names the object
  RecognitionRule       how touches are combined (RECOGNITION_RULES): VoteRule, the paper's;
                        SummedErrorRule
  TouchEvidence         scores candidates by the touch model: felt signal against predicted signals
  ProprioceptionEvidence  the paper's baseline: where the hand stopped against the candidates' scans
"""

from .evidence import ProprioceptionEvidence, TouchEvidence
from .object_recogniser import ObjectRecogniser
from .rules import RECOGNITION_RULES, RecognitionRule, SummedErrorRule, VoteRule

__all__ = ["ObjectRecogniser", "ProprioceptionEvidence", "TouchEvidence", "RECOGNITION_RULES", "RecognitionRule",
           "SummedErrorRule", "VoteRule"]
