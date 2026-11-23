"""
What is done to a depth image and to the touch signal before they meet a model.

  ContactMask, DepthFill      operations on a depth image
  SignalScaler                the scalings of the touch signal for training (SIGNAL_SCALERS):
                              StandardScaler, MinMaxScaler
"""

from .contact_mask import ContactMask
from .depth_fill import DepthFill
from .signal_scalers import SIGNAL_SCALERS, MinMaxScaler, QuantileScaler, SignalScaler, StandardScaler

__all__ = ["ContactMask", "DepthFill", "SIGNAL_SCALERS", "SignalScaler", "MinMaxScaler", "StandardScaler"]
