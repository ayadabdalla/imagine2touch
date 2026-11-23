"""The scalings of the touch signal for training."""

from abc import ABC, abstractmethod

import numpy as np


class SignalScaler(ABC):
    """Maps touch signals (n, 15) to the values a network is trained on, and back."""

    name: str

    @abstractmethod
    def fit(self, signal):
        """Learn `shift` and `scale` from the training signals; returns self."""

    def transform(self, signal):
        return ((np.asarray(signal, float) - self.shift) / self.scale).astype(np.float32)

    def inverse(self, scaled):
        return np.asarray(scaled, float) * self.scale + self.shift

    def comparable(self, signal):
        """Signals in the form in which a felt and a predicted one are compared."""
        return np.asarray(signal, float) / self.scale


class StandardScaler(SignalScaler):
    """Every one of the 15 values to zero mean and unit variance."""

    name = "standard"

    def fit(self, signal):
        signal = np.asarray(signal, float)
        self.shift, self.scale = signal.mean(0), signal.std(0) + 1e-9
        return self


class MinMaxScaler(SignalScaler):
    """All values into [0, 1] by one minimum and one maximum, so the 15 values keep their relative sizes."""

    name = "minmax"

    def fit(self, signal):
        signal = np.asarray(signal, float)
        self.shift, self.scale = signal.min(), signal.max() - signal.min()
        return self


class QuantileScaler(SignalScaler):
    """
    Every one of the 15 values to its rank among the training signals, from 0 (smallest) to 1
    (largest): the scaling of the thesis's training code (a quantile transform).  One extreme
    reading then counts as one rank and cannot squeeze all the others together.
    """

    name = "quantile"

    def fit(self, signal):
        self.sorted = np.sort(np.asarray(signal, float), axis=0)
        self.ranks = np.linspace(0, 1, len(self.sorted))
        return self

    def transform(self, signal):
        signal = np.atleast_2d(np.asarray(signal, float))
        columns = [np.interp(signal[..., j], self.sorted[:, j], self.ranks) for j in range(signal.shape[-1])]
        return np.stack(columns, axis=-1).astype(np.float32)

    def inverse(self, scaled):
        scaled = np.atleast_2d(np.asarray(scaled, float))
        return np.stack([np.interp(scaled[..., j], self.ranks, self.sorted[:, j]) for j in range(scaled.shape[-1])], axis=-1)

    def comparable(self, signal):
        return self.transform(signal).astype(float).reshape(np.shape(signal))


SIGNAL_SCALERS = {cls.name: cls for cls in (StandardScaler, MinMaxScaler, QuantileScaler)}
