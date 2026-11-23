"""The network that names the stamp pressed into the sensor (paper: Table I; thesis: Fig. 10)."""

import numpy as np
import torch
from torch import nn


class ShapeClassifier:
    """
    From one touch signal (15 values) to one of the stamps.

      "deep"   15 - 40 - 100 - 100 - classes, the network of the thesis
      "wide"   15 - 500 - 10 - classes, the network of the paper

    As in the paper's code, the network is given the size of every value of the signal (its absolute
    value), unscaled, and training minimises the cross-entropy with Adam.  `standardise` scales the
    values with the statistics of the training set instead.  `probabilities` returns the softmax,
    (n, classes).
    """

    LAYERS = {"deep": (40, 100, 100), "wide": (500, 10)}

    def __init__(self, architecture="wide", classes=5, epochs=1000, lr=1e-3, batch_size=64, seed=123, standardise=False):
        self.epochs, self.lr, self.batch_size, self.seed, self.standardise = epochs, lr, batch_size, seed, standardise
        torch.manual_seed(seed)
        sizes, layers = (15, *self.LAYERS[architecture]), []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.network = nn.Sequential(*layers, nn.Linear(sizes[-1], classes))

    def _scaled(self, signal):
        signal = np.abs(np.asarray(signal, float))
        return torch.as_tensor((signal - self.mean) / self.std if self.standardise else signal, dtype=torch.float32)

    def fit(self, signal, label):
        self.mean, self.std = np.abs(signal).mean(0), np.abs(signal).std(0) + 1e-9
        X, y = self._scaled(signal), torch.as_tensor(np.asarray(label), dtype=torch.long)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        generator = torch.Generator().manual_seed(self.seed)
        self.network.train()
        for _ in range(self.epochs):
            for batch in torch.randperm(len(X), generator=generator).split(self.batch_size):
                optimiser.zero_grad()
                nn.functional.cross_entropy(self.network(X[batch]), y[batch]).backward()
                optimiser.step()
        return self

    def probabilities(self, signal):
        self.network.eval()
        with torch.no_grad():
            return torch.softmax(self.network(self._scaled(signal)), 1).numpy()

    def predict(self, signal):
        return self.probabilities(signal).argmax(1)
