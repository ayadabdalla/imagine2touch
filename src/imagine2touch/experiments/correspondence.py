"""
Do the image of the surface and the touch signal go together?  (Paper: Fig. 2; thesis: Fig. 18.)

The paper lays the touches out by their images with t-SNE, groups the points of that layout into
five clusters and shows, for each cluster, the mean image next to the mean touch signal: where the surface is under the sensor, the signal differs.
This experiment puts a number on it.

case "cluster"   one record per cluster: its size, where its mean mask lies, and its mean signal
case "explained" the share of the differences between touch signals that the cluster of the image
                 explains, and the same share when images and signals are paired at random
case "side"      the correlation between the side of the mask and the side of the signal
"""

import numpy as np
from openTSNE import TSNE
from sklearn.cluster import KMeans

from imagine2touch.reskin_reading import ReSkinReading
from .metrics import explained_share
from .experiment import Experiment


class Correspondence(Experiment):
    name = "correspondence"

    def clusters(self, masks, seed):
        """The cluster (n,) of every mask, as in the paper's script: the masks laid out by t-SNE, k-means on the layout."""
        flat = masks.reshape(len(masks), -1).astype(float)
        layout = TSNE(n_components=2, perplexity=min(30, (len(flat) - 1) / 3), random_state=seed).fit(flat)
        return KMeans(n_clusters=self.settings.clusters, random_state=seed, n_init=10).fit_predict(np.asarray(layout))

    @staticmethod
    def centre(mask):
        """Where a mask lies: the mean row and column of its filled pixels, each from 0 to 1."""
        rows, cols = np.mgrid[0:mask.shape[0], 0:mask.shape[1]] / (mask.shape[0] - 1)
        weight = max(mask.sum(), 1e-9)
        return float((mask * rows).sum() / weight), float((mask * cols).sum() / weight)

    def run(self, seed):
        rng = np.random.default_rng([seed, 1])
        data = self.bench.selection()
        masks, signal = data.images("mask"), data.signal
        labels = self.clusters(masks, seed)
        records = []
        for k in np.unique(labels):
            mean_mask, mean_signal = masks[labels == k].mean(0), signal[labels == k].mean(0)
            row, col = self.centre(mean_mask)
            records.append(self.record(seed, "cluster", cluster=int(k), touches=int((labels == k).sum()),
                                       filled=float(mean_mask.mean()), row=row, col=col,
                                       **{f"z_{name}": float(mean_signal[3 * i + 2])
                                          for i, name in enumerate(ReSkinReading.MAGNETOMETERS)}))
        shuffled = [explained_share(signal, rng.permutation(labels)) for _ in range(self.settings.shuffles)]
        records.append(self.record(seed, "explained", share=explained_share(signal, labels), shuffled=float(np.mean(shuffled)),
                                   shuffled_max=float(np.max(shuffled)), touches=len(masks)))
        # the side of the mask against the side of the signal (z of top minus bottom, right minus left)
        z = signal[:, 2::3]
        position = np.array([self.centre(mask) for mask in masks])
        sides = (z[:, 2] - z[:, 4], z[:, 1] - z[:, 3])
        best = max(abs(np.corrcoef(position[:, axis], side)[0, 1]) for axis in (0, 1) for side in sides)
        records.append(self.record(seed, "side", correlation=float(best)))
        return records
