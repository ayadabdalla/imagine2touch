"""
Do the image of the surface and the touch signal go together?  (Paper: Fig. 2; thesis: Fig. 18.)

The paper lays the touches out by their images with t-SNE, groups the points of that layout into
five clusters and shows, for each cluster, the mean image next to the mean touch signal: where the surface is under the sensor, the signal differs.
This experiment puts a number on it.

case "cluster"   one record per cluster: its size, where its mean mask lies, the typical strength of the signal
                 at each magnetometer (the median, signal_) and how that compares with the typical
                 strength over all touches (raised_: 0.2 is a fifth more)
case "explained" the share of the differences between touch signals that the cluster of the image
                 explains, and the same share when images and signals are paired at random
case "whole"     the same share when the whole mask is used, not only its cluster: the signal of a
                 touch is predicted from the touches of the same object with the most similar masks,
                 always touches it is not among
case "side"      whether the signal is stronger on the side of the sensor where the contact lies: the correlation
                 along the sensor's top-bottom axis and along its left-right axis

The strength of the signal at a magnetometer is the one the figure draws (TactileImage.strength):
the change of the field in all three directions.  The direction of touch alone would miss half of
the picture: where the contact lies from left to right shows in the field along the sensor's
surface, not in the field along the touch.

The shares are measured on the signal as the model is given it: every value as its rank among all
touches (the configuration's `model.scaling`).  On the raw values a single glitched reading, two
hundred times the usual size, would hold nearly all of the differences.  For the same reason the
cluster's signal is a median and the correlations are taken between ranks.
"""

import numpy as np
from openTSNE import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from imagine2touch.datasets import TactileImage
from imagine2touch.method.preprocessing import SIGNAL_SCALERS
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

    def explained_by_whole_mask(self, masks, signal, shape, seed):
        """
        The share of the differences between the signals of one object's touches that their masks
        explain: every touch is predicted as the mean signal of the touches of that object with the
        most similar masks, among touches set aside from it, and the error left is compared with the
        differences there were.  Summed over the objects.
        """
        flat = masks.reshape(len(masks), -1).astype(np.float32)
        left, total = 0.0, 0.0
        for name in np.unique(shape):
            rows = np.flatnonzero(shape == name)
            folds = KFold(min(self.settings.folds, len(rows)), shuffle=True, random_state=seed)
            for train, test in folds.split(rows):
                train, test = rows[train], rows[test]
                nearest = KNeighborsRegressor(min(self.settings.neighbours, len(train))).fit(flat[train], signal[train])
                left += ((signal[test] - nearest.predict(flat[test])) ** 2).sum()
            total += ((signal[rows] - signal[rows].mean(0)) ** 2).sum()
        return float(1 - left / total)

    def run(self, seed):
        rng = np.random.default_rng([seed, 1])
        data = self.bench.selection()
        masks, signal = data.images("mask"), data.signal
        labels = self.clusters(masks, seed)
        scaled = SIGNAL_SCALERS[self.bench.config.model.scaling]().fit(signal).transform(signal)
        strength = TactileImage.strength(signal)
        records, usual = [], np.median(strength, axis=0)
        for k in np.unique(labels):
            mean_mask, typical = masks[labels == k].mean(0), np.median(strength[labels == k], axis=0)
            row, col = self.centre(mean_mask)
            records.append(self.record(seed, "cluster", cluster=int(k), touches=int((labels == k).sum()),
                                       filled=float(mean_mask.mean()), row=row, col=col,
                                       **{f"signal_{name}": float(value) for name, value in zip(TactileImage.MAGNETOMETERS, typical)},
                                       **{f"raised_{name}": float(value) for name, value in zip(TactileImage.MAGNETOMETERS, typical / usual - 1)}))
        shuffled = [explained_share(scaled, rng.permutation(labels)) for _ in range(self.settings.shuffles)]
        records.append(self.record(seed, "explained", share=explained_share(scaled, labels), shuffled=float(np.mean(shuffled)),
                                   shuffled_max=float(np.max(shuffled)), touches=len(masks)))
        records.append(self.record(seed, "whole", share=self.explained_by_whole_mask(masks, scaled, np.asarray(data.shape), seed)))
        # where the contact lies against where the signal is stronger, as the sensor is laid out (TactileImage.aligned):
        # the camera image's columns run towards the sensor's top, its rows away from the sensor's right
        rank = np.argsort(np.argsort(strength, axis=0), axis=0)
        z = dict(zip(TactileImage.MAGNETOMETERS, rank.T))
        row, col = np.array([self.centre(mask) for mask in masks]).T
        correlation = lambda contact, signal: float(np.corrcoef(contact, signal)[0, 1])
        records.append(self.record(seed, "side", top_bottom=correlation(col, z["top"] - z["bottom"]),
                                   left_right=correlation(1 - row, z["right"] - z["left"])))
        return records
