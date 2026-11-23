"""The measures the experiments report."""

import numpy as np


def relative_error(predicted, signal):
    """Size of the prediction error relative to the size of the signal: 0 is perfect, 1 as wrong as predicting nothing."""
    return float(np.sqrt(((predicted - signal) ** 2).sum() / (signal ** 2).sum()))


def explained_share(signal, groups):
    """Share of the variance of `signal` (n, d) that lies between the groups `groups` (n,)."""
    within = sum(((signal[groups == g] - signal[groups == g].mean(0)) ** 2).sum() for g in np.unique(groups))
    return float(1 - within / ((signal - signal.mean(0)) ** 2).sum())


def certainty_accuracy(probability, label, certainty):
    """
    The measure of the thesis (Eq. 5.1 to 5.3), (classes,): for every class, how often "it is this
    class" is answered correctly over all samples, when a class counts as named above `certainty`.
    """
    named = np.asarray(probability) >= certainty
    truth = np.asarray(label)[:, None] == np.arange(named.shape[1])
    return (named == truth).mean(0)
