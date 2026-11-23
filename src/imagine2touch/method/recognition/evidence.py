"""What one touch says about each candidate object: a cost per candidate, lower for a better fit."""

import numpy as np


class TouchEvidence:
    """
    The paper's method: the model predicts the signal each candidate would give at the place of the
    touch; a candidate's cost is the squared distance of its prediction to the signal that was felt,
    both standardised (Eq. 1 is exp(-cost)).
    """

    @staticmethod
    def cost(felt, predicted):
        """(n, c) for felt signals (n, d) and the predictions for every candidate (n, c, d)."""
        return ((np.asarray(predicted) - np.asarray(felt)[:, None]) ** 2).sum(-1)

    @classmethod
    def from_model(cls, model, felt, images):
        """
        (n, c) for felt signals (n, d) and the image every candidate shows at each touch (n, c, h, w):
        the model predicts a signal for every image, and both sides are put on the model's scale.
        """
        images = np.asarray(images)
        n, c = images.shape[:2]
        predicted = model.predict(images.reshape(n * c, *images.shape[2:])).reshape(n, c, -1)
        return cls.cost(model.standardise(felt), model.standardise(predicted))

    @staticmethod
    def probability(cost):
        """Eq. (1): P(o = o' | touch) = exp(-(felt - predicted)^2)."""
        return np.exp(-np.asarray(cost))


class ProprioceptionEvidence:
    """
    The paper's baseline, the least a robot can sense: a candidate's cost is the distance between
    the place where the hand stopped and the nearest point of that candidate's 3D model.
    """

    @staticmethod
    def cost(position, scans):
        """(c,) for one stopped position and the candidates' ObjectScans."""
        return np.array([scan.distance_to(position) for scan in scans])
