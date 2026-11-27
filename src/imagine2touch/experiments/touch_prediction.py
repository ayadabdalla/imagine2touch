"""
How well does the model predict the touch signal?  (Paper: Sec. II and Fig. 1c, 1d; thesis: Fig. 23.)

The paper's model is trained on flower and hexagon.  One record per input: the loss
on the training objects, on the cutter (validation) and on parallelogram, square and star
(recognition).  The loss is the mean squared error in the scaled form the model is trained on, the
quantity the thesis plots; `_average` is the same loss for always predicting the mean training
signal, which a useful model has to beat.  If the configuration sets `data.max_signal`, there is a
second record per input, "cleaned", for which the touches with a larger signal are left out.
"""

import numpy as np

from .experiment import Experiment


class TouchPrediction(Experiment):
    name = "touch_prediction"

    def run(self, seed):
        filtered = self.bench.config.data.max_signal is not None
        return [self.split_errors(seed, kind, cleaned) for kind in self.settings.inputs for cleaned in ((False, True) if filtered else (False,))]

    def split_errors(self, seed, kind, cleaned):
        bench, split = self.bench, self.bench.config.data
        unfiltered = split.max_signal is None           # then both selections hold every touch, and the shared model is the one to score
        model, data = bench.trained_model(seed, kind, cleaned or unfiltered), bench.selection(cleaned)
        images, signal, shape = data.images(kind), model.standardise(data.signal), data.shape
        mean = signal[np.isin(shape, split.train)].mean(0)
        loss = lambda predicted, truth: float(((predicted - truth) ** 2).mean())
        errors = {}
        for name, objects in (("training", split.train), ("validation", split.validation), ("recognition", split.recognition)):
            rows = np.isin(shape, objects)
            errors[f"{name}_error"] = loss(model.standardise(model.predict(images[rows])), signal[rows])
            errors[f"{name}_average"] = loss(mean, signal[rows])
        return self.record(seed, "object split", model=model.architecture, input=kind, epochs=model.epochs_trained,
                           readings="cleaned" if cleaned else "as recorded", **errors)
