"""
How well does the model predict the touch signal?  (Paper: Sec. II and Fig. 1c, 1d; thesis: Fig. 23.)

The paper's model is trained on flower and hexagon.  One record per input and per way of reading
the data ("as recorded", or "cleaned" by the filter of the configuration, if one is set): the loss
on the training objects, on the cutter (validation) and on parallelogram, square and star
(recognition).  The loss is the mean squared error in the scaled form the model is trained on, the
quantity the thesis plots; `_average` is the same loss for always predicting the mean training
signal, which a useful model has to beat.
"""

import numpy as np

from .experiment import Experiment


class TouchPrediction(Experiment):
    name = "touch_prediction"

    def run(self, seed):
        return [self.split_errors(seed, kind, cleaned) for kind in self.settings.inputs for cleaned in (False, True)]

    def split_errors(self, seed, kind, cleaned):
        bench, split = self.bench, self.bench.config.data
        model, data = bench.trained_model(seed, kind, cleaned), bench.selection(cleaned)
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
