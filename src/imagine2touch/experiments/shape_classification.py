"""
Can the sensor tell shapes smaller than itself apart?  (Paper: Table I.)

The stamps are pressed into the mounted sensor (datasets/shape_classification_dataset.py), and a
small network names the stamp from one touch signal.  Data and split are those of the paper's
code: of every stamp 181 presses train the network, 25 are set aside for validation and 50 test
it.  The paper states "trained on 80%", which is the 181 and the 25 together.

case "balanced split" one record per stamp and one for all stamps: the share of test presses that
                      are named correctly

The trained classifier is kept in <results>/models/shape_classifier_seed<seed>.pt and loaded from
there if the file exists.
"""

from .metrics import certainty_accuracy
from imagine2touch.method.models import ShapeClassifier
from .experiment import Experiment


class ShapeClassification(Experiment):
    name = "shape_classification"

    @property
    def available(self):
        return self.bench.stamps.available

    def classifier(self, seed):
        s = self.settings
        return ShapeClassifier(s.network, len(self.bench.stamps.STAMPS), s.epochs, s.lr, s.batch_size, seed)

    def trained_classifier(self, seed, signal, label):
        """The classifier of a seed: loaded from its file, or trained on `signal` and `label` and saved there."""
        path = self.bench.results / "models" / f"shape_classifier_seed{seed}.pt"
        if path.exists():
            print(f"loading the classifier {path}", flush=True)
            return ShapeClassifier.load(path)
        model = self.classifier(seed).fit(signal, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        return model

    def score(self, seed, case, data, architecture, model, signal, label, **more):
        """Records of a trained classifier on a test set: one per stamp in it and one for all."""
        stamps = self.bench.stamps.STAMPS
        probability = model.probabilities(signal)
        named, certain = probability.argmax(1), certainty_accuracy(probability, label, self.settings.certainty)
        records = [self.record(seed, case, data=data, network=architecture, stamp=stamp, presses=int((label == k).sum()),
                               accuracy=float((named[label == k] == k).mean()), certain=float(certain[k]), **more)
                   for k, stamp in enumerate(stamps) if (label == k).any()]
        records.append(self.record(seed, case, data=data, network=architecture, stamp="all", presses=len(label),
                                   accuracy=float((named == label).mean()), certain=float(certain.mean()), **more))
        return records

    def run(self, seed):
        stamps = self.bench.stamps
        signal, label, train, test = stamps.signal, stamps.stamp, stamps.role == "train", stamps.role == "test"
        model = self.trained_classifier(seed, signal[train], label[train])
        records = self.score(seed, "balanced split", "balanced", self.settings.network, model, signal[test], label[test])
        return records
