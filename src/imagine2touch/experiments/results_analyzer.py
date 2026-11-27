"""
The results of a benchmark run as tables.

    analyzer = ResultsAnalyzer("results/benchmark")
    analyzer.touch_prediction()                  # one of the tables below
    analyzer.records("object_recognition")       # or the records themselves

Numbers are means over the seeds.
"""

from pathlib import Path

import pandas as pd

from imagine2touch.datasets import IMAGE_INPUTS, ShapeClassificationDataset, TactileImage
from imagine2touch.paths import RESULTS_DIR
from .config import Config


class ResultsAnalyzer:
    def __init__(self, results=RESULTS_DIR):
        self.dir = Path(results)

    def records(self, experiment, case=None):
        """The records of an experiment as a table, without the columns that are empty for `case`."""
        path = self.dir / f"{experiment}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run `python scripts/run_benchmark.py` first")
        table = pd.read_csv(path)
        if case is not None:
            table = table[table.case == case].dropna(axis=1, how="all")
        return table

    def shape_classification(self, measure="accuracy"):
        """
        Table I of the paper: a column per stamp and one for all stamps.  `measure` is "accuracy", the
        share of presses named correctly, or "certain", the measure of the thesis.
        """
        table = self.records("shape_classification")
        return table.pivot_table(index="network", columns="stamp", values=measure)[[*ShapeClassificationDataset.STAMPS, "all"]]

    RAISED, LARGEST = "color:#2e9e5b", "background:#e0a100;color:#111;font-weight:bold;padding:0 0.35em;border-radius:3px"
    GROUND = "color:#fff;font-weight:bold;padding:0 0.35em;border-radius:3px;background:"
    AGREEING = dict(vertical=GROUND + "#2a78d6", horizontal=GROUND + "#7a5fc4")       # the contact's place from top to bottom, and from left to right

    @classmethod
    def marked(cls, value, largest, axis=None):
        """
        A share as a percentage for a Markdown table: coloured if above zero, on a coloured ground if
        the largest of its row.  With `axis`, "vertical" or "horizontal", it gets that axis's
        ground instead, framed in the first colour if it is also the largest.
        """
        text = f"{100 * value:+.0f}%"
        style = cls.AGREEING[axis] + (";outline:2px solid #e0a100" if largest else "") if axis else cls.LARGEST if largest else cls.RAISED if value > 0 else None
        return f'<span style="{style}">{text}</span>' if style else text

    @staticmethod
    def place(position, low, high):
        """
        Where along one axis of the sensing surface a point lies, in words; `position` runs from 0
        (`low`) to 1 (`high`).  Within 3.5% of the middle it is "centre", within 10% "slightly" to a side.
        """
        off = position - 0.5
        side = low if off < 0 else high
        return "centre" if abs(off) < 0.035 else f"slightly {side}" if abs(off) < 0.1 else side

    def shape_classification_measures(self):
        """Table I by both measures, one row each: the share of presses named correctly, and the same when a stamp counts as named only above the configuration's certainty."""
        saved = self.dir / "config.yaml"
        certainty = f"{100 * Config(saved).shape_classification.certainty:.0f}%" if saved.exists() else "the set"
        rows = {"this repository, accuracy": "accuracy", f"this repository, accuracy at {certainty} certainty": "certain"}
        return pd.concat([self.shape_classification(measure).set_axis([label]) for label, measure in rows.items()]).rename_axis("measure")

    def clusters(self, seed=None):
        """
        Fig. 2 of the paper in numbers, one row per cluster of masks: how many touches it holds, how much
        of the patch under the sensor is in contact, where on the sensor that contact lies, and how
        the typical signal at each magnetometer (the median over the cluster's touches) compares with
        the typical signal there over all touches; values above the usual are coloured, the largest of a row
        most, and those at the magnetometer on the side the contact lies at in the colour of that axis's title.  The place is
        given as the sensor is laid out (TactileImage), so that it can be read against the names of
        the magnetometers.
        """
        table = self.records("correspondence", "cluster")
        table = table[table.seed == (table.seed.min() if seed is None else seed)].sort_values("filled")
        down, across = 1 - table.col, 1 - table.row             # the camera's image turned as TactileImage.aligned turns it
        raised = table[[f"raised_{name}" for name in TactileImage.MAGNETOMETERS]].to_numpy()
        vertical, horizontal = [self.place(d, "top", "bottom") for d in down], [self.place(a, "left", "right") for a in across]
        titled = lambda title, axis: f'<span style="{self.AGREEING[axis]}">{title}</span>'
        clusters = pd.DataFrame({"touches": table.touches.astype(int).to_numpy(),
                                 "patch in contact": [f"{100 * share:.0f}%" for share in table.filled],
                                 titled("contact, top to bottom", "vertical"): vertical,
                                 titled("contact, left to right", "horizontal"): horizontal},
                                index=pd.Index(table.cluster.astype(int), name="cluster"))
        for k, name in enumerate(TactileImage.MAGNETOMETERS):       # a value agrees if it is above usual at the magnetometer on the contact's side
            agreeing = [("vertical" if name == v.replace("slightly ", "") else "horizontal" if name == h.replace("slightly ", "") else None)
                        if value > 0 else None for value, v, h in zip(raised[:, k], vertical, horizontal)]
            clusters[f"signal at {name}"] = [self.marked(value, value == row.max(), axis) for value, row, axis in zip(raised[:, k], raised, agreeing)]
        usual = table[[f"signal_{name}" for name in TactileImage.MAGNETOMETERS]].to_numpy()[0] / (1 + raised[0])
        clusters.attrs["note"] = "Usual strength: " + ", ".join(f"{value:.1f} {name}" for name, value in zip(TactileImage.MAGNETOMETERS, usual)) + "."
        return clusters

    def correspondence(self):
        """By how much knowing the mask reduces the error of guessing a touch signal: its cluster alone, the whole mask, and chance."""
        table = self.records("correspondence")
        explained, whole, side = (table[table.case == case] for case in ("explained", "whole", "side"))
        percent = lambda share: f"{100 * share:.1f}%"
        groups = table[table.case == "cluster"].cluster.nunique()
        return pd.Series({"touches": int(explained.touches.iloc[0]),
                          f"guess improved by: same cluster (one of {groups})": percent(explained.share.mean()),
                          "guess improved by: ten most similar masks": percent(whole.share.mean()),
                          "guess improved by: at random":
                              f"{percent(explained.shuffled.mean())} on average, {percent(explained.shuffled_max.max())} at most",
                          "signal stronger where the contact lies, top to bottom (0 never, 1 always)": round(side.top_bottom.mean(), 2),
                          "signal stronger where the contact lies, left to right": round(side.left_right.mean(), 2)})

    def touch_prediction(self):
        """
        The model's error, one row per kind of image it was trained on and one column per set of
        objects it was tested on: those it was trained on, the validation object and the recognition
        objects.  The last row is the error of always predicting the mean training signal.  Where
        `data.max_signal` is set, the readings as recorded and the cleaned ones are told apart.
        """
        table = self.records("touch_prediction")
        tested = dict(training="the objects it was trained on", validation="the validation object", recognition="the recognition objects")
        saved = self.dir / "config.yaml"
        if saved.exists():
            data = Config(saved).data
            tested = {role: f"{', '.join(names)} ({note})" for role, names, note in
                      (("training", data.train, "trained on"), ("validation", data.validation, "not trained on"),
                       ("recognition", data.recognition, "not trained on"))}
        keys = ["input", "readings"] if table.readings.nunique() > 1 else ["input"]
        means = table.groupby(keys, sort=False)[[f"{role}_{kind}" for role in tested for kind in ("error", "average")]].mean().reset_index()
        model = means.assign(input=[IMAGE_INPUTS[name].described for name in means.input])
        model = model.rename(columns={"input": "model trained on", **{f"{role}_error": objects for role, objects in tested.items()}})
        mean = means.groupby(keys[1:], sort=False).first().reset_index() if keys[1:] else means.iloc[:1]
        mean = mean.assign(input="no model: always the mean signal")
        mean = mean.rename(columns={"input": "model trained on", **{f"{role}_average": objects for role, objects in tested.items()}})
        columns = ["model trained on", *keys[1:], *tested.values()]
        return pd.concat([model[columns], mean[columns]]).set_index(columns[:len(keys)])

    def object_recognition(self, experiment="object_recognition"):
        """
        Table II of the paper, repeated by a simulated robot: per method, the share of single touches that
        vote for the touched object, per object and, with several objects, on average, and the share of runs that name it.
        `experiment` may also be "tool_recognition".
        """
        table = self.records(experiment)
        per = table.pivot_table(index="rule", columns="shape", values="per_touch")
        if per.shape[1] > 1:            # with a single touched object the mean would repeat its column
            per["mean per touch"] = per.mean(axis=1)
        per["named after all touches"] = table.groupby("rule").after_touches.mean()
        per["chance"] = table.chance.iloc[0]
        per = per.reindex([rule for rule in ("touch model", "proprioception") if rule in per.index]).rename_axis("method")
        saved = self.dir / "config.yaml"
        if saved.exists():          # the trials behind "named after all touches": the configuration's runs for every object and seed
            per.attrs["trials"] = int(Config(saved).object_recognition.runs * (table.rule == table.rule.iloc[0]).sum())
        return per
