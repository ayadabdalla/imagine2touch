"""
The results of a benchmark run as tables.

    analyzer = ResultsAnalyzer("results/benchmark")
    analyzer.touch_prediction()                  # one of the tables below
    analyzer.records("object_recognition")       # or the records themselves

Numbers are means over the seeds.
"""

from pathlib import Path

import pandas as pd

from imagine2touch.datasets import ShapeClassificationDataset
from imagine2touch.paths import RESULTS_DIR


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

    def clusters(self, seed=None):
        """Fig. 2 of the paper in numbers, per cluster of masks: touches, share of the mask that is filled, where it lies, mean signal."""
        table = self.records("correspondence", "cluster")
        table = table[table.seed == (table.seed.min() if seed is None else seed)]
        return table.drop(columns=["experiment", "case", "seed"]).set_index("cluster").sort_values("filled")

    def correspondence(self):
        """How much of the difference between touch signals the image explains, against random pairings."""
        explained = self.records("correspondence", "explained")
        side = self.records("correspondence", "side")
        return pd.Series({"touches": int(explained.touches.iloc[0]),
                          "explained by the cluster of the image": explained.share.mean(),
                          "explained when paired at random": explained.shuffled.mean(),
                          "largest share at random": explained.shuffled_max.max(),
                          "correlation of the sides": side.correlation.mean()})

    def touch_prediction(self):
        """
        The model's error on the training objects, the validation object and the recognition objects,
        each beside the error of the mean training signal (`_average`).
        """
        table = self.records("touch_prediction")
        columns = [f"{name}_{kind}" for name in ("training", "validation", "recognition") for kind in ("error", "average")]
        return table.groupby(["input", "readings"])[columns].mean()

    def object_recognition(self):
        """
        Table II of the paper, repeated by a simulated robot: per method, the share of single touches that
        vote for the touched object, per object and on average, and the share of runs that name it.
        """
        table = self.records("object_recognition")
        per = table.pivot_table(index="rule", columns="shape", values="per_touch")
        per["mean per touch"] = per.mean(axis=1)
        per["named after all touches"] = table.groupby("rule").after_touches.mean()
        per["chance"] = table.chance.iloc[0]
        return per.sort_values("mean per touch", ascending=False)
