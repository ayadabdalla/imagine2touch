"""
The results of a benchmark run: tables to <results>/summary.md and the paper's Fig. 2 to <results>/fig_tsne.

    python scripts/report_benchmark.py [--results results/benchmark]
"""

import argparse
from pathlib import Path

from imagine2touch.experiments import Benchmark, Config, ResultsAnalyzer
from imagine2touch.paths import RESULTS_DIR
from imagine2touch.visualization import CorrespondenceFigure

TABLES = [   # title, table of ResultsAnalyzer, what it shows
    ("Shape classification (paper: Table I)", "shape_classification",
     "Share of test presses named correctly, per stamp: the paper's 1,280 presses, split, network and training.  "
     "Paper: T 0.91, circle 0.90, angle 0.90, triangle 0.95, cross 0.94, all 0.92."),
    ("The clusters of the t-SNE figure (paper: Fig. 2)", "clusters",
     "Per cluster of masks: touches, share of the mask that is filled, where the mask lies (0 to 1 along rows and "
     "columns), and the mean field along the direction of touch at each magnetometer."),
    ("How much of the touch signal the mask explains", "correspondence",
     "Share of the differences between touch signals that the cluster of the mask explains, and the same when "
     "masks and signals are paired at random."),
    ("The touch model on the paper's objects (paper: Sec. II and Fig. 1; thesis: Fig. 23)", "touch_prediction",
     "Trained on flower and hexagon; loss on those, on the cutter (validation) and on parallelogram, square and "
     "star (recognition): the mean squared error of the scaled signal, as the thesis plots it.  `_average` is the "
     "loss of always predicting the mean training signal.  Thesis Fig. 23: 0.06 in training, 0.10 on the three "
     "recognition objects."),
    ("Object recognition (paper: Table II), repeated by a simulated robot", "object_recognition",
     "A simulated robot touches the scan of parallelogram, square or star from the v6 sessions ten times, following the paper's "
     "procedure, once per object.  Every touch counts.  Where the sensor meets the object it feels the signal "
     "recorded at the nearest real touch; where it meets nothing it feels nothing.  Share of single touches that vote "
     "for the touched object, and share of runs that name it.  The paper measured this once on the robot "
     "(primitives, per touch: 50% with the model, 23% with proprioception); those runs were not recorded, so "
     "these numbers are not the paper's."),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()
    results = Path(args.results)
    analyzer, lines = ResultsAnalyzer(results), ["# Imagine2touch: results\n"]
    for title, name, caption in TABLES:
        name, *arguments = (name,) if isinstance(name, str) else name
        try:
            table = getattr(analyzer, name)(*arguments)
        except FileNotFoundError as missing:
            print(f"skipped '{title}': {missing}")
            continue
        table = table.to_frame("value") if table.ndim == 1 else table
        lines += [f"## {title}\n", caption + "\n", table.reset_index().round(3).to_markdown(index=False) + "\n"]
    summary = "\n".join(lines)
    print(summary)
    (results / "summary.md").write_text(summary)

    config = Config(results / "config.yaml") if (results / "config.yaml").exists() else Config()
    touches = Benchmark(config, results=results).selection(cleaned=False)        # every reading, as the paper drew it
    CorrespondenceFigure(touches, config.correspondence.figure, config.correspondence.clusters).embed().draw(results / "fig_tsne")
