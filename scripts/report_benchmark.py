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
     "columns), and the mean size of the change of the field along the direction of touch at each magnetometer."),
    ("How much of the touch signal the mask explains", "correspondence",
     "Share of the differences between touch signals that the cluster of the mask explains, and the same when "
     "masks and signals are paired at random."),
    ("The touch model on the paper's objects (paper: Sec. II and Fig. 1; thesis: Fig. 23)", "touch_prediction",
     "Trained on flower and hexagon; loss on those, on the cutter (validation) and on parallelogram, square and "
     "star (recognition): the mean squared error of the scaled signal, as the thesis plots it.  `_average` is the "
     "loss of always predicting the mean training signal.  Thesis Fig. 23: 0.06 in training, 0.10 on the three "
     "recognition objects."),
    ("Object recognition (paper: Table II), repeated by a simulated robot", "object_recognition",
     "A simulated robot touches the scan of parallelogram, square or star ten times, following the paper's "
     "procedure.  One such round of ten touches is a trial; each object gets twenty trials with each seed's model.  The model is trained on flower and hexagon only, on depth images "
     "of their scans.  Where the sensor meets the object, both methods are scored from the real touch recorded "
     "nearest to it: its signal and its pose.  Share of single touches that vote for the touched object, per "
     "object and on average, and share of trials that name it; means over the seeds.  The paper measured this once "
     "on the robot.  The votes of those runs are kept (original repository, reskin/task/results): for parallelogram, "
     "square and star 40%, 70% and 40% with the model (50% on average) and 30%, 20% and 20% with proprioception "
     "(23%).  The touches themselves were not kept, so those runs cannot be scored again."),
    ("Object recognition, tools (paper: Table II)", ("object_recognition", "tool_recognition"),
     "The same test with the paper's second set of objects, the tools: a box cutter and a hanger, and the "
     "same model.  The robot touches the cutter; the hanger was only scanned, so it cannot be the touched object.  "
     "Chance is 50%.  On the paper's robot (one run per tool): the model voted for the cutter in 60% of the touches and "
     "for the hanger in 80% (Table II: 70%), proprioception in 100% and 60% (Table II: 80%)."),
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
