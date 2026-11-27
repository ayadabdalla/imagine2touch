"""
The results of a benchmark run: tables to <results>/summary.md and the paper's Fig. 2 to <results>/fig_tsne.

    python scripts/report_benchmark.py [--results results/benchmark]
"""

import argparse
from pathlib import Path

import pandas as pd

from imagine2touch.experiments import Benchmark, Config, ResultsAnalyzer
from imagine2touch.paths import RESULTS_DIR
from imagine2touch.visualization import CorrespondenceFigure

TABLES = [   # title, table of ResultsAnalyzer, which value of a column is the best (highest, lowest or none), what it shows
    ("Shape classification (paper: Table I)", "shape_classification_measures", None,
     "Share of test presses named correctly, per stamp.  At 90% certainty a stamp counts as named only if the "
     "classifier gives it at least 90%, and each of the five yes-or-no answers of a press is scored; this reads "
     "higher, since four of the five answers are an easy no."),
    ("The clusters of the t-SNE figure (paper: Fig. 2)", "clusters", None,
     "The masks are sorted into clusters of similar masks.  Per cluster: where the contact lies on the sensor, and "
     "the strength of the signal at each magnetometer against what is usual there (+20%: a fifth stronger than over "
     "all touches).  A contact to one side should raise the magnetometer on that side.\n\n"
     "- Green text: above usual.\n"
     "- Yellow background: the largest value of the row.\n"
     "- Blue background: the magnetometer on the side where the contact lies, from top to bottom, is above usual.\n"
     "- Purple background: the same from left to right.\n"
     "- Blue or purple with a yellow frame: it is also the largest of the row.\n"),
    ("Does the mask help to guess the touch signal?", "correspondence", None,
     "If you know the mask of a touch, can you guess what the sensor felt?  The signal of each touch is hidden and "
     "guessed as the average signal of other touches.  The table says how much better that guess is than one made "
     "without looking at the mask: 0% no better, 100% exact.  The rows differ in which touches are averaged:\n\n"
     "- **Same cluster**: every touch in the same cluster of the table above, a few hundred loosely similar masks.\n"
     "- **Ten most similar masks**: the ten touches of the same object whose masks are nearly the same picture.\n"
     "- **At random**: touches picked without looking at the mask, to show what no help looks like.\n\n"
     "The last two rows ask something else: is the signal stronger on the side of the sensor where the contact lies?"),
    ("The touch model on the paper's objects (paper: Sec. II and Fig. 1)", "touch_prediction", None,
     "Mean squared error of the scaled signal, per kind of image the model was trained on (rows) and set of "
     "objects (columns).  The mean signal has the lower error on new objects, but the error does not see where on "
     "the sensor the signal lies, and the mean cannot tell objects apart; the next tables measure that."),
    ("Object recognition (paper: Table II), repeated by a simulated robot", "object_recognition", "highest",
     "A simulated robot touches parallelogram, square or star ten times and chooses among the three; twenty trials "
     "per object and seed.  The model is trained on flower and hexagon, on depth images of their scans.  Where the "
     "sensor meets the object, both methods are scored from the real touch recorded nearest.  The paper's "
     "robot made one run of ten touches per object.  The number of trials is in brackets.  "
     "The best value of a column has a yellow background, ties included; the paper's rows, under the dashed line, are "
     "not compared."),
    ("Object recognition, tools (paper: Table II)", ("object_recognition", "tool_recognition"), "highest",
     "The same with the paper's tools: the robot touches the cutter and chooses between cutter and hanger.  The "
     "paper's robot made one run; its Table II averages this run with one on the hanger."),
]

NAMED = "named after all touches"
PAPER = {   # the paper's own numbers, added under a table's rows: title of the table -> (trials behind NAMED, row -> column -> value)
    "Shape classification (paper: Table I)": (None, {
        "paper, Table I": dict(T=0.91, circle=0.90, angle=0.90, triangle=0.95, cross=0.94, all=0.92)}),
    # the votes of the paper's runs, kept in the original repository (reskin/task/results)
    "Object recognition (paper: Table II), repeated by a simulated robot": (3, {
        "paper's robot, touch model": {"parallelogram": 0.4, "square": 0.7, "star": 0.4, "mean per touch": 0.5, NAMED: 1.0},
        "paper's robot, proprioception": {"parallelogram": 0.3, "square": 0.2, "star": 0.2, "mean per touch": 0.233, NAMED: 0.0}}),
    "Object recognition, tools (paper: Table II)": (1, {
        "paper's robot, touch model": {"cutter": 0.6, NAMED: 1.0},
        "paper's robot, proprioception": {"cutter": 1.0, NAMED: 1.0}}),
}


def as_text(table, best=None, skip=("chance",)):
    """
    The table with its numbers as text.  With `best` ("highest" or "lowest") the best value of every
    column gets a coloured background; equal values share it.  The trials behind NAMED follow it in brackets.
    """
    values = table.round(3)
    text = values.map(lambda value: "" if pd.isna(value) else f"{value:g}")
    for column in values.columns.difference(skip) if best else ():
        winner = values[column].max() if best == "highest" else values[column].min()
        text.loc[values[column] == winner, column] = f'<span style="{ResultsAnalyzer.LARGEST}">{winner:g}</span>'
    if NAMED in text and "trials" in table.attrs:
        text[NAMED] += f" ({table.attrs['trials']})"
    return text


def with_paper(text, trials, rows):
    """
    The table of text with the paper's rows under its own; `attrs["dashed"]` names the first of them,
    above which a dashed line is drawn.  They take no part in the comparison above them.  A paper's
    row is left out if the table lacks one of its columns.
    """
    rows = {label: values for label, values in rows.items() if set(values) <= set(text.columns)}
    if rows:
        text.attrs["dashed"] = next(iter(rows))
    for label, values in rows.items():
        text.loc[label] = {column: f"{values[column]:g}" if column in values else text[column].iloc[0] if column == "chance" else "" for column in text.columns}
        if NAMED in values:
            text.loc[label, NAMED] += f" ({trials})"
    return text


def as_html(text):
    """The table of text as an HTML table, which Markdown shows as it is, with a dashed line above the row `attrs["dashed"]`."""
    cell = lambda tag, value, style: f'<{tag} style="{style}">{value}</{tag}>'
    rows = ["<tr>" + cell("th", text.index.name or "", "text-align:left") + "".join(cell("th", column, "text-align:right") for column in text.columns) + "</tr>"]
    for label, values in text.iterrows():
        line = "border-top:2px dashed #888;" if label == text.attrs.get("dashed") else ""
        rows.append("<tr>" + cell("td", label, line + "text-align:left") + "".join(cell("td", value, line + "text-align:right") for value in values) + "</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()
    results = Path(args.results)
    analyzer, lines = ResultsAnalyzer(results), ["# Imagine2touch: results\n"]
    for title, name, best, caption in TABLES:
        name, *arguments = (name,) if isinstance(name, str) else name
        try:
            table = getattr(analyzer, name)(*arguments)
        except FileNotFoundError as missing:
            print(f"skipped '{title}': {missing}")
            continue
        caption = "\n\n".join(text.strip() for text in (caption, table.attrs.get("note")) if text)
        if table.ndim == 1:
            table = table.rename_axis("measure").to_frame("value")
        elif best or title in PAPER:
            table = with_paper(as_text(table, best), *PAPER.get(title, (None, {})))
        shown = as_html(table) if table.attrs.get("dashed") else table.reset_index().round(3).to_markdown(index=False)
        lines += [f"## {title}\n", caption + "\n", shown + "\n"]
    summary = "\n".join(lines)
    print(summary)
    (results / "summary.md").write_text(summary)

    config = Config(results / "config.yaml") if (results / "config.yaml").exists() else Config()
    touches = Benchmark(config, results=results).selection(cleaned=False)        # every reading, as the paper drew it
    CorrespondenceFigure(touches, config.correspondence.figure, config.correspondence.clusters).embed().draw(results / "fig_tsne")
