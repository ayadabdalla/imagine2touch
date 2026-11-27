"""The benchmark end to end on simulated data, its configuration and its measures."""

import numpy as np
import pandas as pd
import pytest

from imagine2touch.experiments import Benchmark, Config, EXPERIMENTS, ResultsAnalyzer, certainty_accuracy, explained_share, relative_error


SMALL = ["seeds=[0]", "experiments=[shape_classification, correspondence, touch_prediction, object_recognition]",
         "data.sessions=[square_v_1, hexagon_v_1, star_v_1]", "data.train=[square]", "data.validation=[hexagon]",
         "data.recognition=[star, hexagon]", "model.epochs=3", "model.device=cpu", "correspondence.clusters=2",
         "correspondence.shuffles=5", "object_recognition.touches=2", "object_recognition.runs=1", "object_recognition.targets=40", "object_recognition.reach=1.0", "object_recognition.turn=180",
         "shape_classification.epochs=2"]


def test_every_experiment_runs_and_can_be_resumed(touches, stamps, tmp_path, capsys):
    bench = Benchmark(Config(overrides=SMALL), touches, stamps, tmp_path)
    bench.run()
    for name in bench.config.experiments:
        assert len(pd.read_csv(tmp_path / f"{name}.csv")) > 0
    recognition = pd.read_csv(tmp_path / "object_recognition.csv")
    assert set(recognition["shape"]) == {"star", "hexagon"} and set(recognition.rule) == {"touch model", "proprioception"}
    assert recognition.per_touch.between(0, 1).all() and recognition.after_touches.between(0, 1).all()
    analyzer = ResultsAnalyzer(tmp_path)
    assert analyzer.shape_classification().shape[1] == 6 and len(analyzer.touch_prediction()) == 4
    assert len(analyzer.clusters()) == 2 and "chance" in analyzer.object_recognition()
    bench.run()
    assert "already done" in capsys.readouterr().out


def test_measures():
    signal = np.array([[1.0, 0], [1, 0], [0, 1], [0, 1]])
    assert explained_share(signal, np.array([0, 0, 1, 1])) == pytest.approx(1)      # the groups explain everything
    assert explained_share(signal, np.array([0, 1, 0, 1])) == pytest.approx(0)      # or nothing
    sure = np.array([[0.95, 0.05], [0.6, 0.4]])                                     # one certain answer, one not
    assert certainty_accuracy(sure, np.array([0, 0]), 0.9) == pytest.approx([0.5, 1.0])
    assert relative_error(signal, signal) == 0 and relative_error(0 * signal, signal) == pytest.approx(1)


def test_overrides_reach_the_configuration():
    config = Config(overrides=["model.epochs=7", "seeds=[5]"])
    assert config.model.epochs == 7 and config.seeds == [5] and config.raw["model"]["epochs"] == 7
