"""Runs experiments over seeds, keeps what they measure, and holds the data they share."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imagine2touch.datasets import ShapeClassificationDataset, VisuoTactileDataset
from imagine2touch.method.models import TouchModel
from imagine2touch.paths import DATA_DIR, RESULTS_DIR
from imagine2touch.datasets import IMAGE_INPUTS
from .config import Config
from .touch_selection import TouchSelection


class Benchmark:
    """
    An experiment turns one seed into a list of records (experiments/experiment.py).  The benchmark
    writes them to `<results>/<experiment>.csv` after every seed, so a run can be stopped and
    restarted: finished seeds are not repeated.
    """

    def __init__(self, config=None, touches=None, stamps=None, results=RESULTS_DIR):
        self.config = config or Config()
        self.touches = touches or VisuoTactileDataset(DATA_DIR / self.config.data.file)
        self.stamps = stamps or ShapeClassificationDataset()
        self.results = Path(results)
        self._selections, self._models = {}, {}

    # ------------------------------------------------------------------ data and model shared by the experiments
    def selection(self, cleaned=True):
        """
        The touches of the paper: the six objects of its Fig. 1e, every approach.  `cleaned` leaves out
        the reading that is a glitch of the sensor; the paper's code used every reading as recorded.
        """
        if cleaned not in self._selections:
            data = self.config.data
            self._selections[cleaned] = TouchSelection(self.touches, data.sessions, self.results / "cache",
                                                       data.max_signal if cleaned else None)
        return self._selections[cleaned]

    def touch_model(self, seed, kind=None, architecture=None):
        """An untrained TouchModel for images of `kind`, with the paper's recipe (the configuration's `model`)."""
        m = self.config.model
        device = ("cuda" if torch.cuda.is_available() else "cpu") if m.device == "auto" else m.device
        return TouchModel(architecture or m.architecture, m.scaling, m.epochs, m.lr, m.batch_size, m.image_weight,
                          m.patience, seed, device, binary_images=kind is not None and IMAGE_INPUTS[kind].binary)

    def trained_model(self, seed, kind, cleaned=True):
        """
        The model of the paper, trained once and shared by the experiments: the paper's network and
        recipe on the paper's training objects, with images of `kind`.
        """
        key = (seed, kind, cleaned)
        if key not in self._models:
            data = self.selection(cleaned)
            train = np.isin(data.shape, self.config.data.train)
            print(f"training the model on {int(train.sum())} touches ({kind}, {'cleaned' if cleaned else 'as recorded'})", flush=True)
            self._models[key] = self.touch_model(seed, kind).fit(data.images(kind)[train], data.signal[train])
        return self._models[key]

    # ------------------------------------------------------------------ running
    def run(self, names=None, seeds=None):
        from imagine2touch.experiments import EXPERIMENTS
        self.results.mkdir(parents=True, exist_ok=True)
        self.config.save(self.results / "config.yaml")
        for name in names or self.config.experiments:
            experiment, path = EXPERIMENTS[name](self), self.results / f"{name}.csv"
            if not experiment.available:
                print(f"{name}: skipped, its data is missing (see 'Data' in the README)", flush=True)
                continue
            done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["seed"])
            for seed in seeds or self.config.seeds:
                if seed in set(done.seed):
                    print(f"{name}, seed {seed}: already done", flush=True)
                    continue
                records = pd.DataFrame(experiment.run(seed))
                done = records if done.empty else pd.concat([done, records], ignore_index=True)
                done.to_csv(path, index=False)
                print(f"{name}, seed {seed}: {len(records)} records", flush=True)
