"""
Train the Imagine2touch model on the dataset and save it.

    python -m imagine2touch.experiments.scripts.train_model                                              # the paper's model, on contact masks
    python -m imagine2touch.experiments.scripts.train_model --input scan --out results/models/scan.pt    # on depth images rendered from the scans

Trains on the paper's training objects (flower, hexagon) with the recipe of configs/benchmark.yaml
and reports the error on its validation object (cutter).  Use the model with

    from imagine2touch.method.models import TouchModel
    model = TouchModel.load("results/models/imagine2touch.pt")
    signal = model.predict(images)          # images (n, 48, 48) as datasets.IMAGE_INPUTS make them
"""

import argparse
from pathlib import Path

import numpy as np

from imagine2touch.experiments import Benchmark, relative_error
from imagine2touch.method.models import ARCHITECTURES
from imagine2touch.paths import ROOT
from imagine2touch.datasets import IMAGE_INPUTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="mask", choices=sorted(IMAGE_INPUTS))
    ap.add_argument("--architecture", default="mlp", choices=sorted(ARCHITECTURES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=ROOT / "results" / "models" / "imagine2touch.pt")
    args = ap.parse_args()
    bench = Benchmark()
    data, split = bench.selection(), bench.config.data
    images, signal = data.images(args.input), data.signal
    train, held = np.isin(data.shape, split.train), np.isin(data.shape, split.validation)
    model = bench.touch_model(args.seed, args.input, args.architecture).fit(images[train], signal[train])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"{args.out}: trained {model.epochs_trained} epochs on {int(train.sum())} touches; error on "
          f"{', '.join(split.validation)}, relative to its signal: {relative_error(model.predict(images[held]), signal[held]):.3f}")
