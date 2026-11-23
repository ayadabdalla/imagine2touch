"""
Look at the data.

    python -m imagine2touch.visualization.scripts.visualize_dataset                    # the pictures of the README, into assets/
    python -m imagine2touch.visualization.scripts.visualize_dataset --more             # further pictures, into results/pictures/
    python -m imagine2touch.visualization.scripts.visualize_dataset --touch 1208       # one touch, in a window
    python -m imagine2touch.visualization.scripts.visualize_dataset --touch 1208 --out touch.png

The README's pictures are dataset_overview.png (the objects and where each was touched),
dataset_touches.gif (a walk through touches of every object) and stamps_overview.png (the stamps and
what each does to the sensor).  --more adds two example touches of every object side by side.
"""

import argparse

import matplotlib

from imagine2touch.datasets import ShapeClassificationDataset, VisuoTactileDataset
from imagine2touch.experiments import Config
from imagine2touch.paths import CONFIG, DATA_DIR, ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--touch", type=int, help="draw this one touch (a row of the dataset) instead of the README's pictures")
    ap.add_argument("--out", help="the file for --touch")
    ap.add_argument("--more", action="store_true", help="draw the further pictures")
    ap.add_argument("--config", default=CONFIG, help="the configuration that names the data file and its sessions")
    ap.add_argument("--sessions", nargs="*", help="the sessions to show (default: those of the configuration)")
    args = ap.parse_args()
    if args.touch is None or args.out:          # nothing is shown in a window
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from imagine2touch.visualization import DatasetVisualizer, StampVisualizer
    data = Config(args.config).data
    sessions = args.sessions or data.sessions
    visualizer = DatasetVisualizer(VisuoTactileDataset(DATA_DIR / data.file), sessions, photos=ROOT / "assets" / "objects")
    if args.touch is not None:
        visualizer.touch(args.touch, args.out)
        if not args.out:
            plt.show()
    else:
        visualizer.overview(ROOT / "assets" / "dataset_overview.png")
        visualizer.animation(ROOT / "assets" / "dataset_touches.gif")
        stamps = ShapeClassificationDataset()
        if stamps.available:
            StampVisualizer(stamps, ROOT / "assets" / "stamps").overview(ROOT / "assets" / "stamps_overview.png")
        if args.more:
            folder = ROOT / "results" / "pictures"
            folder.mkdir(parents=True, exist_ok=True)
            visualizer.sessions_overview(sessions, folder / "example_touches.png")
