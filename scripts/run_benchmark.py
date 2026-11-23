"""
Run the offline experiments and write what they measure to results/benchmark/.

    python scripts/run_benchmark.py                              # everything in configs/benchmark.yaml
    python scripts/run_benchmark.py experiments=[correspondence] seeds=[0]
    python scripts/run_benchmark.py --results results/try model.epochs=50

Finished seeds are skipped, so a stopped run continues where it left off.
"""

import argparse

from imagine2touch.experiments import Benchmark, Config
from imagine2touch.paths import CONFIG, RESULTS_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("overrides", nargs="*", help="settings to change, e.g. model.epochs=50")
    args = ap.parse_args()
    Benchmark(Config(args.config, args.overrides), results=args.results).run()
