"""Where the repository keeps its data and results.  Set I2T_DATA or I2T_RESULTS to move them."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "benchmark.yaml"
DATA_DIR = Path(os.environ.get("I2T_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("I2T_RESULTS", ROOT / "results" / "benchmark"))
