"""Every script starts and explains itself; the heavier ones are run in the tests of their packages."""

import subprocess
import sys

import pytest

from imagine2touch.paths import ROOT

FILES = ["scripts/run_benchmark.py", "scripts/report_benchmark.py"]
MODULES = ["imagine2touch.experiments.scripts.train_model", "imagine2touch.visualization.scripts.visualize_dataset",
           "imagine2touch.procedures.scripts.calibrate_camera", "imagine2touch.procedures.scripts.scan_object",
           "imagine2touch.procedures.scripts.collect_data", "imagine2touch.procedures.scripts.recognize_object",
           "imagine2touch.datasets.recordings.recording_reducer", "imagine2touch.datasets.recordings.stamp_recording_converter"]


@pytest.mark.parametrize("script", FILES + MODULES)
def test_script_shows_its_help(script):
    command = [sys.executable, str(ROOT / script)] if script in FILES else [sys.executable, "-m", script]
    shown = subprocess.run(command + ["--help"], capture_output=True, text=True, cwd=ROOT)
    assert shown.returncode == 0 and shown.stdout.startswith("usage:") and "options:" in shown.stdout.replace("optional arguments:", "options:")
