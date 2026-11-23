"""Finds the recordings under a folder."""

import re
from pathlib import Path

from .recording import Recording


class RecordingReader:
    """
    `source` holds one sub-folder per object, each with one or more experiments (datasets/recordings/recording.py).
    Several sources may be given; where two hold a folder of the same name, the later one is read.
    """

    def __init__(self, source, objects=None):
        sources = [source] if isinstance(source, (str, Path)) else list(source)
        self.sources, self.objects = [Path(s) for s in sources], objects      # `objects`: read only these folders

    def recordings(self):
        """Every complete recording, loaded, in the order of the folders and experiments."""
        found = []
        named = {path.name: path for source in self.sources for path in sorted(source.iterdir()) if path.is_dir()}
        for folder in [named[name] for name in sorted(named) if self.objects is None or name in self.objects]:
            numbers = sorted({int(match.group(1)) for path in (folder / f"{folder.name}_tactile").glob("experiment_*_reskin")
                              if (match := re.match(r"experiment_(\d+)_reskin$", path.name))})
            first_image = 1
            for number in numbers:
                recording = Recording(folder, number, first_image)
                first_image += recording.image_count
                if recording.complete:
                    found.append(recording.load())
        return found
