"""A set of touches that experiments share, with the images made from them."""

from pathlib import Path

import numpy as np

from imagine2touch.datasets import IMAGE_INPUTS


class TouchSelection:
    """
    The touches of some sessions of a VisuoTactileDataset.
    Touches in which a value exceeds `max_signal` are left out, as the thesis's training code did
    (`dropped` says how many).  With the paper's limit of 50 that is about a fifth of the touches, among
    them one reading that is a glitch of the sensor, a hundred times larger than any other.

      rows        their indices in the dataset
      sessions    the sessions, as indices into dataset.objects
      signal      (n, 15) their touch signals
      shape       (n,) the name of the touched shape
      images(kind, session)   their images, as the ImageInput `kind` makes them

    Images rendered from scans take minutes; the depth images are rendered once and kept in `cache`,
    and masks are made from them.
    """

    def __init__(self, dataset, sessions, cache=None, max_signal=None):
        self.dataset, self.sessions, self.cache = dataset, dataset.sessions(sessions), cache
        keep = dataset.in_sessions(sessions)
        self.glitches = np.zeros(len(dataset), bool) if max_signal is None else np.abs(dataset.signal).max(1) > max_signal
        self.dropped = int((keep & self.glitches).sum())
        self.rows = np.flatnonzero(keep & ~self.glitches)
        self.tag = sessions if isinstance(sessions, str) else '-'.join(sessions)
        self._images = {}

    def __len__(self):
        return len(self.rows)

    @property
    def signal(self):
        return self.dataset.signal[self.rows].astype(float)

    @property
    def shape(self):
        return np.array(self.dataset.shapes)[self.dataset.shape[self.rows]]

    def images(self, kind, session=None):
        key = (kind, session)
        if key not in self._images:
            source = IMAGE_INPUTS[kind]
            if kind == "scan":
                self._images[key] = self._rendered(source, session)
            elif kind == "scan_mask":
                depth = self.images("scan", session) * self.dataset.grid.empty
                self._images[key] = source.from_depth(depth, self.dataset.grid)
            else:
                self._images[key] = source.build(self.dataset, self.rows)
        return self._images[key]

    def _rendered(self, source, session):
        name = "true" if session is None else self.dataset.objects[session]
        path = None if self.cache is None else Path(self.cache) / f"scan_{self.tag}_{name}.npy"
        if path is not None and path.exists() and len(np.load(path, mmap_mode="r")) == len(self.rows):
            return np.load(path)
        print(f"rendering the scan of {name} at {len(self.rows)} touches", flush=True)
        images = source.build(self.dataset, self.rows, session)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, images)
        return images
