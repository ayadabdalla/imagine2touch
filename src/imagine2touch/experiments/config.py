"""The settings of a benchmark run."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from imagine2touch.paths import CONFIG


class Config:
    """
    A YAML file as nested attributes: `config.model.epochs`.  `overrides` are strings such as
    'model.epochs=50' or 'seeds=[0]' that replace single settings, as on the command line.
    """

    def __init__(self, path=CONFIG, overrides=()):
        self.raw = yaml.safe_load(Path(path).read_text())
        for item in overrides:
            key, value = item.split("=", 1)
            *parents, last = key.split(".")
            node = self.raw
            for parent in parents:
                node = node[parent]
            node[last] = yaml.safe_load(value)
        self.__dict__.update(vars(self._namespace(self.raw)))

    @classmethod
    def _namespace(cls, value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: cls._namespace(item) for key, item in value.items()})
        return value

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.raw, sort_keys=False))
