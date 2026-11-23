"""What an experiment has to be."""

from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    An experiment turns one seed into records: dictionaries with the experiment's name, the seed, a
    `case` and the numbers measured.  `name` is its key in EXPERIMENTS and its section of the
    configuration, available as `self.settings`.
    """

    name: str

    def __init__(self, bench):
        self.bench, self.settings = bench, getattr(bench.config, self.name)

    @property
    def available(self):
        """Whether the data the experiment needs is there."""
        return self.bench.touches.available

    def record(self, seed, case, **values):
        return dict(experiment=self.name, seed=seed, case=case, **values)

    @abstractmethod
    def run(self, seed):
        """The records of one seed."""
