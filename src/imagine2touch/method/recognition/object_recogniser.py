"""Naming an object from a sequence of touches."""

import numpy as np

from .rules import VoteRule


class ObjectRecogniser:
    """
        recogniser = ObjectRecogniser(["square", "star", "cylinder"])
        recogniser.observe(cost)            # after every touch: the cost of each candidate
        recogniser.probabilities()          # the candidates' scores so far, by name
        recogniser.name()                   # the best candidate

    The cost of a touch comes from TouchEvidence or ProprioceptionEvidence; `rule` combines the touches.
    """

    def __init__(self, candidates, rule=None):
        self.candidates, self.rule, self.costs = list(candidates), rule or VoteRule(), []

    def observe(self, cost):
        cost = np.asarray(cost, float)
        if cost.shape != (len(self.candidates),):
            raise ValueError(f"expected one cost per candidate, got shape {cost.shape}")
        self.costs.append(cost)

    @property
    def touches(self):
        return len(self.costs)

    def probabilities(self):
        return dict(zip(self.candidates, self.rule.scores(np.array(self.costs)).tolist()))

    def name(self, rng=None):
        return self.candidates[self.rule.winner(np.array(self.costs), rng)]
