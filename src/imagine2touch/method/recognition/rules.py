"""How the evidence of several touches is combined into one decision."""

from abc import ABC, abstractmethod

import numpy as np


class RecognitionRule(ABC):
    """Turns costs (n touches, c candidates), lower for a better fit, into a score per candidate, higher for a better one."""

    name: str

    @abstractmethod
    def scores(self, cost):
        """(c,) from (n, c); works on stacks of trials too: (..., c) from (..., n, c)."""

    def winner(self, cost, rng=None):
        """The index of the best candidate; ties are broken at random with `rng`, else by order."""
        scores = self.scores(cost)
        best = np.flatnonzero(scores == scores.max())
        return int(best[0] if rng is None or len(best) == 1 else rng.choice(best))

    def win_share(self, cost, candidate):
        """
        For stacks of trials (t, n, c): the share of trials in which `candidate` has the best score.
        A trial that ends in a tie counts as a win in proportion to the candidates tied.
        """
        scores = self.scores(cost)
        best = scores == scores.max(-1, keepdims=True)
        return float((best[..., candidate] / best.sum(-1)).mean())


class VoteRule(RecognitionRule):
    """
    Eq. (2) of the paper: every touch votes for the candidate it fits best, and a candidate's
    probability is its share of the votes.  One bad touch (a slip, a misplaced object) costs one vote
    and no more, which is why the paper prefers it.  Touches that fit several candidates equally
    split their vote.
    """

    name = "votes"

    def scores(self, cost):
        cost = np.asarray(cost)
        best = cost == cost.min(-1, keepdims=True)
        return (best / best.sum(-1, keepdims=True)).mean(-2)


class SummedErrorRule(RecognitionRule):
    """The candidate with the smallest cost summed over the touches: every touch counts by how clearly it speaks."""

    name = "summed error"

    def scores(self, cost):
        return -np.asarray(cost).sum(-2)


RECOGNITION_RULES = {cls.name: cls() for cls in (VoteRule, SummedErrorRule)}
