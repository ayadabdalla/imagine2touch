"""The evidence of a touch, the rules that combine touches, and the recogniser."""

import numpy as np
import pytest

from imagine2touch.method.recognition import ObjectRecogniser, RECOGNITION_RULES, SummedErrorRule, TouchEvidence, VoteRule


def test_recognition_rules():
    felt = np.zeros((3, 2))
    predicted = np.array([[[0.1, 0], [1, 1], [3, 0]],        # touch 1: candidate 0 is nearest
                          [[0.2, 0], [0.1, 0], [3, 0]],      # touch 2: candidate 1
                          [[0.0, 0], [2, 0], [3, 0]]])       # touch 3: candidate 0
    cost = TouchEvidence.cost(felt, predicted)
    assert cost[0] == pytest.approx([0.01, 2, 9])
    assert TouchEvidence.probability(cost).argmax(1).tolist() == [0, 1, 0]
    assert VoteRule().scores(cost) == pytest.approx([2 / 3, 1 / 3, 0])       # Eq. (2): shares of the votes
    assert VoteRule().winner(cost) == 0 and SummedErrorRule().winner(cost) == 0
    assert VoteRule().scores(np.array([[0.5, 0.5]])) == pytest.approx([0.5, 0.5])     # a tie splits the vote
    assert set(RECOGNITION_RULES) == {"votes", "summed error"}
    # over stacks of trials: candidate 0 wins the first trial and loses the second
    trials = np.stack([cost, cost[:, ::-1]])
    assert VoteRule().win_share(trials, 0) == 0.5 and SummedErrorRule().win_share(trials, 2) == 0.5


def test_object_recogniser_collects_touches():
    recogniser = ObjectRecogniser(["square", "star"])
    for cost in ([0.1, 0.9], [0.2, 0.3], [0.8, 0.1]):
        recogniser.observe(cost)
    assert recogniser.touches == 3 and recogniser.name() == "square"
    assert recogniser.probabilities() == pytest.approx({"square": 2 / 3, "star": 1 / 3})
    with pytest.raises(ValueError):
        recogniser.observe([0.1, 0.2, 0.3])
