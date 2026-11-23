"""Recognising an object by touching it, with the real robot or the simulated one (paper: Sec. II and Table II)."""

import numpy as np
from scipy.spatial import cKDTree

from imagine2touch.datasets import IMAGE_INPUTS
from imagine2touch.geometry import ObjectScan, Pose, SensorGrid
from imagine2touch.method.recognition import ObjectRecogniser, ProprioceptionEvidence, SummedErrorRule, TouchEvidence
from .contact_approach import ApproachSettings


class TouchingRecogniser:
    """
    The robot knows the scans of a set of candidate objects and touches the one in front of it, as
    the paper's robot did (scripts/recognize_object_online.py at commit c2ea2c3):

      1. It holds a guess: at first any candidate, then the one that fits the touches so far best.
      2. It aims the next touch at a point of the guessed object's scan, preferring points where the
         other candidates' surfaces face another way: a touch there tells them apart.
      3. It moves there.  If the guess is wrong, the sensor meets the real object somewhere else on
         the way, or nothing at all.
      4. Every candidate gets a cost for the touch, by `evidence`: "touch" compares the felt signal
         with what the model predicts for the candidate where the sensor stopped; "proprioception",
         the paper's baseline, takes the distance from there to the candidate's scan.
      5. After N touches the rule (by default the paper's votes) names the object.

    `collector` does the touching; `candidates` and `normals` map a name to that object's scan and
    its surface normals in the world frame [m]; `model` is a TouchModel trained on images of kind
    `image_input`, "scan" or "scan_mask".  The robot aims at `per_candidate` points of each scan.
    `accept`, if given, is asked for every touch's record whether the touch counts; a refused touch is aimed again.
    """

    FACING = (0.2, 1.0)         # a point counts where the other surface faces between these cosines of its own direction
    LEAST = -1.3                # and where the cosines of all other candidates sum to less than this, negated
    APPROACH = ApproachSettings(step=0.002, max_force=7.0, tolerance=0.005)     # the approach of the paper's recognition script

    def __init__(self, collector, candidates, normals, model=None, image_input="scan_mask", evidence="touch", rule=None,
                 grid=None, per_candidate=300, accept=None, rng=None):
        self.collector, self.model, self.evidence, self.accept = collector, model, evidence, accept
        self.grid, self.rng = grid or SensorGrid(size=48), rng or np.random.default_rng()
        self.image_input = IMAGE_INPUTS[image_input]
        self.names = list(candidates)
        collector.approach.settings = self.APPROACH
        to_base = collector.base_from_world
        self.scans = [ObjectScan.from_raw_points(1000 * to_base.apply(candidates[name])) for name in self.names]
        self.recogniser, self.guesses = ObjectRecogniser(self.names, rule), []
        unit = lambda vectors: np.asarray(vectors, float) / np.linalg.norm(vectors, axis=1, keepdims=True)
        clouds = [np.asarray(candidates[name], float) for name in self.names]
        directions = [unit(normals[name]) for name in self.names]
        # the collector aims at points of one cloud: all candidates, one after the other
        self.first = np.cumsum([0] + [len(cloud) for cloud in clouds])
        collector.cloud, collector.normals = np.concatenate(clouds), np.concatenate(directions)
        self.targets = [self._targets(k, clouds, directions, per_candidate) for k in range(len(clouds))]

    def _targets(self, k, clouds, directions, per_candidate):
        """
        For candidate k: the scan points the robot may aim at, as indices into the collector's cloud,
        and how likely each is drawn.  A point is safe to touch and is weighed by how differently the
        other candidates' surfaces face there.
        """
        safety, half_side = self.collector.safety, self.collector.half_side
        points, facing = clouds[k], directions[k]
        some = np.arange(len(points))[::max(len(points) // per_candidate, 1)]
        safe = np.array([i for i in some if safety.can_touch(points, points[i], facing[i], half_side)], int)
        if len(safe) == 0:
            return safe, None
        alignment, defined = np.zeros(len(safe)), np.ones(len(safe), bool)
        for other in range(len(clouds)):
            if other == k:
                continue
            nearest = cKDTree(clouds[other]).query(points[safe])[1]
            cosine = (directions[other][nearest] * facing[safe]).sum(1)
            defined &= (cosine > self.FACING[0]) & (cosine < self.FACING[1])
            alignment -= cosine
        telling = defined & (alignment > self.LEAST) & (alignment < 0)
        if telling.sum() < 2:                          # nowhere do the surfaces differ in this way: any safe point
            return self.first[k] + safe, None
        weight = alignment[telling] - alignment[telling].min()
        return self.first[k] + safe[telling], weight / weight.sum() if weight.sum() > 0 else None

    # ------------------------------------------------------------------ one touch
    def guess(self):
        """The index of the candidate that fits the touches so far best; before the first touch, any."""
        if self.recogniser.touches == 0:
            return int(self.rng.integers(len(self.names)))
        return SummedErrorRule().winner(np.array(self.recogniser.costs), self.rng)

    def images(self, pose):
        """The image of every candidate for a sensor at `pose` (base frame, m): (c, 48, 48)."""
        pose_mm = Pose(pose.R, 1000 * pose.t)
        depth = np.stack([self.grid.depth_of(scan, pose_mm) for scan in self.scans])
        if self.image_input.binary:
            return self.image_input.from_depth(depth, self.grid)
        return (depth / self.grid.empty).astype(np.float32)

    def cost(self, record):
        """Takes the record of a touch.  Returns the cost of every candidate, (c,)."""
        stopped = Pose.from_matrix(record["pose"])
        if self.evidence == "proprioception":
            return ProprioceptionEvidence.cost(1000 * stopped.t, self.scans)
        felt = record["reading"] - record["ambient"]
        return TouchEvidence.from_model(self.model, felt[None], self.images(stopped)[None])[0]

    def touch(self):
        """Touch once, aiming at the guessed object.  Returns whether the touch could be used."""
        guess = self.guess()
        points, weights = self.targets[guess]
        if len(points) == 0:
            return False
        record = self.collector.touch(int(self.rng.choice(points, p=weights)))
        if record is None or not np.isfinite(record["reading"] - record["ambient"]).all():      # skipped, or no answer
            return False
        if self.accept is not None and not self.accept(record):
            return False
        self.guesses.append(self.names[guess])
        self.recogniser.observe(self.cost(record))
        return True

    def recognise(self, touches=10, attempts=None):
        """Touch `touches` times and name the object.  Returns (name, probabilities by name)."""
        attempts = attempts or 10 * touches
        while self.recogniser.touches < touches and attempts > 0:
            attempts -= 1
            self.touch()
        return self.recogniser.name(self.rng), self.recogniser.probabilities()
