""" Implementation of BasicVote """

import numpy as np
from numpy.distutils.system_info import x11_info

from votes.vote import Vote
from utils.transformations import AffineTransform
from utils.optimizers import ShelfOptimizer, Dichotomy, dichotomy, derivate


class BasicVote(Vote):
    NAME2OPT = {
        "shelf": ShelfOptimizer,
        "dichotomy": Dichotomy
    }

    def __init__(self, ratings, mask, voting_rights, voting_resilience=1,
                 transformation_name='standardization'):
        super().__init__(ratings, mask, voting_rights)
        self.voting_resilience = voting_resilience  # W in the paper
        self.transformation = AffineTransform(name=transformation_name)

    def qr_median(self, scores, weights, voting_resilience=None, opt_name="dichotomy"):
        if voting_resilience is None:
            voting_resilience = self.voting_resilience

        delta = 1.
        bounds = ((min(0, min(scores)), max(0, max(scores))),)
        optimizer = BasicVote.NAME2OPT[opt_name](tolerance=1e-9, max_iter=100)
        function = lambda x: 0.5 * voting_resilience * x ** 2 + (weights.T @ np.abs(x - scores)).sum()
        derivative = None
        if opt_name == "dichotomy":
            derivative = lambda x: derivate(x, weights, scores, delta, voting_resilience)
        # if opt_name == "dichotomy", the argument 'function' passed to minimize is useless
        out = optimizer.minimize(function=function, derivative=derivative, bounds=bounds)

        return out

    def run(self):
        """ run voting algorithm """
        # Basic vote normalisation
        for voter in range(self.n_voters):
            self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

        out = np.zeros(self.n_alternatives)

        for alternative in range(self.n_alternatives):
            scores = np.array(
                [x for voter, x in enumerate(self.ratings[:, alternative]) if
                 self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.qr_median(scores, weights)

        return out
