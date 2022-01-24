""" Implementation of BasicVote """

import numpy as np

from votes.vote import Vote
from utils.transformations import AffineTransform
from utils.optimizers import dichotomy

class BasicVote(Vote):

    def __init__(self, ratings, mask, voting_rights, voting_resilience=1,  
                 transformation_name='standardization'):
        super().__init__(ratings, mask, voting_rights)
        self.voting_resilience = voting_resilience  # W in the paper
        self.transformation = AffineTransform(name=transformation_name)

    def qr_median(self, scores, weights, voting_resilience=None):
        if voting_resilience is None:
            voting_resilience = self.voting_resilience
        delta = 1

        bnds = (min(0, min(scores)), max(0, max(scores)))
        out2 = dichotomy(weights, scores, delta=delta, voting_resilience=self.voting_resilience, bnds=bnds)

        return out2

    def run(self):
        """ run voting algorithm """
        # Basic vote normalisation
        for voter in range(self.n_voters):
            self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

        out = np.zeros(self.n_alternatives)

        for alternative in range(self.n_alternatives):
            scores = np.array(
                [x for voter, x in enumerate(self.ratings[:, alternative]) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.qr_median(scores, weights)

        return out