""" Implementation of BasicVote """
from votes.vote import Vote
from utils.optimizers import ShelfOptimizer
import numpy as np


class BasicVote(Vote):
    def __init__(self, n_alternatives, n_voters, density, p_byzantine=.05, byz_density=.5, voting_resilience=1.,
                 transformation_name="standardization", random_mask=False, seed=123):
        super().__init__(n_alternatives=n_alternatives, n_voters=n_voters, density=density, p_byzantine=p_byzantine,
                         transformation_name=transformation_name, byz_density=byz_density, random_mask=random_mask,
                         seed=seed)
        self.voting_resilience = voting_resilience
        self.voting_rights = np.ones(self.n_voters)

    def qr_median(self, scores, weights, voting_resilience=None):
        if voting_resilience is None:
            voting_resilience = self.voting_resilience
        bnds = ((min(0, min(scores)), max(0, max(scores))),)
        f = lambda x: 0.5 * voting_resilience * x ** 2 + (weights.T @ np.abs(x - scores)).sum()
        opt = ShelfOptimizer(tolerance=1e-12, max_iter=10000)
        out = opt.minimize(f, bnds)

        return out

    def generate_voting_rights(self):
        self.voting_rights = self.rng.lognormal(1., 0.5, self.n_voters)
        self.voting_rights = [(i + 1 if x == 0 else x) for i, x in enumerate(self.voting_rights)]

        byzantine = -1
        total_honest_rights = sum(self.voting_rights[:-1])
        self.voting_rights[byzantine] = total_honest_rights * self.p_byzantine / (1-self.p_byzantine)

    def generate_data(self):
        self.generate_voting_rights()
        ratings, original_preferences, _ = self.generate_data_from_unanimity()
        return ratings, original_preferences

    def run(self, ratings=None, original_preferences=None, voting_rights=None, mask=None, generate_data=True):
        if generate_data:
            ratings, original_preferences = self.generate_data()
        elif voting_rights is not None and mask is not None:
            self.voting_rights = voting_rights
            self.mask = mask
        out = np.zeros(self.n_alternatives)

        for alternative in range(self.n_alternatives):
            scores = np.array(
                [x for voter, x in enumerate(ratings[:, alternative]) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.qr_median(scores, weights)

        print("ratings: {}".format(ratings))

        return out, original_preferences
