""" Votes class """
import numpy as np
from numpy.random import default_rng
from utils.transformations import AffineTransform


class Vote:
    def __init__(self, n_alternatives, n_voters, density, p_byzantine=.05, byz_density=.5,
                 transformation_name="standardization", random_mask=False, seed=123):
        self.n_alternatives = n_alternatives
        self.n_voters = n_voters
        self.rng = default_rng(seed)
        self.density = density
        self.p_byzantine = p_byzantine
        self.byz_density = byz_density
        self.random_mask = random_mask
        self.transformation = AffineTransform(name=transformation_name)
        self.mask = np.zeros((self.n_voters, self.n_alternatives))

    def run(self):
        return

    def regularize_mask(self, mask):
        n, m = mask.shape
        mask[:, :2] = np.ones((n, 2))
        for j in range(m):
            if np.all(mask[:, j] == 0):
                a = self.rng.integers(0, n)
                mask[a, j] = 1
        return mask

    def generate_mask(self, random=False):
        if random:
            mask = self.rng.binomial(1, self.density, (self.n_voters, self.n_alternatives))
            mask = self.regularize_mask(mask)

        else:
            mask = np.zeros((self.n_voters, self.n_alternatives))
            mask[:, :2] = np.ones((self.n_voters, 2))
            for j in range(2, self.n_alternatives):
                mask[(j - 2) // ((self.n_alternatives-2) // (self.n_voters-1)), j] = 1

            byzantine = -1
            mask[byzantine, 2:] = self.rng.binomial(1, self.byz_density, (1, self.n_alternatives-2))

        self.mask = mask
        return

    def generate_data_from_unanimity(self, mehestan=False):
        #TODO: Have the common comparison preference pair be random (now it's 0,1 always)
        self.generate_mask(random=self.random_mask)
        print("mask: {}".format(self.mask))
        original_preferences = self.rng.standard_cauchy(self.n_alternatives)
        original_preferences = (original_preferences - original_preferences.min()) / (
                    original_preferences.max() - original_preferences.min())
        original_preferences = np.sort(original_preferences)
        ratings = np.zeros((self.n_voters, self.n_alternatives))
        bv_ratings = np.zeros((self.n_voters, self.n_alternatives))

        byzantine = self.n_voters-1
        for voter in range(self.n_voters):
            if voter == byzantine:
                ratings[voter] = self.rng.standard_cauchy(self.n_alternatives)
                bv_ratings[voter] = ratings[voter]
                # ratings[voter] = - (original_preferences - original_preferences.mean())
            else:
                scaling = self.rng.lognormal(1., 1.)
                translation = self.rng.normal(0., 10.)
                scaling = 1e-1 if scaling == 0 else scaling
                # print("voter {}'s scalings: {}".format(voter, (scaling, translation)))
                ratings[voter] = scaling * original_preferences + translation
                bv_ratings[voter] = ratings[voter]
            # ratings[voter] = (ratings[voter] - ratings[voter].mean()) / ratings[voter].std()
            if mehestan:
                x, y = sorted(ratings[voter, :2])
                ratings[voter] = (ratings[voter] - x) / (y - x)
                # bv_ratings[voter] = (bv_ratings[voter] - bv_ratings[voter].min()) / (
                #             bv_ratings[voter].max() - bv_ratings[voter].min())

                # bv_ratings[voter] = (bv_ratings[voter] - bv_ratings[voter].mean()) / bv_ratings[voter].std()
                # bv_ratings[voter] = self.transformation.apply(bv_ratings[voter])
                bv_ratings[voter] = self.transformation.sparse_apply(bv_ratings[voter], self.mask[voter, :])
            else:
                # ratings[voter] = (ratings[voter] - ratings[voter].min()) / (ratings[voter].max() - ratings[
                # voter].min())

                # ratings[voter] = (ratings[voter] - ratings[voter].mean()) / ratings[voter].std()
                # ratings[voter] = self.transformation.apply(ratings[voter])
                ratings[voter] = self.transformation.sparse_apply(ratings[voter], self.mask[voter, :])

        return ratings, original_preferences, bv_ratings
