""" Implementation of MajJudement """

import numpy as np
import multiprocess as mp

from utils.optimizers import derivate
from votes.basic_vote import BasicVote


class MajJudement(BasicVote):
    def __init__(self, ratings, mask, voting_rights, voting_resilience=0, transformation_name='standardization',
                 n_proc=1, deltas=[]):
        super().__init__(ratings, mask, voting_rights, voting_resilience, transformation_name=transformation_name,
                         n_proc=n_proc, deltas=deltas)

    def qr_median(self, scores, weights, voting_resilience=0, default_val=0., deltas=[], opt_name="dichotomy"):
        if len(scores) == 0:
            return default_val
        bounds = ((min(0, scores.min()), max(0, scores.max())),)
        optimizer = BasicVote.NAME2OPT[opt_name](tolerance=1e-9, max_iter=100)
        derivative = None
        if opt_name == "dichotomy":
            derivative = lambda x: derivate(x, weights, scores, deltas, voting_resilience, default_val=default_val)
        function = None
        if opt_name == "shelf":
            function = lambda x: (
                    weights.T @ np.abs(x - scores)).sum()
        # if opt_name == "dichotomy", the argument 'function' passed to minimize is useless
        out = optimizer.minimize(function=function, derivative=derivative, bounds=bounds)

        return out

    def run(self):
        """ run voting algorithm """
        pool = mp.Pool(self.n_proc)
        out = self.multi_compute_global_scores(pool)

        return out
