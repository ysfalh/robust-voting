""" Implementation of MajJudement """

import numpy as np
from votes.vote import Vote
from utils.optimizers import ShelfOptimizer, Dichotomy, derivate


class MajJudement(Vote):
    NAME2OPT = {
        "shelf": ShelfOptimizer,
        "dichotomy": Dichotomy
    }

    def __init__(self, ratings, mask, voting_rights, n_proc=1):
        super().__init__(ratings, mask, voting_rights)
        self.n_proc = n_proc

    def median(self, scores, weights, default_val=0., opt_name="shelf"):
        bounds = ((min(0, min(scores)), max(0, max(scores))),)
        optimizer = MajJudement.NAME2OPT[opt_name](tolerance=1e-9, max_iter=100)
        derivative = None
        if opt_name == "dichotomy":
            derivative = lambda x: derivate(x, weights, scores, 1e-15, 0., default_val=default_val)
        function = None
        if opt_name == "shelf":
            function = lambda x: (weights.T @ np.abs(x - scores)).sum()
        # if opt_name == "dichotomy", the argument 'function' passed to minimize is useless
        out = optimizer.minimize(function=function, derivative=derivative, bounds=bounds)

        return out

    def run(self):
        """ run voting algorithm """
        out = np.zeros(self.n_alternatives)

        for alternative in range(self.n_alternatives):
            scores = np.array(
                [x for voter, x in enumerate(self.ratings[:, alternative]) if
                 self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.median(scores, weights)

        return out
