""" Implementation of BasicVote """
import numpy as np
from votes.vote import Vote
from utils.transformations import AffineTransform
from utils.optimizers import ShelfOptimizer, Dichotomy, derivate
import multiprocess as mp


class BasicVote(Vote):
    NAME2OPT = {
        "shelf": ShelfOptimizer,
        "dichotomy": Dichotomy
    }

    def __init__(self, ratings, mask, voting_rights, voting_resilience=1,
                 transformation_name='standardization', n_proc=1, deltas=[]):
        super().__init__(ratings, mask, voting_rights)
        self.voting_resilience = voting_resilience  # W in the paper
        self.transformation = AffineTransform(name=transformation_name)
        self.n_proc = n_proc
        self.deltas = deltas

    def qr_median(self, scores, weights, voting_resilience=None, default_val=0., deltas=[], opt_name="dichotomy"):
        if voting_resilience is None:
            voting_resilience = self.voting_resilience

        bounds = ((min(0, min(scores)), max(0, max(scores))),)
        optimizer = BasicVote.NAME2OPT[opt_name](tolerance=1e-9, max_iter=100)
        derivative = None
        if opt_name == "dichotomy":
            derivative = lambda x: derivate(x, weights, scores, deltas, voting_resilience, default_val=default_val)
        function = None
        if opt_name == "shelf":
            function = lambda x: 0.5 * voting_resilience * (x - default_val) ** 2 + (
                    weights.T @ np.abs(x - scores)).sum()
        # if opt_name == "dichotomy", the argument 'function' passed to minimize is useless
        out = optimizer.minimize(function=function, derivative=derivative, bounds=bounds)

        return out

    def trimmed_mean(self, scores, weights, p_byzantine=0.49):
        # TODO: implement weighted trimmed mean given p_byzantine < 0.5
        return

    def __compute_global_scores(self, alternatives_list, deltas=None, noreg=False):
        # deltas = 1e-10 if noreg else deltas
        deltas = self.deltas # FIXME
        out = []
        voting_resilience = 0. if noreg else self.voting_resilience

        for alternative in alternatives_list:
            scores = np.array([x for voter, x in enumerate(self.ratings[:, alternative])
                               if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out.append(self.qr_median(scores, weights, deltas=deltas, voting_resilience=voting_resilience))

        return out

    def multi_compute_global_scores(self, pool, noreg=False):
        n_proc = pool._processes
        alternatives_lists = [range(p * self.n_alternatives // n_proc, (p + 1) * self.n_alternatives // n_proc) for p in
                              range(n_proc)]

        def f(x):
            return self.__compute_global_scores(x, deltas=self.deltas, noreg=noreg)

        out = sum(pool.map(f, alternatives_lists), [])
        out = np.array(out).flatten()

        return out

    def run(self, noreg=True):
        """ run voting algorithm """
        # Basic vote normalisation
        pool = mp.Pool(self.n_proc)
        for voter in range(self.n_voters):
            self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

        out = self.multi_compute_global_scores(pool)
        out_noreg = self.multi_compute_global_scores(pool, noreg=noreg) if noreg else np.zeros(
            self.n_alternatives)

        return out, out_noreg

# def qr_median(scores, weights, voting_resilience=None, default_val=0., opt_name="dichotomy"):
#     delta = 1e-6
#     bounds = ((min(0, min(scores)), max(0, max(scores))),)
#     optimizer = BasicVote.NAME2OPT[opt_name](tolerance=1e-9, max_iter=100)
#     function = None
#     derivative = None
#     if opt_name == "dichotomy":
#         derivative = lambda x: derivate(x, weights, scores, delta, voting_resilience, default_val=default_val)
#     if opt_name == "shelf":
#         function = lambda x: 0.5 * voting_resilience * (x - default_val) ** 2 + (
#                 weights.T @ np.abs(x - scores)).sum()
#     # if opt_name == "dichotomy", the argument 'function' passed to minimize is useless
#     out = optimizer.minimize(function=function, derivative=derivative, bounds=bounds)
#
#     return out


# scores = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.01863407e-10,
#                    -5.35047716e+00, -3.08402684e+00, -7.37334860e+00, -4.48877521e+01,
#                    2.05494900e+00, -2.28660619e+00, -6.99886337e+00, -8.32755864e+00])
# weights = np.array([0.42111545, 0.42111545, 0.42111545, 0.42111545, 0.42111545, 0.42111545,
#                     0.42111545, 0.42111545, 0.42111545, 0.42111545, 0.18047805, 0.18047805,
#                     0.18047805, 0.18047805, 0.18047805, 0.18047805, 0.18047805, 0.18047805,
#                     0.18047805, 0.18047805])
#
# out_1 = qr_median(scores, weights, voting_resilience=1., default_val=0., opt_name="dichotomy")
# out_2 = qr_median(scores, weights, voting_resilience=1., default_val=0., opt_name="shelf")
# print("Dichotomy output: {}".format(out_1))
# print("Shelf output: {}".format(out_2))
