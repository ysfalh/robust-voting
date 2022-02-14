""" Implementation of Mehestan """
import numpy as np
from votes.basic_vote import BasicVote
import multiprocess as mp


def cl_mean(scores, weights, center, radius):
    clip_scores = np.clip(scores, center - radius, center + radius)
    if sum(weights) == 0:
        return center
    out = (weights.T @ clip_scores) / sum(weights)
    return out


class Mehestan(BasicVote):
    def __init__(self, ratings, mask, voting_rights, voting_resilience=1, transformation_name='standardization',
                 n_proc=1, deltas=[]):
        super().__init__(ratings, mask, voting_rights, voting_resilience, transformation_name=transformation_name,
                         n_proc=n_proc, deltas=deltas)

    def br_mean(self, scores, weights, voting_resilience=None, deltas=[], opt_name="dichotomy"):
        if voting_resilience is None:
            voting_resilience = self.voting_resilience

        center = self.qr_median(scores, weights, voting_resilience=4 * voting_resilience, deltas=deltas,
                                opt_name=opt_name)
        radius = sum(weights) / (4 * voting_resilience) if voting_resilience != 0 else 1e9
        out = cl_mean(scores, weights, center, radius)
        return out

    def __learn_scaling(self, voter, ratings, deltas):
        scores, weights = [], []
        voter_alters = [i for i, x in enumerate(self.mask[voter, :]) if x == 1]

        for voterbis in range(self.n_voters):
            voterbis_alters = [i for i, x in enumerate(self.mask[voterbis, :]) if x == 1]
            alter_inter = [i for i in voterbis_alters if i in voter_alters]
            if len(alter_inter) < 2:
                continue
            r, s = 0, 0
            for i in range(len(alter_inter) - 1):
                a = alter_inter[i]
                for j in range(i + 1, len(alter_inter)):
                    b = alter_inter[j]
                    if (ratings[voter, b] != ratings[voter, a]) and (ratings[voterbis, b] != ratings[voterbis, a]):
                        s += 1
                        r += abs((ratings[voterbis, b] - ratings[voterbis, a]) / (ratings[voter, b] - ratings[voter, a]))

            r = r / s if s != 0 else 0
            scores.append(r - 1)
            weights.append(self.voting_rights[voterbis])

        if len(scores) == 0:
            return 1.

        out = 1 + self.br_mean(np.array(scores), np.array(weights), deltas=deltas,
                               voting_resilience=self.voting_resilience * max(np.abs(ratings[voter])))
        return out

    def __learn_translation(self, voter, ratings, scalings, deltas):
        scores, weights = [], []
        voter_alters = [i for i, x in enumerate(self.mask[voter, :]) if x == 1]

        for voterbis in range(self.n_voters):
            voterbis_alters = [i for i, x in enumerate(self.mask[voterbis, :]) if x == 1]
            alter_inter = [i for i in voterbis_alters if i in voter_alters]
            if len(alter_inter) < 1:
                continue
            r = 0
            for a in alter_inter:
                r += scalings[voterbis] * ratings[voterbis, a] - scalings[voter] * ratings[voter, a]
            r = r / len(alter_inter)
            scores.append(r)
            weights.append(self.voting_rights[voterbis])

        if len(scores) == 0:
            return 0.

        out = self.br_mean(np.array(scores), np.array(weights), deltas=deltas)
        return out

    def __compute_scalings(self, voter_list):
        return [self.__learn_scaling(voter, self.ratings, self.deltas) for voter in voter_list]

    def __compute_translations(self, voter_list, scalings):
        return [self.__learn_translation(voter, self.ratings, scalings, self.deltas) for voter in voter_list]

    def __compute_factors(self, pool):
        n_proc = pool._processes
        voter_lists = [range(p * self.n_voters // n_proc, (p + 1) * self.n_voters // n_proc) for p in range(n_proc)]
        scalings = sum(pool.map(self.__compute_scalings, [voter_lists[p] for p in range(n_proc)]), [])
        translations = sum(
            pool.starmap(self.__compute_translations, [(voter_lists[p], scalings) for p in range(n_proc)]), [])

        return scalings, translations

    def __init_mehestan(self):
        """ pre-normalization """
        for voter in range(self.n_voters):
            self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

    def run(self):
        """ run voting algorithm """
        pool = mp.Pool(self.n_proc)
        self.__init_mehestan()

        scalings, translations = self.__compute_factors(pool)

        for voter in range(self.n_voters):
            self.ratings[voter, :] = scalings[voter] * self.ratings[voter, :] + translations[voter]

        out = self.multi_compute_global_scores(pool)

        return out
