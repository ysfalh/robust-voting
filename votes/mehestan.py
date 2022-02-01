""" Implementation of Mehestan """
import numpy as np
from votes.basic_vote import BasicVote
import multiprocess as mp


def find_pair(mask, weights, ratings, alt_list=None):
    """ find the most rated pair of alternatives in the mask """

    def _count_ratings(mask, i, j, weights, ratings):
        """ counts nb of voters voting for both i and j """
        return sum((mask[:, i] & mask[:, j] & ((ratings[:, i] - ratings[:, j]) != 0)) * weights)

    best_weight, best_i, best_j = 0, 0, 0
    n_alternatives = mask.shape[1]
    if alt_list is None:
        alt_list = range(n_alternatives-1)

    for i in alt_list:
        for j in range(i + 1, n_alternatives):
            new_weight = _count_ratings(mask, i, j, weights, ratings)
            if new_weight > best_weight:
                best_weight, best_i, best_j = new_weight, i, j
    return best_i, best_j, best_weight


def multi_find_pair(mask, weights, ratings, pool):
    n_proc = pool._processes
    n_alternatives = mask.shape[1]
    alternatives_lists = [range(p * n_alternatives // n_proc, (p + 1) * n_alternatives // n_proc) for p in
                          range(n_proc)]

    def f(alt_list):
        return find_pair(mask, weights, ratings, alt_list=alt_list)

    out = max(pool.map(f, alternatives_lists), key=lambda item: item[2])[:2]
    return out


class Mehestan(BasicVote):
    def __init__(self, ratings, mask, voting_rights, voting_resilience=1, transformation_name='standardization',
                 n_proc=1):
        super().__init__(ratings, mask, voting_rights, voting_resilience, transformation_name=transformation_name,
                         n_proc=n_proc)

    def __learn_scaling(self, voter, ratings):
        scores, weights = [], []
        voter_alters = [i for i, x in enumerate(self.mask[voter, :]) if x == 1]

        for voterbis in range(self.n_voters):
            voterbis_alters = [i for i, x in enumerate(self.mask[voterbis, :]) if x == 1]
            alter_inter = [i for i in voterbis_alters if i in voter_alters]
            if len(alter_inter) < 2:
                continue
            a, b = alter_inter[:2]
            r = abs((ratings[voterbis, b] - ratings[voterbis, a]) / (ratings[voter, b] - ratings[voter, a]))
            scores.append(r)
            weights.append(self.voting_rights[voterbis])

        if len(scores) == 0:
            return 1.

        out = self.qr_median(np.array(scores), np.array(weights), voting_resilience=self.voting_resilience * max(
            np.abs(ratings[voter])), default_val=1.)
        # print("S_n: {}".format(out * (max(ratings[0]) - min(ratings[0])) / abs(ratings[0, 0]-ratings[0, 1])))
        return out

    def __learn_translation(self, voter, ratings, scalings):
        scores, weights = [], []
        voter_alters = [i for i, x in enumerate(self.mask[voter, :]) if x == 1]

        for voterbis in range(self.n_voters):
            voterbis_alters = [i for i, x in enumerate(self.mask[voterbis, :]) if x == 1]
            alter_inter = [i for i in voterbis_alters if i in voter_alters]
            if len(alter_inter) < 1:
                continue
            for a in alter_inter:
                r = scalings[voterbis] * ratings[voterbis, a] - scalings[voter] * ratings[voter, a]
                scores.append(r)
                weights.append(self.voting_rights[voterbis] / len(alter_inter))

        if len(scores) == 0:
            return 0.

        out = self.qr_median(np.array(scores), np.array(weights))
        # S = scalings[voter] * (max(ratings[0]) - min(ratings[0])) / abs(ratings[0, 0]-ratings[0, 1])
        return out

    def __compute_scalings(self, voter_list):
        return [self.__learn_scaling(voter, self.ratings) for voter in voter_list]

    def __compute_translations(self, voter_list, scalings):
        return [self.__learn_translation(voter, self.ratings, scalings) for voter in voter_list]

    def __compute_factors(self, pool):
        n_proc = pool._processes
        voter_lists = [range(p * self.n_voters // n_proc, (p + 1) * self.n_voters // n_proc) for p in range(n_proc)]
        scalings = sum(pool.map(self.__compute_scalings, [voter_lists[p] for p in range(n_proc)]), [])
        translations = sum(
            pool.starmap(self.__compute_translations, [(voter_lists[p], scalings) for p in range(n_proc)]), [])

        return scalings, translations

    def __init_mehestan(self, pool):
        """ rescaling using most rated pair of alternatives """
        # a, b, _ = find_pair(self.mask, self.voting_rights, self.ratings)
        a, b = multi_find_pair(self.mask, self.voting_rights, self.ratings, pool)
        # print("Pair chosen by Mehestan: {}".format((a, b)))
        for voter in range(self.n_voters):
            if self.mask[voter][a] and self.mask[voter][b]:
                x, y = sorted(self.ratings[voter, [a, b]])
                self.ratings[voter] = (self.ratings[voter] - x) / (y - x)
            else:  # if pair not rated by voter
                # print('using alternative scaling')
                self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

    def run(self):
        """ run voting algorithm """
        pool = mp.Pool(self.n_proc)
        # mehestan normalisation
        self.__init_mehestan(pool)

        scalings, translations = self.__compute_factors(pool)

        for voter in range(self.n_voters):
            self.ratings[voter, :] = scalings[voter] * self.ratings[voter, :] + translations[voter]

        out = self.multi_compute_global_scores(pool)

        return out
