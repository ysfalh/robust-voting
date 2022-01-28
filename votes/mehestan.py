""" Implementation of Mehestan """

from operator import imod
import numpy as np
from votes.basic_vote import BasicVote


def find_pair(mask, weights, ratings):
    """ find the most rated pair of alternatives in the mask """

    def _count_ratings(mask, i, j, weights, ratings):
        """ counts nb of voters voting for both i and j """
        return sum((mask[:, i] & mask[:, j] & ((ratings[:, i] - ratings[:, j]) != 0)) * weights)

    best_weight, best_i, best_j = 0, 0, 0
    for i in range(len(mask[0]) - 1):
        for j in range(i + 1, len(mask[0])):
            new_weight = _count_ratings(mask, i, j, weights, ratings)
            if new_weight > best_weight:
                best_weight, best_i, best_j = new_weight, i, j
    return best_i, best_j


class Mehestan(BasicVote):
    def __init__(self, ratings, mask, voting_rights, voting_resilience=1, transformation_name='standardization'):
        super().__init__(ratings, mask, voting_rights, voting_resilience, transformation_name=transformation_name)

    def learn_scaling(self, voter, ratings):
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

    def learn_translation(self, voter, ratings, scalings):
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

    def init_mehestan(self):
        """ rescaling using most rated pair of alternatives """
        a, b = find_pair(self.mask, self.voting_rights, self.ratings)
        print("Pair chosen by Mehestan: {}".format((a, b)))
        for voter in range(self.n_voters):
            if self.mask[voter][a] and self.mask[voter][b]:
                x, y = sorted(self.ratings[voter, [a, b]])
                self.ratings[voter] = (self.ratings[voter] - x) / (y - x)
            else:  # if pair not rated by voter
                print('using alternative scaling')
                self.ratings[voter] = self.transformation.sparse_apply(self.ratings[voter], self.mask[voter, :])

    def run(self):
        """ run voting algorithm """

        # mehestan normalisation
        self.init_mehestan()

        out = np.zeros(self.n_alternatives)
        scalings = [self.learn_scaling(voter, self.ratings) for voter in range(self.n_voters)]
        translations = [self.learn_translation(voter, self.ratings, scalings) for voter in range(self.n_voters)]

        for voter in range(self.n_voters):
            self.ratings[voter, :] = scalings[voter] * self.ratings[voter, :] + translations[voter]

        for alternative in range(self.n_alternatives):
            scores = np.array([x for voter, x in enumerate(self.ratings[:, alternative])
                               if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.qr_median(scores, weights)

        return out
