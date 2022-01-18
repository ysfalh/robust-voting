""" Implementation of Mehestan """
from votes.basic_vote import BasicVote
import numpy as np


class Mehestan(BasicVote):
    def __init__(self, n_alternatives, n_voters, density, p_byzantine=.05, byz_density=.5, voting_resilience=1.,
                 transformation_name="standardization", random_mask=False, seed=123):
        super().__init__(n_alternatives=n_alternatives, n_voters=n_voters, density=density, p_byzantine=p_byzantine,
                         byz_density=byz_density, random_mask=random_mask, voting_resilience=voting_resilience,
                         transformation_name=transformation_name, seed=seed)
        self.ratings = []
        self.bv_ratings = []
        self.original_preferences = []

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

        out = self.qr_median(np.array(scores), np.array(weights), voting_resilience=self.voting_resilience * max(ratings[voter]))
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

        out = self.qr_median(np.array(scores), np.array(weights))
        return out

    def regularize_voting_rights(self, ratings):
        byzantine = -1
        n_honest = self.n_voters - 1
        total_byzantine_rights = self.voting_rights[byzantine]
        total_voting_rights = sum(self.voting_rights)
        total_honest_rights = total_voting_rights - total_byzantine_rights
        # print("Byzantine voting rights ratio: {}%".format(total_byzantine_rights / total_voting_rights))
        x, y = sorted(ratings[0, :2])
        w_zero = self.voting_resilience * (max(ratings[0]) - min(ratings[0])) / (y - x)
        alpha = w_zero / (total_honest_rights - 0.5 * total_voting_rights)
        if alpha > 0:
            self.voting_rights = np.array(self.voting_rights) * alpha       # to ensure condition (iii)
        total_byzantine_rights = self.voting_rights[byzantine]
        total_voting_rights = sum(self.voting_rights)
        total_honest_rights = total_voting_rights - total_byzantine_rights
        print("Condition (iii): {}".format(total_honest_rights >= 0.5 * total_voting_rights + w_zero))
        cnd_four = True
        n, m = self.mask.shape
        for j in range(m):
            local_honest_rights = sum([x for i, x in enumerate(self.voting_rights)
                                       if self.mask[i, j] != 0 and i < n_honest])
            local_byzantine_rights = sum(
                [x for i, x in enumerate(self.voting_rights) if self.mask[i, j] != 0 and i >= n_honest])
            bol = (local_honest_rights >= local_byzantine_rights + w_zero)
            # print("j: {} bol: {}".format(j, bol))
            honest_pool = [i for i, _ in enumerate(self.voting_rights) if self.mask[i, j] == 0 and i < n_honest]
            self.rng.shuffle(honest_pool)
            remaining_honests = len(honest_pool)
            while remaining_honests != 0 and not bol:
                # print("honest_pool = {}".format(honest_pool))
                rand_honest = honest_pool[len(honest_pool) - remaining_honests]
                remaining_honests += -1
                self.mask[rand_honest, j] = 1       # to ensure condition (iv)
                local_honest_rights = sum(
                    [x for i, x in enumerate(self.voting_rights) if self.mask[i, j] != 0 and i < n_honest])
                local_byzantine_rights = sum(
                    [x for i, x in enumerate(self.voting_rights) if self.mask[i, j] != 0 and i >= n_honest])
                bol = (local_honest_rights >= local_byzantine_rights + w_zero)

            cnd_four = (cnd_four and bol)

        print("Condition (iv): {}".format(cnd_four))
        print("mask: {}".format(self.mask))


    def run(self):
        self.generate_voting_rights()
        self.ratings, self.original_preferences, self.bv_ratings = self.generate_data_from_unanimity(mehestan=True)
        out = np.zeros(self.n_alternatives)
        scalings = [self.learn_scaling(voter, self.ratings) for voter in range(self.n_voters)]
        translations = [self.learn_translation(voter, self.ratings, scalings) for voter in range(self.n_voters)]

        for voter in range(self.n_voters):
            self.ratings[voter, :] = scalings[voter] * self.ratings[voter, :] + translations[voter]
        self.regularize_voting_rights(self.ratings)

        for alternative in range(self.n_alternatives):
            scores = np.array([x for voter, x in enumerate(self.ratings[:, alternative])
                               if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            weights = np.array(
                [x for voter, x in enumerate(self.voting_rights) if self.mask[voter][alternative] != 0]).reshape(-1, 1)
            out[alternative] = self.qr_median(scores, weights)

        # print("ratings: {}".format(ratings))

        return out, self.original_preferences
