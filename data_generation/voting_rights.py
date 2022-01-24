""" voting rights """

import numpy as np

def generate_voting_rights(n_voters, p_byzantine, rng=None):
    """ generate random voting rights 
    
    p_byzantine : importance of the byzantine voter
    """
    voting_rights = rng.lognormal(1., 0.5, n_voters)
    voting_rights = [(i + 1 if x == 0 else x) for i, x in enumerate(voting_rights)]

    byzantine = -1  # last position
    total_honest_rights = sum(voting_rights[:-1])
    voting_rights[byzantine] = total_honest_rights * p_byzantine / (1 - p_byzantine)

    return voting_rights


def regularize_voting_rights(ratings, voting_rights, mask, voting_resilience=1, rng=None):
        byzantine = -1  # last position (last user)
        n_voters = len(ratings)
        n_honest = n_voters - 1
        total_byzantine_rights = voting_rights[byzantine]
        total_voting_rights = sum(voting_rights)
        total_honest_rights = total_voting_rights - total_byzantine_rights
        # print("Byzantine voting rights ratio: {}%".format(total_byzantine_rights / total_voting_rights))
        x, y = sorted(ratings[0, :2])
        w_zero = voting_resilience * (max(ratings[0]) - min(ratings[0])) / (y - x)  # same as original_preferences
        alpha = w_zero / (total_honest_rights - 0.5 * total_voting_rights)
        if alpha > 0:
            voting_rights = np.array(voting_rights) * alpha       # to ensure condition (iii)
        total_byzantine_rights = voting_rights[byzantine]
        total_voting_rights = sum(voting_rights)
        total_honest_rights = total_voting_rights - total_byzantine_rights
        print("Condition (iii): {}".format(total_honest_rights >= 0.5 * total_voting_rights + w_zero))
        cnd_four = True
        n, m = mask.shape
        for j in range(m):
            local_honest_rights = sum([x for i, x in enumerate(voting_rights)
                                       if mask[i, j] != 0 and i < n_honest])
            local_byzantine_rights = sum(
                [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i >= n_honest])
            bol = (local_honest_rights >= local_byzantine_rights + w_zero)
            honest_pool = [i for i, _ in enumerate(voting_rights) if mask[i, j] == 0 and i < n_honest]
            rng.shuffle(honest_pool)
            remaining_honests = len(honest_pool)
            while remaining_honests != 0 and not bol:
                # print("honest_pool = {}".format(honest_pool))
                rand_honest = honest_pool[len(honest_pool) - remaining_honests]
                remaining_honests += -1
                mask[rand_honest, j] = 1       # to ensure condition (iv)
                local_honest_rights = sum(
                    [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i < n_honest])
                local_byzantine_rights = sum(
                    [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i >= n_honest])
                bol = (local_honest_rights >= local_byzantine_rights + w_zero)

            cnd_four = (cnd_four and bol)

        print("Condition (iv): {}".format(cnd_four))
        return voting_rights, mask