""" voting rights """

import numpy as np


def generate_voting_rights(n_voters, p_byzantine, rng=None):
    """ generate random voting rights 
    
    p_byzantine : importance of the byzantine voter
    """
    voting_rights = rng.random(n_voters)

    byzantine = n_voters - 1  # last position
    total_honest_rights = sum(voting_rights[:byzantine])
    voting_rights[byzantine] = total_honest_rights * p_byzantine / (1 - p_byzantine)

    return voting_rights


def get_density(mask, voting_rights):
    """ returns weighted density of mask excluding the byzantine user (last position) """
    n, m = mask.shape
    weighted = np.sum((mask * voting_rights.reshape((n, 1)))[:-1])
    return weighted / (np.sum(voting_rights[:-1]) * m)


def regularize_voting_rights(
        original_preferences, voting_rights, mask, 
        voting_resilience=1, sm3=0, sm4=0,
        rng=None
    ):
    """ change voting rights to be closer to (SM3) and (SM4) 
    
    sm3, sm4 : between 0 and 1, how much to repect the conditions
    """
    n_voters = len(voting_rights)
    byzantine = n_voters - 1  # last position (last user)
    n_honest = n_voters - 1
    total_byzantine_rights = voting_rights[byzantine]
    total_voting_rights = sum(voting_rights)
    total_honest_rights = total_voting_rights - total_byzantine_rights
    x, y = sorted(original_preferences[:2])  # TODO: Should change depending on pair found by init_mehestan
    tmp_1 = 1 / (y - x)
    tmp_2 = 0
    for i in range(n_voters):
        for j in range(n_voters):
            if j > i and original_preferences[j] != original_preferences[i]:
                tmp_2 = max(tmp_2, abs(tmp_1 - 1 / (abs(original_preferences[j] - original_preferences[i]))))
    tmp = (max(original_preferences) - min(original_preferences)) * max(tmp_1, tmp_2)
    safe_margin = 1e-6
    w_zero = voting_resilience * tmp + safe_margin
    # print("W_0: {}".format(w_zero))

    # ---------- SM3 ----------------------
    alpha = sm3 * w_zero / (total_honest_rights - 0.5 * total_voting_rights)
    if alpha > 0:   # FIXME superior as 0 or 1 ??
        voting_rights = np.array(voting_rights) * alpha  # to ensure condition (iii)  # FIXME why not only on the (a,b) pair ?
    total_byzantine_rights = voting_rights[byzantine]
    total_voting_rights = sum(voting_rights)
    total_honest_rights = total_voting_rights - total_byzantine_rights
    print("Condition (iii): {}".format(total_honest_rights >= 0.5 * total_voting_rights + sm3 * w_zero - safe_margin))

    # --------- SM4 -----------------
    cnd_four = True
    n, m = mask.shape
    for j in range(m):  # for each alternative
        local_honest_rights = sum([x for i, x in enumerate(voting_rights)
                                   if mask[i, j] != 0 and i < n_honest])
        local_byzantine_rights = sum(
            [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i == byzantine])
        bol = (local_honest_rights >= local_byzantine_rights + w_zero * sm4)
        honest_pool = [i for i, _ in enumerate(voting_rights) if mask[i, j] == 0 and i < byzantine]
        rng.shuffle(honest_pool)
        remaining_honests = len(honest_pool)
        while remaining_honests != 0 and not bol:  # while (SM4) not verified for this alternative
            rand_honest = honest_pool[len(honest_pool) - remaining_honests]
            remaining_honests += -1
            mask[rand_honest, j] = 1
            local_honest_rights = sum(
                [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i < byzantine])
            local_byzantine_rights = sum(
                [x for i, x in enumerate(voting_rights) if mask[i, j] != 0 and i == byzantine])
            bol = (local_honest_rights >= local_byzantine_rights + w_zero * sm4)

        cnd_four = (cnd_four and bol)

    print("Condition (iv): {}".format(cnd_four))
    print('weighted empirical density :', get_density(mask, voting_rights))

    return voting_rights, mask
